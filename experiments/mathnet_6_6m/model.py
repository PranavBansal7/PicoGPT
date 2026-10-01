"""A compact, modern decoder-only GPT for the isolated MathNet experiment.

The implementation intentionally uses PyTorch primitives rather than a
``transformers`` model.  It includes Pre-LN residual blocks, RMSNorm, standard
multi-head causal self-attention, Flash/SDPA-compatible attention, tied input
and output embeddings, and an inference-time per-layer KV cache.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Optional, Sequence, TypeAlias

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import ModelConfig


PastKeyValue: TypeAlias = tuple[torch.Tensor, torch.Tensor]
PastKeyValues: TypeAlias = tuple[PastKeyValue, ...]


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel scale."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Accumulate normalization statistics in float32 for mixed precision.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=input_dtype)
        return normalized * self.weight.to(dtype=input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs, as used by rotary positional embeddings."""
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    """RoPE applied to queries and keys without adding learned position weights."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("rotary head dimensions must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.outer(positions.float(), self.inv_freq)
        angles = torch.repeat_interleave(frequencies, repeats=2, dim=-1)
        cos = angles.cos()[None, None, :, :].to(dtype=q.dtype)
        sin = angles.sin()[None, None, :, :].to(dtype=q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    """Fused-QKV, multi-head causal self-attention with an optional KV cache."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(self.head_dim)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[PastKeyValue]]:
        batch_size, tokens, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        shape = (batch_size, tokens, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q, k = self.rope(q, k, positions)

        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.shape[:2] != (batch_size, self.n_head):
                raise ValueError("KV cache batch size or head count does not match input")
            past_length = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # SDPA dispatches to a fused GPU kernel when the Colab runtime supports
        # it.  A cache requires an offset causal mask because new queries start
        # after the cached tokens.
        dropout_p = self.dropout if self.training else 0.0
        if past_length == 0:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            total_tokens = past_length + tokens
            causal_mask = torch.ones(tokens, total_tokens, dtype=torch.bool, device=x.device)
            causal_mask = torch.tril(causal_mask, diagonal=past_length)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, dropout_p=dropout_p)

        y = y.transpose(1, 2).contiguous().view(batch_size, tokens, channels)
        present = (k, v) if use_cache else None
        return self.resid_dropout(self.c_proj(y)), present


class MLP(nn.Module):
    """A compact GELU feed-forward sublayer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.intermediate_size
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    """A Pre-LN transformer block using RMSNorm rather than LayerNorm."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_key_value: Optional[PastKeyValue] = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[PastKeyValue]]:
        attention_output, present = self.attn(
            self.rms_1(x), positions, past_key_value, use_cache=use_cache
        )
        x = x + attention_output
        x = x + self.mlp(self.rms_2(x))
        return x, present


class MathNetGPT(nn.Module):
    """6.622M-parameter decoder-only GPT with RoPE and tied embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.rms_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Tying prevents a second 1.34M-parameter output matrix and is common
        # in compact language models.
        self.lm_head.weight = self.wte.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Sequence[PastKeyValue]] = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[PastKeyValues]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, tokens)")
        _, tokens = input_ids.shape
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("a KV cache must contain one entry per transformer block")
        past_length = 0 if past_key_values is None else past_key_values[0][0].size(2)
        if tokens + past_length > self.config.block_size:
            raise ValueError(
                f"sequence length {tokens + past_length} exceeds block size {self.config.block_size}"
            )

        positions = torch.arange(past_length, past_length + tokens, device=input_ids.device)
        x = self.drop(self.wte(input_ids))
        presents: list[PastKeyValue] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            x, present = block(x, positions, past, use_cache=use_cache)
            if use_cache:
                assert present is not None
                presents.append(present)
        logits = self.lm_head(self.rms_f(x))
        return logits, tuple(presents) if use_cache else None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        """Autoregressively sample while reusing keys and values when possible."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        output = input_ids
        context = output[:, -self.config.block_size :]
        logits, cache = self(context, use_cache=True)
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < values[:, [-1]], -torch.inf)
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            output = torch.cat((output, next_token), dim=1)

            # Once the cache would exceed its learned position range, rebuild it
            # from the most recent context window.  Otherwise decode one token.
            assert cache is not None
            cached_tokens = cache[0][0].size(2)
            if cached_tokens >= self.config.block_size:
                context = output[:, -self.config.block_size :]
                logits, cache = self(context, use_cache=True)
            else:
                logits, cache = self(next_token, cache, use_cache=True)
        return output

    def export_config(self) -> dict[str, object]:
        return asdict(self.config)


def count_parameters(config: ModelConfig) -> int:
    """Construct and count parameters; convenient for a reproducibility check."""
    return MathNetGPT(config).parameter_count
