"""Benchmark cache-off versus KV-cached greedy decoding from one checkpoint.

This is an inference-efficiency benchmark, not an accuracy experiment: both
paths must emit identical greedy tokens from the same checkpoint.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from .config import DEFAULT_MODEL_CONFIG, ModelConfig
from .model import MathNetGPT
from .train import choose_device


@torch.no_grad()
def decode_without_cache(model: MathNetGPT, prompt: torch.Tensor, new_tokens: int) -> torch.Tensor:
    output = prompt
    for _ in range(new_tokens):
        logits, _ = model(output[:, -model.config.block_size :], use_cache=False)
        output = torch.cat((output, logits[:, -1].argmax(dim=-1, keepdim=True)), dim=1)
    return output


@torch.no_grad()
def decode_with_cache(model: MathNetGPT, prompt: torch.Tensor, new_tokens: int) -> torch.Tensor:
    output = prompt
    logits, cache = model(output[:, -model.config.block_size :], use_cache=True)
    for _ in range(new_tokens):
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        output = torch.cat((output, next_token), dim=1)
        assert cache is not None
        if cache[0][0].size(2) >= model.config.block_size:
            logits, cache = model(output[:, -model.config.block_size :], use_cache=True)
        else:
            logits, cache = model(next_token, cache, use_cache=True)
    return output


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_decode(
    decoder: Callable[[MathNetGPT, torch.Tensor, int], torch.Tensor],
    model: MathNetGPT,
    prompt: torch.Tensor,
    new_tokens: int,
    trials: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    durations: list[float] = []
    peak_bytes: list[int] = []
    output: torch.Tensor | None = None
    for _ in range(trials):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
        synchronize(device)
        started = time.perf_counter()
        output = decoder(model, prompt, new_tokens)
        synchronize(device)
        durations.append(time.perf_counter() - started)
        if device.type == "cuda":
            peak_bytes.append(max(0, torch.cuda.max_memory_allocated(device) - before))
    assert output is not None
    median_seconds = statistics.median(durations)
    return output, {
        "median_seconds": median_seconds,
        "tokens_per_second": new_tokens / median_seconds,
        "peak_additional_memory_bytes": float(max(peak_bytes, default=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to run-dir/checkpoints/best.pt")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup-trials", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = args.checkpoint or args.run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device, _ = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint.get("run_config", {}).get("model")
    config = DEFAULT_MODEL_CONFIG if raw_config is None else ModelConfig(**raw_config)
    if not 1 <= args.prompt_tokens <= config.block_size:
        raise ValueError(f"prompt-tokens must be between 1 and {config.block_size}")
    if args.new_tokens < 1:
        raise ValueError("new-tokens must be positive")
    model = MathNetGPT(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    prompt = torch.randint(
        0,
        config.vocab_size,
        (1, args.prompt_tokens),
        generator=torch.Generator(device=device).manual_seed(args.seed),
        device=device,
    )

    for _ in range(args.warmup_trials):
        decode_without_cache(model, prompt, args.new_tokens)
        decode_with_cache(model, prompt, args.new_tokens)
    uncached_output, uncached = time_decode(
        decode_without_cache, model, prompt, args.new_tokens, args.trials, device
    )
    cached_output, cached = time_decode(
        decode_with_cache, model, prompt, args.new_tokens, args.trials, device
    )
    tokens_match = bool(torch.equal(uncached_output, cached_output))
    if not tokens_match:
        raise AssertionError("Cached and uncached greedy decoding produced different tokens.")
    result: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "variant": checkpoint.get("run_config", {}).get("variant", "pre_rms_mha"),
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "trials": args.trials,
        "device": str(device),
        "cache_off": uncached,
        "cache_on": cached,
        "tokens_match": tokens_match,
        "cache_speedup": cached["tokens_per_second"] / uncached["tokens_per_second"],
    }
    output_path = args.run_dir / "decode_benchmark.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(parse_args()), indent=2))
