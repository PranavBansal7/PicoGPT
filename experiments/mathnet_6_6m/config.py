"""Immutable defaults for the MathNet 6.6M experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Configuration chosen to keep the tied-weight GPT near 6.6M parameters."""

    vocab_size: int = 8_192
    block_size: int = 256
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 8
    intermediate_size: int = 960
    dropout: float = 0.10
    bias: bool = False

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head dimension must be even for the chosen architecture")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """The requested fixed-update training budget for a Colab GPU."""

    epochs: int = 41
    steps_per_epoch: int = 100
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.10
    grad_clip: float = 1.0
    eval_batches: int = 20
    seed: int = 13_337

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["total_steps"] = self.total_steps
        values["effective_batch_size"] = self.effective_batch_size
        values["planned_target_tokens"] = (
            self.total_steps * self.effective_batch_size * DEFAULT_MODEL_CONFIG.block_size
        )
        return values


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()

# This is an analytic count for a tied token embedding / output projection.
# The runtime test verifies it against model.parameters().
EXPECTED_PARAMETER_COUNT = 6_622_464

