"""Parameter-matched model configurations for the MathNet ablation study."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from .config import DEFAULT_MODEL_CONFIG, ModelConfig
from .model import MathNetGPT


VARIANT_DESCRIPTIONS: Final[dict[str, str]] = {
    "post_ln_mha": "Post-LN, scale-only LayerNorm, standard multi-head attention.",
    "pre_ln_mha": "Pre-LN, scale-only LayerNorm, standard multi-head attention.",
    "pre_rms_mha": "Pre-LN RMSNorm, standard multi-head attention (the current default).",
    "pre_rms_gqa": "Pre-LN RMSNorm, 8 query / 2 KV-head GQA, parameter matched via a wider MLP.",
}


def get_variant(name: str) -> ModelConfig:
    """Return a configuration with the same trainable-parameter budget.

    GQA with two KV heads saves 98,304 attention parameters per block.  Its
    MLP grows from 960 to 1,152 channels, adding exactly the same count back.
    """
    if name == "post_ln_mha":
        return replace(DEFAULT_MODEL_CONFIG, normalization="layernorm", norm_position="post")
    if name == "pre_ln_mha":
        return replace(DEFAULT_MODEL_CONFIG, normalization="layernorm", norm_position="pre")
    if name == "pre_rms_mha":
        return DEFAULT_MODEL_CONFIG
    if name == "pre_rms_gqa":
        return replace(
            DEFAULT_MODEL_CONFIG,
            attention_type="gqa",
            num_kv_heads=2,
            intermediate_size=1_152,
        )
    valid = ", ".join(VARIANT_DESCRIPTIONS)
    raise ValueError(f"Unknown variant '{name}'. Choose one of: {valid}")


def parameter_count(name: str) -> int:
    return MathNetGPT(get_variant(name)).parameter_count


def names() -> tuple[str, ...]:
    return tuple(VARIANT_DESCRIPTIONS)
