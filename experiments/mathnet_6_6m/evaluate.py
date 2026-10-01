"""Run one internal held-out test evaluation plus a KV-cache correctness check."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import DEFAULT_MODEL_CONFIG
from .model import MathNetGPT
from .train import TokenStore, choose_device, evaluate


@torch.no_grad()
def verify_kv_cache(model: MathNetGPT, device: torch.device) -> dict[str, Any]:
    """Cached token-by-token logits must match a single full-sequence forward pass."""
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(2026)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (2, min(32, model.config.block_size)),
        device=device,
        generator=generator,
    )
    full_logits, _ = model(tokens, use_cache=False)
    cache = None
    cached_logits = []
    for position in range(tokens.size(1)):
        step_logits, cache = model(tokens[:, position : position + 1], cache, use_cache=True)
        cached_logits.append(step_logits)
    cached_logits = torch.cat(cached_logits, dim=1)
    max_abs_error = (full_logits - cached_logits).abs().max().float().item()
    if was_training:
        model.train()
    result = {
        "tokens_checked": tokens.size(1),
        "batch_size": tokens.size(0),
        "max_abs_logit_error": max_abs_error,
        "tolerance": 2e-4,
        "passed": max_abs_error <= 2e-4,
    }
    if not result["passed"]:
        raise AssertionError(f"KV cache logits diverge from full forward: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to run-dir/checkpoints/best.pt")
    parser.add_argument("--test-batches", type=int, default=500)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint or args.run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    output_path = args.run_dir / "test_results.json"
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Test evaluation is intentionally one-time; "
            "delete it only if you are explicitly re-running the experiment."
        )
    device, amp_dtype = choose_device(args.device)
    model = MathNetGPT(DEFAULT_MODEL_CONFIG).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    token_store = TokenStore(args.data_dir, DEFAULT_MODEL_CONFIG.block_size)
    metrics = evaluate(
        model,
        token_store,
        split="test",
        batches=args.test_batches,
        micro_batch_size=args.micro_batch_size,
        device=device,
        amp_dtype=amp_dtype,
        seed=91_001,
    )
    cache_check = verify_kv_cache(model, device)
    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch_completed"],
        "checkpoint_global_step": checkpoint["global_step"],
        "metric_scope": "PicoGPT MathNet-v0 internal held-out next-token LM test; not MathNet-Solve accuracy.",
        "test_batches": args.test_batches,
        "micro_batch_size": args.micro_batch_size,
        "test_loss": metrics["loss"],
        "test_perplexity": metrics["perplexity"],
        "test_tokens": int(metrics["tokens"]),
        "kv_cache_check": cache_check,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_path = args.run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "status": "complete",
                "test_loss": result["test_loss"],
                "test_perplexity": result["test_perplexity"],
                "test_tokens": result["test_tokens"],
                "kv_cache_check": cache_check,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
