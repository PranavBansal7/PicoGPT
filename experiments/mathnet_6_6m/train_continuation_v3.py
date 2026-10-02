"""Train the 6.622M MathNet GPT on a Colab GPU and persist every result.

An "epoch" here is intentionally a *logical epoch*: exactly 100 optimizer
updates.  The requested 41 logical epochs therefore equal 4,100 updates, not
41 guaranteed full passes through the source corpus.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch
from torch.nn import functional as F

from .config import DEFAULT_MODEL_CONFIG, TrainConfig
from .model import MathNetGPT


class TokenStore:
    """Randomly sample contiguous token windows directly from uint16 files."""

    def __init__(self, data_dir: Path, block_size: int) -> None:
        self.block_size = block_size
        self.data: dict[str, np.memmap] = {}
        for split in ("train", "validation", "test"):
            path = data_dir / f"{split}.bin"
            if not path.exists():
                raise FileNotFoundError(f"Missing prepared split: {path}")
            tokens = np.memmap(path, dtype=np.uint16, mode="r")
            if len(tokens) <= block_size:
                raise ValueError(f"{split} has only {len(tokens)} tokens; need more than {block_size}")
            self.data[split] = tokens

    def batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.data[split]
        starts = torch.randint(
            0, len(source) - self.block_size - 1, (batch_size,), generator=generator
        )
        # Copying makes tensors writable and avoids warnings from read-only
        # memmaps.  It is cheap relative to a GPU transformer update.
        x = torch.stack(
            [
                torch.from_numpy(
                    np.array(source[int(index) : int(index) + self.block_size], dtype=np.int64)
                )
                for index in starts
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    np.array(
                        source[int(index) + 1 : int(index) + self.block_size + 1], dtype=np.int64
                    )
                )
                for index in starts
            ]
        )
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        return x.to(device), y.to(device)


class MetricWriter:
    """Append durable JSONL records and materialize a spreadsheet-friendly CSV."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "metrics.jsonl"
        self.handle = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps(record, allow_nan=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def to_csv(self, destination: Path) -> None:
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        fields = sorted({key for row in rows for key in row})
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> tuple[torch.device, Optional[torch.dtype]]:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Enable a GPU in Colab.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return device, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device, None


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]) -> contextlib.AbstractContextManager[None]:
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def make_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:  # PyTorch 2.2 compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


def learning_rate(step: int, config: TrainConfig, schedule_total_steps: int | None = None) -> float:
    total_steps = config.total_steps if schedule_total_steps is None else schedule_total_steps
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if step >= total_steps - 1:
        return config.min_learning_rate
    decay_ratio = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(decay_ratio, 0.0), 1.0)))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


@torch.no_grad()
def evaluate(
    model: MathNetGPT,
    token_store: TokenStore,
    *,
    split: str,
    batches: int,
    micro_batch_size: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    seed: int,
) -> dict[str, float]:
    """Evaluate fixed random windows so epoch-to-epoch validation is comparable."""
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weighted_loss = 0.0
    tokens_seen = 0
    for _ in range(batches):
        x, y = token_store.batch(split, micro_batch_size, device, generator)
        with autocast_context(device, amp_dtype):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        token_count = y.numel()
        weighted_loss += loss.float().item() * token_count
        tokens_seen += token_count
    if was_training:
        model.train()
    mean_loss = weighted_loss / tokens_seen
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0)), "tokens": float(tokens_seen)}


def save_checkpoint(
    path: Path,
    *,
    model: MathNetGPT,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch_completed: int,
    global_step: int,
    best_validation_loss: float,
    run_config: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch_completed": epoch_completed,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "run_config": run_config,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: MathNetGPT,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    # Do not restore the RNG state from the 4,100-step checkpoint.
    # The checkpoint's RNG tensor is uint8, but this runtime rejects it through
    # torch.set_rng_state(). Model/optimizer/scaler state is still restored.
    print(
        "Resuming model/optimizer/scaler from checkpoint; "
        "starting a fresh deterministic RNG stream for continuation.",
        flush=True,
    )

    continuation_seed = 13_337 + 4_100
    set_seed(continuation_seed)
    return (
        int(checkpoint["epoch_completed"]),
        int(checkpoint["global_step"]),
        float(checkpoint["best_validation_loss"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--force-lr", type=float, default=None, help="Force constant LR overriding scheduler (float)")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=41)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13_337)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--resume", type=Path, help="Resume from a checkpoint written by this script.")
    parser.add_argument("--schedule-total-steps", type=int, help="Explicit total-step target for the LR schedule.")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    train_config = TrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_batches=args.eval_batches,
        seed=args.seed,
    )
    schedule_total_steps = (
        train_config.total_steps
        if args.schedule_total_steps is None
        else args.schedule_total_steps
    )
    if schedule_total_steps <= train_config.warmup_steps:
        raise ValueError("schedule total steps must exceed warmup steps")
    print("1/6 Setting seed...", flush=True)
    set_seed(train_config.seed)
    print("2/6 Choosing device...", flush=True)
    device, amp_dtype = choose_device(args.device)
    print(f"   Device: {device}, AMP: {amp_dtype}", flush=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    data_metadata_path = args.data_dir / "data_metadata.json"
    if not data_metadata_path.exists():
        raise FileNotFoundError("Prepare MathNet data before training (data_metadata.json is missing).")
    data_metadata = json.loads(data_metadata_path.read_text(encoding="utf-8"))
    if data_metadata["tokenizer"]["vocab_size"] != DEFAULT_MODEL_CONFIG.vocab_size:
        raise ValueError("Prepared tokenizer vocabulary does not match the fixed 6.622M model config.")
    for artifact_name in ("data_metadata.json", "split_manifest.jsonl", "tokenizer.json"):
        artifact_path = args.data_dir / artifact_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Prepared data artifact is missing: {artifact_path}")
        shutil.copy2(artifact_path, args.run_dir / artifact_name)

    print("3/6 Building model...", flush=True)
    model = MathNetGPT(DEFAULT_MODEL_CONFIG).to(device)
    print(f"   Parameters: {model.parameter_count:,}", flush=True)
    print("4/6 Loading token store...", flush=True)
    token_store = TokenStore(args.data_dir, DEFAULT_MODEL_CONFIG.block_size)
    print("   Token store ready.", flush=True)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=train_config.weight_decay,
            fused=device.type == "cuda",
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=train_config.weight_decay,
        )
    scaler = make_scaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)

    run_config: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model.export_config(),
        "parameter_count": model.parameter_count,
        "training": train_config.to_dict(),
        "runtime": {
            "device": str(device),
            "amp_dtype": str(amp_dtype) if amp_dtype else None,
            "torch_version": torch.__version__,
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "data_directory": str(args.data_dir),
        "metric_interpretation": "Held-out next-token cross-entropy and exp(loss); not MathNet problem-solving accuracy.",
    }
    run_config["training"]["learning_rate_schedule_total_steps"] = schedule_total_steps
    run_config["training"]["resume_mode"] = args.resume is not None
    run_config["training"]["initial_global_step"] = 0
    if args.resume is not None:
        run_config["training"]["initial_global_step"] = None

    (args.run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    print("5/6 Loading checkpoint...", flush=True)
    if args.resume is not None:
        start_epoch, global_step, best_validation_loss = load_checkpoint(
            args.resume, model=model, optimizer=optimizer, scaler=scaler, device=device
        )

    print("6/6 Checkpoint loaded; starting training loop...", flush=True)
    print(f"   Starting global step: {global_step}", flush=True)
    metrics = MetricWriter(args.run_dir)
    train_generator = torch.Generator(device="cpu").manual_seed(train_config.seed)
    started_at = time.perf_counter()
    try:
        for epoch_index in range(start_epoch, train_config.epochs):
            model.train()
            epoch_loss_sum = 0.0
            epoch_tokens = 0
            for step_in_epoch in range(1, train_config.steps_per_epoch + 1):
                lr = learning_rate(
                    global_step,
                    train_config,
                    schedule_total_steps=schedule_total_steps,
                )

                # Force LR override if provided via CLI (--force-lr)
                if getattr(args, 'force_lr', None) is not None:
                    lr = float(args.force_lr)
                    try:
                        for _pg in optimizer.param_groups:
                            _pg['lr'] = lr
                    except Exception:
                        pass
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                update_loss_sum = 0.0
                for _ in range(train_config.gradient_accumulation_steps):
                    x, y = token_store.batch(
                        "train", train_config.micro_batch_size, device, train_generator
                    )
                    with autocast_context(device, amp_dtype):
                        logits, _ = model(x)
                        unscaled_loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                        )
                        loss = unscaled_loss / train_config.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    update_loss_sum += unscaled_loss.detach().float().item()
                scaler.unscale_(optimizer)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip).item()
                )
                scaler.step(optimizer)
                scaler.update()

                update_loss = update_loss_sum / train_config.gradient_accumulation_steps
                update_tokens = (
                    train_config.effective_batch_size * DEFAULT_MODEL_CONFIG.block_size
                )
                epoch_loss_sum += update_loss * update_tokens
                epoch_tokens += update_tokens
                global_step += 1
                metrics.write(
                    {
                        "kind": "train_step",
                        "epoch": epoch_index + 1,
                        "step_in_epoch": step_in_epoch,
                        "global_step": global_step,
                        "loss": update_loss,
                        "perplexity": math.exp(min(update_loss, 20.0)),
                        "learning_rate": lr,
                        "gradient_norm_before_clip": gradient_norm,
                        "tokens": update_tokens,
                        "elapsed_seconds": time.perf_counter() - started_at,
                    }
                )
                print(
                    f"Epoch {epoch_index + 1:03d}/{args.epochs:03d} | "
                    f"Step {global_step:04d}/{args.schedule_total_steps or global_step} | "
                    f"Loss {update_loss:.4f} | "
                    f"PPL {math.exp(min(update_loss,20.0)):.2f} | "
                    f"LR {lr:.2e} | "
                    f"Grad {gradient_norm:.3f}",
                    flush=True,
                )

                print(
                    f"Epoch {epoch_index + 1:03d}/{args.epochs:03d} | "
                    f"Step {global_step:04d}/{args.schedule_total_steps or global_step} | "
                    f"Loss {update_loss:.4f} | "
                    f"PPL {math.exp(min(update_loss,20.0)):.2f} | "
                    f"LR {lr:.2e} | "
                    f"Grad {gradient_norm:.3f}",
                    flush=True,
                )

                print(
                    f"Epoch {epoch_index + 1:03d}/121 | "
                    f"Step {global_step:04d}/8100 | "
                    f"Loss {update_loss:.4f} | "
                    f"PPL {math.exp(min(update_loss, 20.0)):.2f} | "
                    f"LR {lr:.2e} | "
                    f"Grad {gradient_norm:.3f}",
                    flush=True,
                )

            validation = evaluate(
                model,
                token_store,
                split="validation",
                batches=train_config.eval_batches,
                micro_batch_size=train_config.micro_batch_size,
                device=device,
                amp_dtype=amp_dtype,
                seed=train_config.seed + 1,
            )
            epoch_train_loss = epoch_loss_sum / epoch_tokens
            epoch_record = {
                "kind": "epoch_summary",
                "epoch": epoch_index + 1,
                "global_step": global_step,
                "train_loss": epoch_train_loss,
                "train_perplexity": math.exp(min(epoch_train_loss, 20.0)),
                "validation_loss": validation["loss"],
                "validation_perplexity": validation["perplexity"],
                "validation_tokens": int(validation["tokens"]),
                "elapsed_seconds": time.perf_counter() - started_at,
            }
            metrics.write(epoch_record)

            save_checkpoint(
                checkpoints_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch_completed=epoch_index + 1,
                global_step=global_step,
                best_validation_loss=min(best_validation_loss, validation["loss"]),
                run_config=run_config,
            )
            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch_completed=epoch_index + 1,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    run_config=run_config,
                )
            if (epoch_index + 1) % 5 == 0:
                save_checkpoint(
                    checkpoints_dir / f"epoch_{epoch_index + 1:02d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch_completed=epoch_index + 1,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    run_config=run_config,
                )

        # Save a compact inference-only artifact in addition to resumable state.
        fp16_state = {name: value.detach().cpu().half() for name, value in model.state_dict().items()}
        torch.save(
            {"model_config": model.export_config(), "model_state_dict": fp16_state},
            args.run_dir / "model_fp16.pt",
        )
        summary = {
            "status": "training_complete_pending_test_evaluation",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "epochs_completed": train_config.epochs,
            "optimizer_steps_completed": global_step,
            "best_validation_loss": best_validation_loss,
            "best_validation_perplexity": math.exp(min(best_validation_loss, 20.0)),
            "elapsed_seconds": time.perf_counter() - started_at,
            "next_step": "Run evaluate.py on checkpoints/best.pt to record the one-time internal test metric.",
        }
        (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        metrics.close()
    metrics.to_csv(args.run_dir / "metrics.csv")
    return json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    parsed = parse_args()
    print(json.dumps(train(parsed), indent=2))
