"""Create plots and companion results documentation from actual run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_epoch_rows(metrics_path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("kind") == "epoch_summary"]


def make_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    fig, (loss_axis, ppl_axis) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    loss_axis.plot(epochs, [row["train_loss"] for row in rows], label="train", color="#2563eb")
    loss_axis.plot(epochs, [row["validation_loss"] for row in rows], label="validation", color="#dc2626")
    loss_axis.set(title="Cross-entropy loss", xlabel="Logical epoch", ylabel="nats / token")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()
    ppl_axis.plot(epochs, [row["train_perplexity"] for row in rows], label="train", color="#2563eb")
    ppl_axis.plot(epochs, [row["validation_perplexity"] for row in rows], label="validation", color="#dc2626")
    ppl_axis.set(title="Perplexity", xlabel="Logical epoch", ylabel="exp(loss)")
    ppl_axis.grid(alpha=0.25)
    ppl_axis.legend()
    fig.suptitle("PicoGPT MathNet 6.62M — actual run metrics")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _number(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def make_results_markdown(
    run_config: dict[str, Any],
    data_metadata: dict[str, Any],
    summary: dict[str, Any],
    test: dict[str, Any] | None,
) -> str:
    model = run_config["model"]
    training = run_config["training"]
    split = data_metadata["split"]
    dataset = data_metadata["dataset"]
    test_loss = test.get("test_loss") if test else None
    test_ppl = test.get("test_perplexity") if test else None
    test_tokens = test.get("test_tokens") if test else "—"
    cache = "passed" if test and test["kv_cache_check"]["passed"] else "not run"
    completed_steps = summary.get("optimizer_steps_completed")
    completed_steps_display = f"{completed_steps:,}" if isinstance(completed_steps, int) else "—"
    return f'''# PicoGPT — MathNet 6.62M experiment

This is a self-contained scaled experiment. It does **not** modify or import
the original PicoGPT model/training files. Values below are generated from
saved run artifacts, never pre-filled estimates.

## Architecture

| Setting | Value |
|---|---:|
| Trainable parameters | {run_config['parameter_count']:,} |
| Tokenizer | Byte-level BPE, {model['vocab_size']:,} tokens; fit on train split only |
| Decoder blocks | {model['n_layer']} |
| Width / heads / head dimension | {model['n_embd']} / {model['n_head']} / {model['n_embd'] // model['n_head']} |
| MLP width | {model['intermediate_size']} |
| Context length | {model['block_size']} |
| Normalization | Pre-LN RMSNorm |
| Attention | causal multi-head attention, RoPE, PyTorch SDPA |
| Decode optimization | per-layer inference KV cache |
| Token embeddings | tied input/output embeddings |

## Training protocol

| Setting | Value |
|---|---:|
| Logical epochs × steps | {training['epochs']} × {training['steps_per_epoch']} |
| Optimizer updates | {training['total_steps']:,} |
| Microbatch × accumulation | {training['micro_batch_size']} × {training['gradient_accumulation_steps']} |
| Effective sequences/update | {training['effective_batch_size']} |
| Planned target tokens | {training['planned_target_tokens']:,} |
| AdamW peak/min LR | {training['learning_rate']:.1e} / {training['min_learning_rate']:.1e} |
| Warmup | {training['warmup_steps']} steps |

“Epoch” means a fixed 100-update logical epoch, **not** a full pass through
MathNet. Loss and perplexity are held-out next-token language-model metrics,
not mathematical problem-solving accuracy.

## Dataset provenance

- Dataset: `{dataset['id']}` / `{dataset['config']}`, pinned revision `{dataset['revision']}`
- Prepared documents: `{split['encoded_document_counts']}`
- Token counts: `{split['token_counts']}`
- Split method: `{split['method']}`

MathNet includes image-bearing questions. This text-only GPT replaces inline
`attached_image_N.png` references with `<|image|>` and excludes image bytes.
The public release exposes one train split, so this experiment creates
deterministic internal 80/10/10 splits; they are not MathNet-Solve partitions.

## Results

| Metric | Actual result |
|---|---:|
| Best validation loss | {_number(summary.get('best_validation_loss'))} |
| Best validation perplexity | {_number(summary.get('best_validation_perplexity'))} |
| Completed optimizer updates | {completed_steps_display} |
| Internal held-out test loss | {_number(test_loss)} |
| Internal held-out test perplexity | {_number(test_ppl)} |
| Test tokens evaluated | {test_tokens} |
| KV-cache validation | {cache} |

`loss_perplexity.png` is generated from `metrics.jsonl`. Configuration, split
manifest, tokenizer hash, checkpoints, and raw metrics remain beside this file.

## Dataset citation and license note

MathNet: Shaden Alshammari et al., *MathNet: a Global Multimodal Benchmark for
Mathematical Reasoning and Retrieval*, ICLR 2026. Dataset:
https://huggingface.co/datasets/ShadenA/MathNet

The dataset card lists CC BY 4.0 for material without separately asserted
rights; source competition/country rights can take precedence. Use
`split_manifest.jsonl` for provenance rather than redistributing corpus text.
'''


def main(run_dir: Path) -> None:
    rows = load_epoch_rows(run_dir / "metrics.jsonl")
    if not rows:
        raise RuntimeError("No epoch_summary records found; train before creating a report.")
    make_plot(rows, run_dir / "loss_perplexity.png")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    data_metadata = json.loads((run_dir / "data_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    test_path = run_dir / "test_results.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else None
    report = make_results_markdown(config, data_metadata, summary, test)
    (run_dir / "RESULTS.md").write_text(report, encoding="utf-8")
    (run_dir / "README.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    main(parser.parse_args().run_dir)
