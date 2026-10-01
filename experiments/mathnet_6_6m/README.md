# PicoGPT MathNet 6.62M — isolated Colab experiment

This directory is a new, standalone experiment. It never imports or edits the
original `model/`, `data/`, `train.py`, `generate.py`, or root `README.md`.
That preserves the original ~816K tiny-Shakespeare teaching model intact.

## Target model

The model has exactly **6,622,464 trainable parameters**:

| Component | Design |
|---|---|
| Tokenizer | 8,192-token byte-level BPE, trained only on the internal train partition |
| Transformer | 6 decoder blocks; width 256; 8 attention heads; 960-wide GELU MLP |
| Attention | Causal multi-head attention with RoPE and PyTorch scaled-dot-product attention |
| Stability | Pre-LN residual design with RMSNorm |
| Decode path | Per-layer KV cache, used only for generation |
| Parameter saving | Tied input/output embedding matrix and bias-free projections |

KV caching changes inference work, not the number of parameters. Training uses
the full causal-attention path and never stores a cache in the computation graph.

## Dataset and split

The run uses the official [`ShadenA/MathNet`](https://huggingface.co/datasets/ShadenA/MathNet)
dataset pinned to revision `33e6b3bc254e6f0f0c1479b4252f5e9dd551d56c`.
MathNet is multimodal; this is text-only training, so inline
`attached_image_N.png` Markdown is replaced with `<|image|>` and image bytes
are excluded. It formats usable rows as problem → solution documents.

The public release exposes a `train` split. `prepare_data.py` creates a
deterministic provenance-grouped 80/10/10 internal split, saves the IDs and
source metadata in `split_manifest.jsonl`, and trains BPE only on the internal
training split. The validation/test partitions are explicitly **not** official
MathNet-Solve evaluation data.

The dataset card describes CC BY 4.0 for material without separately asserted
rights; original competition/country copyright can supersede it. Keep the
manifest and cite MathNet; do not redistribute a derived corpus without
reviewing its source rights.

## Requested training budget

| Setting | Value |
|---|---:|
| Logical epochs | 41 |
| Optimizer updates per logical epoch | 100 |
| Total optimizer updates | 4,100 |
| Microbatch × gradient accumulation | 16 × 4 |
| Effective sequences/update | 64 |
| Sequence length | 256 |
| Planned target tokens | 67,174,400 |
| Optimizer | AdamW, β=(0.9, 0.95), weight decay 0.1 |
| LR schedule | 200-step warmup to 3e-4, cosine decay to 3e-5 |

In this project an “epoch” is deliberately a **100-update logical epoch**, not
a claim of a full corpus pass. The exact corpus exposure is recorded once data
preparation completes.

## Run in Google Colab

Open [mathnet_6_6m_colab.ipynb](mathnet_6_6m_colab.ipynb), choose a GPU runtime,
and run its cells in order. The notebook saves its run under Google Drive:

```text
MyDrive/PicoGPT_MathNet/runs/<run_name>/
├── run_config.json, data_metadata.json, split_manifest.jsonl
├── tokenizer.json, train.bin, validation.bin, test.bin
├── metrics.jsonl, metrics.csv, loss_perplexity.png
├── checkpoints/best.pt, checkpoints/last.pt, checkpoints/epoch_*.pt
├── model_fp16.pt, test_results.json, summary.json
└── README.md, RESULTS.md
```

The resumable checkpoints include model, optimizer, AMP scaler, CPU/CUDA RNG
state, global step, and configuration. A Colab interruption can resume from
`checkpoints/last.pt` after mounting the same Drive location.

Loss, perplexity, generated samples, and test results must only be filled from
these artifacts after the Colab run. No result is invented in advance.

## Verify the implementation

```bash
python -m unittest experiments.mathnet_6_6m.test_model
```

This verifies the exact parameter count, cached-vs-full logits, and cached
generation shape. The end-of-run evaluator repeats the cache logit check and
writes the outcome alongside the internal held-out next-token loss/perplexity.
