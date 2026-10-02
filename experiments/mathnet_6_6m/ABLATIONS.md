# MathNet optimization ablations

This study separates **model-quality** changes from **inference-efficiency**
changes. Every quality run uses the same prepared data directory, tokenizer,
80/10/10 internal split, 4,100 optimizer updates, sequence length, optimizer,
and three seeds: `1337`, `2026`, and `4242`.

## Quality study

| Variant | Normalization / attention | Parameters |
|---|---|---:|
| `post_ln_mha` | Post-LN scale-only LayerNorm + MHA | 6,622,464 |
| `pre_ln_mha` | Pre-LN scale-only LayerNorm + MHA | 6,622,464 |
| `pre_rms_mha` | Pre-LN RMSNorm + MHA | 6,622,464 |
| `pre_rms_gqa` | Pre-LN RMSNorm + 8Q/2KV GQA | 6,622,464 |

The GQA variant’s MLP width is 1,152 rather than 960. Its smaller K/V
projections save 98,304 parameters per block; the 192 extra MLP channels add
exactly those parameters back. This prevents parameter count from confounding
the GQA comparison.

Run each job into its own immutable output directory:

```bash
python -m experiments.mathnet_6_6m.train \
  --data-dir /content/drive/MyDrive/PicoGPT_MathNet/data \
  --run-dir /content/drive/MyDrive/PicoGPT_MathNet/runs/pre_rms_gqa_seed1337 \
  --variant pre_rms_gqa --seed 1337 \
  --epochs 41 --steps-per-epoch 100
```

Report the mean ± sample standard deviation across seeds for held-out loss,
perplexity, and `validation_top1_accuracy`; choose the checkpoint by validation
loss, then run the internal test evaluator once per selected checkpoint.
Top-1 token accuracy is an LM metric, not mathematical problem-solving
accuracy. A later answer-accuracy benchmark must use prompt-only problems,
normalised final answers, and a separately documented verifier.

## KV-cache efficiency study

KV caching must not change model quality: it reuses previous K/V projections at
decode time. Run it against the **same trained checkpoint** with cache off/on:

```bash
python -m experiments.mathnet_6_6m.benchmark_decode \
  --run-dir /content/drive/MyDrive/PicoGPT_MathNet/runs/pre_rms_gqa_seed1337 \
  --prompt-tokens 128 --new-tokens 128 --trials 10
```

The benchmark writes `decode_benchmark.json` with cached/uncached median decode
time, tokens/s, peak additional GPU memory, and an exact greedy-token equality
check. Repeat at prompt lengths 32, 128, and 256. GQA’s reduced KV-head count
should be assessed primarily in this benchmark because it lowers cache memory.

## Continuation

The canonical `train.py` supports exact resume only for checkpoints it writes:
the model, optimizer, AMP scaler, global RNG, CUDA RNG, and batch-sampler RNG
are all saved. To extend a 4,100-step run to 8,200 total updates, invoke it
with `--epochs 82 --schedule-total-steps 8200 --resume <last.pt>`. Do not use
the removed continuation copies; their sampler-RNG handling was not exact.
