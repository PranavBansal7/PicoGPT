# MathNet optimization ablations

This study keeps model quality and decode efficiency separate:

- Quality runs compare matched 6,622,464-parameter architectures.
- Cache runs compare cache off/on on the **same trained checkpoint**.

All quality jobs use the same prepared data, tokenizer, internal
provenance-grouped 80/10/10 split, sequence length, 4,100 optimizer updates,
and AdamW schedule. Results are held-out next-token language-model metrics;
they are not MathNet-Solve accuracy or answer correctness.

## Architecture matrix

| Variant | Normalization / attention | MLP width | Parameters |
|---|---|---:|---:|
| post_ln_mha | Post-LN scale-only LayerNorm + MHA | 960 | 6,622,464 |
| pre_ln_mha | Pre-LN scale-only LayerNorm + MHA | 960 | 6,622,464 |
| pre_rms_mha | Pre-LN RMSNorm + MHA | 960 | 6,622,464 |
| pre_rms_gqa | Pre-LN RMSNorm + 8Q/2KV GQA | 1,152 | 6,622,464 |

GQA saves 98,304 attention parameters per block through its reduced K/V
projections. Its 192 additional MLP channels add exactly that count back, so
parameter budget does not confound the comparison.

## Status as of 2026-10-02

The reproducibility archive contains the two completed seed-1337 runs below.
Both use the same split, 4,100 updates, effective batch size 64, and model
budget. They differ in normalization **and** attention, so the apparent
improvement cannot be attributed to GQA alone.

| Variant / seed | Status | Best validation loss | Test loss | Test perplexity | Test top-1 token accuracy |
|---|---|---:|---:|---:|---:|
| post_ln_mha / 1337 | complete and evaluated | 2.34068 | 2.34940 | 10.47929 | 0.47370 |
| pre_rms_gqa / 1337 | complete and evaluated | 2.22066 | 2.23694 | 9.36461 | 0.49612 |
| post_ln_mha / 2026 | no completed result recorded | — | — | — | — |
| pre_ln_mha / all seeds | not completed | — | — | — | — |
| pre_rms_mha / all seeds | not completed | — | — | — | — |
| pre_rms_gqa / 2026, 4242 | not completed | — | — | — | — |

Each completed test uses 500 batches × 16 sequences = 2,048,000 tokens. Both
passed cache-logit validation with 1.24e-05 maximum absolute error against a
2e-04 tolerance.

This is a **single-seed paired observation**, not a statistical result. Do
not select a winner, report mean ± standard deviation, or claim a GQA-only
effect until all planned seeds and the missing MHA baselines are complete.

## How to run a quality job

Write each run to an immutable directory:

~~~bash
python -m experiments.mathnet_6_6m.train \
  --data-dir /content/drive/MyDrive/PicoGPT_MathNet/data \
  --run-dir /content/drive/MyDrive/PicoGPT_MathNet/runs/pre_rms_gqa_seed1337 \
  --variant pre_rms_gqa --seed 1337 \
  --epochs 41 --steps-per-epoch 100
~~~

Select a checkpoint by validation loss, then run one internal test evaluation:

~~~bash
python -m experiments.mathnet_6_6m.evaluate \
  --data-dir /content/drive/MyDrive/PicoGPT_MathNet/data \
  --run-dir /content/drive/MyDrive/PicoGPT_MathNet/runs/pre_rms_gqa_seed1337
~~~

The evaluator writes test_results.json and updates summary.json. Preserve the
config, split manifest, tokenizer, metrics, checkpoints, and test result with
the run; those are the evidence required to compare it.

## KV-cache result

The completed pre_rms_gqa seed-1337 checkpoint was measured on a Tesla T4,
generating 128 tokens over ten trials. Exact greedy outputs matched cache
off/on in every case.

| Prompt tokens | Off median | On median | Cache-on/off speed | Additional peak memory: off → on |
|---|---:|---:|---:|---:|
| 32 | 0.8314 s | 0.9372 s | 0.887× | 11.30 MB → 1.35 MB |
| 128 | 0.8427 s | 0.9315 s | 0.905× | 17.21 MB → 5.26 MB |
| 256 | 0.7970 s | 0.7989 s | 0.998× | 17.31 MB → 18.88 MB |

For this model and harness, caching was correct and reduced short-prompt
additional memory, but it did not improve median latency. The 256-token
measurement even used more additional peak memory with cache enabled. That is
a measured behavior of this setup, not a general statement about KV caching.

Run the same benchmark on the same checkpoint whenever changing hardware,
batching, prompt length, model size, or decode implementation:

~~~bash
python -m experiments.mathnet_6_6m.benchmark_decode \
  --run-dir /content/drive/MyDrive/PicoGPT_MathNet/runs/pre_rms_gqa_seed1337 \
  --prompt-tokens 128 --new-tokens 128 --trials 10
~~~

## Continuation policy

The current trainer supports exact resume only for checkpoints it writes:
model, optimizer, AMP scaler, global RNG, CUDA RNG, and batch-sampler RNG are
saved. To extend a current 4,100-step run to 8,200 total updates, use its
last.pt with epochs 82 and schedule-total-steps 8200. Do not use removed
continuation copies whose sampler-RNG handling was not exact.

## Artifact provenance

The generated records are in the shared
[PicoGPT_MathNet Drive folder](https://drive.google.com/drive/folders/1p3RGG7HFzLCC6nLq5nVs9rKWHvk538Tf?usp=sharing).
The 1.39 GB pico_session_export.tar.gz archive contains the completed
seed-1337 Post-LN and GQA artifacts plus decode_benchmark_32.json,
decode_benchmark_128.json, and decode_benchmark_256.json. SHA-256:

~~~text
a94cf611e4bee7f3ab33a68a34444f74f8d78e99a1fbdae83d002cf70cefb25d
~~~
