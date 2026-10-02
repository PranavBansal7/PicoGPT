# PicoGPT

PicoGPT is a from-scratch GPT learning project with two deliberately separate
tracks:

- A compact, character-level tiny-Shakespeare model that makes each
  transformer primitive inspectable.
- An isolated 6.62M-parameter MathNet experiment for reproducible,
  GPU-backed language-model training, evaluation, ablations, and cached
  decoding.

The MathNet code never imports or changes the original teaching model. The
smaller project therefore remains a useful, readable path from neuron to GPT,
while the experiment can evolve without invalidating it.

## What is in the repository

~~~text
.
├── foundations/                  # ML primitives and manual-backprop exercises
├── data/                         # character vocabulary, BPE, and batching
├── model/                        # original attention, transformer, and GPT modules
├── train.py, generate.py         # original tiny-Shakespeare workflow
└── experiments/mathnet_6_6m/     # standalone 6.62M MathNet experiment
    ├── prepare_data.py           # pinned source data and deterministic split
    ├── model.py                  # RoPE, MHA/GQA, RMSNorm/LayerNorm, KV cache
    ├── train.py, evaluate.py     # resumable training and held-out LM evaluation
    ├── benchmark_decode.py       # cache-off/cache-on decoding benchmark
    ├── chat.py                   # EOS-aware problem-to-solution interface
    └── ABLATIONS.md              # protocol and recorded experiment status
~~~

## Original teaching model

The original track builds a decoder-only GPT from individual, tested pieces:

**neuron → backprop → MLP → tokenizer → embeddings → attention → transformer
block → GPT → training loop → generation**

It includes hand-derived gradients, BPE, causal and multi-head attention,
Grouped Query Attention, a KV cache, LayerNorm/BatchNorm/RMSNorm, AdamW
training, and autoregressive sampling. The original tiny-Shakespeare example
uses an approximately 816K-parameter character model.

The fixed-precision output rounding in model/gpt.py is useful for the original
numerical tests but prevents meaningful gradients. Keep it for those
deterministic test outputs; skip it for real training.

## MathNet experiment: current status

The MathNet work is a **text-only next-token language-model experiment**, not
a MathNet-Solve accuracy result. It uses ShadenA/MathNet revision
33e6b3bc254e6f0f0c1479b4252f5e9dd551d56c, replaces attached-image Markdown
with the image sentinel token, and creates a deterministic provenance-grouped
internal 80/10/10 split. Its held-out partitions are not official benchmark
splits.

Prepared data provenance:

| Item | Value |
|---|---:|
| Source rows loaded | 27,817 |
| Documents: train / validation / test | 19,934 / 2,662 / 1,686 |
| Tokens: train / validation / test | 12,851,747 / 1,906,401 / 1,213,495 |
| Tokenizer | 8,192-token byte-level BPE trained on train only |
| Trainable parameters per ablation variant | 6,622,464 |

### Completed long reference run

A completed reference continuation reached 16,100 optimizer updates on a Tesla
T4 with bfloat16. It is a useful language-model checkpoint, but its older
run configuration does not record the later variant field, so it must **not**
be used to rank the ablation variants.

| Metric | Recorded result |
|---|---:|
| Best validation loss / perplexity | 1.84157 / 6.30644 |
| Internal held-out test loss / perplexity | 1.87108 / 6.49528 |
| Test tokens evaluated | 2,048,000 |
| KV-cache logit check | passed; max absolute error 1.91e-05 (tolerance 2e-04) |

### Completed single-seed comparison

Both rows below use seed 1337, 4,100 updates, the same prepared split,
sequence length, optimizer schedule, and 6,622,464-parameter budget. This is
an informative **single-seed observation**, not a multi-seed conclusion and
not evidence that GQA alone caused the difference: normalization and attention
change together between these variants.

| Variant | Best validation loss | Test loss | Test perplexity | Test top-1 token accuracy |
|---|---:|---:|---:|---:|
| Post-LN LayerNorm + MHA | 2.34068 | 2.34940 | 10.47929 | 0.47370 |
| Pre-LN RMSNorm + 8Q/2KV GQA | 2.22066 | 2.23694 | 9.36461 | 0.49612 |

Each run evaluated 2,048,000 internal held-out tokens and passed the
cached-vs-full logit check with a maximum absolute error of 1.24e-05.
Pre-LN LayerNorm, Pre-LN RMSNorm + MHA, and the remaining seeds are still
needed before reporting a mean, standard deviation, or a preferred variant.

### KV-cache benchmark

The archived Pre-LN RMSNorm + GQA checkpoint was decoded for 128 new tokens on
a Tesla T4 over ten trials. Cache-on and cache-off greedy tokens matched in
every benchmark. The figures below are measured, not estimates.

| Prompt tokens | Cache-off median | Cache-on median | Cache-on/off speed | Additional peak memory: off → on |
|---|---:|---:|---:|---:|
| 32 | 0.8314 s | 0.9372 s | 0.887× | 11.30 MB → 1.35 MB |
| 128 | 0.8427 s | 0.9315 s | 0.905× | 17.21 MB → 5.26 MB |
| 256 | 0.7970 s | 0.7989 s | 0.998× | 17.31 MB → 18.88 MB |

For this small model and benchmark harness, caching proved correctness and
lower short-prompt additional memory, but it did **not** produce a latency
speedup. Treat cache performance as workload-dependent; do not generalize
these numbers to a larger model or another device.

## Reproducibility artifacts

The source code lives here; generated datasets, checkpoints, metrics, and
reports are kept out of Git. The shared
[PicoGPT_MathNet Drive folder](https://drive.google.com/drive/folders/1p3RGG7HFzLCC6nLq5nVs9rKWHvk538Tf?usp=sharing)
contains prepared data and run directories. Its 1.39 GB
pico_session_export.tar.gz archive includes the completed seed-1337 Post-LN
and GQA runs plus the three decode benchmarks. Archive SHA-256:

~~~text
a94cf611e4bee7f3ab33a68a34444f74f8d78e99a1fbdae83d002cf70cefb25d
~~~

See experiments/mathnet_6_6m/README.md for the full protocol and
experiments/mathnet_6_6m/ABLATIONS.md for the exact comparison boundaries.

## Quick start

For the original learning modules, install the repository dependencies, build
a character vocabulary from a corpus, create model.gpt.GPT, and use train.py
plus generate.py. The modules are intentionally composable rather than hidden
behind a large framework.

For the MathNet experiment, open
experiments/mathnet_6_6m/mathnet_6_6m_colab.ipynb in a GPU runtime and follow
the cells in order. The notebook prepares the pinned data, writes all run
artifacts to the selected Drive directory, and can resume from checkpoints
written by the current experiment trainer.

## License

MIT License

Copyright (c) 2026 Pranav Bansal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
