# PicoGPT MathNet 6.62M

This directory is a standalone, GPU-oriented experiment. It does not import or
modify the original model/, data/, train.py, generate.py, or root teaching
workflow.

It trains a causal language model on text-formatted MathNet problem-to-solution
documents. It is **not** a MathNet-Solve evaluation and must not be described
as mathematical answer accuracy.

## Model family

Every quality-study variant has exactly 6,622,464 trainable parameters,
achieved with tied input/output embeddings and bias-free projections.

| Component | Default / study design |
|---|---|
| Tokenizer | 8,192-token byte-level BPE fit on the internal train partition only |
| Decoder | 6 blocks, width 256, 8 query heads, context length 256 |
| Positional encoding | RoPE |
| Attention back end | PyTorch scaled-dot-product causal attention |
| Normalization study | Post-LN LayerNorm, Pre-LN LayerNorm, or Pre-LN RMSNorm |
| Attention study | MHA or 8-query-head / 2-KV-head GQA |
| GQA parameter matching | MLP width 1,152 rather than 960 |
| Generation | Per-layer KV cache with an EOS stopping path |

The cache is used only while generating. Training runs the full causal
attention path and does not retain a cache in the training graph.

## Dataset and provenance

The source is ShadenA/MathNet, config all, pinned to revision
33e6b3bc254e6f0f0c1479b4252f5e9dd551d56c. The preparer loads 27,817 rows and
formats each usable record as a special-token-delimited problem → solution
document.

MathNet is multimodal. This experiment is intentionally text only: Markdown
references to attached images are replaced with the image sentinel token, and
image bytes are excluded. The public release supplies one train split, so
prepare_data.py creates an internal, SHA-256 seeded, provenance-grouped
80/10/10 split (seed 13337):

| Split | Documents | Tokens |
|---|---:|---:|
| Train | 19,934 | 12,851,747 |
| Validation | 2,662 | 1,906,401 |
| Test | 1,686 | 1,213,495 |

These internal partitions are not MathNet-Solve evaluation data. The saved
split manifest, tokenizer hash, and source revision are part of the
reproducibility record. Review source rights before redistributing a derived
corpus; the dataset card's CC BY 4.0 note does not remove underlying
competition or country rights.

## Recorded experiment status

### Reference continuation

One completed 6.62M reference run reached 16,100 optimizer updates on a Tesla
T4 with bfloat16. Its older run configuration predates the explicit variant
field used by this study, so it is documented as a reference checkpoint rather
than included in any ablation ranking.

| Metric | Result |
|---|---:|
| Best validation loss / perplexity | 1.84157 / 6.30644 |
| Internal held-out test loss / perplexity | 1.87108 / 6.49528 |
| Test tokens | 2,048,000 |
| Cache logit equality check | passed; 1.91e-05 maximum absolute error |

### Seed-1337 quality comparison

The two completed rows below are directly paired by seed, prepared data,
4,100 optimizer updates, sequence length, optimizer schedule, and parameter
budget. They differ in **both** normalization and attention, so the table is a
single-seed architecture comparison—not an isolated GQA causal claim.

| Variant | Best validation loss | Test loss | Test perplexity | Test top-1 token accuracy |
|---|---:|---:|---:|---:|
| post_ln_mha | 2.34068 | 2.34940 | 10.47929 | 0.47370 |
| pre_rms_gqa | 2.22066 | 2.23694 | 9.36461 | 0.49612 |

Both test evaluations cover 2,048,000 internal held-out tokens. Cached and
uncached logits agreed within 1.24e-05 maximum absolute error, below the
2e-04 tolerance.

The remaining variants and seeds have not completed, so do not report a
mean ± standard deviation, a winning architecture, or mathematical
problem-solving accuracy. See ABLATIONS.md for the status boundary.

### KV-cache measurements

The completed pre_rms_gqa checkpoint was decoded for 128 new tokens on a Tesla
T4 over ten trials. Greedy cache-on and cache-off token sequences matched in
all three cases.

| Prompt length | Cache off | Cache on | Cache-on/off speed | Additional peak memory: off → on |
|---|---:|---:|---:|---:|
| 32 | 0.8314 s | 0.9372 s | 0.887× | 11.30 MB → 1.35 MB |
| 128 | 0.8427 s | 0.9315 s | 0.905× | 17.21 MB → 5.26 MB |
| 256 | 0.7970 s | 0.7989 s | 0.998× | 17.31 MB → 18.88 MB |

This measured cache implementation lowers short-prompt additional memory in
this harness but does not improve latency on this small model. The result is
workload- and device-specific; benchmark a different model or serving setup
again rather than extrapolating it.

## Training protocol

Each standard ablation run uses 41 logical epochs × 100 updates, for 4,100
optimizer updates. A logical epoch is a fixed update count, not a claim of a
full pass through the corpus.

| Setting | Value |
|---|---:|
| Microbatch × gradient accumulation | 16 × 4 |
| Effective sequences/update | 64 |
| Sequence length | 256 |
| Planned target tokens | 67,174,400 |
| Optimizer | AdamW, β=(0.9, 0.95), weight decay 0.1 |
| Learning-rate schedule | 200-step warmup to 3e-4, cosine decay to 3e-5 |

Checkpoints written by the current trainer include model, optimizer, AMP
scaler, CPU/CUDA RNG state, batch-sampler RNG state, global step, and
configuration. Resume only a checkpoint made by this current trainer when
claiming exact resume behavior.

## Run in Colab

Open mathnet_6_6m_colab.ipynb, select a GPU runtime, and run cells in order.
The notebook prepares its data and writes artifacts beneath the selected Drive
root:

~~~text
MyDrive/PicoGPT_MathNet/
├── data/
│   ├── data_metadata.json, split_manifest.jsonl, tokenizer.json
│   └── train.bin, validation.bin, test.bin
└── runs/<run_name>/
    ├── run_config.json, metrics.jsonl, metrics.csv, summary.json
    ├── checkpoints/best.pt, checkpoints/last.pt, checkpoints/epoch_*.pt
    ├── model_fp16.pt, test_results.json
    └── decode_benchmark_<prompt_tokens>.json
~~~

An example quality run:

~~~bash
python -m experiments.mathnet_6_6m.train \
  --data-dir /content/drive/MyDrive/PicoGPT_MathNet/data \
  --run-dir /content/drive/MyDrive/PicoGPT_MathNet/runs/pre_rms_gqa_seed1337 \
  --variant pre_rms_gqa --seed 1337 \
  --epochs 41 --steps-per-epoch 100
~~~

Run evaluate.py once for the selected checkpoint, and use
benchmark_decode.py to measure cache behavior with the same checkpoint. Do
not compare cache speed to a separate run.

## Generation interface

The corpus has bos, problem, solution, image, and eos special tokens. The
EOS-aware chat helper formats a question as a problem-to-solution prompt and
stops when it samples EOS or reaches its token limit.

~~~bash
python -m experiments.mathnet_6_6m.chat \
  --run-dir /content/drive/MyDrive/PicoGPT_MathNet/runs/<completed_run> \
  --question "Solve for x: 3x - 5 = 16."
~~~

This is not a chat-tuned assistant. Its generated solution text may be wrong
and requires independent verification.

## Local terminal demo

Use demo_chat.py when you want a simple, presentable back-and-forth terminal
demo. It displays numbered turns and supports :help, :settings, :new, and
:quit. It deliberately sends each entered question as a fresh
problem-to-solution prompt: the model was not trained on conversational
history, and keeping the conversation in the prompt would quickly exceed its
256-token context window.

### Windows / PowerShell

This demo uses the completed 16,100-step reference checkpoint (internal held-out
test loss 1.87108), not the 4,100-step GQA checkpoint in
pico_session_export.tar.gz. From the
[continue_from_8100 run folder](https://drive.google.com/drive/folders/1gIJkvkLA9CMM9Kkx26SWCHGMLq2xdNle),
download only these two files:

- checkpoints/best.pt
- tokenizer.json

Place them in a local demo_model directory with this structure:

~~~text
PicoGPT/
└── demo_model/
    ├── tokenizer.json
    └── checkpoints/
        └── best.pt
~~~

Then clone the repository and install the small local runtime:

~~~powershell
git clone https://github.com/PranavBansal7/PicoGPT.git
cd PicoGPT
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install tokenizers
.\.venv\Scripts\python.exe -m experiments.mathnet_6_6m.demo_chat --run-dir .\demo_model --device cpu --max-new-tokens 96
~~~

For a CUDA-capable local PyTorch installation, omit the CPU-only PyTorch
command, install the build appropriate for your GPU from pytorch.org, and
replace --device cpu with --device cuda. --device auto prefers CUDA when it is
available.

For a repeatable demo response, add --top-k 1. Otherwise, the default sampling
settings provide a little variety between runs. Ask self-contained questions
such as Solve for x: 3x - 5 = 16.; restate any information needed for a
follow-up.

## Artifact bundle

Generated artifacts are intentionally not committed. The shared
[PicoGPT_MathNet Drive folder](https://drive.google.com/drive/folders/1p3RGG7HFzLCC6nLq5nVs9rKWHvk538Tf?usp=sharing)
contains source data and run directories. The 1.39 GB
pico_session_export.tar.gz archive contains the completed seed-1337 Post-LN
and GQA artifacts, including the three decode benchmarks. Its SHA-256 is:

~~~text
a94cf611e4bee7f3ab33a68a34444f74f8d78e99a1fbdae83d002cf70cefb25d
~~~

## Implementation checks

~~~bash
python -m unittest experiments.mathnet_6_6m.test_model
~~~

The tests verify parameter counts, EOS termination, cached generation shape,
and cached-vs-full GQA logits.
