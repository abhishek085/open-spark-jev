# Model Card

**spark-s1**, release v6 (`spark-s1-4b-v6`), 2026-09-22. Part of Open Spark Jev, an open-source project of the Nokast AI community. This is a
very early release; expect it to change quickly.

## Overview

`spark-s1` is a System-1 decision model: given a state (text or JSON) and a typed question with options defined at request time, it returns a
probability for every option and a confidence from one forward pass, without generating text. Each release is a public Qwen backbone fine-tuned
with LoRA (r=16, merged into the weights). There is no extra head: the answer is the softmax of the first-token logits restricted to the option
letters, divided by a temperature.

It is not Jev and not affiliated with TypeSafe AI. It follows the same contract and readout idea; its training is a supervised fine-tune, not
TypeSafe's undisclosed method.

## What changed in v6

v6 keeps v5's data, scope and recipe unchanged and swaps the backbone: **Qwen3-4B -> Qwen3.5-4B**. Qwen3.5-4B is a hybrid architecture (24
linear-attention "Gated DeltaNet" layers plus 8 full-attention layers, of 32 total) rather than a uniform transformer. This is the single
variable changed from v5; same 11,792-row / 49-pack training mix, same LoRA recipe (extended to cover the linear-attention layers' own
projections, so all 32 layers are adapted, not just the 8 full-attention ones), same one epoch / lr 5e-5.

The result is the largest jump of any release so far: own-split accuracy +6.2 points over v5-4b, JevBench public-tier Intelligence +8.9 points,
and NVFP4 quantization now costs **no measurable accuracy** (v5-4b's NVFP4 cost ~1.9 Intelligence points; v6-4b's costs none). The backbone swap
is not a clean win everywhere: two of five external Jev-style sets regressed slightly (see Evaluation summary), and both bf16 and NVFP4 inference
are slower in absolute terms than v5's, because only the MLP is quantized and the hybrid backbone carries more total compute in its unquantized
attention paths.

## Intended use

* Bounded, repeated control decisions inside an agent harness: approve, escalate or refuse a proposed tool call, route a request, choose among
  fixed options, decide whether to hand off to a stronger model or a person, moderate content, check a tool call or a draft answer before it is
  used.
* Behind deterministic policy checks, least privilege and human approval (see below), with a conservative auto-allow threshold.
* Local, low-latency use where a generative model plus JSON parsing is too slow or too brittle.

## Out-of-scope use

* As the only authorization control for tool execution, or for any high-impact action without a human in the loop.
* Open-ended conversation, coding, general reasoning, summarisation or any task that needs generated text.
* Decisions with more than 26 options, multi-select, ranking, or non-English input (not evaluated).
* General-purpose text classification (topic, sentiment, NLI, etc.) — out of scope for this release, not trained or evaluated.
* Vulnerability detection in code — not part of this release's data or evaluation (v3/v4 measured near-chance accuracy on it).

## Supported decision tasks

Question types: Choice (up to 26 options), Score (ordered levels), Boolean. v6 trains on the same **49 task packs** as v5, spanning agent
guardrails and harness engineering (tool-call verification, context pruning, response-quality and citation checks, delegation, error recovery,
plan alignment, trace monitoring), security policy (prompt-injection and message-manipulation gating, PII/output-leak checks, authorization),
moderation and routing (content moderation, ticket triage, model routing), and structured records (entity resolution, fraud/risk scoring,
invoice/insurance/SOC-alert routing, financial triage, semantic linting, row validation, semantic grep). Per-pack test counts are small for many
of the newer packs (some under 20 rows); read the aggregate numbers below as the more reliable signal.

## Available variants

| Release id | Backbone | Weights | Notes |
|---|---|---|---|
| `spark-s1-4b-v6` | Qwen3.5-4B | [`abhishek085/spark-s1-4b-v6`](https://huggingface.co/abhishek085/spark-s1-4b-v6) | Current best accuracy |
| `spark-s1-4b-v6-nvfp4` | Qwen3.5-4B | [`abhishek085/spark-s1-4b-v6-nvfp4`](https://huggingface.co/abhishek085/spark-s1-4b-v6-nvfp4) | NVFP4 (MLP-only), 1.40x faster via vLLM on Blackwell hardware, no measurable accuracy cost; needs a GB10/B200-class GPU |

v6 is a 4B-only release; there is no v6 1.7B this round. Earlier releases (`v3`, `v4`, `v5`, simulator-era `sft-v2`, RLCD variants, architecture
variants A0-A4) are described in [docs/MODELS.md](docs/MODELS.md) and are not part of this release.

## Training/data summary

* **Data:** identical to v5 — 11,792 training rows (19,568 examples with option-order augmentation) across the 49 packs above, drawn from
  `os-datagen` ([`datagen-pipeline/`](datagen-pipeline/)): LLM-written scenarios (labels established by code, text written by an LLM and checked
  by an independent verifier model) plus code-only rule packs where both the label and the surface text come from deterministic rules, no LLM in
  the loop. Splits: calibration/locked-test/challenge rows are drawn from scenario families never seen in training.
* **Recipe:** LoRA r=16 (alpha 32) on all attention and MLP projections — extended for v6 to also cover the linear-attention layers' own
  projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`), so all 32 layers are adapted rather than only the 8
  full-attention ones. 32.5M trainable parameters (0.77% of 4.24B total). One epoch, lr 5e-5, cross-entropy plus a Brier regulariser. One seed.
* **Not used:** no real public text, no vulnerable-code data, no outcome-based (RLCD) training on the shipped checkpoint.
* Generator and verifier models: Gemma-4-26B-A4B, Qwen3.6-35B-A3B, Qwen3.6-27B, and NVIDIA-Nemotron-3-Super-120B-A12B for a subset of the
  harness packs (see [NOTICE](NOTICE)).

## Evaluation summary

| Measure | v6-4b | v5-4b | Notes |
|---|---:|---:|---|
| Own locked test / challenge accuracy | **0.929 / 0.911** | 0.867 / 0.861 | Scenario families held out of training |
| 60-case tool-call diagnostic set | 0.900 | 0.917 | Diagnostic, not a locked holdout; regressed slightly |
| Prompt injection, with / without deployment context | **0.924** / 0.867 | 0.894 / 0.903 | External, no training overlap; mixed (ctx up, noctx down) |
| Jev-directory (70 questions) | **0.871** | 0.800 | External |
| Kev decision-v1 (external classification) | **0.800** | 0.755 | External, out-of-domain |
| Kev transfer-v4 (development / locked test, 764 each) | 0.756 / 0.787 | not evaluated | External, out-of-domain; compares to v3-4b's 0.706 / 0.749 |
| JevBench public tiers (easy / standard / hard) | **1.000 / 1.000 / 0.595** | 1.000 / 0.847 / 0.523 | [Benchmark Heaven's JevBench](https://github.com/fstandhartinger/jevbench) v1.2, public items only (231 of 534); not the official JevBench Score |
| JevBench public-proxy Intelligence score | **83.1** | 74.2 | Our own metric from JevBench's own `composite_v12.intelligence()`, judge tier dropped (0 public items) |

Details and the run log: [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/RUNS.md](docs/RUNS.md). Charts: `docs/img/`.

## Calibration

Post-hoc temperature scaling, fitted on the calibration split only, for every question type. Temperatures: choice 2.491, score 1.255,
noul 1.608 (all higher than v5-4b's — the backbone swap changed logit scale, expected). Calibrated ECE on validation: choice 0.009, noul 0.014,
score 0.052. Calibration does not transfer reliably across domains: refit on your own labelled outcomes before relying on thresholds. The
shipped `calibration.json` carries these values.

## Latency methodology

Batch size 1, idle GPU, one NVIDIA DGX Spark (GB10), served via vLLM (the realistic production path for this hybrid backbone — see Known
limitations for why HF Transformers in-process serving is currently much slower). bf16: 74.9 ms p50 / 13.3 decisions/s. NVFP4 (MLP-only): 53.3 ms
p50 / 18.6 decisions/s — a real 1.40x speedup, smaller than v5-4b's 1.66x because only the MLP is quantized and the hybrid backbone carries
relatively more compute in its unquantized attention/linear-attention paths. Both are slower in absolute terms than v5-4b (61.6 ms bf16 / 37.1 ms
NVFP4 via vLLM), but v6-4b-nvfp4 (53.3 ms) is still faster than v5-4b bf16 (61.6 ms) while being far more accurate.

## Known limitations

* **Scope narrowed on purpose**, unchanged from v5: this release is Jev-style decisions only. It is untested on general text classification and
  vulnerable-code detection; treat those as unsupported, not merely weak.
* **HF Transformers in-process serving and training are slow on this backbone.** `causal_conv1d` and `flash-linear-attention` are not installed
  in this environment, so the linear-attention layers fall back to unoptimized reference PyTorch kernels — correct, but far slower (training ran
  ~31-34s/step vs v5-4b's ~18s/step despite having fewer O(n²) attention layers). **vLLM is unaffected**: it has its own native, fast kernels for
  this architecture (`model_type=qwen3_5_text`) and is the recommended serving path. Installing the two packages should close most of the HF gap;
  not yet attempted on this hardware (GB10/Blackwell, compute capability 12.1) so the build risk is unverified.
* **The state-cache optimization (encode the state once, answer several questions cheaply) does not apply to this backbone yet.** It falls back
  to a full forward pass per question (`LinearAttentionLayer` has no `batch_repeat_interleave`) — results are correct, just without the
  multi-question speedup the architecture description advertises for the uniform-attention releases (v3/v5).
* **Backbone swap is not a clean win**: two of five external Jev-style sets regressed slightly versus v5-4b (`ext-injection-noctx` -3.6 points,
  `ext-toolcall-risk` -1.7 points), while own-splits and JevBench improved substantially. Read this as a net win, not a universal one.
* **RLCD made results worse**, not better, in both attempts tried on earlier releases (see v5's CHANGELOG entry); not retried on v6, no
  outcome-based training is in this release.
* Per-pack test counts are small for many of the 49 packs; read per-pack numbers in `docs/BENCHMARKS.md` as noisy.
* Sensitive to how options are defined and to option order; recalibrate and re-test on your own labelled outcomes before relying on thresholds.
* English only; not evaluated adversarially beyond the small adversarial slice.

## Safety and deployment requirements

Do not use a model decision as the only authorization control. Deploy with deterministic policy guardrails
([`open_spark_jev/policy.py`](open_spark_jev/policy.py) is a heuristic starting point, not a security engine), a conservative auto-allow
threshold (default 0.995), logging, least-privilege credentials, sandboxing and egress controls, human approval or escalation for sensitive
actions, and a kill switch. Unsupported input, parsing errors, an unavailable evaluator, low confidence, or a rule conflict must resolve to
`ask` or `deny`, never automatic execution; the bundled gate does this. See [SECURITY.md](SECURITY.md).

## Reproducibility

* Code: this repository; training `scripts/run_v6_4b.sh` (`configs/train/sft_v6.yaml`), evaluation `python -m open_spark_jev.eval.osdg`,
  `python -m open_spark_jev.eval.external`, JevBench `scripts/run_jevbench.sh` / `scripts/run_jevbench_quant.sh`, NVFP4 quantization
  `scripts/quant/ptq_nvfp4.py` (run directly against this repo's own venv with a current `nvidia-modelopt`, not the bundled TensorRT-LLM
  container — its pinned `nvidia-modelopt` predates Qwen3.5 support).
* Data: assembled by `scripts/build_v5_data.py` from `os-datagen` runs plus the code-only packs in `datagen-pipeline/` — unchanged from v5; not
  yet packaged as a single versioned HF dataset release (unlike v1's `abhishek085/spark-s1-osdg-v1`, which is a subset, not the full mix).
* Seeds: single seed per configuration; no confidence intervals reported yet.
* Hardware: one NVIDIA DGX Spark (GB10).

## Versioning and lineage

Release ids are `spark-s1-<size>-v<n>`. Lineage and parents: [docs/model_lineage.yaml](docs/model_lineage.yaml); changes:
[CHANGELOG.md](CHANGELOG.md). Backbone: `Qwen/Qwen3.5-4B` (Apache-2.0). Parent: `spark-s1-4b-v5` (same data/recipe, prior backbone
`Qwen/Qwen3-4B`).
