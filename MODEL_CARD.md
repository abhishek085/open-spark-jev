# Model Card

**spark-s1**, release v5 (`spark-s1-4b-v5`, `spark-s1-1.7b-v5`), 2026-09-22. Part of Open Spark Jev, an open-source project of the Nokast AI
community. This is a very early release; expect it to change quickly.

## Overview

`spark-s1` is a System-1 decision model: given a state (text or JSON) and a typed question with options defined at request time, it returns a
probability for every option and a confidence from one forward pass, without generating text. Each release is a public Qwen3 backbone fine-tuned
with LoRA (r=16, merged into the weights). There is no extra head: the answer is the softmax of the first-token logits restricted to the option
letters, divided by a temperature.

It is not Jev and not affiliated with TypeSafe AI. It follows the same contract and readout idea; its training is a supervised fine-tune, not
TypeSafe's undisclosed method.

## What changed in v5

v5 narrows scope on purpose: training and evaluation are now **Jev-style decisions only** (agent-harness control, tool-call/guardrail gating,
moderation and routing, retrieval gating, and structured records decisions such as entity resolution, fraud/risk and invoice/claims routing).
General-purpose text classification is no longer a training goal. Two earlier release candidates were tried and rejected on the way here:

* **v4** (lower LR, one epoch, more data, still general-purpose) improved on v3 in-domain but regressed on prompt-injection-with-context
  (0.819 -> 0.634) because that pack was under-represented in its mix.
* **RLCD-direct**, tried on both v4's and v5's 1.7B checkpoint on a held-out pool (later filtered to the rows the model actually got wrong or
  was unsure on), made results **worse** across the board both times, including a further collapse on the same injection pack (0.813 -> 0.468 on
  v5). It was not applied to any shipped checkpoint.

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

Question types: Choice (up to 26 options), Score (ordered levels), Boolean. v5 trains on **49 task packs** (up from v3's 15), spanning agent
guardrails and harness engineering (tool-call verification, context pruning, response-quality and citation checks, delegation, error recovery,
plan alignment, trace monitoring), security policy (prompt-injection and message-manipulation gating, PII/output-leak checks, authorization),
moderation and routing (content moderation, ticket triage, model routing), and structured records (entity resolution, fraud/risk scoring,
invoice/insurance/SOC-alert routing, financial triage, semantic linting, row validation, semantic grep). Per-pack test counts are small for many
of the new packs (some under 20 rows); read the aggregate numbers below as the more reliable signal.

## Available variants

| Release id | Backbone | Weights | Notes |
|---|---|---|---|
| `spark-s1-4b-v5` | Qwen3-4B | [`abhishek085/spark-s1-4b-v5`](https://huggingface.co/abhishek085/spark-s1-4b-v5) | Accuracy pick |
| `spark-s1-1.7b-v5` | Qwen3-1.7B | [`abhishek085/spark-s1-1.7b-v5`](https://huggingface.co/abhishek085/spark-s1-1.7b-v5) | Speed pick |
| `spark-s1-4b-v5-nvfp4` | Qwen3-4B | [`abhishek085/spark-s1-4b-v5-nvfp4`](https://huggingface.co/abhishek085/spark-s1-4b-v5-nvfp4) (4.0 GB) | NVFP4 (MLP-only), 1.66x faster via vLLM on Blackwell hardware; needs a GB10/B200-class GPU |

Earlier releases (`v3`, `v4`, simulator-era `sft-v2`, RLCD variants, architecture variants A0-A4) are described in
[docs/MODELS.md](docs/MODELS.md) and are not part of this release. The NVFP4 quantization (1.66x faster on the DGX Spark's Blackwell tensor
cores via vLLM, no loss on JevBench's easy/standard tiers, -4.6 points on the hard tier) is the only accuracy check run on it so far; reproduce
it with `scripts/quant/ptq_nvfp4.py`. The same recipe on the 1.7B cost 9-13 accuracy points across external sets and was not published.

## Training/data summary

* **Data:** 11,792 training rows (19,568 examples with option-order augmentation) across the 49 packs above, drawn from `os-datagen`
  ([`datagen-pipeline/`](datagen-pipeline/)): LLM-written scenarios (labels established by code, text written by an LLM and checked by an
  independent verifier model) plus new code-only rule packs where both the label and the surface text come from deterministic rules, no LLM in
  the loop. Splits: calibration/locked-test/challenge rows are drawn from scenario families never seen in training.
* **Recipe:** LoRA r=16 (alpha 32) on all attention and MLP projections, **1 epoch**, lr **5e-5** (lower and shorter than v3's 3 epochs / 2e-4,
  following the same-style advice that a lower learning rate preserves more base-model capability), cross-entropy plus a Brier regulariser. One
  seed.
* **Not used:** no real public text, no vulnerable-code data, no outcome-based (RLCD) training on the shipped checkpoints.
* Generator and verifier models: Gemma-4-26B-A4B, Qwen3.6-35B-A3B, Qwen3.6-27B, and NVIDIA-Nemotron-3-Super-120B-A12B for a subset of the
  harness packs (see [NOTICE](NOTICE)).

## Evaluation summary

| Measure | 4B | 1.7B | Notes |
|---|---:|---:|---|
| Own locked test / challenge accuracy | 0.867 / 0.861 | 0.695 / 0.671 | Scenario families held out of training |
| 60-case tool-call diagnostic set | 0.917 | 0.900 | Diagnostic, not a locked holdout |
| Prompt injection, with / without deployment context | 0.894 / 0.903 | 0.813 / 0.837 | External, no training overlap |
| Jev-directory (70 questions) | 0.800 | 0.643 | External |
| Kev decision-v1 (external classification) | 0.755 | 0.693 | External, out-of-domain |
| JevBench public tiers (easy / standard / hard) | 1.000 / 0.847 / 0.523 | 0.979 / 0.778 / 0.378 | [Benchmark Heaven's JevBench](https://github.com/fstandhartinger/jevbench) v1.2, public items only (231 of 534); not the official JevBench Score |

Details and the run log: [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/RUNS.md](docs/RUNS.md). Charts: `docs/img/`.

## Calibration

Post-hoc temperature scaling, fitted on the calibration split only, **for every question type this release** (v3 left Boolean/Score at an
unfitted 1.0). Temperatures — 4B: choice 2.026, score 1.149, noul 1.318; 1.7B: choice 1.792, score 1.112, noul 1.608. Calibrated ECE on the
locked test: 4B 0.019, 1.7B 0.033. Calibration does not transfer reliably across domains: refit on your own labelled outcomes before relying on
thresholds. The shipped `calibration.json` carries these values.

## Latency methodology

Batch size 1, idle GPU, one NVIDIA DGX Spark (GB10). Hugging Face Transformers in-process, bf16, median over the 60 diagnostic rows (4B 65.9 ms
p50 / 15.1 decisions/s; 1.7B 29.6 ms / 33.8/s — carried over unchanged from v3: same backbone and prompt set, and LoRA-merged weights don't
change inference cost). Also measured fresh on the 4B via vLLM: bf16 61.6 ms p50 / 16.2/s; NVFP4 (MLP-only) 37.1 ms p50 / 27.1/s — a real 1.66x
speedup from quantization on this hardware.

## Known limitations

* **Scope narrowed on purpose**: this release is Jev-style decisions only. It is untested on general text classification and vulnerable-code
  detection; treat those as unsupported, not merely weak.
* **RLCD made results worse**, not better, in two separate attempts (see "What changed in v5" above); no outcome-based training is in this
  release.
* **NVFP4 quantization costs real accuracy on the 1.7B** (9-13 points across external sets) and was not shipped; the 4B tolerates it much
  better but has only been accuracy-checked on JevBench so far, not the full own-splits/external battery.
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

* Code: this repository; training `scripts/run_v5_4b.sh` / `scripts/run_v5_17b.sh` (`configs/train/sft_v5.yaml` overrides), evaluation
  `python -m open_spark_jev.eval.osdg`, `python -m open_spark_jev.eval.external`, JevBench `scripts/run_jevbench.sh`.
* Data: assembled by `scripts/build_v5_data.py` from `os-datagen` runs plus the code-only packs in `datagen-pipeline/`; not yet packaged as a
  single versioned HF dataset release (unlike v1's `abhishek085/spark-s1-osdg-v1`, which is a subset, not the full v5 mix).
* Seeds: single seed per configuration; no confidence intervals reported yet.
* Hardware: one NVIDIA DGX Spark (GB10).

## Versioning and lineage

Release ids are `spark-s1-<size>-v<n>`. Lineage and parents: [docs/model_lineage.yaml](docs/model_lineage.yaml); changes:
[CHANGELOG.md](CHANGELOG.md). Backbones: `Qwen/Qwen3-4B`, `Qwen/Qwen3-1.7B` (Apache-2.0).
