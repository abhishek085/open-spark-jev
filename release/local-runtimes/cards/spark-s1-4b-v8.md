---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: text-classification
tags:
  - decision-model
  - agent
  - bf16
  - spark-s1
datasets:
  - nvidia/When2Call
  - MadeAgents/xlam-irrelevance-7.5k
  - Team-ACE/ToolACE
  - Lakera/gandalf_ignore_instructions
---

# Model Card

**spark-s1**, release v8, **bf16 build** (`spark-s1-4b-v8`), the full-precision weights that [`spark-s1-4b-v8-nvfp4`](https://huggingface.co/abhishek085/spark-s1-4b-v8-nvfp4) (2026-10-02) was quantised from. Published 2026-10 so the model can run on GPUs without NVFP4 support, be converted to GGUF ([`spark-s1-4b-v8-GGUF`](https://huggingface.co/abhishek085/spark-s1-4b-v8-GGUF)) and be served by hosted inference providers. Part of Open Spark Jev, an open-source project of the Nokast AI community. This is a
very early release; expect it to change quickly.

## Overview

`spark-s1` is a System-1 decision model: given a state (text or JSON) and a typed question with options defined at request time, it returns a
probability for every option and a confidence from one forward pass, without generating text. Each release is a public Qwen backbone fine-tuned
with LoRA (r=16, merged into the weights). There is no extra head: the answer is the softmax of the first-token logits restricted to the option
letters, divided by a temperature.

It is not Jev and not affiliated with TypeSafe AI. It follows the same contract and readout idea; its training is a supervised fine-tune, not
TypeSafe's undisclosed method.

## What changed in v8

v8 is a **weight average** (50/50, same architecture and tensor shapes) of two checkpoints trained from the same Qwen3.5-4B backbone:
`spark-s1-4b-v6` and a new SFT run on a much wider data mix. It then ships with a **re-fitted calibration temperature**. No new architecture,
no new head, same single-forward-pass letter-logit readout. This repo holds the bf16 weights; the NVFP4 (MLP-only) build is `spark-s1-4b-v8-nvfp4`.

* **New decision types.** Agent-harness decisions: which tool to call next (or `finish` when the request is done), whether to call a tool, ask the
  user, or say no tool fits, how risky a terminal command is under a stated policy, and whether a tool's output tries to steer the agent off the
  user's task.
* **Better calibrated.** v6 predicts wrong answers with high confidence more often than it should (JevBench-hard answers that are wrong at
  >=0.9 confidence: v6-nvfp4 21, v8-nvfp4 11; on our held-out agent and public agent sets, questions answered wrongly at >=0.9 confidence fall from
  14.6-14.8% to 0.2-1.6%). v8's scalar temperature is higher than v6's shipped one (2.5 vs 1.481).
* **Not a clean win, and not a replacement for v6 everywhere.** It adds coverage and calibration; it does not improve general accuracy. On JevBench
  hard, v8-nvfp4 (0.586) is *below* the published v6-nvfp4 (0.622); the bf16 builds tie (0.595). Long-document and calendar-arithmetic questions did
  not improve (see Known limitations).

**Who this is for.** Use v8-nvfp4 when decisions come from an agent harness (tool choice, `finish`, call/ask/refuse, command risk, screening
tool output), for example behind a locally running agent or a decision-measurement app such as JevControl, and when confident wrong answers cost you
more than a few points of hard-tier accuracy. Prefer `spark-s1-4b-v6-nvfp4` if your decisions resemble JevBench's hard long-document reasoning.

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
* Documents longer than about 2,000 tokens were not part of the new training data (rows over the training length limit were dropped), so v8 is not
  validated for long-document reasoning; JevBench's long-policy and multi-hop hard families did not improve over v6.

## Supported decision tasks

Question types: Choice (up to 26 options), Score (ordered levels), Boolean. v8 keeps v6's 49 task packs and adds:

* **Next action in an agent loop** over a stated tool list and step history, including `finish` and `clarify`.
* **Tool-call gating:** call a tool / ask the user for a missing argument / no listed tool fits.
* **Tool selection** among up to 12 candidate tools.
* **Terminal-command risk** (read-only / mutating / destructive / privileged / exfiltration) under an explicit policy text.
* **Tool-output screening:** does a tool result try to instruct the agent (prompt injection) or is it ordinary content.
* **Five reasoning-style programmatic families** (effective dating, policy precedence, sampling probability, ambiguous routing, weighted
  tradeoff), generated by code with held-out skins and phrasings.

## Available variants

| Release id | Backbone | Weights | Notes |
|---|---|---|---|
| `spark-s1-4b-v8` | Qwen3.5-4B | `abhishek085/spark-s1-4b-v8` | bf16, 8.4 GB, any recent GPU (this repo) |
| `spark-s1-4b-v8-nvfp4` | Qwen3.5-4B | `abhishek085/spark-s1-4b-v8-nvfp4` | NVFP4 (MLP-only), 4.9 GB, needs a GB10/B200-class GPU |
| `spark-s1-4b-v8-GGUF` | Qwen3.5-4B | `abhishek085/spark-s1-4b-v8-GGUF` | Q8_0 / Q6_K / bf16 GGUF for llama.cpp and Ollama |

The bf16 numbers in this card ("v8 bf16" columns) are for exactly these weights. Earlier releases (`v3`, `v4`, `v5`, `v6`) are described in [docs/MODELS.md](https://github.com/abhishek085/open-spark-jev/blob/main/docs/MODELS.md) and the CHANGELOG.

## Training/data summary

* **v6 component:** unchanged: 11,792 rows over 49 packs ([`datagen-pipeline/`](https://github.com/abhishek085/open-spark-jev/blob/main/datagen-pipeline/)), LoRA r=16 (merged), one epoch, lr 5e-5.
* **New component** (trained from the Qwen3.5-4B base, same LoRA recipe, one epoch, cross-entropy plus Brier regulariser with weight 1.0; 32,500 rows):
  8,000 rows sampled from the v5 mix; 8,000 rows from the five programmatic families above (code-computed labels, verified by independent
  re-derivation, about 35% soft-target rows); 6,000 synthetic Hermes-style agent-decision rows (code-computed labels, held-out phrasing
  and injection templates); about 8,000 rows converted from public datasets (see below, 25% of the mix); 2,500 replay rows whose targets are v6's own
  predicted distributions (KL regularisation toward v6).
* **Public data (minority share, reported separately in evaluation):** `nvidia/When2Call` (CC-BY-4.0), `MadeAgents/xlam-irrelevance-7.5k`
  (CC-BY-4.0), `Team-ACE/ToolACE` (Apache-2.0), `Lakera/gandalf_ignore_instructions` (MIT). Converted into typed decision rows by
  `scripts/data/build_agent_public.py`. Attribution and caveats: [NOTICE](https://github.com/abhishek085/open-spark-jev/blob/main/NOTICE). The When2Call test split is evaluation-only.
* **Guards:** every training file was checked against fingerprints of the JevBench public items and of all our external evaluation sets
  (`scripts/tools/benchmark_overlap.py`): zero matches.
* **Not used:** no outcome-based (RLCD) training, no data from JevBench, no private user traces.
* Generator and verifier models for the LLM-written packs: see [NOTICE](https://github.com/abhishek085/open-spark-jev/blob/main/NOTICE).

## Evaluation summary

All numbers are from this repository's own runs: single seed, **v8-nvfp4 vs v6-nvfp4 (the published v6 build), the same quantisation recipe,
both served with vLLM under the same harness**, each at its own shipped calibration. JevBench public-test items are used for internal comparison
only; this is not the official JevBench Score.

| Measure | v6-nvfp4 (published) | v8-nvfp4 | Notes |
|---|---:|---:|---|
| JevBench public tiers (easy / standard / hard) | 1.000 / 0.972 / 0.622 | 1.000 / 0.986 / 0.586 | JevBench v1.2 public items only (231 of 534). **Hard accuracy is lower than v6-nvfp4's.** |
| JevBench public-proxy Intelligence | 83.2 | 82.2 | `composite_v12.intelligence()`, judge tier dropped |
| JevBench hard: Brier / ECE | 0.617 / 0.280 | 0.623 / 0.194 | Shipped temperatures |
| JevBench hard: wrong at >=0.8 / >=0.9 confidence | 26 / 21 | 20 / 11 | Of 111 items |

External and held-out sets at each model's shipped temperature (v6 1.481, v8 2.5). External rows use the standard external harness; the Hermes-style, public-agent and programmatic rows are choice questions scored from option logits. None of these sets is part of v6's training data:

| Measure | v6 bf16 | v8 bf16 (this repo) | v8-nvfp4 |
|---|---:|---:|---:|
| Prompt injection, with context: accuracy / Brier / ECE | 0.924 / 0.124 / 0.055 | 0.937 / 0.113 / 0.044 | 0.935 / 0.108 / 0.041 |
| Prompt injection, no context: accuracy / Brier / ECE | 0.867 / 0.242 / 0.116 | 0.890 / 0.193 / 0.090 | 0.884 / 0.204 / 0.097 |
| Jev-directory (70): accuracy / Brier / ECE | 0.871 / 0.212 / 0.116 | 0.914 / 0.178 / 0.070 | 0.886 / 0.205 / 0.064 |
| Kev decision-v1: accuracy / Brier / ECE | 0.800 / 0.344 / 0.149 | 0.802 / 0.316 / 0.109 | 0.795 / 0.324 / 0.112 |
| 60-case tool-call diagnostic: accuracy / Brier / ECE | 0.900 / 0.143 / 0.074 | 0.900 / 0.122 / 0.059 | 0.900 / 0.140 / 0.053 |
| Hermes-style agent decisions (our generator, held-out phrasing): accuracy / Brier / wrong at >=0.9 | 0.760 / 0.416 / 14.8% | 0.953 / 0.073 / 0.2% | 0.940 / 0.094 / 0.2% |
| Public agent evals (When2Call test, held-out ToolACE tools, Gandalf): accuracy / Brier / wrong at >=0.9 | 0.764 / 0.401 / 14.6% | 0.797 / 0.282 / 0.9% | 0.791 / 0.299 / 1.6% |
| Programmatic families (bf16 only; not run on NVFP4): accuracy / Brier / wrong at >=0.9 | 0.745 / 0.408 / 7.7% | 0.851 / 0.228 / 0.3% | not run |

The Hermes-style and programmatic sets come from the same generators as v8's new training data (held-out phrasing, skins and templates), so
they measure how well the new families were learned, not general ability. The public agent evals use data v8 never trained on (When2Call test is
not in the training set; ToolACE evaluation tools are disjoint from training tools).

Kev transfer-v4 (development / locked test, 764 each; external, out-of-domain) was measured on the bf16 build only: v8 0.762 / 0.775 (Brier 0.353 / 0.327)
vs v6 0.756 / 0.787 (0.409 / 0.363): development up, locked test 1.2 points lower, better calibrated on both.

### What quantisation cost (v8 NVFP4 vs the bf16 build it came from)

| Measure | v8 bf16 | v8 NVFP4 |
|---|---:|---:|
| JevBench easy / standard / hard | 1.000 / 1.000 / 0.595 | 1.000 / 0.986 / 0.586 |
| JevBench Intelligence | 83.1 | 82.2 |
| JevBench hard wrong at >=0.9 | 10 | 11 |
Per-set differences are in the table above (bf16 vs NVFP4 columns): about one point or less except Jev-directory (0.914 to 0.886).

Details and the run log: [docs/BENCHMARKS.md](https://github.com/abhishek085/open-spark-jev/blob/main/docs/BENCHMARKS.md), [docs/RUNS.md](https://github.com/abhishek085/open-spark-jev/blob/main/docs/RUNS.md).

## Calibration

Post-hoc scalar temperatures, shipped in `calibration.json`: **choice 2.5, score 1.192, noul 1.495**. The choice temperature was selected as the
single value that balances ECE, Brier and confident-wrong rate across five datasets that exclude JevBench (own v5 splits, the programmatic
families, Hermes-style decisions, public agent evals, long-document set); it was not fitted on JevBench. Calibrated ECE per-dataset fits are in the
run logs. v6's *shipped* choice temperature is 1.481 (the figure of 2.491 in v6's card is the own-split fit, not the shipped value); re-fitting v6 to
~3 would remove most of its confident-wrong answers too, which is why v8's temperature is much higher. Calibration does not transfer reliably across
domains: refit on your own labelled outcomes before relying on thresholds. A higher temperature lowers confidence everywhere: on the easy
JevBench tiers v8's Brier is 0.0007 vs v6's 0.000, a small price.

## Latency methodology

Not re-measured for v8. v8 has the same architecture and parameter count as v6, so speed should match v6-nvfp4's measured 53.3 ms p50 / 18.6 decisions/s (and 74.9 ms bf16).

## Known limitations

* **JevBench hard accuracy: v8-nvfp4 0.586 vs v6-nvfp4 0.622 (the bf16 builds tie at 0.595).** The families v8 gained on are agent and programmatic decisions; JevBench's hard long-policy (v6 7/19, v8 7/19),
  multi-hop (16/18, 15/18) and temporal-numeric (1/15, 0/15) families did not improve, and temporal-numeric is near zero for every release.
* **Long documents.** Rows over the 2,048-token training limit were dropped, so the new data contains no long documents. A follow-up run that added
  4,500 synthetic long-document rows learned its own data (0.80 vs 0.49 accuracy) but lowered JevBench-hard to 0.523, so it was not shipped.
* **Own locked challenge accuracy (bf16 build) is 0.8 points below v6** (0.902 vs 0.910); locked test is equal (0.928 vs 0.929). Not re-run on NVFP4.
* **Quantisation costs a little:** JevBench Intelligence 83.1 (bf16) to 82.2, Jev-directory 0.914 to 0.886. v6's quantisation cost nothing, v8's does not.
* **Not measured for this release:** v8 latency, the programmatic families and own v5 splits on the NVFP4 build, v6-nvfp4 on the external and held-out sets (the table compares to v6 bf16 there).
* **Weight averaging is a heuristic.** v8 is a 50/50 average chosen after trying three-way averaging (worse) and the unaveraged new run (worse on
  JevBench hard, 0.532). It is validated empirically, not derived.
* **Synthetic and LLM-generated sources.** The Hermes-style set is generated by our own code; the public datasets are partly LLM-generated.
  Gains on our own generators overstate real-world gains.
* **Calibration of the quantised variant** uses the same scalar temperatures as the bf16 build; the held-out tables above are measured on the NVFP4 build itself.
* Everything v6's card lists still applies: HF in-process serving is slow on this hybrid backbone (use vLLM), the state-cache optimisation does not
  apply, sensitivity to option order and definition, English only, per-pack test counts are small, single seed, no confidence intervals.
* A statistical caution: JevBench hard has 111 items (standard error about 4.7 points), so differences of a few points there are within noise.

## Safety and deployment requirements

Do not use a model decision as the only authorization control. Deploy with deterministic policy guardrails
([`open_spark_jev/policy.py`](https://github.com/abhishek085/open-spark-jev/blob/main/open_spark_jev/policy.py) is a heuristic starting point, not a security engine), a conservative auto-allow
threshold (default 0.995), logging, least-privilege credentials, sandboxing and egress controls, human approval or escalation for sensitive
actions, and a kill switch. Unsupported input, parsing errors, an unavailable evaluator, low confidence, or a rule conflict must resolve to
`ask` or `deny`, never automatic execution; the bundled gate does this. See [SECURITY.md](https://github.com/abhishek085/open-spark-jev/blob/main/SECURITY.md).

## Reproducibility

* Code: this repository. v8 weights = uniform average of `checkpoints/v6-4b` and the SFT run `configs/train/sft_v7_11.yaml`
  (`python -m open_spark_jev.experimental.model_soup`), then `calibration.json` as above. Data: `scripts/data/build_agent_public.py`,
  `datagen-pipeline/src/os_datagen/programmatic/v2_all.py`, `.../v2_hermes.py`, mix `data/mixes/v7_11_train.jsonl`.
* Evaluation: `python -m open_spark_jev.eval.osdg`, `python -m open_spark_jev.eval.external`, `scripts/run_jevbench.sh`; calibration report
  `scripts/analysis/calib_report.py`, `scripts/analysis/jevbench_calib.py`. NVFP4: `scripts/quant/ptq_nvfp4.py` (NVFP4_MLP_ONLY_CFG, calibration
  prompts `runs/quant/calib_prompts_v8.jsonl`), served with vLLM and scored with `scripts/run_jevbench_quant.sh`.
* Weights hash: this repo's bf16 `model.safetensors` sha256 `79fac11d7ceb9ec6c4e124559d013bc02a9299b7c3e17896830203eb4e681ce9`; the NVFP4 build's is `dccffb72a26e747aa983c969da334e6c6066cab7465a9b4222901380028f2439`.
* Seeds: single seed per configuration; no confidence intervals. Hardware: one NVIDIA DGX Spark (GB10).

## Versioning and lineage

Release ids are `spark-s1-<size>-v<n>`. Lineage and parents: [docs/model_lineage.yaml](https://github.com/abhishek085/open-spark-jev/blob/main/docs/model_lineage.yaml); changes:
[CHANGELOG.md](https://github.com/abhishek085/open-spark-jev/blob/main/CHANGELOG.md). Backbone: `Qwen/Qwen3.5-4B` (Apache-2.0). Parents: `spark-s1-4b-v6` (averaged) and an unreleased SFT run on the
v7.11 mix.
