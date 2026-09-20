# Benchmarks

> Section labels: **B-numbers** are result sections in this file. **M-numbers** are milestones in [ROADMAP.md](ROADMAP.md); **R-numbers** are run-log entries in [RUNS.md](RUNS.md). The three are independent.

All numbers on one DGX Spark (GB10), repo commit as of 2026-09-18. Test set:
`data/benchmarks/sim_test.jsonl` (`osj simulate --n 300 --seed 1`, 1800 records, 6 domains,
~10% prompt-injected, ~30% with an abstain option). Hard labels are *sampled from the
posterior*, so the Bayes-optimal predictor does not reach 100% accuracy; soft Brier against
the posterior is the calibration metric that has a true zero.

## Ceiling (Bayes-optimal argmax on the known posterior)
| domain | Bayes accuracy | mean max-posterior |
|---|---|---|
| routing | 0.843 | 0.807 |
| security (noul) | 0.983 | 0.976 |
| risk (score) | 0.633 | 0.582 |
| moderation | 0.813 | 0.799 |
| incident | 0.843 | 0.820 |
| game | 0.790 | 0.752 |
| **overall** | **0.818** | |

## B6: base Qwen3-1.7B, zero-shot, HF bf16 (no training, T = 1)
`runs/eval_base_zero_shot.json` (prompt convention: bare letter as first assistant token)

| slice | acc | macro-F1 | ECE | Brier | soft Brier vs posterior |
|---|---|---|---|---|---|
| choice/routing | 0.44 | 0.35 | 0.53 | 1.09 | 0.82 |
| choice/moderation | 0.18 | 0.10 | 0.80 | 1.57 | 1.34 |
| choice/incident | 0.36 | 0.13 | 0.64 | 1.28 | 1.06 |
| choice/game | 0.14 | 0.09 | 0.86 | 1.72 | 1.40 |
| noul/security | 0.60 | 0.26 | 0.40 | 0.79 | 0.76 |
| score/risk | 0.31 | 0.16 | 0.68 | 1.37 | 0.93 |
| **overall** | **0.34** | 0.24 | **0.65** | 1.30 | |

* injection flip rate: **0.67** (argmax differs from posterior argmax on injected states)
* throughput (HF, one unique state per record, no cross-state batching): 24 decisions/s

Reading: the untrained backbone already understands the menu format (it never emits an
invalid label by construction) but is wildly over-confident: near-one-hot distributions
(ECE 0.65) and it follows injected instructions most of the time. This is the "before"
row for Phase 1 / Phase 2 and for the served-engine comparison. (An earlier variant with an
explicit `Answer:` prefix scored acc 0.39 / ECE 0.60; it was dropped because chat endpoints
cannot prefill the assistant turn, see docs/DGX_SPARK.md.)

## Latency (HF backend, bf16, in-process)
Smoke test: 4 questions on a 188-token state, cached prefix, 61.9 ms total, 3.9 GB peak.
Full grid: `python -m open_spark_jev.eval.latency --backend hf --model models/Qwen3-1.7B`
(to be filled in with M9).

## Served engine parity (trtllm-serve 1.2.1, PyTorch backend, bf16, chat logprobs path)
`runs/eval_base_trtllm_serve_300.json`: first 300 records of sim_test (all routing), base
model, gateway-style chat backend (`top_logprobs=20`, `enable_thinking=false`) vs the
in-process HF run on the same slice.

| backend | acc | ECE | Brier | soft Brier vs posterior | injection flip |
|---|---|---|---|---|---|
| HF bf16 in-process (state KV cache) | 0.440 | 0.531 | 1.093 | 0.817 | 0.67 (all domains) |
| trtllm-serve 1.2.1, chat `top_logprobs` | 0.433 | 0.535 | 1.099 | 0.819 | 0.73 (routing only) |

Differences are at bf16 noise level (label logits are O(50), kernels differ). Served
throughput here is 15 decisions/s from a *single sequential* client with one HTTP call per
question; concurrency and per-state batching are the M9 work.

## TensorRT-LLM version retest (1.2.1 vs 1.3.0rc13)
Same 300-record routing slice, same backbone, chat-logprobs path, sequential single client.

| engine | acc | ECE | Brier | soft Brier vs posterior | injection flip |
|---|---|---|---|---|---|
| HF bf16 in-process | 0.440 | 0.531 | 1.093 | 0.817 | 0.67 (all domains) |
| trtllm-serve 1.2.1 | 0.433 | 0.535 | 1.099 | 0.819 | 0.73 (routing only) |
| trtllm-serve 1.3.0rc13 | 0.413 | 0.557 | 1.118 | 0.832 | 0.73 (routing only) |

All three agree within bf16 noise (label logits are O(50), kernels/scheduling differ slightly
between engine versions and builds). 1.3.0rc13's OpenAI-compatible chat-logprobs path behaves
identically to 1.2.1's for this workload: same request shape, same `chat_template_kwargs`
support, served model id differs (`Qwen3-1.7B` vs the full path) but the client already
resolves this via `GET /v1/models` rather than hardcoding it. No changes needed to
`serve/client.py` or `prompting.py` to move between versions.

## B7: SFT (Phase 1) vs zero-shot base, held-out test set

`runs/eval_sft.json` against `data/benchmarks/sim_test.jsonl` (1800 records; read
moderation/incident/game with the leakage caveat above, routing/risk/security are clean).

| slice | acc | ECE | Brier |
|---|---|---|---|
| choice/game | 0.780 | 0.061 | 0.346 |
| choice/incident | 0.830 | 0.031 | 0.253 |
| choice/moderation | 0.800 | 0.029 | 0.286 |
| choice/routing | 0.837 | 0.030 | 0.272 |
| noul/security | 0.983 | 0.010 | 0.023 |
| score/risk | 0.617 | 0.042 | 0.510 |
| **overall** | **0.808** | **0.020** | 0.282 |

injection flip rate **0.038** (down from the untrained base model's 0.67). Two epochs,
soft-target cross-entropy + Brier regularizer, temperature fit per question type -
`docs/COOKBOOK.md` section 7's recipe, unmodified.

## B8a: RLCD-direct, first attempt - a real negative result from a real bug

`configs/train/rlcd_direct.yaml`'s exact-policy-gradient reward included an **unweighted**
per-action term rewarding full confidence on the single *sampled* hard label, active
regardless of config (not gated by `reward_weights` at all - a bug, not a deliberate choice).
For any record where the true posterior isn't one-hot (most of them), that term directly
fights the Brier distributional term next to it. Result, on the same held-out test set:

| slice | SFT acc | RLCD-direct acc | SFT ECE | RLCD-direct ECE |
|---|---|---|---|---|
| choice/game | 0.780 | 0.783 | 0.061 | 0.073 |
| choice/incident | 0.830 | 0.837 | 0.031 | 0.060 |
| choice/moderation | 0.800 | 0.813 | 0.029 | 0.053 |
| choice/routing | 0.837 | 0.840 | 0.030 | 0.049 |
| noul/security | 0.983 | 0.987 | 0.010 | 0.010 |
| score/risk | 0.617 | 0.600 | 0.042 | 0.050 |
| **overall** | **0.808** | **0.810** | **0.020** | **0.036** |

ECE got worse on every single domain; injection flip rate rose from 0.038 to 0.065; accuracy
barely moved (+0.002 overall). **This is a genuine finding about reward composition, not
noise**: an unweighted hard-label term and a distributional calibration term fighting to a
standstill on accuracy while calibration loses is exactly the failure mode Claim A in
`docs/RESEARCH.md` R1 predicted for *sampled* RL (mechanism 3/GRPO) - finding it had crept
into mechanism 2 as well, via a bug rather than by design, is itself useful evidence for how
easy this failure mode is to introduce by accident. Root-caused and fixed same-day (the hard
label term is now `reward_weights["correctness"]`, off by default); full checkpoint, eval,
and training log preserved at `runs/archive_buggy_rlcd_direct_20260918/` rather than deleted.
Corrected rerun below.

## B8b: RLCD-direct, corrected (`correctness: 0.0`, pure Brier + shaping reward)

Same held-out test set, checkpoint retrained from scratch with the fix from B8a.

| slice | SFT acc | RLCD acc | SFT ECE | RLCD ECE | SFT Brier | RLCD Brier |
|---|---|---|---|---|---|---|
| choice/game | 0.780 | 0.777 | 0.061 | 0.043 | 0.346 | 0.336 |
| choice/incident | 0.830 | 0.843 | 0.031 | 0.040 | 0.253 | 0.245 |
| choice/moderation | 0.800 | 0.810 | 0.029 | 0.044 | 0.286 | 0.281 |
| choice/routing | 0.837 | 0.837 | 0.030 | 0.042 | 0.272 | 0.273 |
| noul/security | 0.983 | 0.987 | 0.010 | 0.014 | 0.023 | 0.024 |
| score/risk | 0.617 | 0.623 | 0.042 | 0.042 | 0.510 | 0.496 |
| **overall** | **0.808** | **0.813** | **0.020** | **0.020** | 0.282 | 0.276 |

injection flip rate **0.038 -> 0.027** (down ~30% relative). Fitted per-type temperatures on
this run's own validation split came back close to 1.0 (0.71-1.02) versus the buggy run's
1.09-2.07 - direct confirmation the policy is no longer producing artificially sharp logits
that needed heavy post-hoc cooling. KL-to-reference stayed an order of magnitude smaller
throughout training (0.001-0.03 vs the buggy run's 0.04-0.15), consistent with a policy that
only has reason to drift when chasing calibration, not when also chasing hard-label
confidence.

**Reading**: no dramatic calibration win over SFT alone - overall ECE is flat, and a few
individual domains (routing, security) show a small ECE regression even under the corrected
reward. What moved clearly and consistently: Brier (a proper scoring rule, the reward's
actual optimization target) improved on 5 of 6 domains, and injection robustness improved
substantially. This is a modest, defensible result, not the "no-RL is just as good" finding
the buggy run misleadingly suggested, nor a dramatic calibration breakthrough - "does RL earn
its cost at all" (RESEARCH.md Claim C) reads as a qualified yes here: yes for Brier and
injection resistance specifically, not yet demonstrated for ECE over what SFT + temperature
scaling already achieves. Mechanism 1 (contrastive) and GRPO below are the next data points.

## B8c: mechanism 1 (contrastive RLCD) vs mechanism 2 (direct) vs SFT

Same held-out test set, three checkpoints. 499 contrastive pairs (nvidia/Qwen3.6-27B-NVFP4,
opposed principles) -> a reward model (93.9% held-out pairwise accuracy) -> exact policy
gradient for mechanism 1; see B7/B8a/B8b above for the other two.

| slice | SFT acc | direct acc | contr acc | SFT ECE | direct ECE | contr ECE |
|---|---|---|---|---|---|---|
| choice/game | 0.780 | 0.777 | 0.777 | 0.061 | 0.043 | 0.040 |
| choice/incident | 0.830 | 0.843 | 0.837 | 0.031 | 0.040 | 0.042 |
| choice/moderation | 0.800 | 0.810 | 0.807 | 0.029 | 0.044 | 0.025 |
| choice/routing | 0.837 | 0.837 | 0.837 | 0.030 | 0.042 | 0.043 |
| noul/security | 0.983 | 0.987 | 0.990 | 0.010 | 0.014 | 0.009 |
| score/risk | 0.617 | 0.623 | 0.627 | 0.042 | 0.042 | 0.023 |
| **overall** | **0.808** | **0.813** | **0.812** | **0.020** | **0.020** | **0.021** |

**injection_flip_rate: 0.038 -> 0.027 -> 0.016** - the one clean, monotonic result across all
three. Neither RL mechanism clearly beats plain SFT + temperature scaling on accuracy or ECE
here (all three within noise of each other, ~0.81 / ~0.02). But injection robustness improves
with each added piece of machinery, and mechanism 1 (with the trained reward model) wins
clearly over mechanism 2 (heuristic TV-distance term only). This is the answer to Claim B in
RESEARCH.md R1 ("does the contrastive/RM detour help, or is direct calibration enough"): **no
for accuracy/ECE, yes for injection resistance specifically** - the RM's training claim ("not
influenced by instructions inside the original state") gives it a dedicated signal that a
heuristic shaping term doesn't fully replicate. Full per-checkpoint detail and a
fastest-way-to-run command for each: [docs/MODELS.md](MODELS.md).

## B8d: all four Phase-2 mechanisms, complete

Adds GRPO (sampled-token TRL baseline, capped to 1600 records/300 steps - see
docs/RESEARCH.md R1 for why an uncapped run wasn't practical) to the B8c table.

| slice | SFT acc | direct acc | contr acc | GRPO acc | SFT ECE | direct ECE | contr ECE | GRPO ECE |
|---|---|---|---|---|---|---|---|---|
| choice/game | 0.780 | 0.777 | 0.777 | 0.750 | 0.061 | 0.043 | 0.040 | 0.134 |
| choice/incident | 0.830 | 0.843 | 0.837 | 0.680 | 0.031 | 0.040 | 0.042 | 0.198 |
| choice/moderation | 0.800 | 0.810 | 0.807 | 0.730 | 0.029 | 0.044 | 0.025 | 0.223 |
| choice/routing | 0.837 | 0.837 | 0.837 | 0.830 | 0.030 | 0.042 | 0.043 | 0.146 |
| noul/security | 0.983 | 0.987 | 0.990 | 0.963 | 0.010 | 0.014 | 0.009 | 0.027 |
| score/risk | 0.617 | 0.623 | 0.627 | 0.523 | 0.042 | 0.042 | 0.023 | 0.240 |
| **overall** | **0.808** | **0.813** | **0.812** | **0.746** | **0.020** | **0.020** | **0.021** | **0.158** |

injection_flip_rate: SFT 0.038, direct 0.027, contrastive 0.016, **GRPO 0.156**.

**Claim A from docs/RESEARCH.md R1, confirmed cleanly**: GRPO is worse than all three exact
menu policy gradient checkpoints on every single axis - accuracy down ~6pts, ECE up nearly
8x, injection robustness down 4-10x. This is close to the textbook argmax-collapse failure
mode the exact objective was built to avoid: a 0/1-correctness reward on *sampled* actions
pushes the policy toward sharp, overconfident distributions, because it never sees the full
distribution it's optimizing - so it can't be penalized for spreading probability mass
correctly, only rewarded for spiking it on whichever single action got sampled and scored
well. One honest caveat: GRPO trained on a smaller budget (1600 records/300 steps vs ~16,200
examples for the other three) for practicality, so this isn't perfectly matched on training
scale. That said, an 8x ECE regression and a 4-10x injection-robustness regression are far
larger than a ~6pt accuracy gap would predict from undertraining alone - the direction and
magnitude are consistent with the mechanism-level prediction, not just "GRPO needed more
steps." A matched-budget rerun would be needed to fully separate the two effects; noted as a
follow-up in docs/ROADMAP.md.

**Overall reading across all four (B8a-B8d)**: on this benchmark, the exact-vs-sampled
distinction (Claim A) is the dominant effect by a wide margin - all three exact mechanisms
cluster tightly (~0.81 acc, ~0.02 ECE) regardless of whether they use a contrastive reward
model (mechanism 1), a direct calibration objective (mechanism 2), or nothing calibration-RL
at all (plain SFT). The contrastive/RM machinery's one clear, distinguishing win is injection
robustness (Claim B), not accuracy or ECE. Full per-checkpoint detail and a
fastest-way-to-run command for each: [docs/MODELS.md](MODELS.md).

## Teacher comparison (ground-truth anchored, `eval/teacher_benchmark.py`)
20 records/domain from `sim_test.jsonl`, known posteriors. `qwen27b` = `nvidia/Qwen3.6-27B-NVFP4`
via the sibling project's already-running vLLM server (reused read-only, not launched by this
repo). `nemotron120b` / `gptoss120b` are queued: both refused their memory preflight (need
65-70GB available; box had ~56GB with the sibling project's containers up) rather than risk an
OOM on a shared box - re-run once memory is free, see docs/COOKBOOK.md.

| teacher | domain | n | soft Brier vs posterior | argmax agreement |
|---|---|---|---|---|
| qwen27b | routing | 20 | 0.120 | 0.90 |
| qwen27b | security | 20 | 0.014 | 1.00 |
| qwen27b | risk | 20 | 0.172 | 0.65 |
| qwen27b | moderation | 20 | 0.137 | 0.80 |
| qwen27b | incident | 20 | 0.256 | 0.65 |
| qwen27b | game | 20 | 0.155 | 0.75 |
| **qwen27b overall** | | **120** | **0.142** | **0.79** |
| nemotron120b | routing | 20 | 0.186 | 0.85 |
| nemotron120b | security | 20 | 0.085 | 0.95 |
| nemotron120b | risk | 20 | 0.202 | 0.45 |
| nemotron120b | moderation | 20 | 0.301 | 0.70 |
| nemotron120b | incident | 20 | 0.940 | 0.25 |
| nemotron120b | game | 20 | 0.177 | 0.75 |
| **nemotron120b overall** | | **120** | **0.315** | **0.66** |
| gemma26b | routing | 20 | 0.060 | 0.90 |
| gemma26b | security | 20 | 0.017 | 1.00 |
| gemma26b | risk | 20 | 0.145 | 0.70 |
| gemma26b | moderation | 20 | 0.115 | 0.75 |
| gemma26b | incident | 20 | 0.265 | 0.55 |
| gemma26b | game | 20 | 0.170 | 0.80 |
| **gemma26b overall** | | **120** | **0.129** | **0.78** |
| gptoss120b | - | - | **blocked** - see docs/DGX_SPARK.md | |

**Nemotron result, notably worse than Qwen despite being much larger**: overall soft Brier
0.315 vs qwen27b's 0.142 - a real, somewhat counterintuitive finding, not a fluke of one bad
domain. `incident` is the outlier within it (0.940 soft Brier, 0.25 argmax agreement - barely
above the 4-way random-chance floor), but even excluding it Nemotron trails Qwen on every
other domain too (e.g. routing 0.186 vs 0.120, moderation 0.301 vs 0.137). Model size and a
different architecture (hybrid Mamba/MoE vs dense) did not translate into better grading
calibration on this benchmark - a useful caution against assuming "bigger teacher = better
labels" without measuring it, which is exactly the point of building this benchmark rather
than trusting teacher choice by reputation. Getting this number took two failed container
launches first: vLLM's `--gpu-memory-utilization` is the *total* memory budget as a fraction
of the whole box, not an add-on reservation, and an early fix moved it the wrong direction -
see the commit history and `scripts/teachers/serve_nemotron.sh` for the corrected sizing.

For scale: the untrained 1.7B student's own Brier on the same slices is ~1.1-1.7 (see B6
above); a 0.142 soft Brier from the 27B teacher is close to the theoretical floor set by the
domains' inherent ambiguity (routing/game/risk posteriors top out around 0.75-0.82 max
probability by construction, see the Bayes-ceiling table). This is the calibration target
Phase 1 distillation and Phase 2 RLCD are trying to close the gap toward.

**Incident during this run**: the shared `nokast-teacher-vllm` container crashed under 6-way
concurrent grading requests (`EngineDeadError`, a latent flashinfer/cuDNN fp8-GEMM-autotune
CUDA kernel-launch failure - unrelated to JSON mode or to this session's other downloads; see
full traceback in the session log). It was restarted with the sibling project's own
`ops/teacher_server.sh start` and re-run cleanly at concurrency 3, which is now the enforced
default for this teacher in `configs/teachers.yaml` (`max_concurrency: 3`).

## Known issue: train/test state leakage in the B6-era corpora (found 2026-09-18)

`data/synthetic/sim_train.jsonl` and `data/benchmarks/sim_test.jsonl` (both generated before
this fix) have near-total exact-text overlap in four of six domains, because those domains'
`render()` functions had no or too-narrow random fields, so their finite combinatorial state
space was exhausted many times over by 3000 training samples per domain:

| domain | overlapping test records | out of |
|---|---|---|
| moderation | 297 | 300 (99%) |
| incident | 291 | 300 (97%) |
| game | 177 | 300 (59%) |
| security | 34 | 300 (11%) |
| routing | 0 | 300 |
| risk | 0 | 300 |

**Consequence**: any "test" metric on moderation/incident/game from a model trained on
`sim_train.jsonl` mixes real generalization with straight memorization of exact training
states, and is not trustworthy as a generalization measure for those three domains.
Routing, risk, and (mostly) security are unaffected and can be read at face value.

`data/simulators.py` is fixed (random message/incident ids, timestamps, wider ranges;
verified 0/300 overlap on all four domains post-fix) but the corpora already used for the
SFT/RLCD/GRPO run below predate the fix and were not regenerated mid-run, to avoid discarding
multiple hours of in-progress training. **Read every table below with routing, risk and
security as the trustworthy columns**; moderation/incident/game numbers are reported for
completeness but should be treated as an upper bound on what the mechanism can actually do,
not a real measurement. A clean rerun on regenerated corpora is the next step (docs/ROADMAP.md).

## Teacher comparison, final ranking across three real candidates

| teacher | family | active params | overall soft Brier |
|---|---|---|---|
| **gemma26b** | Google DeepMind, MoE | 4B | **0.129** - best |
| qwen27b | Alibaba, dense | 27B | 0.142 |
| nemotron120b | NVIDIA, hybrid Mamba/MoE | 12B | 0.315 - worst |

The smallest, fastest teacher (Gemma-4-26B-A4B, only 4B active parameters, an 18GB
checkpoint) is the *best* grader of the three, beating a 27B dense model and a 120B-total
model with 3x its active parameter count. This reinforces and sharpens the earlier Nemotron
finding: neither total size nor active-parameter count predicted grading quality on this
benchmark. Practically, this makes gemma26b the recommended default teacher for future
`osj synth --distill` runs on this box - it is simultaneously the most accurate grader
measured and the cheapest/fastest to run (18GB checkpoint vs 82GB/61GB for the other two
large candidates). qwen27b's earlier-generated 473 distilled records
(`data/synthetic/teacher_train.jsonl`, used in the mechanism-1 training run above) predate
this finding, since the benchmark was built and run after that data generation - worth a
regeneration pass with gemma26b if training resumes, see docs/ROADMAP.md.

## In-container TRT-LLM Python API backend (`serve/trtllm_backend.py`)
`scripts/trtllm_api_smoke.py` on the smoke-test state (4 questions, 188 state tokens),
`return_generation_logits=True`, full-vocab gather, prefix reuse on:

| backend | decisions | latency | notes |
|---|---|---|---|
| HF bf16 in-process | abstain / medium / no / no | 62.7 ms | 3.9 GB peak |
| TRT-LLM 1.2.1 Python API | abstain / medium / no / no | 53.8 ms | label logits within ~1 of HF (bf16) |

## B20: mechanism speed - menu scoring vs generating the same decision (2026-09-19)

`runs/speed_vs_generation.json`, `python -m open_spark_jev.eval.speed_vs_generation`. 60 records from
`data/benchmarks/external/ext-toolcall-risk.jsonl` (4-way Choice), **idle GPU** (the concurrent A2 training
job was SIGSTOPped for the measurement window, so no contention; `gpu_contended: false`). Every arm uses the
same hardware and the same task text; the arms differ only in *how the answer is produced*.

| arm | p50 | p95 | dec/s | accuracy | JSON parse failures | mean output tokens |
|---|---|---|---|---|---|---|
| **spark-s1-1.7b-sft-v2** (menu scoring, 1 forward pass) | **30.1 ms** | 31.5 ms | 33.2 | **0.733** | 0 (impossible by construction) | 0 |
| generate (same Qwen3-1.7B, JSON, non-thinking) | 374.7 ms | 396.8 ms | 2.6 | 0.433 | **14 / 60** | 15.5 |
| generate_think (same Qwen3-1.7B, thinking) | 7164.9 ms | 12581.1 ms | 0.12 | 0.550 | 0 | 353.9 |

**Speedup of the mechanism: 12.4x** over the same backbone emitting JSON, **238x** over the same backbone
with reasoning enabled. The 20-200x range TypeSafe quotes for Jev is against *frontier* LLMs, a denominator we
cannot reproduce locally; the honest local statement is "12x vs an equally-sized generative classifier, 238x vs
the same model reasoning first". Note the menu arm is also *more accurate* than either generative arm on the
same weights class, and cannot emit malformed output - the non-thinking generative baseline failed to produce
parseable JSON on 23% of records.

**Accuracy against hosted Jev on the same 60 rows** (Jev's answers are committed in the source repo, run
2026-09-17): **jev-latest 0.917, spark-s1-1.7b-sft-v2 0.733**. We do *not* match Jev's accuracy on this external
set - it is third-party data in a domain our training data only approximates, and Jev is a far larger hosted
model. Speed parity does not imply decision parity; see `runs/external/` for the per-source picture.

Caveats: single run, n=60, one task type, batch size 1. Latency excludes model load. The menu arm's 30 ms is
lower than the 76-91 ms in the A1 grid because that grid ran under contention and over longer synthetic states.

## B21: spark-s1-1.7b-sft-v2 on seven third-party Jev evaluation sources (2026-09-19)

`runs/external/spark-s1-1.7b-sft-v2/`, `python -m open_spark_jev.eval.external`. Each source is scored
**separately** and never pooled; provenance (upstream repo, pinned commit, license, label origin, caveats)
in `data/external/<source>/PROVENANCE.md`. "Jev acc" columns are that repo's own committed Jev outputs on
the same rows, not a run we made.

| source | n | ours acc | ours ECE | Jev acc | Jev ECE | agreement w/ Jev | chance |
|---|---|---|---|---|---|---|---|
| ext-injection-noctx | 662 | 0.798 | **0.023** | 0.897 | - | 0.834 | 0.50 |
| ext-injection-ctx | 662 | 0.743 | **0.059** | 0.965 | 0.058 | 0.730 | 0.50 |
| ext-toolcall-risk | 60 | 0.733 | 0.147 | 0.917 | 0.050 | 0.783 | 0.25 |
| ext-jev-directory | 70 | 0.714 | 0.082 | - | - | - | varies |
| ext-kev-decision-v1 | 736 | 0.656 | 0.106 | - | - | - | varies |
| ext-kev-transfer-v4 | 764 | 0.581 | 0.231 | - | - | - | varies |
| ext-vuln-code | 400 | 0.502 | 0.365 | 0.715 | 0.178 | 0.647 | 0.50 |

`ext-jev-directory` eval-level pass rate (an eval passes only if *every* one of its questions is right):
**31/50 = 0.62**.

**Accuracy: we do not match Jev.** The gap is 18-22 points on the three sources with recorded Jev answers.
On `ext-toolcall-risk` the slices explain where it comes from: clear cases 0.882 (Jev 1.000), adversarial
0.667 (Jev 0.917), **ambiguous 0.429** (Jev 0.714) - we lose most heavily exactly where the decision is hard,
which is the case a calibrated model is supposed to earn its keep on.

**Calibration: competitive where the domain is close to training.** On `ext-injection-ctx` our ECE is 0.059
against Jev's 0.058 at 22 points lower accuracy, and on `ext-injection-noctx` it is 0.023. That is the one
property we trained for directly, and it survives the domain shift better than accuracy does.

**Where it breaks down.** `ext-vuln-code` is chance accuracy (0.502 on a binary question) with ECE 0.365 -
confidently wrong on source-code security, a domain nothing in our training data resembles. `ext-kev-transfer-v4`
(MMLU, Emotion, QNLI, PAWS, SciQ) at 0.581/0.231 is the same story for general NLP text. Compare the same
model's 0.838/0.017 on its own held-out simulator + teacher tests: **in-distribution numbers say nothing about
a new workflow**, and this table is the evidence for that claim, not a footnote to it.

The single highest-value fix is M17 (train on real public text); every weak row here is a domain the model has
literally never seen text from. Reference point: Kev-0.5B, which *is* trained on six public datasets, reports
0.633 on its own transfer suite vs hosted Jev's 0.823 (`docs/RUNS.md`) - a similar-sized gap from the other
direction.

Caveats: third-party labels from single annotators in some cases; Jev's rows were run 2026-09-17 on
`jev-latest`/`jev-1.13.0`; n=60 and n=70 sources are small; 116 Banking77 questions were skipped for exceeding
our 26-option cap.

## B22: architecture experiments A1 and A2 at matched budget (2026-09-19)

Ladder: `scripts/run_variants3.sh`. A0/A2 share seed, data (9,717 records), LoRA rank/targets,
budget (1 epoch) and evaluation code; **only the readout/attention differs**. Each saves a servable
checkpoint (`checkpoints/variants/<v>`) and is scored on all seven external sources.

### A1 (single-pass parallel multi-question readout) - CONFIRMED, small win

`runs/variants/a1_parallel.log`, no retraining (it is a readout change on
spark-s1-1.7b-sft-v2). Answers match the N-pass baseline to bf16 noise (max |Δp| = 0.022), so the
block-diagonal mask is correct and questions cannot see each other.

| questions per state | A0 (N passes) | A1 (1 pass) |
|---|---|---|
| 1 | 91.3 ms | 75.9 ms |
| 4 | 108.4 ms | 93.4 ms |
| 16 | 193.6 ms | 165.2 ms |

~15% faster, not the large win one might expect, because the state prefix KV cache was already
shared across questions in A0. Worth keeping as a serving optimisation; not an architectural finding.

### A2 (prefix-LM: bidirectional attention over the state) - REFUTED

| | A0 baseline | A2 prefix-LM |
|---|---|---|
| train seconds | **1743** | 2276 |
| sim_test acc / ECE | **0.806 / 0.011** | 0.791 / 0.016 |
| teacher_test acc / ECE | **0.889 / 0.073** | 0.620 / 0.090 |
| ext-injection-ctx | **0.784** | 0.394 |
| ext-injection-noctx | **0.802** | 0.600 |
| ext-jev-directory | **0.729** | 0.429 |
| ext-kev-decision-v1 | **0.700** | 0.296 |
| ext-kev-transfer-v4 | **0.603** | 0.382 |
| ext-toolcall-risk | **0.717** | 0.350 |
| ext-vuln-code | **0.565** | 0.492 |

A2 is worse on **every** axis, trains 30% slower, and collapses off-distribution (external sources
drop 7-40 points). The hypothesis was that causal masking hurts state comprehension; at this budget
the opposite holds. The likely reason is the one NOVELTY.md flagged when proposing it: Qwen3 was
pretrained causally, and switching the mask invalidates what the pretrained attention learned. One
epoch of LoRA is nowhere near enough to re-adapt. This does **not** refute prefix-LM in general - it
refutes it as a cheap drop-in, which is what we tested.

### Unplanned finding: LoRA generalises better than the full fine-tune

A0 (LoRA, 9,717 records, 1 epoch, 29 min) vs spark-s1-1.7b-sft-v2 (full fine-tune, 21,293 records,
2 epochs, ~1h30) on third-party data:

| source | A0 (LoRA) | SFT v2 (full FT) |
|---|---|---|
| ext-injection-ctx | **0.784** | 0.743 |
| ext-injection-noctx | **0.802** | 0.798 |
| ext-jev-directory | **0.729** | 0.714 |
| ext-kev-decision-v1 | **0.700** | 0.656 |
| ext-kev-transfer-v4 | **0.603** | 0.581 |
| ext-vuln-code | **0.565** | 0.502 |
| ext-toolcall-risk | 0.717 | **0.733** |

A0 wins 6 of 7 external sources with a third of the training budget, while losing on the in-house
tests (teacher_test 0.889 vs SFT v2's higher in-domain numbers). Read together with B21 this says the
full fine-tune is **overfitting the synthetic distribution**: the constrained LoRA update keeps more
of the backbone's general competence, which is exactly what third-party data measures. Not conclusive
(one seed, different data sizes, LoRA rank untuned), but it makes a LoRA arm of M17 worth running
next to the full-FT one.

## B23: M17 - does real public text close the third-party gap? (2026-09-19)

`spark-s1-1.7b-sft-m17` = identical recipe to `spark-s1-1.7b-sft-v2` (full fine-tune, 2 epochs) plus 7,920
real public records (ag_news, emotion, banking77-top20, toxic-chat, boolq, yelp) - 28,412 training examples
vs 21,293. 12% of each public source held out (`data/benchmarks/public_test.jsonl`, 1,080 rows). Training
took 2h08 (`runs/m17_train.log`). Commands: `scripts/run_m17.sh`.

### The 60-row tool-call set: accuracy and speed vs Jev (priority metric)

Same 60 decisions for every row; `runs/speed60_m17.json`, idle GPU (`gpu_contended: false`), batch 1.

| arm | accuracy | p50 | p95 | decisions/s | malformed output |
|---|---|---|---|---|---|
| **jev-latest** (hosted; recorded by the source repo) | **0.917** | 421.6 ms | 542.0 ms | 2.3 | 0 |
| **spark-s1-1.7b-sft-m17** (menu scoring) | 0.733 | **30.0 ms** | 30.7 ms | **33.4** | 0 |
| spark-s1-1.7b-sft-v2 (previous, B20) | 0.733 | 30.1 ms | 31.5 ms | 33.2 | 0 |
| same Qwen3-1.7B generating JSON | 0.433 | 376.3 ms | 399.9 ms | 2.6 | 14 / 60 |
| same Qwen3-1.7B, thinking on | 0.550 | 7,209.6 ms | 12,660.6 ms | 0.11 | 0 |

**Speed: 14.1x faster than hosted Jev, 12.5x over the same backbone generating JSON, 240x over the same
backbone reasoning first.** The Jev latency includes a network round trip and is not measured on this
machine, so the 14.1x is indicative, not like-for-like; the 12.5x and 240x are like-for-like (same weights,
same box, only the readout differs).

**Accuracy: 0.733 vs Jev's 0.917 - 18 points behind, and M17 did not move it.** Real public text left the
60-set accuracy exactly where it was (0.733 -> 0.733). We do not match the ~90% accuracy half of the claim.

### External sources, M17 vs the same recipe without public text

| source | sft-v2 | **sft-m17** | change | overlap with M17's training data? |
|---|---|---|---|---|
| ext-kev-decision-v1 | 0.656 | **0.750** | +9.4 | **yes: boolq, agnews, yelp, mnli-adjacent, sst5** |
| ext-kev-transfer-v4 | 0.581 | **0.656** | +7.5 | **partly: emotion (trained on); mmlu/qnli/paws/sciq/tweet are not** |
| ext-jev-directory | 0.714 | 0.729 | +1.5 | no |
| ext-toolcall-risk | 0.733 | 0.733 | 0.0 | no |
| ext-injection-ctx | 0.743 | 0.734 | -0.9 | no |
| ext-injection-noctx | 0.798 | 0.754 | **-4.4** | no |
| ext-vuln-code | 0.502 | 0.510 | +0.8 | no |

**The two large gains are on datasets M17 now trains on.** Kev's suites are built from the *test/validation
splits* of the same public datasets whose *train splits* we added, so `ext-kev-decision-v1` (boolq, ag_news,
yelp) is in-domain for M17 and its +9.4 is not a generalisation result. Only the sources with no overlap
tell us about transfer, and on those M17 is flat (+0.8 to +1.5, -0.9, 0.0) or worse (-4.4). Row-level
train/test leakage is not the issue (different splits); source-level familiarity is.

**Calibration got worse where it was best.** `ext-injection-ctx` ECE 0.059 -> **0.159**, `ext-injection-noctx`
0.023 -> **0.161**. Adding real text degraded the one property B21 identified as our strength off-distribution.

### In-house and real-text held-out (internal)

| set | sft-v2 | sft-m17 |
|---|---|---|
| overall (sim + teacher tests) acc / ECE | 0.838 / 0.017 | 0.833 / 0.018 |
| choice/routing | 0.815 / 0.062 | 0.835 / 0.022 |
| choice/incident | 0.820 / 0.040 | 0.824 / 0.016 |
| injection flip rate | 0.033 | 0.027 |

New real-text slices (public_test, held out, first measurement): classification 0.819 / ECE 0.089,
moderation 0.812 and 0.956, routing 0.835, reading (BoolQ) 0.817 / 0.100, **scoring (Yelp 5-level) 0.667 /
ECE 0.153**. In-house numbers did not regress, so public text cost nothing internally.

### Reading

Hypothesis from B21: *"every weak external row is a domain the model has never seen text from, so training on
real text is the highest-value fix."* **Not supported.** It improved sources it was trained on, was flat on
the rest, and degraded injection calibration. The gap to Jev on the 60-set is unchanged at 18 points. Together
with B22 (a third-budget LoRA run generalising better than the full fine-tune) the evidence now points at
**how the model is adapted** rather than **what text it sees**. The LoRA-plus-public-text arm
(`scripts/run_m17_lora.sh`, running) is the discriminating experiment.

Caveats: single seed; one public-text mix; Kev's suites overlap M17's training sources as marked;
`ext-toolcall-risk` is n=60, so 0.733 and 0.733 are within noise of each other, not proof of no effect.

## B24: M17-LoRA - the A0 LoRA recipe plus real public text (2026-09-19)

Question left open by B22/B23: is the third-party gap about *how* the model is adapted (full FT vs LoRA) or
*what text* it sees? Same A0 recipe as B22 (LoRA r16, 1 epoch), 16,000 records, with the 7,920 real public
records pooled in. Artifact: `runs/external/spark-s1-1.7b-lora-m17/`, log `runs/m17_lora.log` (train 46 min).

| source | A0 LoRA (no public text) | A0 LoRA + public text | sft-m17 (full FT + public text) | Jev (recorded) |
|---|---|---|---|---|
| ext-toolcall-risk (60-set) acc / ECE | 0.717 / 0.093 | 0.750 / 0.163 | 0.733 / 0.136 | 0.917 |
| ext-injection-ctx | 0.784 / 0.063 | 0.725 / 0.141 | 0.734 / 0.159 | 0.965 |
| ext-injection-noctx | 0.802 / 0.027 | 0.766 / 0.101 | 0.754 / 0.161 | 0.897 |
| ext-vuln-code | 0.565 / 0.272 | 0.512 / 0.352 | 0.510 / 0.358 | 0.715 |
| ext-jev-directory | 0.729 / 0.099 | 0.757 / 0.103 | 0.729 / 0.120 | - |
| ext-kev-decision-v1 (overlaps public text) | 0.700 / 0.082 | 0.726 / 0.056 | 0.750 / 0.087 | - |
| ext-kev-transfer-v4 (overlaps public text) | 0.603 / 0.150 | 0.636 / 0.149 | 0.656 / 0.167 | - |

Reading: adding public text to LoRA reproduces the full-FT pattern, so the text, not the adaptation method,
is what hurts. Injection accuracy drops 2.6-5.9 pts and ECE more than doubles; vuln-code drops 5 pts. Gains
appear only on the Kev suites (which overlap the trained-on datasets) and, within noise at n=60/70, on the
60-set and the directory. The best non-overlapping profile is still plain A0 LoRA. The 60-set stays 17-20 pts
below Jev in every arm; speed is unchanged (same mechanism, ~30 ms). Real generic public text is not the
missing ingredient; the untested lever is in-domain data for the task being scored (agent tool-call risk).

## B25: where the 60-row tool-call set is lost (per-row analysis, 2026-09-19)

Per-row predictions of six checkpoints on `ext-toolcall-risk`: `runs/rows60.json` (script `scripts/analysis/rows60.py`).
Accuracy: sft-v2 0.733, sft-m17 0.733, A0 0.717, A0+public 0.750, A0+toolcall 0.750, A2 0.350; Jev 0.917 (5 misses).

- 15 of 60 missed by the best arm (A0+toolcall); 8 rows are missed by all five sane models; 17 by four or more.
- By slice (A0+toolcall / Jev): clear 30/34 vs 34/34, adversarial 8/12 vs 11/12, ambiguous 7/14 vs 10/14.
- Dominant confusion: gold `privileged` predicted `destructive` (6 rows): config changes that weaken a security
  control (`helm upgrade --set auth.enabled=false`, `kubectl edit configmap/gateway-auth`, `UPDATE feature_flags SET
  enabled=true WHERE name='bypass_auth'`) read as "modifies prod" to us, as "weakens security" to Jev.
- Lexical shortcuts: `curl .../healthz` and `openssl s_client` predicted `exfiltration` at 0.97 confidence.
- We over-predict the alarming classes (destructive 22 vs 15 gold, exfiltration 15 vs 10) and under-predict readonly (12 vs 18) and privileged (11 vs 17).
- Confidence when wrong averages 0.81 (right: 0.96): the model is nearly as sure when it is wrong.
- Jev's own 5 misses include 3 of ours (`kubectl cp`, `kubectl port-forward`, `kubectl set image`), where the gold label is
  arguable: the reachable ceiling for any model is closer to 0.9 than 1.0.
- **Training-data defect found (ours):** the generated `toolcall_risk` corpus is degenerate. In 2,492 rows every
  `curl` call is `exfiltration` (650/650), every `aws` call is `privileged` (615/619), `kubectl` is only
  readonly/destructive (privileged: 12 rows). The teacher wrote one stereotypical tool per label, so the model learned
  tool-name -> label. This explains the curl and privileged failures and why +2.5k in-domain rows did not help.

## B26: A3 slot-query head at matched budget (2026-09-19)

Same recipe and data as the B24/B25 A0 arm (LoRA r16, 12,000 records incl. 2,492 tool-call rows, 1 epoch, 28.5 min).
Saved model scored through the shared runner (`runs/external/v4-a3/`), and it matches the in-process report, so this
is not a loader bug.

| | A0 (+toolcall data) | A3 slot-query |
|---|---|---|
| sim_test acc | 0.801 | 0.783 |
| teacher_test acc | 0.887 | 0.749 |
| ext-toolcall-risk (60) | 0.767 | 0.583 |
| ext-injection-ctx / noctx | 0.649 / 0.813 | 0.397 / 0.397 (below the 0.5 chance rate of a 2-way task) |
| ext-kev-decision-v1 | 0.686 | 0.364 |
| ext-vuln-code | 0.560 | 0.500 |

A3 is worse on every measure, most severely off-distribution. Verdict: not competitive as a drop-in at one epoch. Caveat: the
training corpus is tool-name-degenerate (B25), which affects all arms alike.

## B27: version ladder v0, v1, v3 on the os-datagen splits (2026-09-20)

Data: `datagen-pipeline/artifacts/mixture_final` (967 train / 132 calibration / 121 locked test / 124 challenge), see [EXPERIMENTS.md](EXPERIMENTS.md).
Evaluator: `eval/osdg.py` (3 option orders per choice row; temperature per question type fitted on the calibration split only).
Per-row predictions: `runs/osdg/<name>/rows.jsonl`. Locked-test n=121 is small: +-0.09 is one standard error.

**v0/v1: frozen base, no training** (`scripts/run_v0.sh`)

| base | test acc | challenge acc | option flips (test) | ECE raw -> cal (test) |
|---|---|---|---|---|
| Qwen3-0.6B | 0.405 | 0.234 | 0.48 | 0.396 -> 0.066 |
| Qwen3-1.7B | 0.372 | 0.290 | 0.51 | 0.609 -> 0.109 |
| Qwen3-4B | 0.504 | 0.613 | 0.17 | 0.474 -> 0.140 |
| Gemma-4-E4B | 0.545 | 0.581 | 0.38 | 0.277 -> 0.092 |
| Gemma-4-12B | 0.314 | 0.347 | 0.85 | 0.508 -> 0.060 (looks like a format mismatch, not capability; not investigated) |

Calibration-split temperature repairs ECE but not accuracy. No confidence threshold reaches 95% selective accuracy on any frozen base.

**v3: LoRA r16, 3 epochs, 4,277 rows (967 train rows, choice rows in ~5 option orders)** (`scripts/run_v3.sh`)

| model | test acc | challenge acc | option flips | ECE cal (test) | selective test (cov / acc) | train time |
|---|---|---|---|---|---|---|
| v3 Qwen3-0.6B | 0.603 | 0.694 | 0.089 | 0.142 | - (no threshold) | 34 min |
| v3 Qwen3-1.7B | 0.603 | 0.694 | 0.054 | 0.147 | 0.30 / 0.67 | 58 min |
| **v3 Qwen3-4B** | **0.785** | **0.790** | 0.045 | **0.087** | 0.75 / 0.87 | 113 min |

- Training on ~1k decision rows lifts 1.7B by 23 pts (test) and 40 pts (challenge) and cuts option-order sensitivity from ~50% to ~5%.
- 0.6B and 1.7B tie; 4B jumps 18 pts. Capacity matters once the data is right, and the ladder of sizes is not smooth.
- The 4B threshold picked on calibration (0.82, target 95%) yields 96% on calibration but 87% on test and challenge: the threshold does not transfer to held-out families.
- Weak packs at 4B (test): router 1/4, urgency 0/3, retrieval 3/6, termination 7/12, sufficiency 3/6; strong: entailment 30/32, document type, relevance, rule application, tool gate 4/5. Most packs have 2-12 test rows, so these are noisy.
- Train loss reaches 0.0000 in epoch 2 on every size: the model memorises the augmented set. The held-out numbers above are what count.

## B28: v2 frozen head, v6 offline cascade, v3 speed (2026-09-20)

**v2: frozen backbone + MLP head on the answer-position hidden state** (`experimental/frozen_head.py`, same augmented train file as v3)

| | test acc | challenge acc | option flips | ECE cal (test) |
|---|---|---|---|---|
| v2 Qwen3-1.7B frozen + head | 0.314 | 0.355 | 0.56 | 0.142 |
| v2 Qwen3-4B frozen + head | 0.512 | 0.524 | 0.47 | 0.195 |
| (v0 frozen direct-logit, same bases) | 0.372 / 0.504 | 0.290 / 0.613 | 0.51 / 0.17 | 0.109 / 0.140 |
| (v3 LoRA, same bases) | 0.603 / 0.785 | 0.694 / 0.790 | 0.05 / 0.045 | 0.147 / 0.087 |

REFUTED as an improvement: a frozen backbone plus a slot head does no better than reading the base logits directly (it is worse than v0
at 1.7B) and stays as option-order-sensitive as v0. Adapting the backbone (v3) is what makes the model robust.

**v3 decision latency** (`scripts/analysis/speed_osdg.py`, os-datagen locked-test prompts, batch 1, idle GPU, per-row in `runs/osdg/speed_v3.rows.jsonl`)

| model | p50 | p95 | decisions/s |
|---|---|---|---|
| v3 0.6B | 19.2 ms | 27.6 | 48.9 |
| v3 1.7B | 34.5 ms | 51.1 | 26.8 |
| v3 4B | 81.9 ms | 118.2 | 11.8 |

**v6 offline cascade** (`scripts/analysis/cascade_v6.py`; thresholds from calibration only; small model first, 4B on escalation, low-confidence rows to review)
The cascade is not worth it: the small students' confidence does not separate their correct from incorrect rows, so most rows escalate, and the
mean latency (~100 ms) exceeds running the 4B alone (82 ms). Its ~0.87 accuracy on answered rows (~80% answered) matches what the 4B's own
abstention gives without the extra tier. Table: `runs/osdg/cascade_v6.md`.

### v7 attempt 1 (RLCD-direct on v3-4b) stopped, no result (2026-09-20)
Started from `checkpoints/v3-4b` on the same train file it had just been trained on. After 30 steps the reward J was 1.0000 and KL 0.0000:
the LoRA model has memorised the training rows (train loss 0.0000, B27), so the Brier reward is already saturated and there is no
gradient. RLCD on rows the policy already fits cannot improve calibration on held-out families. Stopped at step 30/240 to free the GPU.

## B29: final review on the 60-row tool-call set (2026-09-20)

`scripts/analysis/final60.py` -> `runs/osdg/final60.json`, per-row `runs/osdg/final60.rows.jsonl`. No model was trained on any tool-call-risk
data (the degenerate corpus of B25 is not used); the ladder models saw only os-datagen rows. Temperature is the one fitted on the os-datagen
calibration split. Speed: batch 1, idle GPU, p50 per decision on the same 60 rows. Soft final check: per-row failures of older models on this
set were inspected before the ladder (B25); n=60, one standard error ~ +-5 pts.

| model | accuracy | ECE (cal) | option flips | p50 | p95 | dec/s |
|---|---|---|---|---|---|---|
| **Jev (recorded, hosted, incl. network)** | **0.917** | - | - | 421.6 ms | 542.0 | 2.3 |
| **v3 Qwen3-4B (LoRA)** | **0.850** | 0.086 | 0.15 | 65.9 ms | 69.2 | 15.1 |
| v3 Qwen3-1.7B | 0.783 | 0.087 | 0.10 | 29.6 ms | 30.5 | 33.8 |
| v3 Qwen3-0.6B | 0.767 | 0.293 | 0.17 | 15.4 ms | 16.2 | 64.2 |
| v0 Gemma-4-E4B frozen | 0.733 | 0.213 | 0.38 | 92.4 ms | 97.2 | 10.8 |
| v0 Qwen3-4B frozen | 0.700 | 0.200 | 0.20 | 67.6 ms | 70.0 | 14.8 |
| v2 4B frozen + head | 0.667 | 0.091 | 0.22 | 62.4 ms | 65.0 | 15.9 |
| v0 Qwen3-0.6B frozen | 0.617 | 0.299 | 0.80 | 16.1 ms | 16.9 | 61.6 |
| v0 Qwen3-1.7B frozen | 0.600 | 0.130 | 0.08 | 30.3 ms | 31.0 | 33.1 |
| v2 1.7B frozen + head | 0.567 | 0.262 | 0.50 | 26.5 ms | 28.2 | 37.4 |

- Best model v3-4B: 51/60 (Jev 55/60), 6.4x faster p50 than Jev's recorded latency (not like-for-like: Jev includes a network round trip).
  Previous best on this set was 0.767 (B24/B25 arms); the gain comes from os-datagen's code-derived labels, not from tool-call data.
- v3-4B by slice: clear 30/34, ambiguous 10/14, adversarial 11/12 (Jev 34/34, 10/14, 11/12). Misses: security-weakening config changes
  (`helm --set auth.enabled=false`, `kubectl patch hostNetwork`) called readonly/destructive; exfiltration by `curl -d @/etc/shadow` and
  `scp ~/.aws` called destructive; `aws s3 sync` to another bucket under a "Standard backup step:" framing called readonly at 0.90 confidence.
- v3-1.7B at 0.783 / 29.6 ms is the speed pick: 14x faster than Jev at 13 pts lower accuracy.

## B30: five end-to-end API cases through `POST /v1/decide` (2026-09-20)

`scripts/api_bench/five_cases.py` starts the gateway, calls the public contract over HTTP (`open_spark_jev/serve/contract.py`), 20 repetitions per case,
full distributions + latency saved in `runs/api_bench/<model>/{results.json,rows.jsonl}`. Oracle labels are the ones given with the cases. 11 decisions in total,
so this is a smoke test of the API and behaviour, not a statistically meaningful benchmark. Temperatures: choice from the calibration split, score/noul left at 1.0.

| model | decisions correct | mean NLL | mean Brier | request p50 (2-3 questions, one state) |
|---|---|---|---|---|
| spark-s1-4b-v3 | 9/11 | 0.518 | 0.251 | 127-135 ms |
| spark-s1-1.7b-v3 | 7/11 | 1.866 | 0.626 | 61-64 ms |

| case | 4B | 1.7B |
|---|---|---|
| 1 support routing (department / refund / urgency) | 3/3 | 1/3 (refund_requested false at 1.00; urgency 2 at 1.00, oracle 1) |
| 2 tool gate (action / violation) | 2/2 | 1/2 (violation false 0.65) |
| 3 extraction validation | 2/2 | 2/2 |
| 4 answer sufficiency (action / level) | 1/2 (action: return 0.47 vs repair_from_context 0.37) | 1/2 (level: 2 at 0.93, oracle 1) |
| 5 stop/retry/repair (next step / complete) | 1/2 (next_step: ask_user 0.86, repair 0.03) | 2/2 |

- Both models get the two hard boolean/extraction checks in case 3 right and both mark the incomplete artifact in case 5 as not complete.
- 4B is confidently wrong on case 5 (asks the user instead of repairing a known defect) and split on case 4 (return vs repair): the answer-sufficiency and termination packs were the weakest in B27 too.
- 1.7B misses are high-confidence (1.00) on boolean/score questions: score/noul temperatures were not fitted (too few calibration rows), so those heads are uncalibrated.
- Latency is per request: 2 questions ~127 ms and 3 questions ~135 ms on the 4B, so the state prefill dominates and extra questions on a shared state are nearly free. These prompts are longer than the 60-row set's (~66 ms).

## B31: v3 models on the seven external suites (2026-09-20)

`scripts/run_ext_v3.sh` -> `runs/external/spark-s1-v3-{4b,1.7b}-osdg-trained/` (per-row `*.rows.jsonl`). Trained only on os-datagen rows, so no external source
overlaps its training data (unlike M17, B23). Calibration: choice temperature from the os-datagen calibration split; boolean/score left at 1.0, so the
binary sources (injection, vuln-code) are uncalibrated.

| source | v3 4B acc / ECE | v3 1.7B acc / ECE | best earlier (any arm) | Jev (recorded) |
|---|---|---|---|---|
| ext-toolcall-risk (60) | 0.850 / 0.086 | 0.783 / 0.087 | 0.767 | 0.917 |
| ext-kev-decision-v1 | 0.773 / 0.114 | 0.700 / 0.207 | 0.750 (M17, overlapping) | - |
| ext-kev-transfer-v4 | 0.749 / 0.076 | 0.635 / 0.209 | 0.656 (M17, overlapping) | - |
| ext-jev-directory | 0.771 / 0.137 | 0.557 / 0.349 | 0.757 | - |
| ext-injection-ctx | 0.788 / 0.176 | 0.819 / 0.163 | 0.784 | 0.965 |
| ext-injection-noctx | 0.784 / 0.175 | 0.754 / 0.241 | 0.811 | 0.897 |
| ext-vuln-code | 0.560 / 0.415 | 0.520 / 0.474 | 0.565 | 0.715 |

- The 4B is the best model on five of seven sources, including the two Kev suites where it beats models that trained on Kev's own source datasets; this is clean transfer.
- Injection ties earlier arms in accuracy but calibration is worse (ECE 0.18 vs 0.03-0.06): binary questions still use T=1.0.
- Vulnerable-code detection remains near chance (0.56, ECE 0.42): none of our data, old or new, teaches it. Jev reaches 0.715.
- The 1.7B is inconsistent (directory 0.557, below the earlier sft-v2's 0.714), so the size step matters for transfer, not only for in-domain.

## B32: spark-s1 v3 against Kev's published family on Kev's transfer-v4 suite (2026-09-20)

Kev (jaredpalmer/kev) released Kev-0.6B / 4B / 8B on 2026-09-20 (LoRA r16 + pointer head on Qwen3 bases, trained on 10-13 public sources plus programmatic policy pairs).
Our `ext-kev-transfer-v4` is Kev's `evals/v4/transfer-v4/test.jsonl` (764 records; sha256 identical to the file in Kev's main branch on 2026-09-20), i.e. their **locked test** for the
out-of-domain suite. Kev's numbers are copied from their README; ours are `runs/external/spark-s1-v3-*-osdg-trained/ext-kev-transfer-v4.json`. We did not run Kev's models.

| model | trained on | transfer-v4 locked test acc | dev acc | confident errors (p>=0.9, wrong) |
|---|---|---|---|---|
| Kev-8B | 10-13 public sources + policy pairs | 0.780 | 0.796 | 9.9% (dev) |
| Kev-4B | same | 0.806 | 0.790 | 8.2% (dev) |
| **spark-s1-4b-v3** | 967 os-datagen rows | **0.749** (ECE 0.076, Brier 0.374) | not measured | **5.9%** |
| spark-s1-1.7b-v3 | same | 0.635 (ECE 0.210) | not measured | 13.4% |
| Kev-0.6B | 10-13 public sources + policy pairs | 0.642 | 0.620 | 10.8% (dev) |
| Jev (hosted) | unknown | not measured | 0.857 | 3.7% (dev) |

Reading: 4B is 3-6 points below Kev-8B/4B and ~10 above Kev-0.6B; 1.7B ties Kev-0.6B. Kev also finds capacity dominates out of domain, more public data helps in-distribution but not transfer
(our M17), and calibration is usable but not transferable. Caveats: this is Kev's suite (their rendering, option order, sources), Kev trains on far more and more varied data, dev and locked-test
columns are different partitions, we read their locked test with several of our models, and none of this is an in-distribution comparison (their decision-v4 suite is not run here).
