# Benchmarks

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

## M6: base Qwen3-1.7B, zero-shot, HF bf16 (no training, T = 1)
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

## M7: SFT (Phase 1) vs zero-shot base, held-out test set

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

## M8a: RLCD-direct, first attempt - a real negative result from a real bug

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

## M8b: RLCD-direct, corrected (`correctness: 0.0`, pure Brier + shaping reward)

Same held-out test set, checkpoint retrained from scratch with the fix from M8a.

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

## M8c: mechanism 1 (contrastive RLCD) vs mechanism 2 (direct) vs SFT

Same held-out test set, three checkpoints. 499 contrastive pairs (nvidia/Qwen3.6-27B-NVFP4,
opposed principles) -> a reward model (93.9% held-out pairwise accuracy) -> exact policy
gradient for mechanism 1; see M7/M8a/M8b above for the other two.

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

## M8d: all four Phase-2 mechanisms, complete

Adds GRPO (sampled-token TRL baseline, capped to 1600 records/300 steps - see
docs/RESEARCH.md R1 for why an uncapped run wasn't practical) to the M8c table.

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

**Overall reading across all four (M8a-M8d)**: on this benchmark, the exact-vs-sampled
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

For scale: the untrained 1.7B student's own Brier on the same slices is ~1.1-1.7 (see M6
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

## Known issue: train/test state leakage in the M6-era corpora (found 2026-09-18)

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

## M20: mechanism speed - menu scoring vs generating the same decision (2026-09-19)

`runs/speed_vs_generation.json`, `python -m open_spark_jev.eval.speed_vs_generation`. 60 records from
`data/benchmarks/external/ext-toolcall-risk.jsonl` (4-way Choice), **idle GPU** (the concurrent A2 training
job was SIGSTOPped for the measurement window, so no contention; `gpu_contended: false`). Every arm uses the
same hardware and the same task text; the arms differ only in *how the answer is produced*.

| arm | p50 | p95 | dec/s | accuracy | JSON parse failures | mean output tokens |
|---|---|---|---|---|---|---|
| **menu** (Spark-S1 SFT v2, 1 forward pass) | **30.1 ms** | 31.5 ms | 33.2 | **0.733** | 0 (impossible by construction) | 0 |
| generate (same Qwen3-1.7B, JSON, non-thinking) | 374.7 ms | 396.8 ms | 2.6 | 0.433 | **14 / 60** | 15.5 |
| generate_think (same Qwen3-1.7B, thinking) | 7164.9 ms | 12581.1 ms | 0.12 | 0.550 | 0 | 353.9 |

**Speedup of the mechanism: 12.4x** over the same backbone emitting JSON, **238x** over the same backbone
with reasoning enabled. The 20-200x range TypeSafe quotes for Jev is against *frontier* LLMs, a denominator we
cannot reproduce locally; the honest local statement is "12x vs an equally-sized generative classifier, 238x vs
the same model reasoning first". Note the menu arm is also *more accurate* than either generative arm on the
same weights class, and cannot emit malformed output - the non-thinking generative baseline failed to produce
parseable JSON on 23% of records.

**Accuracy against hosted Jev on the same 60 rows** (Jev's answers are committed in the source repo, run
2026-09-17): **jev-latest 0.917, Spark-S1 SFT v2 0.733**. We do *not* match Jev's accuracy on this external
set - it is third-party data in a domain our training data only approximates, and Jev is a far larger hosted
model. Speed parity does not imply decision parity; see `runs/external/` for the per-source picture.

Caveats: single run, n=60, one task type, batch size 1. Latency excludes model load. The menu arm's 30 ms is
lower than the 76-91 ms in the A1 grid because that grid ran under contention and over longer synthetic states.
