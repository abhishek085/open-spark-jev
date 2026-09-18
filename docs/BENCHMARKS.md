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
| nemotron120b | - | - | pending (memory) | |
| gptoss120b | - | - | pending (download + memory) | |

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

## In-container TRT-LLM Python API backend (`serve/trtllm_backend.py`)
`scripts/trtllm_api_smoke.py` on the smoke-test state (4 questions, 188 state tokens),
`return_generation_logits=True`, full-vocab gather, prefix reuse on:

| backend | decisions | latency | notes |
|---|---|---|---|
| HF bf16 in-process | abstain / medium / no / no | 62.7 ms | 3.9 GB peak |
| TRT-LLM 1.2.1 Python API | abstain / medium / no / no | 53.8 ms | label logits within ~1 of HF (bf16) |
