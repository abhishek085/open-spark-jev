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

## Served engine parity (trtllm-serve 1.2.1, PyTorch backend, bf16)
Pending: see the section appended after the first served run.
