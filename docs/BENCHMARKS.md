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
`runs/eval_base_zero_shot.json`

| slice | acc | macro-F1 | ECE | Brier | soft Brier vs posterior |
|---|---|---|---|---|---|
| choice/routing | 0.44 | 0.37 | 0.55 | 1.10 | 0.82 |
| choice/moderation | 0.44 | 0.25 | 0.56 | 1.11 | 0.83 |
| choice/incident | (see json) | | | | |
| choice/game | (see json) | 0.13 | 0.63 | 1.27 | 1.05 |
| noul/security | 0.60 | 0.27 | 0.40 | 0.80 | 0.77 (noul Brier 0.18) |
| score/risk | 0.35 | 0.19 | 0.64 | 1.29 | 0.77 |
| **overall** | **0.39** | 0.26 | **0.60** | 1.21 | |

* injection flip rate: **0.60** (argmax differs from posterior argmax on injected states)
* throughput (HF, one unique state per record, no batching across states): 24 decisions/s

Reading: the untrained backbone already understands the menu format (it never emits an
invalid label by construction) but is wildly over-confident: near-one-hot distributions
(ECE 0.60) and it follows injected instructions most of the time. This is the "before"
row for Phase 1 / Phase 2 and for the served-engine comparison.

## Latency (HF backend, bf16, in-process)
Smoke test: 4 questions on a 188-token state, cached prefix, 61.9 ms total, 3.9 GB peak.
Full grid: `python -m open_spark_jev.eval.latency --backend hf --model models/Qwen3-1.7B`
(to be filled in with M9).

## Served engine parity (trtllm-serve 1.2.1, PyTorch backend, bf16)
Pending: see the section appended after the first served run.
