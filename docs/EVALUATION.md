# Evaluation

How spark-s1 is evaluated, what each number means, and what is still missing before a release can be called validated. Detailed results live in [BENCHMARKS.md](BENCHMARKS.md); this page is the protocol.

## Four kinds of data, never mixed

| Set | Used for | Rule |
|---|---|---|
| **Training** | fit weights | Nothing else may be fitted on it. |
| **Calibration** | fit temperatures and confidence thresholds | Independent of training; never train on it. |
| **Development / diagnostic** | look at failures, compare variants, choose what to build next | Once you have inspected it or chosen models by it, it is no longer a blind test. |
| **Locked release test** | one final measurement per released candidate | Read once; never tune, fit or look at rows afterwards. |

In this repository: the os-datagen splits (`train`, `calibration`, `test_locked`, `challenge`) follow this protocol. Kev's suites are third-party sets we only score on. The **60-case tool-call set** is a *diagnostic* benchmark: we read its per-row failures before the version ladder was designed (BENCHMARKS.md B25), and we have scored many models on it, so it is not a locked holdout and its numbers carry selection bias. Use it to show behaviour, not to claim generalisation.

## Group-aware splits

Reshuffled option orders, paraphrases and semantic siblings of one scenario must stay on one side of a split. os-datagen assigns the split at scenario generation time (own entity pools, template families, wording and seeds; calibration and test use held-out scenario families) and every run — including `--dry-run` — writes `reports/split_isolation.json` as proof; it is a generated run artifact (`datagen-pipeline/artifacts/`, gitignored), not a file checked into this repository, so run the pipeline to produce your own copy. [`docs/examples/split_isolation.example.json`](examples/split_isolation.example.json) is one concrete copy, checked in so it can be read without a GPU: a full `--dry-run` (all 34 task packs, the offline fake client, `ok: true`, zero violations) that shows the report's actual shape — the id/template/entity-pool checks, the held-out scenario families per pack, and the near-duplicate check — without needing to regenerate it; it is a dry-run example of the report's structure, not evidence about the real generator's output. Two lessons from our own history: simulator corpora once leaked 97-99% of test states into training, and the training script's validation split contained shuffled copies of training rows, which produced a badly fitted temperature (0.065). Any new data must be split by scenario group, never by row.

## What a future locked release holdout should contain

* Tool-call risk decisions written by people who did not build the model, covering every risk family: read-only inspection, destructive and mutating operations, privilege and security weakening, credential and data exfiltration, cloud copy/sync, opaque or obfuscated payloads, and benign-looking framings.
* Several tools and stacks (shell, kubectl, cloud CLIs, SQL, git, HTTP APIs), several environments, and paired minimal edits (a benign and a harmful version of the same call).
* Hundreds of rows per family, not tens, so per-family error bars are narrower than the differences we care about; labels double-checked by two reviewers with disagreements recorded.
* Frozen, hashed, stored outside the training repository, and read once per release candidate.

## Metrics that matter

Accuracy alone is not enough for a gate. Report all of these per release, on the locked test where it exists:

* **False auto-allow rate**: the share of rows that should not run automatically but were auto-allowed. This is the safety number. Report it for the model alone and for model plus policy, at each auto-allow threshold.
* **Per-class precision and recall**, and accuracy by risk family.
* **Calibration**: ECE and Brier score, per question type, and after fitting on a separate calibration split.
* **Latency**: p50 and p95, batch size, hardware, dtype, and whether the state cache is used.
* **Option-order invariance**: the share of answers that change when options are reshuffled.
* **Provenance**: benchmark revision or hash, model revision, code commit, seeds.
* **Model-only versus model-plus-policy** metrics, kept separate (below).

## Model-only versus model-plus-policy

Model-only metrics measure the network. Model-plus-policy metrics measure what a harness would actually do: the deterministic rules ([`policy.py`](../open_spark_jev/policy.py)) can only make the outcome more restrictive, and the threshold turns a probability into `allow` or `ask`. Never quote a model-plus-policy number as a model accuracy, and never claim the policy makes the model safe: it is heuristic, and it also over-triggers.

## Measured so far on the 60-case diagnostic set (n=60, 18 read-only calls)

Auto-allow means P(read-only) at or above the threshold (temperature already applied). "Model plus policy" additionally requires the rules to find nothing.

| Model | Threshold | Auto-allowed (model alone) | False auto-allow (model alone) | Auto-allowed (model + policy) | False auto-allow (model + policy) |
|---|---|---:|---:|---:|---:|
| spark-s1-4b-v3 | 0.90 | 13 of 18 | 0 | 11 of 18 | 0 |
| spark-s1-4b-v3 | 0.95 | 3 | 0 | 3 | 0 |
| spark-s1-4b-v3 | 0.99 and 0.995 | 0 | 0 | 0 | 0 |
| spark-s1-1.7b-v3 | 0.90 to 0.995 | 0 | 0 | 0 | 0 |

Reading it honestly: **at the default threshold of 0.995 neither model ever auto-allows a call on this set**, so everything goes to `ask`. That is safe and useless as an automation gain. The 4B model becomes useful only near 0.90 (13 of the 18 read-only calls, with no false auto-allow on 60 rows, which is too few to conclude a low rate). The rules alone leave 3 of the 42 non-read-only calls without any finding (they would not be caught by the rules) and over-trigger on 3 of the 18 read-only calls. Files: `runs/osdg/false_auto_allow_60.json` (regenerate with `scripts/analysis/false_auto_allow.py`), `runs/osdg/final60.rows.jsonl`.

## Release gates (template)

A release should not be called validated until every line is checked. Status for v3:

| Gate | Status |
|---|---|
| Locked holdout for the target task family exists, frozen and hashed | TODO: planned, does not exist yet |
| False auto-allow rate with a confidence interval, model alone and model plus policy | TODO: measured on n=60 only |
| Per-family accuracy with at least 100 rows per family | TODO |
| ECE and Brier for Choice, Boolean and Score after calibration on an independent split | Partial: Choice only; Boolean and Score not fitted |
| Option-order invariance under the deployed prompt | Measured (15% of rows flip on the 60-set, 4B) |
| Multiple seeds and bootstrap intervals | TODO: single seed |
| Independent third-party benchmark | Done for Kev transfer-v4 (B32), not a substitute for a security holdout |
| p50 and p95 on stated hardware and configuration | Done for DGX Spark, HF Transformers, batch 1 |

## Benchmark result files

Runs are published as JSON matching [`benchmark_run.schema.json`](benchmark_run.schema.json) (required: `run_id`, `model`, `benchmark`, `benchmark_revision`, `hardware`, `accuracy`, `p50_latency_ms`, `decisions_per_second`, `notes`; optional: `ece_calibrated`, `option_order_flip_rate`, `p95_latency_ms`, `false_auto_allow_by_threshold`). The current diagnostic runs are in [`open_spark_jev/serve/lab_data/benchmark_runs.json`](../open_spark_jev/serve/lab_data/benchmark_runs.json) and drive the Decision Lab's benchmark table; every value there traces to `runs/osdg/final60.json`. Submit your own with the **Benchmark result** issue template.
