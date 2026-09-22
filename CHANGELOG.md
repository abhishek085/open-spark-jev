# Changelog

Format follows Keep a Changelog. Release ids are `spark-s1-<size>-v<n>`.

## [0.2.0] - 2026-09-22

### Added
- `spark-s1-4b-v5`, `spark-s1-1.7b-v5`: retrained on 49 Jev-style-only task packs (11,792 rows / 19,568 examples, up from v3's 967), lower LR / one epoch, every question type calibrated. `os-datagen` code-only rule packs (label and text both from deterministic rules, no LLM). New harness-engineering task packs (tool-call verification, context pruning, response-quality/citation checks, delegation, error recovery, plan alignment, trace monitoring) and records packs (entity resolution, fraud/risk, invoice/insurance/SOC routing).
- JevBench (public tiers) evaluation: `scripts/run_jevbench.sh`, `scripts/run_jevbench_quant.sh`.
- NVFP4 post-training quantization: `scripts/quant/ptq_nvfp4.py`, `scripts/quant/make_v5_calib_prompts.py`, `scripts/quant/endpoint_latency.py`.

### Changed
- Scope narrowed on purpose to Jev-style decisions only; general-purpose text classification is no longer a training goal (was in v3/v4).

### Findings (not shipped)
- An interim v4 candidate (more data, same broad scope) regressed prompt-injection-with-context (0.819 -> 0.634); v5's rebalanced data recovered it (0.813).
- RLCD-direct, tried twice (v4 and v5, the second time on a pool filtered to rows the model got wrong or was unsure on), made results worse both times, not better.
- NVFP4 (MLP-only) quantization: 1.66x faster on the 4B via vLLM with a small hard-tier cost (JevBench); on the 1.7B it cost 9-13 accuracy points across external sets and was not shipped.

## [Unreleased]

### Added
- Decision Lab (`osj lab`): local UI for proposed tool-call decisions with fixtures, recorded real model outputs, policy trace and benchmark view; works without weights.
- `POST /v1/gate` and `open_spark_jev.classify_tool_call`: model distribution plus deterministic policy, fail closed.
- `open_spark_jev/policy.py`: heuristic rules for sensitive paths, network egress, security weakening, mutation, cloud copy and opaque input.
- Charts against Kev and Jev (`docs/img/`), results B24-B32 in BENCHMARKS.md, MODEL_CARD.md, docs/EVALUATION.md, docs/API.md, docs/INTEGRATIONS.md, contribution, release and launch docs.
- Issue and pull request templates.

### Changed
- README reorganised for first-run clarity; the previous README is preserved in docs/PROJECT_OVERVIEW.md.
- `scripts/fetch_checkpoints.sh` downloads a released model repo into `checkpoints/<release-id>/`.

## [0.1.0] - 2026-09-20

### Added
- `spark-s1-4b-v3` and `spark-s1-1.7b-v3`: Qwen3 backbones with LoRA (merged), trained on 967 code-labelled os-datagen rows, published on Hugging Face.
- os-datagen data factory (`datagen-pipeline/`) and the `spark-s1-osdg-v1` dataset.
- Typed `POST /v1/decide` contract (choice, boolean, score) with per-decision probabilities, confidence, margin, entropy and latency.
- Evaluation harness: per-source external suites, os-datagen splits, option-permutation robustness, per-row outputs, speed benchmark.

### Known issues
- Misclassifies security-weakening configuration changes, credential exfiltration and backup-framed cloud copy; vulnerable-code detection is near chance.
- Boolean and Score temperatures are not fitted; the 60-case benchmark is diagnostic and small.
