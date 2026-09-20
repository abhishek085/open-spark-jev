# Changelog

Format follows Keep a Changelog. Release ids are `spark-s1-<size>-v<n>`.

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
