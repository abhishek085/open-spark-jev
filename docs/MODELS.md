# Model registry

Every trained variant of open-spark-Jev, in one place: what experiment it represents,
how it compares to Jev, and the fastest way to run it. Generated from
`docs/model_lineage.yaml` by `python -m open_spark_jev.model_registry render` -- edit that file
and regenerate, don't hand-edit this one.

**Input:** an application state (text/JSON/logs) plus one or more typed questions -- Choice (pick one of N options), Score (place on an ordered rubric), or Noul (a yes/no claim). **Output:** a structured answer per question -- selected option / rubric level / yes-no -- each carrying a full probability distribution and a confidence score. Never free text, never a malformed answer; the answer space is closed by construction.

| id | date | mechanism | parent | overall acc | overall ECE | card |
|---|---|---|---|---|---|---|
| sft-qwen3-1.7b | 2026-09-18 | Phase 1: supervised fine-tune of Qwen3-1.7B on 16,200 simula | - | 0.808 | 0.020 | [sft-qwen3-1.7b](checkpoints/sft-qwen3-1.7b/MODEL_CARD.md) |
| rlcd-direct-qwen3-1.7b | 2026-09-18 | Phase 2, mechanism 2 (direct calibration objective): exact m | sft-qwen3-1.7b | 0.813 | 0.020 | [rlcd-direct-qwen3-1.7b](checkpoints/rlcd-direct-qwen3-1.7b/MODEL_CARD.md) |
| rlcd-direct-qwen3-1.7b-BUGGY-archived | 2026-09-18 | Same intent as rlcd-direct-qwen3-1.7b (direct calibration ob | sft-qwen3-1.7b | 0.810 | 0.036 | [rlcd-direct-qwen3-1.7b-BUGGY-archived](runs/archive_buggy_rlcd_direct_20260918/checkpoint/MODEL_CARD.md) |
| rlcd-contrastive-qwen3-1.7b | 2026-09-19 | Phase 2, mechanism 1 (academic RLCD, Yang et al. 2023): 499  | sft-qwen3-1.7b | 0.812 | 0.021 | [rlcd-contrastive-qwen3-1.7b](checkpoints/rlcd-contrastive-qwen3-1.7b/MODEL_CARD.md) |

Full per-domain numbers and the honest before/after on any bug fixes are in [docs/BENCHMARKS.md](BENCHMARKS.md); the architecture side (non-training-mechanism experiments) is tracked separately in [docs/NOVELTY.md](NOVELTY.md).
