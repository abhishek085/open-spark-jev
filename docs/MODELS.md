# Model registry

Release models (`spark-s1-4b-v3`, `spark-s1-1.7b-v3`) are described in [../MODEL_CARD.md](../MODEL_CARD.md); the rows below are the full research lineage.

Every trained variant of open-spark-Jev, in one place: what experiment it represents,
how it compares to Jev, and the fastest way to run it. Generated from
`docs/model_lineage.yaml` by `python -m open_spark_jev.model_registry render` -- edit that file
and regenerate, don't hand-edit this one.

**Input:** an application state (text/JSON/logs) plus one or more typed questions -- Choice (pick one of N options), Score (place on an ordered rubric), or Noul (a yes/no claim). **Output:** a structured answer per question -- selected option / rubric level / yes-no -- each carrying a full probability distribution and a confidence score. Never free text, never a malformed answer; the answer space is closed by construction.

| id | date | mechanism | parent | overall acc | overall ECE | checkpoint dir |
|---|---|---|---|---|---|---|
| sft-qwen3-1.7b | 2026-09-18 | Phase 1: supervised fine-tune of Qwen3-1.7B on 16,200 simula | - | 0.808 | 0.020 | `checkpoints/sft-qwen3-1.7b` |
| rlcd-direct-qwen3-1.7b | 2026-09-18 | Phase 2, mechanism 2 (direct calibration objective): exact m | sft-qwen3-1.7b | 0.813 | 0.020 | `checkpoints/rlcd-direct-qwen3-1.7b` |
| rlcd-direct-qwen3-1.7b-BUGGY-archived | 2026-09-18 | Same intent as rlcd-direct-qwen3-1.7b (direct calibration ob | sft-qwen3-1.7b | 0.810 | 0.036 | `runs/archive_buggy_rlcd_direct_20260918/checkpoint` |
| rlcd-contrastive-qwen3-1.7b | 2026-09-19 | Phase 2, mechanism 1 (academic RLCD, Yang et al. 2023): 499  | sft-qwen3-1.7b | 0.812 | 0.021 | `checkpoints/rlcd-contrastive-qwen3-1.7b` |
| grpo-qwen3-1.7b | 2026-09-19 | Ablation baseline: sampled single-token GRPO (TRL), reward = | sft-qwen3-1.7b | 0.746 | 0.158 | `checkpoints/grpo-qwen3-1.7b` |
| v3-4b | 2026-09-20 | spark-s1-4b-v3. Qwen3-4B, LoRA r16 (merged) trained 3 epochs | - | - | - | `checkpoints/v3-4b` |
| v3-1.7b | 2026-09-20 | spark-s1-1.7b-v3. Same recipe and data as v3-4b on a Qwen3-1 | - | - | - | `checkpoints/v3-1.7b` |

Full per-domain numbers and the honest before/after on any bug fixes are in [docs/BENCHMARKS.md](BENCHMARKS.md); the architecture side (non-training-mechanism experiments) is tracked separately in [docs/NOVELTY.md](NOVELTY.md).
