# Models

Generated from `docs/model_lineage.yaml` by `python -m open_spark_jev.model_registry render`; edit that file, not this one.

**Input:** an application state (text/JSON/logs) plus one or more typed questions -- Choice (pick one of N options), Score (place on an ordered rubric), or Noul (a yes/no claim). **Output:** a structured answer per question -- selected option / rubric level / yes-no -- each carrying a full probability distribution and a confidence score. Never free text, never a malformed answer; the answer space is closed by construction.

## Released models

Details, limits and pinned revisions: [MODEL_CARD.md](../MODEL_CARD.md).

| release id | date | weights | recipe |
|---|---|---|---|
| `spark-s1-4b-v3` | 2026-09-20 | [abhishek085/spark-s1-4b-v3](https://huggingface.co/abhishek085/spark-s1-4b-v3) | spark-s1-4b-v3. Qwen3-4B, LoRA r16 (merged) trained 3 epochs with cross-entropy plus a Brier regulariser on th... |
| `spark-s1-1.7b-v3` | 2026-09-20 | [abhishek085/spark-s1-1.7b-v3](https://huggingface.co/abhishek085/spark-s1-1.7b-v3) | spark-s1-1.7b-v3. Same recipe and data as v3-4b on a Qwen3-1.7B backbone (58 min training). Choice temperature... |

## Research checkpoints (not released)

Earlier experiments on simulator and teacher-generated data, kept for comparison. Their numbers predate the os-datagen data and are not comparable with the released models; see [BENCHMARKS.md](BENCHMARKS.md) (B6-B8 and later) for how they were measured.

| id | date | mechanism | parent | overall acc | overall ECE | checkpoint dir |
|---|---|---|---|---|---|---|
| sft-qwen3-1.7b | 2026-09-18 | Phase 1: supervised fine-tune of Qwen3-1.7B on 16,200 simula | - | 0.808 | 0.020 | `checkpoints/sft-qwen3-1.7b` |
| rlcd-direct-qwen3-1.7b | 2026-09-18 | Phase 2, mechanism 2 (direct calibration objective): exact m | sft-qwen3-1.7b | 0.813 | 0.020 | `checkpoints/rlcd-direct-qwen3-1.7b` |
| rlcd-direct-qwen3-1.7b-BUGGY-archived | 2026-09-18 | Same intent as rlcd-direct-qwen3-1.7b (direct calibration ob | sft-qwen3-1.7b | 0.810 | 0.036 | `runs/archive_buggy_rlcd_direct_20260918/checkpoint` |
| rlcd-contrastive-qwen3-1.7b | 2026-09-19 | Phase 2, mechanism 1 (academic RLCD, Yang et al. 2023): 499  | sft-qwen3-1.7b | 0.812 | 0.021 | `checkpoints/rlcd-contrastive-qwen3-1.7b` |
| grpo-qwen3-1.7b | 2026-09-19 | Ablation baseline: sampled single-token GRPO (TRL), reward = | sft-qwen3-1.7b | 0.746 | 0.158 | `checkpoints/grpo-qwen3-1.7b` |

Architecture experiments (A0-A6) are tracked in [NOVELTY.md](NOVELTY.md); the run log is [RUNS.md](RUNS.md).
