# spark-s1 experiment ladder (v0-v7) and how it maps to A0-A6

Data for this ladder: the `os-datagen` splits only (`datagen-pipeline/artifacts/mixture_final`, exported to
`data/synthetic/osdg_train.jsonl`, `data/benchmarks/osdg_{calibration,test_locked,challenge}.jsonl`). Train fits weights, calibration
fits temperature/thresholds, test_locked is read once per model, challenge is reported separately. The 60-row tool-call set
(`ext-toolcall-risk`) is the final review after training and testing are finished. Caveat: its per-row failures were analysed
before this ladder (BENCHMARKS.md B25), so it is a soft, not a blind, final check. Other external suites remain reporting-only.

Naming: `spark-s1-<size>-v<n>[-<variant>]`. A0-A6 are architecture variants; v0-v7 are the version ladder. They overlap:

| version | idea | A-variant overlap | status before this ladder | plan / decision |
|---|---|---|---|---|
| v0 | frozen direct-logit base, no training | - | Qwen3-1.7B zero-shot on our simulators only (B6) | **run**: 0.6B / 1.7B / 4B / Gemma-4-E4B / 12B on the osdg splits, 3 option orders each |
| v1 | + temperature on calibration, confidence, abstention | - | temperature yes (fit on a slice of training data); no held-out calibration split, no abstention | **run** with v0 (`eval/osdg.py`): calibration-split temperature, selective prediction |
| v2 | frozen backbone + small head | A3/A4 are heads on a LoRA-tuned backbone | A3 refuted (B26) | **done: refuted** (B28) |
| v3 | LoRA student, randomized options | **A0** | LoRA r16 on 1.7B done (B22, B24); options fixed order; osdg data never used | **run**: LoRA on osdg_train with option-permutation augmentation; sizes 0.6B / 1.7B / 4B |
| v4 | per-candidate cross-encoder | none | not done | **deferred**: costs K passes and loses the single-pass speed; head variants have underperformed (B22, B26). Revisit only if v3 fails on unseen option wording |
| v5 | multi-question shared-state | **A1** (+ A2 prefix-LM) | A1 confirmed (~15% faster), A2 refuted | **skip re-run**: osdg rows are single-question; A1 already measured (B22) |
| v6 | cascade with escalation | - | not done | **run offline** on per-row outputs: 1.7B student -> 4B scorer -> review, cost vs accuracy |
| v7 | outcome-trained calibration (RLCD) | RLCD-direct / contrastive / GRPO | simulator-era runs; v2 retrains unfinished | **run RLCD-direct only** on the best v3 model (Brier + confident-wrong penalty); skip contrastive/GRPO (no evidence they beat direct) |
| - | A4 evidential head | head variant | implemented, not run | **skip**: same family as A3 (refuted) |
| - | A5 routed adapters | mixture of LoRAs | implemented, not run | **skip**: ~1k train rows over 15 packs is too little per adapter |
| - | A6 joint multi-question | correlated questions | implemented, not run | **skip**: osdg has no correlated multi-question states; needs its own simulator |

Every eval writes per-row data (`runs/osdg/<name>/rows.jsonl`). Results are recorded in BENCHMARKS.md and RUNS.md as they land.
