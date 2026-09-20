# Evaluation protocol

```
train → fit the decision model      calibration → fit temperature/calibrator
locked test → NLL, Brier, ECE, per-class calibration, selective risk      challenge → robustness and safety
```

`os-datagen evaluate --dataset D --model-endpoint URL --model NAME [--mode candidate_logprob|constrained_generation]`
renders each record with the choice/score/boolean template (`training/render.py`), records the inference mode, and writes
`predictions.jsonl` **per row** (raw output, raw scores, probabilities, selection, truth match, latency), `metrics.json`,
`slices.json`, `reliability.json`, `selective_risk.json`, `cost_quality_frontier.json` (router) and `report.md`.

Metrics: top-1 and acceptable-set accuracy, macro accuracy, NLL, Brier, ECE (+ reliability bins), per-option precision/recall,
selective risk vs confidence threshold, cost-quality frontier, slices by namespace, pack, scenario family, difficulty, option
count, length and truth tier.

`constrained_generation` produces a smoothed one-hot when the endpoint exposes no logprobs and labels it as such
(`probabilities_source`); use `candidate_logprob` for scoring-based probabilities. Baselines `baseline://uniform` and
`baseline://prior` give a floor. **No calibration claim is made** unless `--calibration-dataset` was used and
`metrics.calibration.fitted_on` names it; temperature is fit on that split only.

Also: `os-datagen option-order` (position-sensitivity check, see validation_and_provenance.md). Calibration and locked-test splits use held-out scenario families, so measured
calibration reflects a shift in scenario structure, not just new wording.
