# Examples

```bash
# per-family generation with a real local model (trial first)
os-datagen generate --task-pack foundation_semantic_entailment_v1 --models configs/models.local.yaml --n-train 6 --n-calibration 1 --n-locked-test 1 --n-challenge 2 --out artifacts/trial
os-datagen report --run artifacts/trial
os-datagen inspect --run artifacts/trial --record-id osj-sem-000001-v001
python examples/generate_foundation_data.py     # dry-run over the foundation packs
python examples/generate_harness_data.py        # dry-run over the harness packs
python examples/run_sandbox_rollouts.py         # sandbox + Wilson-bound routing for the router
python examples/evaluate_candidate_model.py     # baselines (or your endpoint via env)
python examples/inspect_dataset.py artifacts/harness_dry/accepted_train.jsonl
```
