"""Evaluate baselines (or a real endpoint: set OSJ_EVAL_ENDPOINT / OSJ_EVAL_MODEL) on a locked-test file."""
import json
import os
import sys

from os_datagen.evaluation.runner import run_evaluation

dataset = sys.argv[1] if len(sys.argv) > 1 else "artifacts/harness_dry/accepted_test_locked.jsonl"
endpoint = os.environ.get("OSJ_EVAL_ENDPOINT", "baseline://uniform")
model = os.environ.get("OSJ_EVAL_MODEL", "uniform")
print(json.dumps(run_evaluation(dataset and __import__("pathlib").Path(dataset), endpoint, model, __import__("pathlib").Path("artifacts/eval_example"))["overall"], indent=1))
