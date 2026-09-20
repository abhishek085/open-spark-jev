"""Run the Decision Lab fixtures through the tool gate. Nothing is executed: commands are only text.

  python examples/tool_gate.py                                   # rules only, no weights (never returns allow)
  python examples/tool_gate.py --model checkpoints/spark-s1-4b-v3
"""
import argparse
import json
import pathlib

from open_spark_jev import classify_tool_call

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None, help="checkpoint path; omit for rules only")
ap.add_argument("--threshold", type=float, default=0.995)
a = ap.parse_args()
fx = json.loads((pathlib.Path(__file__).parent.parent / "open_spark_jev/serve/lab_data/tool_calls.json").read_text())
for f in fx:
    r = classify_tool_call(tool=f["tool"], command=f["command"], context=f["context"], model=a.model or "none", auto_allow_threshold=a.threshold)
    print(f"{r.policy_action:5}  {f['command'][:70]:70}  model={r.choice}")
