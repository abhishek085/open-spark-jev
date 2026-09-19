"""Reproduces the Banking77 alignment comparison against an independent (non-TypeSafe)
evaluation of Jev, documented in full in docs/JEV_COMPARISON.md.

What this does, precisely:
1. Loads data/external/jev_baselines_eval/banking77_test.csv and replicates
   df.sample(n=300, random_state=0) exactly as ickma2311/jev-baselines-eval's code/common.py
   does -- verified to reproduce their recorded gold labels with 0 mismatches on all 208 of
   Jev's real result rows.
2. Reduces the label space to the top-K most frequent intents in the Banking77 training set
   (K=20 by default, matching this repo's own data/public.py banking77() adapter and its
   reason for existing: our Choice primitive is capped at 26 options). This is NOT
   Jev's actual task -- Jev answered the full 77-way question in one call; our model cannot
   represent more than 26 options at all. Reducing the label set is the only way to construct
   *any* comparable task, and it makes ours strictly easier (fewer options), a fact reported
   alongside every number below, not hidden.
3. Filters Jev's real 208-item result rows to just the ones whose gold label survives that
   reduction (a real subset, not manufactured to hit a target size), and recomputes Jev's
   accuracy on exactly that subset from its own raw per-item data (not just quoting Jev's
   reported all-77-way number, which is not the same task).
4. Sends the identical items, in the identical Jev wire format (state = "Bank customer
   message: {text}", a single "choice" question over the reduced label set), to our own
   checkpoint via the gateway's own request-translation code (serve/gateway.py's _from_jev),
   so the request shape our model receives is provably the same code path a real
   /v1/evaluate call would use, not a hand-approximation.

Usage:
  python -m open_spark_jev.model_registry render  # unrelated, just an example of running from repo root
  .venv/bin/python scripts/jev_alignment_comparison.py --model checkpoints/rlcd-direct-qwen3-1.7b --top-k 20
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from open_spark_jev.calibration import summary
from open_spark_jev.model import MenuScorer
from open_spark_jev.schema import State
from open_spark_jev.serve.gateway import JevQuestion, _from_jev

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external" / "jev_baselines_eval"


def human(label: str) -> str:
    return label.replace("_", " ")


def load_aligned_subset(top_k: int) -> tuple[list[str], list[dict]]:
    train = pd.read_csv(EXT / "banking77_train.csv")
    top_labels = [c for c, _ in Counter(train.category).most_common(top_k)]

    test = pd.read_csv(EXT / "banking77_test.csv").sample(n=300, random_state=0).reset_index(drop=True)
    jev_rows = [json.loads(line) for line in open(EXT / "results_b0.jsonl") if json.loads(line)["method"] == "jev"]

    mismatches = sum(1 for d in jev_rows if test.category[d["i"]] != d["gold"])
    if mismatches:
        raise RuntimeError(f"{mismatches} gold-label mismatches vs recorded Jev results -- sampling did not reproduce exactly")

    aligned = [d for d in jev_rows if d["gold"] in top_labels]
    items = [{"i": d["i"], "text": test.text[d["i"]], "gold": d["gold"], "jev_pred": d["pred"],
              "jev_conf": d["conf"], "jev_correct": d["correct"]} for d in aligned]
    return top_labels, items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default="runs/jev_comparison")
    a = ap.parse_args()

    top_labels, items = load_aligned_subset(a.top_k)
    print(f"aligned subset: {len(items)} of Jev's 208 real result items have a gold label in the top-{a.top_k} set")
    jev_acc = sum(it["jev_correct"] for it in items) / len(items)
    print(f"Jev accuracy on this subset (recomputed from their raw per-item data): {jev_acc:.4f} ({sum(it['jev_correct'] for it in items)}/{len(items)})")

    criteria = {lab: human(lab) for lab in top_labels}
    m = MenuScorer(a.model)
    results = []
    for it in items:
        state = State(content=f"Bank customer message: {it['text']}")
        jq = JevQuestion(type="choice", instructions="Which intent best describes the customer's message?", criteria=criteria)
        q = _from_jev("intent", jq)
        ans = m.decide(state, [q])[0]
        results.append({**it, "our_pred": ans.selected, "our_conf": ans.confidence,
                        "our_correct": ans.selected == it["gold"], "our_probs": dict(zip(ans.labels, ans.probs))})

    our_acc = sum(r["our_correct"] for r in results) / len(results)
    print(f"our model accuracy on the identical items + identical reduced option set: {our_acc:.4f} ({sum(r['our_correct'] for r in results)}/{len(results)})")

    probs = [[r["our_probs"].get(lab, 0.0) for lab in top_labels] for r in results]
    golds = [top_labels.index(r["gold"]) for r in results]
    s = summary(probs, golds)
    print("our model calibration on this subset:", json.dumps(s, indent=2))

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "our_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"n": len(items), "top_k": a.top_k, "jev_accuracy": jev_acc, "our_accuracy": our_acc, "our_calibration": s}, f, indent=2)
    print(f"wrote {out_dir}/our_results.json and {out_dir}/summary.json")


if __name__ == "__main__":
    main()
