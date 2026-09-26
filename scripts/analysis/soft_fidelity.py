#!/usr/bin/env python
"""Distribution fidelity on the programmatic splits: are the probabilities themselves right?

Accuracy is the wrong lens for two of the five families. `underdetermined` deliberately has k options
tied at 1/k, so a correct model's argmax is arbitrary and its accuracy is ~1/k however well calibrated
it is; `probability_exact` has a gold posterior strictly between 0 and 1. For both, the question is not
"did it pick the gold label" but "does its distribution match the gold distribution".

Reports, per family, mean TVD to the gold distribution (JevBench's own calibration measure on its
probability items) and soft Brier, alongside accuracy for reference. Temperatures are the ones the
eval fitted on the calibration split.

  python scripts/analysis/soft_fidelity.py --rows runs/osdg/v7.2-4b-progsplits/rows.jsonl \
      --source data/synthetic/prog_v1 --summary runs/osdg/v7.2-4b-progsplits/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def softmax(z):
    z = np.asarray(z, float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--source", required=True, help="directory holding <split>.jsonl with target.dist")
    ap.add_argument("--summary", help="summary.json, to reuse its fitted temperatures")
    ap.add_argument("--out")
    a = ap.parse_args()

    T = {}
    if a.summary:
        T = json.load(open(a.summary)).get("temperature_fit_on_calibration", {})

    gold: dict[str, dict] = {}
    for split in ("calibration", "test_locked", "challenge"):
        p = os.path.join(a.source, f"{split}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            gold[r["id"]] = {"dist": r["target"]["dist"], "family": r["meta"]["scenario_family"],
                             "soft": r["meta"]["soft_target"], "split": split}

    agg: dict[tuple, dict] = defaultdict(lambda: {"tvd": [], "sbrier": [], "hit": [], "conf": []})
    missing = 0
    for line in open(a.rows):
        r = json.loads(line)
        g = gold.get(r["id"])
        if g is None:
            missing += 1
            continue
        if r.get("perm", 0) != 0:      # one permutation per item; order effects are reported by the eval
            continue
        p = softmax(np.asarray(r["logits"], float) / T.get(r["qtype"], 1.0))
        t = np.array([g["dist"].get(l, 0.0) for l in r["labels"]], float)
        s = t.sum()
        if s <= 0:
            continue
        t = t / s
        k = (r["split"], g["family"])
        agg[k]["tvd"].append(0.5 * float(np.abs(p - t).sum()))
        agg[k]["sbrier"].append(float(((p - t) ** 2).sum()))
        agg[k]["hit"].append(float(int(np.argmax(p)) == int(np.argmax(t))))
        agg[k]["conf"].append(float(p.max()))

    report: dict = {"rows": a.rows, "temperatures": T, "unmatched_rows": missing, "by_split_family": {}}
    for (split, fam), v in sorted(agg.items()):
        report["by_split_family"].setdefault(split, {})[fam] = {
            "n": len(v["tvd"]), "mean_tvd": round(float(np.mean(v["tvd"])), 4),
            "soft_brier": round(float(np.mean(v["sbrier"])), 4),
            "argmax_matches_gold_mode": round(float(np.mean(v["hit"])), 4),
            "mean_conf": round(float(np.mean(v["conf"])), 4)}
    for split, fams in report["by_split_family"].items():
        tot = [x for f in fams.values() for x in [f["mean_tvd"]] * f["n"]]
        report["by_split_family"][split]["ALL"] = {"n": sum(f["n"] for f in fams.values()),
                                                  "mean_tvd": round(float(np.mean(tot)), 4)}
    print(json.dumps(report, indent=1))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
