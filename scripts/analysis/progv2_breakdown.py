#!/usr/bin/env python
"""Accuracy on prog_v2 (osdg rows.jsonl) broken down by governing-date rule, skin and split, all perms averaged.

Rows are one per (item, option-permutation); accuracy = argmax(logits) == gold_idx. Joined to
data/synthetic/prog_v2/<split>.jsonl on id for meta.world.{rule,skin,format}. Chance = mean 1/len(labels).

  python scripts/analysis/progv2_breakdown.py v6-4b v7.6-4b
"""

import collections, json, sys


def acc_table(name):
    recs = {}
    for split in ("calibration", "test_locked", "challenge"):
        for l in open(f"data/synthetic/prog_v2/{split}.jsonl"):
            r = json.loads(l)
            recs[r["id"]] = r["meta"]["world"]
    rows = [json.loads(l) for l in open(f"runs/osdg/{name}-progv2/rows.jsonl")]
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        w = recs[r["id"]]
        hit = int(max(range(len(r["logits"])), key=lambda i: r["logits"][i]) == r["gold_idx"])
        chance = 1.0 / len(r["labels"])
        for key in (
            ("split", r["split"]),
            ("rule", w["rule"]),
            ("skin", w["skin"]),
            ("split+rule", f"{r['split']}/{w['rule']}"),
            ("all", "all"),
        ):
            a = agg[key]
            a[0] += hit
            a[1] += 1
            a[2] += chance
    return agg


if __name__ == "__main__":
    names = sys.argv[1:]
    tabs = {n: acc_table(n) for n in names}
    for group in ("all", "split", "rule", "split+rule", "skin"):
        keys = sorted({k for t in tabs.values() for k in t if k[0] == group})
        print(f"--- by {group} (accuracy | n) ---")
        print(f"{'':28s}" + "".join(f"{n:>14s}" for n in names) + "   chance")
        for k in keys:
            line = f"{k[1]:28s}"
            ch = None
            for n in names:
                a = tabs[n].get(k)
                line += f"{a[0] / a[1]:9.3f} ({a[1] // 3:>3d})" if a else " " * 14
                if a:
                    ch = a[2] / a[1]
            print(line + (f"   {ch:.2f}" if ch else ""))
