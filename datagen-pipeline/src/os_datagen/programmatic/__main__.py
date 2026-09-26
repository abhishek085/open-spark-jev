"""Write programmatic-family splits as training JSONL, verifying every row's gold first.

  python -m os_datagen.programmatic --out ../data/synthetic/prog_v1 --n-train 1200 --n-eval 300
  python -m os_datagen.programmatic --selfcheck        # verify only, write nothing
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from .families import FAMILIES, SPLITS, generate, verify


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="directory for <split>.jsonl (omit with --selfcheck)")
    ap.add_argument("--n-train", type=int, default=1200, help="rows per family in the train split")
    ap.add_argument("--n-eval", type=int, default=300, help="rows per family in each eval split")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    fams = [f for f in a.families.split(",") if f]
    unknown = [f for f in fams if f not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown families: {unknown}; known: {list(FAMILIES)}")

    if a.selfcheck:
        rows = [r for f in fams for s in SPLITS for r in generate(f, s, 60)]
        ok, bad = verify(rows)
        print(f"verified {ok}/{len(rows)} rows")
        soft = Counter(r["meta"]["scenario_family"] for r in rows if r["meta"]["soft_target"])
        print("rows with a genuinely soft target:", dict(soft))
        if bad:
            print("FAILURES:", bad[:20])
            raise SystemExit(1)
        return

    if not a.out:
        raise SystemExit("--out is required unless --selfcheck")
    os.makedirs(a.out, exist_ok=True)
    totals: dict[str, int] = {}
    for split in SPLITS:
        n = a.n_train if split == "train" else a.n_eval
        rows = [r for f in fams for r in generate(f, split, n)]
        ok, bad = verify(rows)
        if bad:
            raise SystemExit(f"{split}: gold verification failed for {len(bad)} rows: {bad[:10]}")
        ids = {r["id"] for r in rows}
        if len(ids) != len(rows):
            raise SystemExit(f"{split}: duplicate ids")
        path = os.path.join(a.out, f"{split}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        totals[split] = len(rows)
        soft = sum(r["meta"]["soft_target"] for r in rows)
        print(f"{path}: {len(rows)} rows ({ok} gold-verified, {soft} soft-target) "
              f"families={dict(Counter(r['meta']['scenario_family'] for r in rows))}")
    print("totals", totals)


if __name__ == "__main__":
    main()
