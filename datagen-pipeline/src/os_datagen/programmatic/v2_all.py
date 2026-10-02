"""Generate every v2 programmatic family into one directory (train / calibration / test_locked / challenge).

Each split is SHUFFLED with a fixed seed: prog_v1's files were grouped by family, which silently biased any head-of-file
--limit sample to a single family (it corrupted two analyses on 2026-09-29). Every row is verified by its family's
independent verify_row before anything is written; the manifest records counts per family / skin / layout and the
verification result.

  python -m os_datagen.programmatic.v2_all --out ../data/synthetic/prog_v2_full --n-train 4000 --n-eval 300
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random

from . import v2, v2_policy, v2_routing, v2_sampling, v2_tradeoff

FAMILIES = {
    v2.FAMILY: v2,
    v2_policy.FAMILY: v2_policy,
    v2_sampling.FAMILY: v2_sampling,
    v2_routing.FAMILY: v2_routing,
    v2_tradeoff.FAMILY: v2_tradeoff,
}
SPLITS = ("train", "calibration", "test_locked", "challenge")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fams = [f for f in a.families.split(",") if f]
    os.makedirs(a.out, exist_ok=True)
    manifest = {
        "families": fams,
        "n_train_per_family": a.n_train,
        "n_eval_per_family": a.n_eval,
        "seed": a.seed,
        "splits": {},
    }
    for split in SPLITS:
        n = a.n_train if split == "train" else a.n_eval
        rows, bad = [], []
        for f in fams:
            mod = FAMILIES[f]
            for i in range(n):
                r = mod.gen(split, i)
                why = mod.verify_row(r)
                if why:
                    bad.append(f"{r['id']}: {why}")
                rows.append(r)
        if bad:
            raise SystemExit(f"{split}: {len(bad)} rows failed verification, e.g. {bad[:3]}")
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{split}: duplicate ids")
        random.Random(f"{a.seed}|{split}").shuffle(rows)
        with open(os.path.join(a.out, f"{split}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "rows": len(rows),
            "verified": len(rows),
            "by_family": dict(collections.Counter(r["meta"]["scenario_family"] for r in rows)),
            "by_skin": dict(collections.Counter(r["meta"]["world"]["skin"] for r in rows)),
            "by_layout": dict(
                collections.Counter(
                    r["meta"]["world"].get("layout") or r["meta"]["world"].get("format") for r in rows
                )
            ),
            "soft_target_rows": sum(r["meta"]["soft_target"] for r in rows),
        }
        print(
            f"{split}: {len(rows)} rows, all verified, {manifest['splits'][split]['soft_target_rows']} soft-target"
        )
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
