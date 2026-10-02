"""RLDM-9: build an advantage-weighted regression (AWR, Peng et al. 2019) dataset from eval logs
this project already produces and keeps -- see the standing per-row-save rule referenced in
project memory. Every prior RL attempt here (RLCD-direct/contrastive, GRPO) was a live rollout
against a policy being actively trained; this is the first offline-RL attempt: no rollout loop at
all, just re-weighting supervised regression by how good each *already-logged* prediction was.

Source: eval/osdg.py's rows.jsonl for several sibling checkpoints (v7.2-v7.6), calibration split,
permutation 0 only (avoids inflating the pool with near-duplicate permuted copies of the same
item). Each row is one (state, question, logged prediction) tuple from a genuinely different
"behavior policy" -- the checkpoint that produced it -- which is closer to AWR's intended setting
(a mixture of behavior policies, not one) than logging a single checkpoint would be.

Per item: reward = 1 - Brier(p_logged, gold_onehot) -- the same proper-scoring-rule reward family
already used by RLCD-direct's Brier term (docs/RESEARCH.md), so "what counts as good" is
unchanged; only the learning mechanism (offline AWR instead of online exact-policy-gradient) is
new. Baseline is the mean reward *within the same scenario_family*, since some families are
inherently harder and a single global baseline would unfairly zero out or invert the advantage of
every correct answer on a hard family. weight = exp(clip(advantage / temperature, max=weight_clip)).

  python -m open_spark_jev.experimental.awr_data \
      --rows runs/osdg/v7.2-4b-progsplits/rows.jsonl runs/osdg/v7.3-4b-progsplits/rows.jsonl \
             runs/osdg/v7.4-4b-progsplits/rows.jsonl runs/osdg/v7.5-4b-progsplits/rows.jsonl \
             runs/osdg/v7.6-4b-progsplits/rows.jsonl \
      --records data/synthetic/prog_v1/calibration.jsonl \
      --out data/synthetic/awr_v1_calibration.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def softmax(z):
    m = max(z)
    ex = [math.exp(x - m) for x in z]
    s = sum(ex)
    return [x / s for x in ex]


def brier(p, gold_idx):
    return sum((pi - (1.0 if i == gold_idx else 0.0)) ** 2 for i, pi in enumerate(p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rows", nargs="+", required=True, help="rows.jsonl from one or more sibling checkpoints' osdg eval"
    )
    ap.add_argument(
        "--records", required=True, help="source jsonl the rows' ids resolve against (state/question text)"
    )
    ap.add_argument(
        "--split",
        default="calibration",
        help="which osdg split to pull from rows.jsonl (never test_locked/challenge -- those stay held-out)",
    )
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--weight-clip", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for path in a.rows:
        src = path.split("/")[-2] if "/" in path else path
        for line in open(path):
            r = json.loads(line)
            if r["split"] == a.split and r["perm"] == 0:
                r["_source"] = src
                rows.append(r)
    print(f"pooled {len(rows)} rows from {len(a.rows)} checkpoints", flush=True)

    for r in rows:
        p = softmax(r["logits"])
        r["_reward"] = 1.0 - brier(p, r["gold_idx"])

    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r["_reward"])
    baseline = {f: sum(v) / len(v) for f, v in by_family.items()}
    print("per-family baseline reward:", {f: round(v, 4) for f, v in baseline.items()}, flush=True)

    for r in rows:
        adv = r["_reward"] - baseline[r["family"]]
        r["_advantage"] = adv
        r["_weight"] = min(math.exp(adv / a.temperature), a.weight_clip)

    recs_by_id = {json.loads(ln)["id"]: json.loads(ln) for ln in open(a.records)}
    n_missing = 0
    out = []
    for r in rows:
        rec = recs_by_id.get(r["id"])
        if rec is None:
            n_missing += 1
            continue
        out.append(
            {
                "id": r["id"],
                "source_checkpoint": r["_source"],
                "family": r["family"],
                "reward": round(r["_reward"], 4),
                "advantage": round(r["_advantage"], 4),
                "weight": round(r["_weight"], 4),
                "record": rec,
            }
        )
    if n_missing:
        print(f"WARNING: {n_missing} rows had no matching record in {a.records}", flush=True)

    with open(a.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")

    ws = [o["weight"] for o in out]
    print(f"wrote {len(out)} weighted examples to {a.out}")
    print(f"weight stats: min={min(ws):.3f} mean={sum(ws) / len(ws):.3f} max={max(ws):.3f}")


if __name__ == "__main__":
    main()
