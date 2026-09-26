#!/usr/bin/env python
"""v7 calibration study on already-collected per-permutation logits (no GPU, no retraining).

Reads an osdg rows.jsonl (one row per item x option-permutation, with raw label logits) and compares
calibration strategies, all fitted on the *calibration* split and reported on test_locked + challenge:

  none           raw softmax, T = 1
  global_T       one temperature, NLL-fitted
  per_type_T     one temperature per question type (what v6 ships)
  per_group_T    per (question type x difficulty) temperature
  perm_avg       mean distribution over the item's option permutations (order-invariant), then per-type T
  perm_avg_T     perm_avg, with T refitted on the averaged distribution
  disagree       logistic recalibrator on distribution features + cross-permutation disagreement

Metrics: accuracy, ECE (10 equal-width bins, top-label), Brier, NLL, and JevBench's ECE-half of the
calibration axis, 100 * (1 - ECE / 0.5), so numbers are comparable to the leaderboard.

  python scripts/analysis/calib_v7_study.py --rows runs/osdg/v6-4b-v5splits/rows.jsonl --out runs/osdg/calib_v7_study.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar

BINS = 10


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ece(conf, correct, n=BINS):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    b = np.minimum((conf * n).astype(int), n - 1)
    out = 0.0
    for i in range(n):
        m = b == i
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def metrics(probs, correct_idx, n_labels):
    """probs: list of arrays (ragged). correct_idx: index of gold within each array."""
    conf = np.array([p.max() for p in probs])
    hit = np.array([float(int(np.argmax(p)) == g) for p, g in zip(probs, correct_idx)])
    nll = float(-np.mean([np.log(max(p[g], 1e-12)) for p, g in zip(probs, correct_idx)]))
    brier = float(np.mean([((p - np.eye(len(p))[g]) ** 2).sum() for p, g in zip(probs, correct_idx)]))
    e = ece(conf, hit)
    return {"n": len(probs), "accuracy": float(hit.mean()), "ece": e, "brier": brier, "nll": nll,
            "mean_conf": float(conf.mean()), "ece_axis_half": round(100 * max(0.0, 1 - e / 0.5), 1)}


def fit_T(logit_list, gold_idx):
    """Temperature minimising NLL."""
    if not logit_list:
        return 1.0

    def f(logT):
        T = float(np.exp(logT))
        return -np.mean([np.log(max(softmax(np.asarray(z) / T)[g], 1e-12)) for z, g in zip(logit_list, gold_idx)])

    r = minimize_scalar(f, bounds=(np.log(0.25), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def load(path):
    """Group rows by item id; each item has one entry per permutation, probs keyed by label name."""
    items: dict[tuple, dict] = {}
    for line in open(path):
        r = json.loads(line)
        # an id can carry more than one question (different option sets), so key on both
        key = (r["id"], tuple(sorted(r["labels"])))
        it = items.setdefault(key, {"id": r["id"], "split": r["split"], "qtype": r["qtype"], "pack": r["pack"],
                                        "difficulty": r.get("difficulty", "na"), "family": r.get("family", "na"),
                                        "gold": r["gold"], "acceptable": r.get("acceptable") or [r["gold"]], "perms": []})
        it["perms"].append({"labels": r["labels"], "logits": np.asarray(r["logits"], float)})
    for it in items.values():   # identical permutations are repeated across eval passes; keep one of each
        seen, uniq = set(), []
        for p in it["perms"]:
            k = tuple(p["labels"])
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        it["perms"] = uniq
    return list(items.values())


def canon(it):
    """Canonical (sorted) label order for the item."""
    return sorted(it["perms"][0]["labels"])


def perm_probs(it, T=1.0, which=0):
    """Distribution from one permutation, mapped onto the canonical label order."""
    p = it["perms"][which]
    d = softmax(p["logits"] / T)
    m = dict(zip(p["labels"], d))
    return np.array([m[l] for l in canon(it)])


def avg_probs(it, T=1.0):
    return np.mean([perm_probs(it, T, k) for k in range(len(it["perms"]))], axis=0)


def gold_index(it):
    return canon(it).index(it["gold"])


def disagreement(it):
    """How much the permutations disagree: mean pairwise TVD, and whether the argmax label flips."""
    ds = [perm_probs(it, 1.0, k) for k in range(len(it["perms"]))]
    if len(ds) < 2:
        return 0.0, 0.0, 0.0
    tvd = [0.5 * np.abs(ds[i] - ds[j]).sum() for i in range(len(ds)) for j in range(i + 1, len(ds))]
    tops = {int(np.argmax(d)) for d in ds}
    sd = float(np.mean([d.max() for d in ds]) - min(d.max() for d in ds))
    return float(np.mean(tvd)), float(len(tops) > 1), sd


def features(p, it):
    s = np.sort(p)[::-1]
    ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
    tvd, flip, sd = disagreement(it)
    return [float(s[0]), float(s[0] - (s[1] if len(s) > 1 else 0.0)), ent, float(np.log(len(p))), tvd, flip, sd]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = load(a.rows)
    by_split = defaultdict(list)
    for it in items:
        by_split[it["split"]].append(it)
    print({k: len(v) for k, v in by_split.items()}, "perms/item:", sorted({len(i["perms"]) for i in items}))

    cal = by_split["calibration"]
    evals = {k: v for k, v in by_split.items() if k != "calibration"}

    # --- fit temperatures on calibration only -------------------------------------------------
    def fit_on(group_fn, rows):
        g = defaultdict(lambda: ([], []))
        for it in rows:
            z, gi = it["perms"][0]["logits"], it["perms"][0]["labels"].index(it["gold"])
            g[group_fn(it)][0].append(z)
            g[group_fn(it)][1].append(gi)
        return {k: fit_T(v[0], v[1]) for k, v in g.items()}

    T_global = fit_on(lambda it: "all", cal)
    T_type = fit_on(lambda it: it["qtype"], cal)
    T_group = fit_on(lambda it: (it["qtype"], it["difficulty"]), cal)
    print("global T", {k: round(v, 3) for k, v in T_global.items()})
    print("per-type T", {k: round(v, 3) for k, v in T_type.items()})
    print("per-group T", {str(k): round(v, 3) for k, v in sorted(T_group.items(), key=str)})

    # temperature refitted on permutation-averaged distributions (grid, since avg isn't a softmax)
    def fit_T_avg(rows, key):
        g = defaultdict(list)
        for it in rows:
            g[key(it)].append(it)
        out = {}
        for k, its in g.items():
            best, bT = 1e9, 1.0
            for T in np.linspace(0.5, 6.0, 56):
                nll = -np.mean([np.log(max(avg_probs(it, T)[gold_index(it)], 1e-12)) for it in its])
                if nll < best:
                    best, bT = nll, float(T)
            out[k] = bT
        return out

    T_avg_type = fit_T_avg(cal, lambda it: it["qtype"])
    print("per-type T (on perm-avg)", {k: round(v, 3) for k, v in T_avg_type.items()})

    # --- disagreement recalibrator, fitted on calibration ------------------------------------
    from sklearn.linear_model import LogisticRegression
    Xc = np.array([features(avg_probs(it, T_avg_type.get(it["qtype"], 1.0)), it) for it in cal])
    yc = np.array([int(np.argmax(avg_probs(it, T_avg_type.get(it["qtype"], 1.0))) == gold_index(it)) for it in cal])
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    clf = LogisticRegression(max_iter=3000).fit((Xc - mu) / sd, yc)

    strategies = {
        "none": lambda it: perm_probs(it, 1.0),
        "global_T": lambda it: perm_probs(it, T_global["all"]),
        "per_type_T": lambda it: perm_probs(it, T_type.get(it["qtype"], 1.0)),
        "per_group_T": lambda it: perm_probs(it, T_group.get((it["qtype"], it["difficulty"]), 1.0)),
        "perm_avg": lambda it: avg_probs(it, T_type.get(it["qtype"], 1.0)),
        "perm_avg_T": lambda it: avg_probs(it, T_avg_type.get(it["qtype"], 1.0)),
    }

    report: dict = {"rows": a.rows, "temperatures": {"global": T_global, "per_type": T_type,
                                                    "per_group": {str(k): v for k, v in T_group.items()},
                                                    "per_type_on_perm_avg": T_avg_type},
                    "disagree_coef": dict(zip(["p_max", "margin", "entropy", "log_n_opts", "perm_tvd", "perm_flip", "conf_spread"],
                                              [round(float(c), 3) for c in clf.coef_[0]])),
                    "splits": {}}

    for split, rows in sorted(evals.items()):
        gi = [gold_index(it) for it in rows]
        nl = [len(canon(it)) for it in rows]
        res = {}
        for name, fn in strategies.items():
            res[name] = metrics([fn(it) for it in rows], gi, nl)
        # the disagreement recalibrator only replaces the *confidence*, not the argmax
        base = [avg_probs(it, T_avg_type.get(it["qtype"], 1.0)) for it in rows]
        conf = [float(1 / (1 + np.exp(-(clf.coef_[0] @ ((np.array(features(p, it)) - mu) / sd) + clf.intercept_[0]))))
                for p, it in zip(base, rows)]
        hit = np.array([float(int(np.argmax(p)) == g) for p, g in zip(base, gi)])
        res["disagree"] = {"n": len(rows), "accuracy": float(hit.mean()), "ece": ece(conf, hit),
                           "mean_conf": float(np.mean(conf)),
                           "ece_axis_half": round(100 * max(0.0, 1 - ece(conf, hit) / 0.5), 1),
                           "note": "recalibrated top-label confidence only (argmax from perm_avg_T)"}
        report["splits"][split] = res
        print(f"\n== {split} (n={len(rows)})")
        for name, m in res.items():
            print(f"  {name:12s} acc={m['accuracy']:.4f} ece={m['ece']:.4f} conf={m['mean_conf']:.3f} "
                  f"axis={m['ece_axis_half']:5.1f}" + (f" brier={m['brier']:.4f} nll={m['nll']:.4f}" if "brier" in m else ""))

    json.dump(report, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
