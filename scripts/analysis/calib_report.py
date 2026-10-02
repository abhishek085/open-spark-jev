"""Uniform calibration report from osdg rows.jsonl: accuracy (soft-aware), ECE, Brier, NLL, confident-wrong rates,
per family and overall, on non-calibration splits, using the temperature fitted on the run's own calibration split.
Usage: python scripts/analysis/calib_report.py runs/osdg/<name> [runs/osdg/<name2> ...]"""

import collections, json, math, sys
from pathlib import Path


def softmax(z, T):
    z = [x / T for x in z]
    m = max(z)
    p = [math.exp(x - m) for x in z]
    s = sum(p)
    return [x / s for x in p]


def report(run):
    run = Path(run)
    T = json.load(open(run / "summary.json"))["temperature_fit_on_calibration"]["choice"]
    agg = collections.defaultdict(lambda: collections.Counter())
    bins = collections.defaultdict(lambda: [[0, 0.0, 0.0] for _ in range(10)])
    for l in open(run / "rows.jsonl"):
        r = json.loads(l)
        if r["qtype"] != "choice" or r["split"] == "calibration" or r.get("perm", 0) != 0:
            continue
        p = softmax(r["logits"], T)
        i = max(range(len(p)), key=p.__getitem__)
        c = p[i]
        ok = r["labels"][i] in r["acceptable"]
        gp = sum(p[j] for j, lb in enumerate(r["labels"]) if lb in r["acceptable"])
        for k in (r.get("family") or r.get("pack") or "?", "ALL"):
            a = agg[k]
            a["n"] += 1
            a["ok"] += ok
            a["conf"] += c
            a["brier"] += sum(
                (p[j] - (r["labels"][j] in r["acceptable"]) / len(r["acceptable"])) ** 2
                for j in range(len(p))
            )
            a["nll"] += -math.log(max(gp, 1e-9))
            a["cw8"] += (not ok) and c >= 0.8
            a["cw9"] += (not ok) and c >= 0.9
            a["wrong"] += not ok
            b = bins[k][min(9, int(c * 10))]
            b[0] += 1
            b[1] += ok
            b[2] += c
    out = {}
    for k, a in agg.items():
        n = a["n"]
        ece = sum(b[0] / n * abs(b[1] / b[0] - b[2] / b[0]) for b in bins[k] if b[0])
        out[k] = dict(
            n=n,
            acc=a["ok"] / n,
            ece=ece,
            brier=a["brier"] / n,
            nll=a["nll"] / n,
            conf=a["conf"] / n,
            cw80=a["cw8"] / n,
            cw90=a["cw9"] / n,
            wrong_conf80=a["cw8"] / max(1, a["wrong"]),
        )
    return out


if __name__ == "__main__":
    res = {p: report(p) for p in sys.argv[1:]}
    fams = sorted({f for r in res.values() for f in r}, key=lambda f: (f != "ALL", f))
    for f in fams:
        print(f"\n{f}")
        for p, r in res.items():
            if f in r:
                x = r[f]
                print(
                    f"  {Path(p).name:34s} n={x['n']:5d} acc={x['acc']:.3f} ece={x['ece']:.3f} brier={x['brier']:.3f} nll={x['nll']:.3f} conf={x['conf']:.3f} cw@.8={x['cw80']:.3f} cw@.9={x['cw90']:.3f} (wrong->conf {x['wrong_conf80']:.2f})"
                )
