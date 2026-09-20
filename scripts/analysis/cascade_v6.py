"""spark-s1-v6 offline cascade on saved per-row outputs (perm 0): small student answers when calibrated confidence >= t1,
else the 4B scorer answers when its confidence >= t2, else the row goes to review (counts as not auto-decided).
Thresholds are picked on the CALIBRATION split, reported on test_locked and challenge. Latency uses measured p50s."""
import json, sys
import numpy as np
from open_spark_jev.calibration import softmax

S = json.load(open("runs/osdg/speed_v3.json"))

def load(name):
    T = json.load(open(f"runs/osdg/{name}/summary.json"))["temperature_fit_on_calibration"]
    out = {}
    for l in open(f"runs/osdg/{name}/rows.jsonl"):
        r = json.loads(l)
        if r["perm"] != 0: continue
        p = softmax(np.array(r["logits"]) / T[r["qtype"]])
        out.setdefault(r["split"], {})[r["id"]] = (int(np.argmax(p)), float(p.max()), r["gold_idx"])
    return out

def run(small, big, lat_small, lat_big, t1, t2, split):
    ids = list(small[split])
    acc = auto = 0; lat = 0.0
    for i in ids:
        ps, cs, g = small[split][i]; pb, cb, _ = big[split][i]
        lat += lat_small
        if cs >= t1: pred, ok = ps, True
        else:
            lat += lat_big
            pred, ok = pb, cb >= t2
        if ok:
            auto += 1; acc += int(pred == g)
    n = len(ids)
    return {"auto_rate": round(auto / n, 3), "auto_acc": round(acc / auto, 3) if auto else None, "overall_acc_if_review_wrong": round(acc / n, 3), "mean_ms": round(lat / n, 1)}

big = load("v3-4b-osdg")
res = {}
for sname, sn in (("0.6b", "v3-0.6b-osdg"), ("1.7b", "v3-1.7b-osdg")):
    small = load(sn)
    for t1 in (0.99, 0.95, 0.9, 0.8, 0.0):
        # t2 chosen on calibration: smallest threshold giving >=0.9 auto accuracy for the escalated rows
        best = None
        for t2 in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
            m = run(small, big, S[f"v3-{sname}"]["p50_ms"], S["v3-4b"]["p50_ms"], t1, t2, "calibration")
            if m["auto_acc"] and m["auto_acc"] >= 0.9 and (best is None or m["auto_rate"] > best[1]["auto_rate"]): best = (t2, m)
        t2 = best[0] if best else 0.9
        res[f"{sname} t1={t1} t2={t2}"] = {s: run(small, big, S[f"v3-{sname}"]["p50_ms"], S["v3-4b"]["p50_ms"], t1, t2, s) for s in ("test_locked", "challenge")}
# baselines: 4B alone
res["4b alone (no review)"] = {s: run(big, big, 0.0, S["v3-4b"]["p50_ms"], 2.0, 0.0, s) for s in ("test_locked", "challenge")}
lines = ["| policy | test auto / acc-of-auto / mean ms | challenge auto / acc-of-auto / mean ms |", "|---|---|---|"]
for k, v in res.items():
    f = lambda m: f"{m['auto_rate']} / {m['auto_acc']} / {m['mean_ms']}"
    lines.append(f"| {k} | {f(v['test_locked'])} | {f(v['challenge'])} |")
open("runs/osdg/cascade_v6.md", "w").write("\n".join(lines) + "\n"); json.dump(res, open("runs/osdg/cascade_v6.json", "w"), indent=1)
print("\n".join(lines))
