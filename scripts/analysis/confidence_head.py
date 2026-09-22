"""Does a fitted correctness calibrator beat the temperature-scaled softmax maximum as an auto-decide gate? Uses per-row logits from
runs/osdg/<name>/rows.jsonl (perm 0 only for scoring; calibration rows from every permutation for fitting)."""
import json, sys
import numpy as np
from open_spark_jev import confidence as C
from open_spark_jev.calibration import ece

name = sys.argv[1]
rows = [json.loads(l) for l in open(f"runs/osdg/{name}/rows.jsonl")]
T = json.load(open(f"runs/osdg/{name}/summary.json"))["temperature_fit_on_calibration"]
cal = [r for r in rows if r["split"] == "calibration"]
model = C.fit(cal, T)
def raw(r): return C.features(r["logits"], r["qtype"], T[r["qtype"]])[0]
def new(r): return C.predict(model, r["logits"], r["qtype"])
def auroc(score, y):
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1; return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))
for split in ("calibration", "test_locked", "challenge"):
    rs = [r for r in rows if r["split"] == split and r["perm"] == 0]
    y = np.array([int(np.argmax(r["logits"]) == r["gold_idx"]) for r in rs])
    a, b = np.array([raw(r) for r in rs]), np.array([new(r) for r in rs])
    e = lambda s: ece([[x, 1 - x] for x in s], [0 if yy else 1 for yy in y])  # noqa: E731
    out = {"n": len(rs), "acc": round(float(y.mean()), 3), "auroc_raw": round(auroc(a, y), 3), "auroc_head": round(auroc(b, y), 3), "ece_raw": round(e(a), 3), "ece_head": round(e(b), 3)}
    for cov in (0.5, 0.7):
        k = int(len(rs) * cov)
        out[f"sel_acc@{int(cov*100)}%cov_raw"] = round(float(y[np.argsort(-a)[:k]].mean()), 3)
        out[f"sel_acc@{int(cov*100)}%cov_head"] = round(float(y[np.argsort(-b)[:k]].mean()), 3)
    print(split, json.dumps(out))
