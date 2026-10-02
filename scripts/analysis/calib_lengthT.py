"""Option 3: post-hoc, length-aware temperature. T(len) = exp(a + b*log(max(len,64)/512)), fit per checkpoint on the *calibration* splits of
non-JevBench datasets (objective: NLL + lam * confident-wrong penalty), then applied to JevBench hard (held out, never fitted on).
Usage: python scripts/analysis/calib_lengthT.py v6-4b v7.6-4b v7.11-4b ..."""

import glob, json, math, os, sys, itertools
from pathlib import Path

RUNS = {
    "progv2full": "data/synthetic/prog_v2_full",
    "hermes": "data/synthetic/prog_hermes_v1",
    "pubagent": "data/public_eval",
    "long": "data/synthetic/prog_long_v1",
}
JB = os.environ.get("JEVBENCH_DIR", "../external/jevbench") + "/datasets/public/hard.jsonl"
SHIPPED = {}


def shipped_T(ck):
    try:
        return json.load(open(f"checkpoints/{ck}/calibration.json"))["temperature"]["choice"]
    except Exception:
        return 1.0


def lens(d):
    out = {}
    for f in glob.glob(f"{d}/*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            out[r["id"]] = len(r["state"]["content"]) // 3  # ~tokens
    return out


def sm(z, T):
    z = [x / T for x in z]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def load_cal(ck, splits=tuple(__import__("os").environ.get("CAL_SPLITS", "calibration").split(","))):
    rows = []
    for k, d in RUNS.items():
        p = Path(f"runs/osdg/{ck}-{k}/rows.jsonl")
        if not p.exists():
            continue
        L = lens(d)
        for l in open(p):
            r = json.loads(l)
            if r["qtype"] != "choice" or r["split"] not in splits or r.get("perm", 0) != 0:
                continue
            g = [j for j, lb in enumerate(r["labels"]) if lb in r["acceptable"]]
            rows.append((r["logits"], g, L.get(r["id"], 300), k))
    return rows


def T_of(a, b, n):
    return math.exp(a + b * math.log(max(n, 64) / 512))


def obj(rows, a, b, lam):
    nll = pen = 0
    for z, g, n, _ in rows:
        p = sm(z, T_of(a, b, n))
        nll += -math.log(max(sum(p[j] for j in g), 1e-9))
        i = max(range(len(p)), key=p.__getitem__)
        if i not in g:
            pen += max(0, p[i] - 0.7) ** 2
    return nll / len(rows) + lam * pen / len(rows)


def fit(rows, lam=3.0):
    best = None
    for a in [x / 10 for x in range(-5, 31, 2)]:
        for b in [x / 10 for x in range(-5, 11, 1)]:
            v = obj(rows, a, b, lam)
            if best is None or v < best[0]:
                best = (v, a, b)
    return best[1], best[2]


def jb_rows():
    out = []
    for l in open(JB):
        r = json.loads(l)
        out.append((r["id"], r["expected"], r["provenance"].get("approx_state_tokens", 500)))
    return out


def eval_jb(ck, a, b):
    Ts = shipped_T(ck)
    gold = {i: (e, n) for i, e, n in jb_rows()}
    res = []
    for l in open(f"runs/jevbench/{ck}/hard.results.jsonl"):
        r = json.loads(l)
        if not (r.get("valid") and r.get("probs")):
            continue
        e, n = gold[r["task_id"]]
        ks = list(r["probs"])
        if e not in ks:
            continue
        lg = [Ts * math.log(max(r["probs"][k], 1e-12)) for k in ks]
        T = T_of(a, b, n) if a is not None else Ts
        p = sm(lg, T)
        i = max(range(len(p)), key=p.__getitem__)
        c = p[i]
        res.append(
            (
                ks[i] == e,
                c,
                sum((p[j] - (ks[j] == e)) ** 2 for j in range(len(p))),
                -math.log(max(p[ks.index(e)], 1e-12)),
            )
        )
    n = len(res)
    bins = [[0, 0, 0.0] for _ in range(10)]
    for ok, c, _, _ in res:
        bb = bins[min(9, int(c * 10))]
        bb[0] += 1
        bb[1] += ok
        bb[2] += c
    ece = sum(b_[0] / n * abs(b_[1] / b_[0] - b_[2] / b_[0]) for b_ in bins if b_[0])
    return dict(
        acc=sum(r[0] for r in res) / n,
        brier=sum(r[2] for r in res) / n,
        nll=sum(r[3] for r in res) / n,
        ece=ece,
        cw8=sum((not r[0]) and r[1] >= 0.8 for r in res),
        cw9=sum((not r[0]) and r[1] >= 0.9 for r in res),
    )


if __name__ == "__main__":
    for ck in sys.argv[1:]:
        rows = load_cal(ck)
        if not rows:
            print(ck, "no calibration rows")
            continue
        a, b = fit(rows)
        f = lambda d: " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())
        print(
            f"{ck:12s} fit on {len(rows)} cal rows ({sorted({r[3] for r in rows})}) a={a:.1f} b={b:.1f}  (T@512={T_of(a, b, 512):.2f}, T@3000={T_of(a, b, 3000):.2f}; shipped T={shipped_T(ck):.2f})"
        )
        print(f"   JevBench hard shipped : {f(eval_jb(ck, None, None))}")
        print(f"   JevBench hard lengthT : {f(eval_jb(ck, a, b))}")
