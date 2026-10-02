"""Temperature-fair calibration on JevBench tiers from stored probs. Rescales p -> p^(1/tau) (renormalised) and reports each checkpoint
(a) as shipped (tau=1) and (b) at its own best tau (oracle: optimistic, but equal for all). Accuracy is T-independent.
Usage: python scripts/analysis/jevbench_calib.py hard v6-4b v7.6-4b v7.11-4b"""

import json, math, os, sys

JB = os.environ.get("JEVBENCH_DIR", "../external/jevbench") + "/datasets/public"


def load(ck, tier):
    gold = {}
    for l in open(f"{JB}/{tier}.jsonl"):
        r = json.loads(l)
        gold[r["id"]] = r["expected"]
    out = []
    for l in open(f"runs/jevbench/{ck}/{tier}.results.jsonl"):
        r = json.loads(l)
        if r.get("valid") and r.get("probs") and r["task_id"] in gold:
            out.append((r["probs"], gold[r["task_id"]]))
    return out


def score(rows, tau):
    n = len(rows)
    bri = nll = ok_n = cw8 = cw9 = 0
    bins = [[0, 0, 0.0] for _ in range(10)]
    for probs, gold in rows:
        ks = list(probs)
        lp = [math.log(max(probs[k], 1e-12)) / tau for k in ks]
        m = max(lp)
        e = [math.exp(x - m) for x in lp]
        s = sum(e)
        p = [x / s for x in e]
        i = max(range(len(p)), key=p.__getitem__)
        c = p[i]
        ok = ks[i] == gold
        bri += sum((p[j] - (ks[j] == gold)) ** 2 for j in range(len(p)))
        nll += -math.log(max(p[ks.index(gold)], 1e-12)) if gold in ks else 0
        ok_n += ok
        b = bins[min(9, int(c * 10))]
        b[0] += 1
        b[1] += ok
        b[2] += c
        cw8 += (not ok) and c >= 0.8
        cw9 += (not ok) and c >= 0.9
    ece = sum(b[0] / n * abs(b[1] / b[0] - b[2] / b[0]) for b in bins if b[0])
    return dict(n=n, acc=ok_n / n, brier=bri / n, nll=nll / n, ece=ece, cw80=cw8, cw90=cw9)


if __name__ == "__main__":
    tier = sys.argv[1]
    for ck in sys.argv[2:]:
        rows = load(ck, tier)
        a = score(rows, 1.0)
        best = min(((t, score(rows, t)) for t in [x / 20 for x in range(5, 401)]), key=lambda z: z[1]["nll"])
        f = lambda d: (
            f"acc={d['acc']:.3f} brier={d['brier']:.3f} nll={d['nll']:.3f} ece={d['ece']:.3f} wrong@.8={d['cw80']} wrong@.9={d['cw90']}"
        )
        print(f"{ck:12s} n={a['n']} shipped: {f(a)}\n{'':12s} best-tau={best[0]:.2f}: {f(best[1])}")
