"""Calibration by state length from eval_longctx.py output: acc, mean confidence, ECE (10 bins), Brier, flips and probability shift versus length 0 on the SAME items."""
import json, sys, collections

rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for r in rows: by[r["len_target"]].append(r)
base = {r["id"]: r for r in by[0]}

def stats(rs):
    n = len(rs); acc = sum(r["pred"] == r["gold"] for r in rs) / n; conf = sum(r["conf"] for r in rs) / n
    ece = 0.0
    for b in range(10):
        bin_ = [r for r in rs if min(int(r["conf"] * 10), 9) == b]
        if bin_: ece += len(bin_) / n * abs(sum(r["pred"] == r["gold"] for r in bin_) / len(bin_) - sum(r["conf"] for r in bin_) / len(bin_))
    brier = sum(sum((p - (k == r["gold"])) ** 2 for k, p in r["probs"].items()) for r in rs) / n
    return n, acc, conf, ece, brier

print(f"{'len':>6} {'tok p50':>8} {'n':>4} {'acc':>6} {'conf':>6} {'ECE':>6} {'Brier':>6} | same-items@0: {'acc':>6} {'conf':>6} {'ECE':>6} | {'flips':>5} {'mean|dp|':>8}")
for L in sorted(by):
    rs = by[L]; n, acc, conf, ece, brier = stats(rs)
    ids = [r for r in rs if r["id"] in base]; b = [base[r["id"]] for r in ids]
    _, bacc, bconf, bece, _ = stats(b)
    flips = sum(x["pred"] != y["pred"] for x, y in zip(ids, b))
    dp = sum(abs(x["probs"].get(x["pred"], 0) - y["probs"].get(x["pred"], 0)) for x, y in zip(ids, b)) / len(ids)
    toks = sorted(r["input_tokens"] for r in rs)[n // 2]
    print(f"{L:>6} {toks:>8} {n:>4} {acc:6.3f} {conf:6.3f} {ece:6.3f} {brier:6.3f} |               {bacc:6.3f} {bconf:6.3f} {bece:6.3f} | {flips:>5} {dp:8.3f}")
