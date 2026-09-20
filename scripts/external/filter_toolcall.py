"""Filter data/synthetic/toolcall_risk.jsonl into a train file that provably does not contain the
60 ext-toolcall-risk test calls (exact or near-duplicate), plus a small in-domain holdout."""
import json, re, random

def norm(t): return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).split()
def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

test = [norm(json.loads(l)["state"]["content"]) for l in open("data/benchmarks/external/ext-toolcall-risk.jsonl")]
recs = [json.loads(l) for l in open("data/synthetic/toolcall_risk.jsonl")]
keep, dropped, seen = [], {"suspect": 0, "leak": 0, "dup": 0}, set()
for r in recs:
    if r.get("meta", {}).get("suspect"):
        dropped["suspect"] += 1; continue
    c = norm(r["state"]["content"])
    if any(jac(c, t) >= 0.7 for t in test):
        dropped["leak"] += 1; continue
    k = " ".join(c)
    if k in seen:
        dropped["dup"] += 1; continue
    seen.add(k); keep.append(r)
random.Random(0).shuffle(keep)
hold, train = keep[:300], keep[300:]
for f, rows in (("data/synthetic/toolcall_risk_train.jsonl", train), ("data/benchmarks/toolcall_risk_holdout.jsonl", hold)):
    open(f, "w").write("".join(json.dumps(r) + "\n" for r in rows))
from collections import Counter
print(len(recs), "->", len(train), "train +", len(hold), "holdout; dropped", dropped)
print(Counter(r["target"]["label"] for r in train))
