"""Score a checkpoint on an RL pool and keep only the rows it gets wrong or is unsure about.

RLCD's policy gradient needs rows where the current policy can still improve; on v4, the held-out pool
was mostly rows the SFT model already fit (reward ~0.99 at step 0), so RLCD moved almost nothing. This
scores the pool with the checkpoint and writes back only rows below a confidence/correctness bar.

  python scripts/filter_rl_pool_hard.py --model checkpoints/v5-1.7b --pool data/synthetic/v5_rl_pool.jsonl --out data/synthetic/v5_rl_pool_hard.jsonl
"""
import argparse
import json

import numpy as np

from open_spark_jev.data.corpus import read_jsonl
from open_spark_jev.model import MenuScorer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=float, default=0.7, help="keep rows where the model's probability on the gold option is below this")
a = ap.parse_args()

scorer = MenuScorer(a.model, use_state_cache=False)
recs = read_jsonl(a.pool)
kept = []
for r in recs:
    q = r.question_obj()
    ans = scorer.decide(r.state_obj(), [q], temperature=1.0)[0]
    gold = r.target["label"]
    p_gold = dict(zip(q.labels, ans.probs)).get(gold, 0.0)
    wrong = q.labels[int(np.argmax(ans.probs))] != gold
    if wrong or p_gold < a.margin:
        kept.append(r)

with open(a.out, "w") as f:
    for r in kept:
        f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
print(f"pool {len(recs)} -> hard {len(kept)} ({100 * len(kept) / max(1, len(recs)):.0f}%), written to {a.out}")
