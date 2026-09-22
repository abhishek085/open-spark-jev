"""Build NVFP4 calibration prompts from the v5 train data, rendered exactly as at inference (same tokenizer-facing text
the model sees via open_spark_jev.prompting), stratified across task packs so the new harness/use-case packs are
represented (not just the old, larger ones).

  python scripts/quant/make_v5_calib_prompts.py --data data/synthetic/v5_train.jsonl --out runs/quant/calib_prompts_v5.jsonl --per-pack 20
"""
import argparse
import collections
import json
import random

from open_spark_jev.data.corpus import read_jsonl
from open_spark_jev.prompting import render_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-pack", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

recs = read_jsonl(a.data)
by_pack = collections.defaultdict(list)
for r in recs:
    by_pack[r.domain].append(r)

rng = random.Random(a.seed)
out = []
for pack, rs in by_pack.items():
    rng.shuffle(rs)
    out += rs[: a.per_pack]
rng.shuffle(out)

with open(a.out, "w") as f:
    for r in out:
        text = render_prompt(r.state_obj(), r.question_obj())
        f.write(json.dumps({"text": text, "pack": r.domain}, ensure_ascii=False) + "\n")
print(f"packs {len(by_pack)}, prompts {len(out)}, written to {a.out}")
