"""Per-pack real-endpoint smoke test with a RANDOM seed (servers already up, configs/models.dev.yaml).
  python examples/smoke_pack.py PACK [N_WORLDS=12] [SEED]
Prints acceptance, top rejection reasons and random accepted/rejected samples (model input -> expected output).
Writes artifacts/smoke/<pack>_<seed>/ (a normal run directory) plus samples.md."""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from os_datagen.config import load_config
from os_datagen.generation.clients import make_clients
from os_datagen.generation.pipeline import Pipeline
from os_datagen.schemas.common import SPLITS
from os_datagen.training.render import render_prompt, truth_sets
from os_datagen.utils.jsonl import read_jsonl

pack = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
seed = int(sys.argv[3]) if len(sys.argv) > 3 else int(time.time()) % 100000
cfg = load_config(None, "configs/models.dev.yaml")
gen, ver, judge = make_clients(cfg, False)
out = Path("artifacts/smoke") / f"{pack}_{seed}"
per = {"train": max(1, n - 6), "calibration": 2, "locked_test": 2, "challenge": 2}
p = Pipeline(cfg, out, gen, ver, judge)
s = p.run([pack], per, seed)
b = s["by_pack"][pack]
print(f"\n=== {pack} seed={seed}: accepted {b['accepted']}/{b['candidates']} ({b['acceptance_rate']:.0%})  labels={b['label_distribution']}")
reasons: Counter[str] = Counter()
for r in read_jsonl(out / "rejected.jsonl"):
    for x in r["reasons"]:
        reasons[":".join(x.split(":")[:2])] += 1
print("top rejection reasons:", dict(reasons.most_common(6)))
rng = random.Random(seed)
acc = [r for sp in SPLITS for r in read_jsonl(out / {"train": "accepted_train.jsonl", "calibration": "accepted_calibration.jsonl", "locked_test": "accepted_test_locked.jsonl", "challenge": "accepted_challenge.jsonl"}[sp])]
lines = [f"# {pack} (seed {seed}): accepted {b['accepted']}/{b['candidates']}\n"]
for r in rng.sample(acc, min(2, len(acc))):
    pref, accs = truth_sets(r)
    lines.append(f"\n## ACCEPTED {r['record_id']} [{r['split']}/{r['meta']['scenario_family']}]\n\n### MODEL INPUT\n```\n{render_prompt(r)}```\n### EXPECTED OUTPUT: `{pref}` (acceptable {accs}; {r['truth']['label_quality']}; reason {r['truth']['reason_code']})\n")
rej = list(read_jsonl(out / "rejected.jsonl"))
cand = {c["candidate_id"]: c for c in read_jsonl(out / "candidates_raw.jsonl")}
for r in rng.sample(rej, min(1, len(rej))):
    lines.append(f"\n## REJECTED {r['candidate_id']} reasons={r['reasons']}\n```\n{json.dumps(cand[r['candidate_id']]['surface'], ensure_ascii=False, indent=1)[:1500]}\n```\n")
(out / "samples.md").write_text("".join(lines))
print("".join(lines)[:6000])
