"""Random accepted samples (model input -> expected output) plus one random rejected candidate per pack, for human review.
  python examples/sample_review.py RUN_DIR [N_ACCEPTED_PER_PACK=2] [SEED]  -> RUN_DIR/samples_for_review.md"""
import json
import random
import sys
from pathlib import Path

from os_datagen.training.render import render_prompt, truth_sets
from os_datagen.utils.jsonl import read_jsonl

run = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else None)
by: dict[str, list] = {}
for f in ("accepted_train", "accepted_calibration", "accepted_test_locked", "accepted_challenge"):
    for r in read_jsonl(run / f"{f}.jsonl"):
        by.setdefault(r["task_pack"], []).append(r)
rej: dict[str, list] = {}
cand = {c["candidate_id"]: c for c in read_jsonl(run / "candidates_raw.jsonl")}
for r in read_jsonl(run / "rejected.jsonl"):
    rej.setdefault(r["task_pack"], []).append(r)
out = [f"# Review samples from {run.name}\n"]
for pk in sorted(by):
    out.append(f"\n\n---\n# {pk}\n")
    for r in rng.sample(by[pk], min(n, len(by[pk]))):
        pref, acc = truth_sets(r)
        out.append(f"\n## ACCEPTED {r['record_id']} [{r['split']} / {r['meta']['scenario_family']} / {r['quality']['difficulty']}]\n\n```\n{render_prompt(r)}```\n"
                   f"**EXPECTED:** `{pref}` (acceptable {acc}; tier {r['truth']['label_quality']}; reason {r['truth']['reason_code']}; supportability {r['quality']['supportability']})\n")
    if rej.get(pk):
        x = rng.choice(rej[pk])
        out.append(f"\n## REJECTED {x['candidate_id']} reasons={x['reasons']}\n```\n{json.dumps((cand[x['candidate_id']]['surface'] or {}), ensure_ascii=False)[:900]}\n```\n")
(run / "samples_for_review.md").write_text("".join(out))
print(f"wrote {run / 'samples_for_review.md'} ({len(''.join(out))} chars)")
