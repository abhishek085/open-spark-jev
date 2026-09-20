"""Diagnose rejections of a run: top reasons per pack and expected-vs-extracted for mismatched fields.
Usage: python examples/diagnose_run.py RUN_DIR [PACK_SUBSTR] [N_EXAMPLES]"""
import collections
import json
import sys

from os_datagen.schemas.scenario import ScenarioWorld
from os_datagen.taskpacks.registry import get_pack

run, sub, n = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else ""), int(sys.argv[3]) if len(sys.argv) > 3 else 2
worlds = {json.loads(line)["world"]["scenario_id"]: json.loads(line)["world"] for line in open(f"{run}/scenario_worlds.jsonl")}
cands = {json.loads(line)["candidate_id"]: json.loads(line) for line in open(f"{run}/candidates_raw.jsonl")}
reasons: dict = collections.defaultdict(collections.Counter)
shown: collections.Counter = collections.Counter()
for line in open(f"{run}/validation_results.jsonl"):
    r = json.loads(line)
    if sub not in r["task_pack"] or r["stage"] != "rejected":
        continue
    for x in r["reasons"]:
        reasons[r["task_pack"]][":".join(x.split(":")[:2])] += 1
    v = r["verification"]
    if v.get("mismatched") or v.get("unverifiable"):
        if shown[r["task_pack"]] >= n:
            continue
        shown[r["task_pack"]] += 1
        w = ScenarioWorld(**worlds[r["candidate_id"].split("-v")[0].replace("osj-", "")])
        exp = get_pack(r["task_pack"]).expected_extraction(w)
        print(f"\n## {r['candidate_id']} [{w.scenario_family}] {r['reasons']}")
        for m in (v.get("mismatched") or []) + (v.get("unverifiable") or []):
            k = m.split(":")[0]
            print(f"   {k}: expected={json.dumps(exp.get(k))[:200]}  got={json.dumps(v['extracted'].get(k))[:200]}")
        s = cands[r["candidate_id"]]["surface"]
        print("   TEXT:", json.dumps(s, ensure_ascii=False)[:500])
print()
for p, c in reasons.items():
    print(p, dict(c.most_common(6)))
