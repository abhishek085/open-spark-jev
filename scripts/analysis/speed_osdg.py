"""Decision latency of saved checkpoints on the os-datagen locked-test rows (batch 1, idle GPU, one question per state).
Per-row latencies are saved. The 60-row tool-call set is deliberately not used here (reserved for the final review)."""
import json, statistics, subprocess, sys, time
import torch
from open_spark_jev.data.corpus import read_jsonl
from open_spark_jev.model import MenuScorer

out = {}
recs = read_jsonl("data/benchmarks/osdg_test_locked.jsonl")
rows = []
for name, path in [(a.split("=")[0], a.split("=")[1]) for a in sys.argv[1:]]:
    sc = MenuScorer(path, use_state_cache=False)
    for r in recs[:5]:
        sc.decide(r.state_obj(), [r.question_obj()])
    lat = []
    for r in recs:
        st, q = r.state_obj(), r.question_obj()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        sc.decide(st, [q]); torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms); rows.append({"model": name, "id": r.id, "ms": round(ms, 2)})
    s = sorted(lat)
    out[name] = {"n": len(lat), "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(s[int(0.95 * len(s))], 1), "decisions_per_s": round(1000 / statistics.mean(lat), 1)}
    print(name, out[name], flush=True)
    del sc; torch.cuda.empty_cache()
json.dump(out, open("runs/osdg/speed_v3.json", "w"), indent=1)
open("runs/osdg/speed_v3.rows.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
