"""Per-decision latency of a served model over the OpenAI-compatible endpoint (batch 1, sequential, 60-call set). Saves per-row latency."""
import json, statistics, sys, time
from open_spark_jev.data.corpus import read_jsonl
from open_spark_jev.serve.client import OpenAICompletionsBackend

url, name = sys.argv[1], sys.argv[2]
recs = read_jsonl("data/benchmarks/external/ext-toolcall-risk.jsonl")
be = OpenAICompletionsBackend(url)
for r in recs[:5]:
    be.decide(r.state_obj(), [r.question_obj()])
lat = []
for r in recs:
    t0 = time.perf_counter(); be.decide(r.state_obj(), [r.question_obj()]); lat.append((time.perf_counter() - t0) * 1000)
s = sorted(lat)
res = {"name": name, "n": len(lat), "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(s[int(0.95 * len(s))], 1), "decisions_per_s": round(1000 / statistics.mean(lat), 1)}
print(json.dumps(res))
open(f"runs/quant/latency_{name}.rows.jsonl", "w").write("".join(json.dumps({"i": i, "ms": round(x, 2)}) + "\n" for i, x in enumerate(lat)))
json.dump(res, open(f"runs/quant/latency_{name}.json", "w"))
