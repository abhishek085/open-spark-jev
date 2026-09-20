"""Record real spark-s1 allow/ask/deny outputs for the Decision Lab fixtures (no command is ever executed)."""
import json, platform, statistics, sys, time
import torch
from open_spark_jev import gate as G
from open_spark_jev.model import MenuScorer

fixtures = json.load(open("open_spark_jev/serve/lab_data/tool_calls.json"))
out = {"_meta": {"recorded": time.strftime("%Y-%m-%d"), "hardware": "NVIDIA DGX Spark (GB10), bf16, HF Transformers, batch 1", "note": "Real inference on the fixtures' proposed calls; risk_posture mode uses the benchmarked four-way posture question mapped to allow/ask/deny; direct mode asks allow/ask/deny outright and was NOT benchmarked."}}
for rid, path in (("spark-s1-4b-v3", "checkpoints/v3-4b"), ("spark-s1-1.7b-v3", "checkpoints/v3-1.7b")):
    sc = MenuScorer(path, use_state_cache=False)
    out[rid] = {}
    for f in fixtures:
        st = G.GateState(tool=f["tool"], command=f["command"], context=f["context"])
        out[rid][f["id"]] = {}
        for mode in ("risk_posture", "direct"):
            G.gate(st, sc, 0.995, rid, mode)  # warm
            lat = []
            for _ in range(5):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                r = G.gate(st, sc, 0.995, rid, mode)
                torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1000)
            out[rid][f["id"]][mode] = {"choice": r.choice, "probabilities": {k: round(v, 6) for k, v in r.probabilities.items()}, "confidence": round(r.confidence, 6),
                                       "posture_probabilities": {k: round(v, 6) for k, v in (r.posture_probabilities or {}).items()} or None, "latency_ms": round(statistics.median(lat), 1)}
            print(rid, f["id"], mode, r.choice, round(r.confidence, 3), r.policy_action, flush=True)
    del sc; torch.cuda.empty_cache()
json.dump(out, open("open_spark_jev/serve/lab_data/recorded_outputs.json", "w"), indent=1)
