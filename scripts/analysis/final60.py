"""Final review: every finished model on the 60-row tool-call set (ext-toolcall-risk), accuracy + calibration + speed vs Jev's recorded run.
Temperature = the one fitted on the os-datagen CALIBRATION split for that model (runs/osdg/<name>/summary.json); the saved
checkpoint temperature is NOT used (v3 checkpoints carry a bad one). Speed: batch 1, idle GPU, latency per decision. Per-row saved."""
import json, os, statistics, time
import numpy as np, torch
from open_spark_jev.calibration import ece, brier, nll, softmax
from open_spark_jev.data.corpus import read_jsonl
from open_spark_jev.eval.external import _jev_probs
from open_spark_jev.eval.osdg import permute
from open_spark_jev.model import MenuScorer

MODELS = [  # name, checkpoint/base, osdg run for temperature, head
    ("v3-4b", "checkpoints/v3-4b", "v3-4b-osdg", None), ("v3-1.7b", "checkpoints/v3-1.7b", "v3-1.7b-osdg", None),
    ("v3-0.6b", "checkpoints/v3-0.6b", "v3-0.6b-osdg", None),
    ("v2-4b-frozen-head", "Qwen/Qwen3-4B", "v2-4b-osdg", "checkpoints/v2-4b/head.pt"),
    ("v2-1.7b-frozen-head", "models/Qwen3-1.7B", "v2-1.7b-osdg", "checkpoints/v2-1.7b/head.pt"),
    ("v0-qwen3-4b", "Qwen/Qwen3-4B", "v0-qwen3-4b", None), ("v0-gemma4-e4b", "google/gemma-4-E4B-it", "v0-gemma4-e4b", None),
    ("v0-qwen3-1.7b", "models/Qwen3-1.7B", "v0-qwen3-1.7b", None), ("v0-qwen3-0.6b", "Qwen/Qwen3-0.6B", "v0-qwen3-0.6b", None),
]
recs = read_jsonl("data/benchmarks/external/ext-toolcall-risk.jsonl")
gold = [r.label_index() for r in recs]
labels0 = recs[0].question_obj().labels
jev = [_jev_probs(r, r.question_obj().labels) for r in recs]
jev_acc = float(np.mean([int(np.argmax(j) == g) for j, g in zip(jev, gold)]))
out = {"jev": {"acc": round(jev_acc, 4), "p50_ms": 421.6, "p95_ms": 542.0, "note": "recorded hosted run incl. network, themsquared/jev-benchmark"}}
rows = []
for name, path, osdg, head in MODELS:
    sc = MenuScorer(path, use_state_cache=False)
    hs = None
    if head:
        from open_spark_jev.experimental.frozen_head import Head, HeadScorer
        ck = torch.load(head, map_location="cpu")
        h = Head(sc.lm.config.hidden_size); h.load_state_dict(ck["head"]); hs = HeadScorer(sc, h)
    dec = (hs or sc).decide
    T = json.load(open(f"runs/osdg/{osdg}/summary.json"))["temperature_fit_on_calibration"]["choice"]
    def logits(rec):
        a = dec(rec.state_obj(), [rec.question_obj()], temperature=1.0, return_logits=True)[0]
        return np.array(a.raw_logits if a.raw_logits is not None else np.log(np.maximum(a.probs, 1e-12)))
    for r in recs[:3]: dec(r.state_obj(), [r.question_obj()])
    lat, Z = [], []
    for r in recs:
        torch.cuda.synchronize(); t0 = time.perf_counter(); z = logits(r); torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000); Z.append(z)
    flips = []
    for r in recs:
        picks = {r.question_obj().labels[int(np.argmax(logits(permute(r, p))))] if p == 0 else permute(r, p).question_obj().labels[int(np.argmax(logits(permute(r, p))))] for p in (0, 1, 2)}
        flips.append(len(picks) > 1)
    P = [softmax(z / T) for z in Z]; P1 = [softmax(z) for z in Z]
    acc = float(np.mean([int(np.argmax(p) == g) for p, g in zip(P, gold)]))
    ag = float(np.mean([int(np.argmax(p) == np.argmax(j)) for p, j in zip(P, jev)]))
    s = sorted(lat)
    out[name] = {"acc": round(acc, 4), "ece_cal": round(ece(P, gold), 4), "ece_raw": round(ece(P1, gold), 4), "brier_cal": round(brier(P, gold), 4),
                 "option_flips": round(float(np.mean(flips)), 3), "agree_w_jev": round(ag, 3), "T": round(T, 3),
                 "p50_ms": round(statistics.median(lat), 1), "p95_ms": round(s[int(0.95 * len(s))], 1), "dec_per_s": round(1000 / statistics.mean(lat), 1)}
    for r, p, l in zip(recs, P, lat):
        rows.append({"model": name, "id": r.id, "slice": r.domain.split("/")[-1], "gold": labels0[0] and r.question_obj().labels[r.label_index()], "pred": r.question_obj().labels[int(np.argmax(p))], "probs": [round(float(x), 4) for x in p], "ms": round(l, 1)})
    print(name, out[name], flush=True)
    del sc, hs; torch.cuda.empty_cache()
json.dump(out, open("runs/osdg/final60.json", "w"), indent=1)
open("runs/osdg/final60.rows.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
