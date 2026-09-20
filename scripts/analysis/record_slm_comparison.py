"""Build open_spark_jev/serve/lab_data/slm_comparison.json for the Decision Lab: real measurements only.
 - aggregate: 60-call set, per size: untrained model generating JSON / thinking (runs/slm_baseline_*.json), the same untrained model read out by option letters
   (runs/osdg/final60.json, v0), and spark-s1 (final60).
 - fixtures: the untrained same-size model's raw JSON output for each Lab fixture (recorded here, batch 1, greedy)."""
import json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from open_spark_jev import gate as G
from open_spark_jev.data.corpus import Record
from open_spark_jev.eval.speed_vs_generation import gen_prompt, parse_choice

f60 = json.load(open("runs/osdg/final60.json"))
fx = json.load(open("open_spark_jev/serve/lab_data/tool_calls.json"))
SIZES = {"4b": ("Qwen/Qwen3-4B", "runs/slm_baseline_4b.json", "v0-qwen3-4b", "v3-4b", "spark-s1-4b-v3"),
         "1.7b": ("models/Qwen3-1.7B", "runs/slm_baseline_1.7b.json", "v0-qwen3-1.7b", "v3-1.7b", "spark-s1-1.7b-v3")}
out = {"_meta": {"recorded": time.strftime("%Y-%m-%d"), "hardware": "NVIDIA DGX Spark (GB10), bf16, HF Transformers, batch 1, idle GPU",
                 "baseline": "The generative SLM is the same-size untrained Qwen3 model, prompted zero-shot to return JSON for the same four-way risk-posture question (greedy, at most 40 new tokens).",
                 "caveat": "This isolates two things: reading options out as probabilities instead of generating text, and training on decision data. A fine-tuned generative model or a much larger model would score differently. 60 calls, about +/-5 points; the set is a diagnostic."},
       "aggregate": {}, "fixtures": {}}
for z, (base, bj, v0, v3, rid) in SIZES.items():
    d = json.load(open(bj)); arms = {a["arm"].split(" (")[0]: a for a in d["arms"]}
    g, t = arms["generate"], arms["generate_think"]
    spark_p50 = f60[v3]["p50_ms"]
    row = lambda name, acc, bad, p50, what, dps, tok: {"name": name, "accuracy": acc, "malformed": bad, "p50_ms": p50, "what": what,  # noqa: E731
                                                       "decisions_per_s": dps, "tokens": tok, "time_vs_spark": round(p50 / spark_p50, 1)}
    out["aggregate"][z] = {"rows": [
        row(f"Generative SLM, JSON output (untrained Qwen3-{z.upper() if z=='4b' else z.upper()})", g["accuracy"], f"{g['json_parse_failures']} of {g['n']}", g["latency_ms"]["p50"], "Writes a JSON answer token by token, then it is parsed.", g["decisions_per_second"], g["mean_output_tokens"]),
        row("Generative SLM, thinking on", t["accuracy"], f"{t['json_parse_failures']} of {t['n']}", t["latency_ms"]["p50"], "Reasons first, then answers.", t["decisions_per_second"], t["mean_output_tokens"]),
        row("Same untrained model, option-letter readout", f60[v0]["acc"], "0 of 60", f60[v0]["p50_ms"], "One forward pass over the option letters; no training.", f60[v0]["dec_per_s"], 0),
        row(f"{rid} (trained, option-letter readout)", f60[v3]["acc"], "0 of 60", f60[v3]["p50_ms"], "One forward pass, trained on code-labelled decisions.", f60[v3]["dec_per_s"], 0)]}
for z, (base, bj, v0, v3, rid) in SIZES.items():
    tok = AutoTokenizer.from_pretrained(base); lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("cuda").eval()
    for f in fx:
        rec = Record.from_json({"id": f["id"], "domain": "lab", "source": "lab", "state": {"content": f"Agent tool call: {f['command']}"},
                                "question": {"type": "choice", "prompt": G.POSTURE_PROMPT, "options": G.POSTURES}, "target": {"label": "readonly"}})
        text = tok.apply_chat_template(gen_prompt(rec), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to("cuda")
        lat = []
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                o = lm.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1000)
        new = o[0][ids["input_ids"].shape[1]:]; raw = tok.decode(new, skip_special_tokens=True).strip()
        choice, _ = parse_choice(raw, G.POSTURES)
        out["fixtures"].setdefault(f["id"], {})[z] = {"raw": raw, "choice": choice, "action": G.POSTURE_TO_ACTION.get(choice) if choice else None, "ms": sorted(lat)[1], "tokens": int(len(new))}
        print(z, f["id"], repr(raw[:80]), choice, flush=True)
    del lm; torch.cuda.empty_cache()
json.dump(out, open("open_spark_jev/serve/lab_data/slm_comparison.json", "w"), indent=1)
