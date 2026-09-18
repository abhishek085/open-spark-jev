# Roadmap

| # | milestone | status |
|---|---|---|
| M0 | repo, schema, prompting, HF menu scorer with KV reuse, calibration metrics | done |
| M1 | known-posterior simulators (6 domains), public adapters, corpus format | done |
| M2 | Phase 1 SFT loop with soft targets + temperature fitting | code done, first run pending |
| M3 | Phase 2 RLCD: pairs, Noul reward model, exact menu policy gradient, GRPO baseline | code done, needs a local teacher up |
| M4 | Gateway (`/v1/decide`, Jev-format `/v1/evaluate`), OpenAI-completions backend | done |
| M5 | TRT-LLM on Spark: container pulled (1.2.1), serve/quantize scripts | scripts done, first serve run pending |
| M6 | Baseline numbers: base Qwen3-1.7B zero-shot on sim_test | pending |
| M7 | SFT run + eval, BENCHMARKS.md | pending |
| M8 | RLCD run (needs teacher on :8010) + ablation vs GRPO | pending |
| M9 | FP8 / NVFP4 engines, temperature re-fit, latency grid | pending |
| M10 | Comparison table vs LLM+regex baseline and Jev-format public examples | pending |

## Open questions for the owner
1. Priority domains for the first real evaluation: routing + moderation + security (current default), or something closer to your production workloads?
2. Teacher for synthetic data / RLCD pairs: `nvidia/Qwen3.6-27B-NVFP4` via vLLM on :8010 (as in sibling projects) or a Nemotron? Both fit beside the 1.7B student.
3. Single Spark (assumed) or two? A second node only helps by hosting the teacher.
4. Latency target: per-question p95 under 50 ms via the gateway is the working goal; confirm.
