# Roadmap

| # | milestone | status |
|---|---|---|
| M0 | repo, schema, prompting, HF menu scorer with KV reuse, calibration metrics | done |
| M1 | known-posterior simulators (6 domains), public adapters, corpus format | done |
| M2 | Phase 1 SFT loop with soft targets + temperature fitting | code done, first run pending |
| M3 | Phase 2, two mechanisms (contrastive RLCD + direct calibration objective), GRPO baseline, temperature-only control | code done, see docs/RESEARCH.md R1 |
| M4 | Gateway (`/v1/decide`, Jev-format `/v1/evaluate`), OpenAI-completions backend | done |
| M5 | TRT-LLM on Spark: container 1.2.1, `trtllm-serve` chat-logprobs path and Python-API backend both verified against HF | done (bf16; quantized engines pending) |
| M6 | Baseline numbers: base Qwen3-1.7B zero-shot on sim_test (HF + served parity) | done, docs/BENCHMARKS.md |
| M7 | SFT run + eval, BENCHMARKS.md | pending |
| M8 | Run all four Phase-2 mechanisms (rlcd_direct, rlcd_contrastive, GRPO, temperature-only) and compare | in progress, see docs/BENCHMARKS.md |
| M8b | Regenerate sim_train/sim_test on the fixed (leakage-free) simulators and rerun SFT + all four Phase-2 mechanisms for trustworthy moderation/incident/game numbers | pending, blocks a fully clean M8 |
| M9 | FP8 / NVFP4 engines, temperature re-fit, latency grid | pending |
| M10 | Comparison table vs LLM+regex baseline and Jev-format public examples | pending |
| M11 | Multi-teacher registry + ground-truth teacher benchmark (`configs/teachers.yaml`, `eval/teacher_benchmark.py`) | done; qwen27b scored (soft Brier 0.142), nemotron120b/gptoss120b queued on memory (gpt-oss-120b now downloaded, ~61GB, ready to launch once ~65-70GB is free) |
| M12 | Reusability cookbook for retraining on other domains (`docs/COOKBOOK.md`) | done |
| M14 | Architecture-experiment ledger + A1 (single-pass parallel multi-question readout) prototype (`docs/NOVELTY.md`) | A1 implemented, measurement queued behind the in-flight SFT/RLCD/GRPO run |
| M15 | A3: slot-query menu head, our candidate novel architecture (`docs/NOVELTY.md`) | proposed, not yet built |
| M13 | Retest full serving path on TensorRT-LLM 1.3.0rc13 | done, parity confirmed, docs/BENCHMARKS.md |

## Open questions for the owner
1. Priority domains for the first real evaluation: routing + moderation + security (current default), or something closer to your production workloads?
2. Teacher for synthetic data / mechanism-1 pairs: `nvidia/Qwen3.6-27B-NVFP4` via vLLM on :8010 (as in sibling projects), `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`, or `openai/gpt-oss-120b` -- all three registered in configs/teachers.yaml.
3. Single Spark (assumed) or two? A second node only helps by hosting the teacher.
4. Latency target: per-question p95 under 50 ms via the gateway is the working goal; confirm.
