# Local serving (single DGX Spark / GB10)

The pipeline requires only an OpenAI-compatible `/v1/chat/completions` per role. It never depends on vLLM internals.
Treat the 128 GB unified memory as shared by OS, containers, weights, KV cache, batches and training: **run one model
role at a time** (generate → verify → judge → evaluate → later train). Use the scheduler block of the config
(`fail_if_memory_headroom_gb_below`) so a phase refuses to start when memory is tight.

## Illustrative launch (not used by core code; exact flags vary by model, quantisation, vLLM release and GB10 support)

```bash
vllm serve <local-model-id-or-path> --host 127.0.0.1 --port 8000 --dtype auto --gpu-memory-utilization 0.80
```

`scripts/serve.sh` wraps the Docker pattern proven on this box (`vllm/vllm-openai:nightly-aarch64`, shared HF cache):
`scripts/serve.sh start generator|verifier|judge`, `stop`, `status`. With `start_cmd`/`stop_cmd` set in the model config,
the scheduler calls these itself per phase.

## Models used for the first real runs (`configs/models.local.yaml`)

| role | model | notes |
|---|---|---|
| generator | `nvidia/Gemma-4-26B-A4B-NVFP4` | MoE (~4B active), NVFP4; fast surface rendering |
| verifier | `nvidia/Qwen3.6-35B-A3B-NVFP4` | different family from the generator (correlated errors are the main risk); temperature 0 |
| judge | optional; `nvidia/Qwen3.6-27B-NVFP4` via `scripts/serve.sh start judge` | only for unresolved cases |

Pointing generator and verifier at the same endpoint works but logs a warning and marks records lower-confidence.
Point any role at Gemma/Qwen or other servers by editing `base_url` and `model`; `api_key_env` names an environment variable
(secret values are never written to artifacts). Guard temperature during sustained GPU work (see the repo's thermal guard).

## One role at a time, several instances of it

Run ONE model role at a time (generator, then verifier, ...), but as N parallel instances of that model sized to memory, and round-robin requests over them
(`base_urls` in the model config). `scripts/serve.sh` starts instances one after another (`OSJ_INSTANCES`, `OSJ_GPU_UTIL` per instance, `OSJ_PORT` base port) so engine memory
profiling never races. Measured on the GB10 with the MoE NVFP4 models: 3 generator instances gave ~1.3x the throughput of one (decode is memory-bandwidth-bound, so extra
instances mostly cost memory), and two instances (~26 GB each) use ~72 GB while running at ~65 C GPU / ~50 W. Prefer 2 instances for tuning loops.

Tuning loop: `scripts/dev_cycle.sh GEN_RUN VER_RUN N` (generate-only with the generator up → swap → `reverify` with the verifier up, including the supportability audit).
Comparator/verifier changes only need `os-datagen reverify --run OLD --out NEW` on saved generations (no generator needed).
