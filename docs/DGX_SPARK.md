# DGX Spark deployment

Target: one DGX Spark (GB10, Blackwell SM121, 121 GiB unified memory, CUDA 13 driver).

## Stack
| layer | choice | why |
|---|---|---|
| training | repo venv: PyTorch cu130 aarch64 wheels, Transformers, TRL, PEFT | no container needed for a 1.7B model; full fine-tune fits |
| inference | `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` (`deploy/spark/env.sh`) | PyTorch backend, Qwen3 supported, FP8/NVFP4 paths. NVIDIA's Spark playbook currently lists `1.3.0rc13`; override `TRTLLM_IMAGE` to track it |
| API | `trtllm-serve` on :8355 → gateway on :8400 | OpenAI-compatible upstream, typed System One surface on top |

## Setup
```bash
scripts/setup_env.sh                       # .venv with CUDA torch + deps
scripts/download_weights.sh Qwen/Qwen3-1.7B   # -> models/Qwen3-1.7B (repo-local HF cache in .hf/)
deploy/spark/pull_trtllm.sh                # container + GPU check
.venv/bin/python scripts/smoke_test.py     # HF backend sanity + latency line
```

## Serve
```bash
deploy/spark/serve.sh checkpoints/rlcd-qwen3-1.7b     # trtllm-serve, PyTorch backend
deploy/spark/gateway.sh checkpoints/rlcd-qwen3-1.7b   # loads calibration.json, fronts :8355
deploy/spark/smoke_curl.sh
```
`configs/serve/trtllm_extra_options.yaml` is tuned for menu scoring: prefix reuse on, CUDA
graphs with padding, overlap scheduler off (decode is one token), 35% of memory (the model is
small; leave room for a teacher or a second workload).

## Quantization
```bash
deploy/spark/quantize.sh configs/quant/fp8.yaml    models/Qwen3-1.7B   # default
deploy/spark/quantize.sh configs/quant/nvfp4.yaml  models/Qwen3-1.7B   # fastest, check calibration
```
Then **re-fit temperatures on the quantized engine**:
```bash
python -m open_spark_jev.eval.benchmark --backend openai --base-url http://localhost:8355/v1 \
  --data data/benchmarks/sim_test.jsonl --out runs/eval_fp8.json
```
and compare ECE / soft-Brier against the bf16 HF run. Expect FP8 to be within noise; NVFP4
at 1.7B typically needs a slightly higher temperature. Calibrate ModelOpt on decision
prompts (`calib_source`), not on generic web text.

## Latency model (what to expect)
Per request = 1 prefill of the state (once, cached) + N suffix prefills of ~80 tokens + N
single-token reads. At 1.7B bf16 on GB10 a 512-token state with 4 questions is a few tens of
milliseconds in-process; through trtllm-serve with prefix reuse the per-question cost is
dominated by HTTP + scheduler overhead, which batching questions per state amortises.
`python -m open_spark_jev.eval.latency` produces the grid; fill `docs/BENCHMARKS.md` from it.

## Serving-path facts verified on 1.2.1
* `/v1/completions` rejects `logprobs` ("logprobs is not supported"); `/v1/chat/completions`
  accepts `logprobs: true, top_logprobs: 20` and `chat_template_kwargs: {enable_thinking: false}`.
* Assistant-message prefill is not continued (the template closes the message), which is why
  the prompt convention makes the answer letter the *first* assistant token.
* `usage.prompt_tokens_details.cached_tokens` shows prefix reuse working when questions on
  the same state are sent back-to-back.
* The Python LLM API exposes `SamplingParams(logprobs=..., return_generation_logits=True)`;
  `serve/trtllm_backend.py` uses that for full-vocab gathering.

## Gotchas specific to this box
* `~/.triton` may be root-owned; export `TRITON_CACHE_DIR=$PWD/.triton_cache` (scripts do).
* GB10 is SM121: FlashAttention-3 is unsupported; TRT-LLM's PyTorch backend picks its own
  kernels, and for HF use `sdpa` (default) rather than `flash_attention_2`.
* `nvidia-smi` reports `[N/A]` for memory on Spark; use `free -g` (unified memory) and
  `torch.cuda.max_memory_allocated()`.
* Only one big serving container fits comfortably alongside training; stop other model
  servers before Phase 2 (which loads policy + reference + reward model).
* The machine's shared `~/.cache/huggingface` may be root-owned; this repo uses `.hf/` as
  `HF_HOME` and `models/` for weights.

## Multi-Spark
Not needed for a 1.7B model. If you stack two Sparks, keep the decision model on one and
use the second for the teacher (synthetic data / RLCD pairs) rather than tensor-parallel.
