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

## Known blocker: openai/gpt-oss-120b is not usable on this box (as of 2026-09-19)

`openai/gpt-oss-120b` requires OpenAI's "harmony" chat/response format, implemented by the
`openai_harmony` Python package (a Rust extension). Serving it via vLLM's
`/v1/chat/completions` fails at first request with:
```
openai_harmony.HarmonyError: error downloading or loading vocab file: failed to download or load vocab file
```
This is a confirmed, currently open upstream bug specific to ARM64/DGX Spark - see
[vllm-project/vllm#22525](https://github.com/vllm-project/vllm/issues/22525),
[openai/harmony#101](https://github.com/openai/harmony/issues/101), and
[NVIDIA/dgx-spark-playbooks#17](https://github.com/NVIDIA/dgx-spark-playbooks/issues/17) (the
last one is this exact box/error combination, still open with no confirmed fix). The
container has working internet access and the vocab file downloads fine manually; the
failure is in `openai_harmony`'s own loader, not connectivity.

Two workarounds were tried and both failed for real reasons, not lack of effort:
1. **Pre-download the vocab file and set `TIKTOKEN_RS_CACHE_DIR`** - a fix reported to at
   least partially help elsewhere. Did not resolve it here; the loader still failed the same
   way with the file present and the env var set.
2. **Bypass chat entirely via `/v1/completions`** with a hand-built prompt. This avoids the
   crash (the endpoint responds), but produces useless output: `/v1/completions` tokenizes
   the prompt as plain text, so even correctly-typed harmony special tokens
   (`<|start|>`, `<|channel|>`, `<|message|>`, ...) are not recognized as real control
   tokens - the model sees garbled text instead of structured conversation turns and produces
   degenerate/incoherent completions. Properly reproducing harmony's tokenization would
   require its own token-ID-level encoder, i.e. reimplementing the broken library - judged
   out of proportion for a third candidate teacher when two others (`qwen27b`, `nemotron120b`)
   are already working and give a real comparison.

**Current status**: `openai/gpt-oss-120b` is downloaded (`scripts/download_gptoss.sh`,
~61GB in `~/.cache/huggingface-osj`) and its launch script
(`scripts/teachers/serve_gptoss.sh`) is correct and starts the model successfully; only
*serving it in a way that produces usable output* is blocked. `configs/teachers.yaml` marks
it `status: blocked` rather than removing it, so this is visible rather than silently absent.
Revisit if `openai_harmony`/vLLM ship a fix for the ARM64 vocab-loading bug, or if this
project moves to a serving stack with its own harmony support that does not depend on that
package (e.g. a version of `llama.cpp`, already present elsewhere on this box, with native
harmony template support).

## Thermal management (hard rule for this box)

The Spark's small form factor throttles/protectively shuts down around 92-95C package/SoC
or GPU temperature. Sustained near-100% GPU utilization for hours -- exactly what SFT/RLCD/GRPO
training back to back looks like -- can drive it there; a suspected thermal shutdown (not
confirmed, no crash log was available, but the timing and load pattern matched) interrupted
this project's own training pipeline once. **Installed once as a cron-managed service, not something to launch or watch per job**:

```bash
bash scripts/ops/install_thermal_cron.sh   # one-time; installs the crontab entries below
```

This installs two crontab entries: `@reboot` starts the guard immediately after any reboot
(including a thermal one - directly closes the gap that caused this project's own incident),
and `*/2 * * * *` restarts it if it's ever not running for any other reason. Nobody needs to
launch it manually or watch it in a session again; cron owns its lifecycle from here on -
checking on it in a Monitor loop is wasted effort, since the daemon and its watchdog already
self-heal without supervision.

The daemon itself (`scripts/ops/thermal_guard.sh`, invoked by the watchdog, not run directly)
polls GPU temp (`nvidia-smi`) and the max of the ACPI thermal zones (SoC/package) every 15s,
and SIGSTOPs every process matching `python -m open_spark_jev.train` once either sensor hits
91C (configurable via `THERMAL_PAUSE_C` in the watchdog's `nohup bash "$GUARD" ...` call),
resuming with SIGCONT only once both are back under 85C (`THERMAL_RESUME_C` - hysteresis,
avoids rapid pause/resume flapping). SIGSTOP freezes a process without killing it or releasing
its GPU memory/context, so a paused training run resumes exactly where it left off once the
box cools. To protect a different kind of job (e.g. a teacher-serving container), edit the
pattern argument in `scripts/ops/thermal_guard_watchdog.sh`.

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
