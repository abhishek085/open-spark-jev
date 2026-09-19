#!/usr/bin/env bash
# Launch the gpt-oss-120b teacher (vLLM, OpenAI-compatible, native MXFP4) on :8013.
set -euo pipefail
cd "$(dirname "$0")/../.."
. scripts/teachers/_common.sh
NAME=osj-teacher-gptoss
PORT=8013
MODEL=openai/gpt-oss-120b
# See scripts/teachers/serve_nemotron.sh for the full explanation: --gpu-memory-utilization
# is vLLM's TOTAL memory budget (weights + KV cache + activations) as a fraction of the box's
# entire system memory, not an add-on. This checkpoint is 61GB on a 121GB box (~0.50 of
# total); 0.75 (91GB budget, ~30GB over the checkpoint) leaves comfortable KV cache headroom
# for a near-exclusive-use benchmark call.
GPU_UTIL="${OSJ_TEACHER_GPU_UTIL:-0.75}"

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME already running"; exit 0
fi
# Remove a leftover crashed/stopped container with the same name -- without --rm (dropped
# deliberately, see below) a crash leaves the container and its logs around for `docker logs
# $NAME` to inspect; the trade-off is we have to clear it ourselves before relaunching, or
# `docker run` fails with "name already in use". Print its last lines first so a crash isn't
# silently discarded on the next retry.
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "found a stopped $NAME from a previous run -- last 20 log lines before removing it:"
  docker logs --tail 20 "$NAME" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi
HF_CACHE="$HOME/.cache/huggingface-osj"   # own cache: ~/.cache/huggingface/hub is root-owned, see download_gptoss.sh
if [ ! -d "$HF_CACHE/hub/models--openai--gpt-oss-120b" ]; then
  echo "gpt-oss-120b not downloaded yet; run scripts/download_gptoss.sh first" >&2
  exit 1
fi
mem_preflight 92   # ~0.75 x 121GB budget, near-exclusive box use required (see GPU_UTIL comment above)
docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HF_CACHE":/root/.cache/huggingface \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len 8192 --max-num-seqs 16 \
  --async-scheduling
wait_healthy "$PORT" "$NAME"
echo "gpt-oss-120b teacher up: http://localhost:$PORT/v1  (stop: docker stop $NAME)"
