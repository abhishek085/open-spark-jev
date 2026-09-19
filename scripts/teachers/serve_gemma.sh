#!/usr/bin/env bash
# Launch the Gemma-4-26B-A4B-NVFP4 teacher (vLLM, OpenAI-compatible) on :8014.
# Already cached on this box (18GB NVFP4 checkpoint) -- no download needed.
set -euo pipefail
cd "$(dirname "$0")/../.."
. scripts/teachers/_common.sh
NAME=osj-teacher-gemma
PORT=8014
MODEL=nvidia/Gemma-4-26B-A4B-NVFP4
# See scripts/teachers/serve_nemotron.sh for the full explanation of what this flag means
# (total memory budget as a fraction of the WHOLE box, not an add-on). This checkpoint is only
# 18GB, so 0.40 (48GB budget) leaves generous headroom without needing near-exclusive use.
GPU_UTIL="${OSJ_TEACHER_GPU_UTIL:-0.40}"

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME already running"; exit 0
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "found a stopped $NAME from a previous run -- last 20 log lines before removing it:"
  docker logs --tail 20 "$NAME" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi
mem_preflight 55
docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len 8192 --max-num-seqs 16 \
  --trust-remote-code
wait_healthy "$PORT" "$NAME"
echo "Gemma teacher up: http://localhost:$PORT/v1  (stop: docker stop $NAME)"
