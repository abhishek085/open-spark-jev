#!/usr/bin/env bash
# Launch the gpt-oss-120b teacher (vLLM, OpenAI-compatible, native MXFP4) on :8013.
set -euo pipefail
cd "$(dirname "$0")/../.."
. scripts/teachers/_common.sh
NAME=osj-teacher-gptoss
PORT=8013
MODEL=openai/gpt-oss-120b
GPU_UTIL="${OSJ_TEACHER_GPU_UTIL:-0.55}"

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME already running"; exit 0
fi
HF_CACHE="$HOME/.cache/huggingface-osj"   # own cache: ~/.cache/huggingface/hub is root-owned, see download_gptoss.sh
if [ ! -d "$HF_CACHE/hub/models--openai--gpt-oss-120b" ]; then
  echo "gpt-oss-120b not downloaded yet; run scripts/download_gptoss.sh first" >&2
  exit 1
fi
mem_preflight 70
docker run -d --name "$NAME" --rm --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HF_CACHE":/root/.cache/huggingface \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len 8192 --max-num-seqs 16 \
  --async-scheduling
wait_healthy "$PORT" "$NAME"
echo "gpt-oss-120b teacher up: http://localhost:$PORT/v1  (stop: docker stop $NAME)"
