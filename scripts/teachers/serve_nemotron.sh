#!/usr/bin/env bash
# Launch the Nemotron-3-Super-120B-A12B-NVFP4 teacher (vLLM, OpenAI-compatible) on :8012.
# Ephemeral: this repo's own container, safe to start/stop freely. Stop with stop_teacher.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
. scripts/teachers/_common.sh
NAME=osj-teacher-nemotron
PORT=8012
MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
GPU_UTIL="${OSJ_TEACHER_GPU_UTIL:-0.55}"

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME already running"; exit 0
fi
mem_preflight 65
docker run -d --name "$NAME" --rm --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len 8192 --max-num-seqs 16 \
  --trust-remote-code
wait_healthy "$PORT" "$NAME"
echo "Nemotron teacher up: http://localhost:$PORT/v1  (stop: docker stop $NAME)"
