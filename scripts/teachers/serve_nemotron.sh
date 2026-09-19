#!/usr/bin/env bash
# Launch the Nemotron-3-Super-120B-A12B-NVFP4 teacher (vLLM, OpenAI-compatible) on :8012.
# Ephemeral: this repo's own container, safe to start/stop freely. Stop with stop_teacher.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
. scripts/teachers/_common.sh
NAME=osj-teacher-nemotron
PORT=8012
MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
GPU_UTIL="${OSJ_TEACHER_GPU_UTIL:-0.25}"  # benchmark-only load (concurrency 3, max_model_len 8192): a much smaller KV pool than sustained-serving defaults leaves headroom for the checkpoint itself on this unified-memory box -- see docs/DGX_SPARK.md for the incident that motivated this

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
mem_preflight 65
docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len 8192 --max-num-seqs 16 \
  --trust-remote-code
wait_healthy "$PORT" "$NAME"
echo "Nemotron teacher up: http://localhost:$PORT/v1  (stop: docker stop $NAME)"
