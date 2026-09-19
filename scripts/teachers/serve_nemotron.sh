#!/usr/bin/env bash
# Launch the Nemotron-3-Super-120B-A12B-NVFP4 teacher (vLLM, OpenAI-compatible) on :8012.
# Ephemeral: this repo's own container, safe to start/stop freely. Stop with stop_teacher.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
. scripts/teachers/_common.sh
NAME=osj-teacher-nemotron
PORT=8012
MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
# --gpu-memory-utilization sets vLLM's TOTAL memory budget (weights + KV cache + activations)
# as a fraction of the box's ENTIRE system memory, not an add-on reservation on top of the
# checkpoint. This model's checkpoint alone is 82GB on a 121GB box (~0.68 of total) -- the
# budget must exceed that just to hold the weights, before any KV cache at all. A first real
# run set this too LOW (0.55, then an even lower "fix" of 0.12) on the mistaken assumption
# that a smaller value saves memory; both crashed with "Available KV cache memory: -58.23
# GiB" / "No available memory for the cache blocks" -- the actual error vLLM raises when the
# budget doesn't cover the weights. 0.85 (103GB budget, ~21GB over the checkpoint) is sized
# for near-exclusive use of the box, which is appropriate here since nothing else should be
# running during a teacher benchmark call -- see docs/DGX_SPARK.md for the full incident.
GPU_UTIL="${OSJ_TEACHER_GPU_UTIL:-0.85}"

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
mem_preflight 105  # ~0.85 x 121GB budget, near-exclusive box use required (see GPU_UTIL comment above)
docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  vllm/vllm-openai:nightly-aarch64 \
  --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len 8192 --max-num-seqs 16 \
  --trust-remote-code
wait_healthy "$PORT" "$NAME"
echo "Nemotron teacher up: http://localhost:$PORT/v1  (stop: docker stop $NAME)"
