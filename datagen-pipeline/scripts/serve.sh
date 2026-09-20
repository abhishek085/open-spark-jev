#!/usr/bin/env bash
# Illustrative helper (NOT used by the pipeline core): serve ONE model role at a time on the GB10 via vLLM in Docker,
# as N parallel instances of that model to batch requests (sized to memory).
#   scripts/serve.sh start generator|verifier|judge   scripts/serve.sh stop generator|...   scripts/serve.sh status
# Env: OSJ_INSTANCES (default 1), OSJ_GPU_UTIL per instance (default 0.35), OSJ_PORT base port, OSJ_MODEL, OSJ_MAX_LEN, OSJ_MAX_SEQS.
# Instances start one after another (waiting for health) so the engines' memory profiling never races.
set -euo pipefail
ROLE="${2:-generator}"
case "$ROLE" in
  generator) MODEL_DEF="nvidia/Gemma-4-26B-A4B-NVFP4"; PORT_DEF=8000 ;;
  verifier)  MODEL_DEF="nvidia/Qwen3.6-35B-A3B-NVFP4";  PORT_DEF=8001 ;;
  judge)     MODEL_DEF="nvidia/Qwen3.6-27B-NVFP4";      PORT_DEF=8002 ;;
  *) echo "role must be generator|verifier|judge"; exit 1 ;;
esac
MODEL="${OSJ_MODEL:-$MODEL_DEF}"; PORT0="${OSJ_PORT:-$PORT_DEF}"; N="${OSJ_INSTANCES:-1}"
UTIL="${OSJ_GPU_UTIL:-0.35}"; MAXLEN="${OSJ_MAX_LEN:-8192}"; IMAGE="${OSJ_IMAGE:-vllm/vllm-openai:nightly-aarch64}"
name() { echo "osj-datagen-$ROLE-$1"; }
case "${1:-}" in
  start)
    for i in $(seq 0 $((N-1))); do
      NAME="$(name $i)"; PORT=$((PORT0+i))
      docker ps --format '{{.Names}}' | grep -qx "$NAME" && { echo "$NAME already running"; continue; }
      docker rm "$NAME" >/dev/null 2>&1 || true
      docker run -d --name "$NAME" --gpus all --ipc host --shm-size 64m -p "${PORT}:8000" \
        -v /home/admin/.cache/huggingface:/root/.cache/huggingface -v /home/admin/.cache/vllm:/root/.cache/vllm \
        -v /home/admin/.cache/flashinfer:/root/.cache/flashinfer \
        "$IMAGE" "$MODEL" --host 0.0.0.0 --port 8000 --served-model-name "$MODEL" \
        --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" --max-num-seqs "${OSJ_MAX_SEQS:-16}" >/dev/null
      echo "started $NAME ($MODEL) on :$PORT"
      for t in $(seq 1 120); do curl -s -m 3 "localhost:$PORT/v1/models" >/dev/null 2>&1 && break
        docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "$NAME DIED"; docker logs --tail 8 "$NAME" 2>&1 | cut -c1-220; exit 1; }
        sleep 5; done
    done ;;
  stop) for c in $(docker ps -a --format '{{.Names}}' | grep "^osj-datagen-$ROLE-"); do docker stop "$c" >/dev/null 2>&1; docker rm "$c" >/dev/null 2>&1; echo "stopped $c"; done ;;
  status) docker ps -a --filter "name=osj-datagen" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' ;;
  *) echo "usage: $0 {start|stop|status} [role]"; exit 1 ;;
esac
