#!/usr/bin/env bash
# (Re)start the vLLM server behind the System One gateway (container name must match thermal_guard's osj-* glob).
# Reconstructed from the running container's docker inspect on 2026-10-10; the image is pinned by id so a moved :nightly tag cannot change the build.
#   MAX_LEN=65536 deploy/spark/vllm_jev.sh      # default 16384 = the original deployment
set -euo pipefail
. "$(dirname "$0")/env.sh"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.10}"
ENGINE="${ENGINE:-$OSJ_ROOT/engines/spark-s1-4b-v8-MLP_ONLY_CFG}"
docker rm -f osj-jev-vllm >/dev/null 2>&1 || true
docker run -d --name osj-jev-vllm --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 -v "$ENGINE:/model:ro" \
  sha256:bb1349271d44a317fc7d95a046b184a26e8827431f71f6f040ccdd7b0a0d4a7d \
  --model /model --served-model-name spark-s1-4b-v8-nvfp4 --host 127.0.0.1 --port 8355 \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN" --max-num-seqs 16 --trust-remote-code \
  ${EXTRA_ARGS:-}
