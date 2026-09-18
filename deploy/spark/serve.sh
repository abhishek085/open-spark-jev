#!/usr/bin/env bash
# Serve the backbone with trtllm-serve (PyTorch backend) as an OpenAI-compatible endpoint.
#   deploy/spark/serve.sh [model_dir]
set -euo pipefail
. "$(dirname "$0")/env.sh"
MODEL=${1:-$OSJ_MODEL_DIR}
REL=${MODEL#$OSJ_ROOT/}
docker run --name osj-trtllm --rm -it --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$OSJ_ROOT":/work -w /work "$TRTLLM_IMAGE" \
  trtllm-serve "/work/$REL" \
    --backend pytorch \
    --host 0.0.0.0 --port "$OSJ_SERVE_PORT" \
    --max_batch_size "$OSJ_MAX_BATCH" \
    --max_num_tokens 16384 \
    --trust_remote_code \
    --extra_llm_api_options /work/configs/serve/trtllm_extra_options.yaml
