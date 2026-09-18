#!/usr/bin/env bash
# Alternative to serve.sh + gateway.sh: run the gateway *inside* the TRT-LLM container on the
# Python LLM API (full-vocab first-token logits, no HTTP hop per question).
set -euo pipefail
. "$(dirname "$0")/env.sh"
MODEL=${1:-$OSJ_MODEL_DIR}
docker run --name osj-gateway-trtllm --rm -it --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$OSJ_ROOT":/work -w /work -e PYTHONPATH=/work "$TRTLLM_IMAGE" \
  bash -c "pip install -q pydantic fastapi 'uvicorn[standard]' httpx pyyaml >/dev/null 2>&1; \
           python3 -m open_spark_jev.serve.gateway --backend trtllm --model /work/${MODEL#$OSJ_ROOT/} --port $OSJ_GATEWAY_PORT"
