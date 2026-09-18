#!/usr/bin/env bash
# Start the System One gateway in front of trtllm-serve.
set -euo pipefail
. "$(dirname "$0")/env.sh"
cd "$OSJ_ROOT"
. .venv/bin/activate
python -m open_spark_jev.serve.gateway --backend openai \
  --upstream "http://localhost:$OSJ_SERVE_PORT/v1" --model "${1:-$OSJ_MODEL_DIR}" --port "$OSJ_GATEWAY_PORT"
