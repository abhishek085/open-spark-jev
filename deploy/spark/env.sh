# Shared settings for the DGX Spark deployment scripts. Source this file.
export TRTLLM_IMAGE=${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.1}   # NVIDIA's Spark playbook also lists 1.3.0rc13
export OSJ_ROOT=${OSJ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export OSJ_MODEL_DIR=${OSJ_MODEL_DIR:-$OSJ_ROOT/models/Qwen3-1.7B}     # or a checkpoints/rlcd-* dir
export OSJ_SERVE_PORT=${OSJ_SERVE_PORT:-8355}
export OSJ_GATEWAY_PORT=${OSJ_GATEWAY_PORT:-8400}
export OSJ_MAX_BATCH=${OSJ_MAX_BATCH:-64}
