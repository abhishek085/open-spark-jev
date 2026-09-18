#!/usr/bin/env bash
# Pull the TensorRT-LLM release container (arm64, CUDA 13) and verify it sees the GB10.
set -euo pipefail
. "$(dirname "$0")/env.sh"
docker pull "$TRTLLM_IMAGE"
docker run --rm --gpus all "$TRTLLM_IMAGE" nvidia-smi
docker run --rm --gpus all "$TRTLLM_IMAGE" python3 -c "import tensorrt_llm; print('tensorrt_llm', tensorrt_llm.__version__)"
