#!/usr/bin/env bash
# Post-training quantization with NVIDIA ModelOpt inside the TRT-LLM container.
#   deploy/spark/quantize.sh configs/quant/fp8.yaml   [input_model_dir] [output_dir]
# Produces an HF-format checkpoint with quantized weights that trtllm-serve loads directly
# (PyTorch backend), so the menu-scoring prompts and label logprobs are unchanged.
set -euo pipefail
. "$(dirname "$0")/env.sh"
CFG=${1:-$OSJ_ROOT/configs/quant/fp8.yaml}
IN=${2:-$OSJ_MODEL_DIR}
QFORMAT=$(grep '^qformat:' "$CFG" | awk '{print $2}')
KVQ=$(grep '^kv_cache_qformat:' "$CFG" | awk '{print $2}')
CALIB=$(grep '^calib_size:' "$CFG" | awk '{print $2}')
OUT=${3:-$OSJ_ROOT/engines/$(basename "$IN")-$QFORMAT}
mkdir -p "$OUT"
# The ModelOpt examples ship inside the container under /app/tensorrt_llm/examples/quantization.
docker run --rm --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$OSJ_ROOT":/work -w /work "$TRTLLM_IMAGE" bash -c "
    set -e
    pip install -q nvidia-modelopt[hf] 2>/dev/null || true
    python3 /app/tensorrt_llm/examples/quantization/quantize.py \
      --model_dir /work/${IN#$OSJ_ROOT/} \
      --qformat $QFORMAT --kv_cache_dtype $KVQ --calib_size $CALIB \
      --output_dir /work/${OUT#$OSJ_ROOT/}
  "
cp "$IN"/calibration.json "$OUT"/ 2>/dev/null || true
echo "quantized checkpoint: $OUT  (re-fit temperatures: see docs/DGX_SPARK.md)"
