#!/usr/bin/env bash
# Benchmark a checkpoint (HF backend) or a served endpoint on the simulator test set.
#   scripts/eval.sh hf checkpoints/rlcd-qwen3-1.7b
#   scripts/eval.sh openai http://localhost:8355/v1
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
mkdir -p runs
if [ "${1:-hf}" = "hf" ]; then
  python -m open_spark_jev.eval.benchmark --backend hf --model "${2:-models/Qwen3-1.7B}" --data data/benchmarks/sim_test.jsonl --out "runs/eval_$(basename "${2:-base}").json"
else
  python -m open_spark_jev.eval.benchmark --backend openai --base-url "${2:-http://localhost:8355/v1}" --data data/benchmarks/sim_test.jsonl --out runs/eval_served.json
fi
