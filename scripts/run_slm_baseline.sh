#!/usr/bin/env bash
# Same-backbone comparison on the 60-case set: spark-s1 readout vs the untrained backbone generating JSON (and with thinking).
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
python -m open_spark_jev.eval.speed_vs_generation --menu-model checkpoints/v3-1.7b --base-model models/Qwen3-1.7B --out runs/slm_baseline_1.7b.json > runs/slm_baseline_1.7b.log 2>&1 && echo "$(date '+%F %T') OK 1.7b" | tee -a runs/slm_baseline.log || echo "$(date '+%F %T') FAIL 1.7b" | tee -a runs/slm_baseline.log
python -m open_spark_jev.eval.speed_vs_generation --menu-model checkpoints/v3-4b --base-model Qwen/Qwen3-4B --out runs/slm_baseline_4b.json > runs/slm_baseline_4b.log 2>&1 && echo "$(date '+%F %T') OK 4b" | tee -a runs/slm_baseline.log || echo "$(date '+%F %T') FAIL 4b" | tee -a runs/slm_baseline.log
echo "$(date '+%F %T') SLM BASELINE DONE" | tee -a runs/slm_baseline.log
