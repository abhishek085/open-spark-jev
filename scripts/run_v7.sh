#!/usr/bin/env bash
# spark-s1-v7: RLCD-direct on the best v3 model, then the osdg evaluation.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
echo "$(date '+%F %T') START train_v7_4b" | tee -a runs/osdg/v7.log
python -m open_spark_jev.train.rlcd policy --config configs/train/rlcd_direct_osdg.yaml > runs/osdg/train_v7_4b.log 2>&1 && echo "$(date '+%F %T') OK    train_v7_4b" | tee -a runs/osdg/v7.log || { echo "$(date '+%F %T') FAIL  train_v7_4b" | tee -a runs/osdg/v7.log; exit 1; }
python -m open_spark_jev.eval.osdg --model checkpoints/v7-4b --name v7-4b-osdg --perms 3 > runs/osdg/eval_v7_4b.log 2>&1 && echo "$(date '+%F %T') OK    eval_v7_4b" | tee -a runs/osdg/v7.log || echo "$(date '+%F %T') FAIL  eval_v7_4b" | tee -a runs/osdg/v7.log
echo "$(date '+%F %T') V7 DONE" | tee -a runs/osdg/v7.log
