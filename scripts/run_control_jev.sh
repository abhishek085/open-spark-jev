#!/usr/bin/env bash
# control-jev: train + evaluate the small (Qwen3.5-0.8B) purpose-built decision model for
# JevControl's own task shape. Waits for the running v7.x pipeline to clear the GPU first.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HOME="$PWD/.hf" HF_HUB_OFFLINE=1
mkdir -p runs/control
LOG=runs/control/control.log

step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a $LOG
  if "$@" > "runs/control/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a $LOG
  else echo "$(date '+%F %T') FAIL  $name (see runs/control/$name.log)" | tee -a $LOG; fi; }

while pgrep -f "scripts/run_v7_4.sh" > /dev/null || pgrep -f "scripts/run_v7_5.sh" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') v7.x pipeline finished, starting control-jev" | tee -a $LOG

step train_control_jev python -m open_spark_jev.train.sft --config configs/train/sft_control_jev.yaml
step eval_control_jev python -m open_spark_jev.eval.osdg --model checkpoints/control-jev \
  --name control-jev-splits --data-prefix data/synthetic/control_v1/ --perms 3

echo "$(date '+%F %T') CONTROL-JEV DONE" | tee -a $LOG
