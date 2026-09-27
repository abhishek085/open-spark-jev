#!/usr/bin/env bash
# control-jev-es: train the extra-small (ModernBERT-base) sibling. Waits for control-jev to finish
# training so the two don't contend for the GPU.
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

while pgrep -f "scripts/run_control_jev.sh" > /dev/null; do sleep 30; done
echo "$(date '+%F %T') control-jev finished, starting control-jev-es" | tee -a $LOG

step train_control_jev_es python -m open_spark_jev.train.sft_control_es --config configs/train/sft_control_jev_es.yaml
step eval_control_jev_es python -m open_spark_jev.eval.osdg --model checkpoints/control-jev-es \
  --name control-jev-es-splits --data-prefix data/synthetic/control_v1/ --perms 3

echo "$(date '+%F %T') CONTROL-JEV-ES DONE" | tee -a $LOG
