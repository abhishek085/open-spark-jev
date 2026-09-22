#!/usr/bin/env bash
# RLCD-direct on top of v4-1.7b; evaluate like the SFT run.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
mkdir -p runs/v4
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/v4/v4_rlcd.log
  if "$@" > "runs/v4/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/v4/v4_rlcd.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/v4/$name.log)" | tee -a runs/v4/v4_rlcd.log; fi; }
step train_v4_rlcd python -m open_spark_jev.train.rlcd policy --config configs/train/rlcd_direct_v4.yaml
step eval_v4_rlcd python -m open_spark_jev.eval.osdg --model checkpoints/v4-1.7b-rlcd --name v4-1.7b-rlcd-v4splits --data-prefix data/benchmarks/v4_ --perms 3
step ext_v4_rlcd python -m open_spark_jev.eval.external --model checkpoints/v4-1.7b-rlcd --name v4-1.7b-rlcd-external
echo "$(date '+%F %T') RLCD DONE" | tee -a runs/v4/v4_rlcd.log
