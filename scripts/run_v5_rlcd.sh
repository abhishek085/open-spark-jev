#!/usr/bin/env bash
# RLCD-direct on top of v5-1.7b (hard-filtered pool); evaluate on the v5 splits and the Jev-style external sets, same as the SFT run.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
mkdir -p runs/v5
JEV="ext-toolcall-risk ext-jev-directory ext-injection-ctx ext-injection-noctx ext-kev-decision-v1"
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/v5/v5_rlcd.log
  if "$@" > "runs/v5/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/v5/v5_rlcd.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/v5/$name.log)" | tee -a runs/v5/v5_rlcd.log; fi; }
step train_v5_rlcd python -m open_spark_jev.train.rlcd policy --config configs/train/rlcd_direct_v5.yaml
step eval_v5_rlcd python -m open_spark_jev.eval.osdg --model checkpoints/v5-1.7b-rlcd --name v5-1.7b-rlcd-v5splits --data-prefix data/benchmarks/v5_ --perms 3
step ext_v5_rlcd python -m open_spark_jev.eval.external --model checkpoints/v5-1.7b-rlcd --name v5-1.7b-rlcd-external --sources $JEV
echo "$(date '+%F %T') RLCD DONE" | tee -a runs/v5/v5_rlcd.log
