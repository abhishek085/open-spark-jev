#!/usr/bin/env bash
# v5 (Jev-style only) on the 4B, same lower-LR / 1-epoch recipe that won on the 1.7B: SFT, evaluate on the v5 splits
# and the Jev-style external sets, compare with v3-4b (scored on the same v5 splits for a fair comparison).
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
mkdir -p runs/v5
JEV="ext-toolcall-risk ext-jev-directory ext-injection-ctx ext-injection-noctx ext-kev-decision-v1"
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/v5/v5_4b.log
  if "$@" > "runs/v5/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/v5/v5_4b.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/v5/$name.log)" | tee -a runs/v5/v5_4b.log; fi; }
step train_v5_4b python -m open_spark_jev.train.sft --config configs/train/sft_v5.yaml --set model=Qwen/Qwen3-4B output_dir=checkpoints/v5-4b
step eval_v5_4b python -m open_spark_jev.eval.osdg --model checkpoints/v5-4b --name v5-4b-v5splits --data-prefix data/benchmarks/v5_ --perms 3
step ext_v5_4b python -m open_spark_jev.eval.external --model checkpoints/v5-4b --name v5-4b-jev-external --sources $JEV
step eval_v3_4b_v5splits python -m open_spark_jev.eval.osdg --model checkpoints/v3-4b --name v3-4b-v5splits --data-prefix data/benchmarks/v5_ --perms 3
step ext_v3_4b python -m open_spark_jev.eval.external --model checkpoints/v3-4b --name v3-4b-jev-external --sources $JEV
echo "$(date '+%F %T') V5 4B DONE" | tee -a runs/v5/v5_4b.log
