#!/usr/bin/env bash
# v5 (Jev-style only) on the 1.7B: SFT lr 5e-5 / 1 epoch, evaluate on the v5 splits and the Jev-style external sets, compare with v3 and v4.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
mkdir -p runs/v5
JEV="ext-toolcall-risk ext-jev-directory ext-injection-ctx ext-injection-noctx ext-kev-decision-v1"
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/v5/v5.log
  if "$@" > "runs/v5/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/v5/v5.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/v5/$name.log)" | tee -a runs/v5/v5.log; fi; }
step train_v5_1.7b python -m open_spark_jev.train.sft --config configs/train/sft_v5.yaml --set model=models/Qwen3-1.7B output_dir=checkpoints/v5-1.7b
step eval_v5_1.7b python -m open_spark_jev.eval.osdg --model checkpoints/v5-1.7b --name v5-1.7b-v5splits --data-prefix data/benchmarks/v5_ --perms 3
step ext_v5_1.7b python -m open_spark_jev.eval.external --model checkpoints/v5-1.7b --name v5-1.7b-jev-external --sources $JEV
step eval_v4_1.7b python -m open_spark_jev.eval.osdg --model checkpoints/v4-1.7b --name v4-1.7b-v5splits --data-prefix data/benchmarks/v5_ --perms 3
step eval_v3_1.7b python -m open_spark_jev.eval.osdg --model checkpoints/v3-1.7b --name v3-1.7b-v5splits --data-prefix data/benchmarks/v5_ --perms 3
echo "$(date '+%F %T') V5 1.7B DONE" | tee -a runs/v5/v5.log
