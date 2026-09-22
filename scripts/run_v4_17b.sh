#!/usr/bin/env bash
# v4 strategy on the 1.7B: assemble data, SFT (lr 5e-5, 1 epoch), evaluate on the v4 splits + the external suites, compare with v3-1.7b.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
mkdir -p runs/v4
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/v4/v4.log
  if "$@" > "runs/v4/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/v4/v4.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/v4/$name.log)" | tee -a runs/v4/v4.log; fi; }
step train_v4_1.7b python -m open_spark_jev.train.sft --config configs/train/sft_v4.yaml --set model=models/Qwen3-1.7B output_dir=checkpoints/v4-1.7b
step eval_v4_1.7b python -m open_spark_jev.eval.osdg --model checkpoints/v4-1.7b --name v4-1.7b-v4splits --data-prefix data/benchmarks/v4_ --perms 3
step eval_v3_1.7b python -m open_spark_jev.eval.osdg --model checkpoints/v3-1.7b --name v3-1.7b-v4splits --data-prefix data/benchmarks/v4_ --perms 3
step ext_v4_1.7b python -m open_spark_jev.eval.external --model checkpoints/v4-1.7b --name v4-1.7b-external
echo "$(date '+%F %T') V4 1.7B DONE" | tee -a runs/v4/v4.log
