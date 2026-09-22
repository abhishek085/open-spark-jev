#!/usr/bin/env bash
# v6: v5's recipe and data (Jev-style only, lower-LR/1-epoch) on Qwen3.5-4B instead of Qwen3-4B. Evaluate on the v5
# splits and the Jev-style external sets, compare against v5-4b (Qwen3) and v3-4b, both scored on the same v5 splits.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
mkdir -p runs/v6
JEV="ext-toolcall-risk ext-jev-directory ext-injection-ctx ext-injection-noctx ext-kev-decision-v1"
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/v6/v6_4b.log
  if "$@" > "runs/v6/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/v6/v6_4b.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/v6/$name.log)" | tee -a runs/v6/v6_4b.log; fi; }
step train_v6_4b python -m open_spark_jev.train.sft --config configs/train/sft_v6.yaml
step eval_v6_4b python -m open_spark_jev.eval.osdg --model checkpoints/v6-4b --name v6-4b-v5splits --data-prefix data/benchmarks/v5_ --perms 3
step ext_v6_4b python -m open_spark_jev.eval.external --model checkpoints/v6-4b --name v6-4b-jev-external --sources $JEV
echo "$(date '+%F %T') V6 4B DONE" | tee -a runs/v6/v6_4b.log
