#!/usr/bin/env bash
# v7.4: does diversifying prog_v1's fixed boilerplate (temporal_numeric's standing note,
# long_policy's clause text) fix v7.2's JevBench-hard regression (0.595 -> 0.559, ECE 0.265 -> 0.344)?
# Same recipe and lambda_brier as v7.2 -- only the data changed -- so this is a clean A/B on the
# diversification fix alone. Waits for part2 and the v7.3 JevBench retry to clear the GPU first.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HOME="$PWD/.hf" HF_HUB_OFFLINE=1
mkdir -p runs/v7
JEV="ext-toolcall-risk ext-jev-directory ext-injection-ctx ext-injection-noctx ext-kev-decision-v1"
LOG=runs/v7/v7x.log

step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a $LOG
  if "$@" > "runs/v7/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a $LOG
  else echo "$(date '+%F %T') FAIL  $name (see runs/v7/$name.log)" | tee -a $LOG; fi; }

while pgrep -f "scripts/run_v7x_part2.sh" > /dev/null || pgrep -f "scripts/retry_jevbench_v73.sh" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') part2 + v7.3 retry finished, starting v7.4" | tee -a $LOG

step train_v7.4 python -m open_spark_jev.train.sft --config configs/train/sft_v7_4.yaml
step eval_v7.4_prog python -m open_spark_jev.eval.osdg --model checkpoints/v7.4-4b \
  --name v7.4-4b-progsplits --data-prefix data/synthetic/prog_v1/ --perms 3
step eval_v7.4_v5 python -m open_spark_jev.eval.osdg --model checkpoints/v7.4-4b \
  --name v7.4-4b-v5splits --data-prefix data/benchmarks/v5_ --perms 3
step ext_v7.4 python -m open_spark_jev.eval.external --model checkpoints/v7.4-4b \
  --name v7.4-4b-jev-external --sources $JEV
step jevbench_v7.4 scripts/run_jevbench.sh v7.4-4b v7.4-4b
step soft_fidelity_v7.4 python scripts/analysis/soft_fidelity.py \
  --rows runs/osdg/v7.4-4b-progsplits/rows.jsonl --source data/synthetic/prog_v1 \
  --summary runs/osdg/v7.4-4b-progsplits/summary.json --out runs/v7/soft_fidelity_v7.4-4b-progsplits.json

echo "$(date '+%F %T') V7.4 DONE" | tee -a $LOG
