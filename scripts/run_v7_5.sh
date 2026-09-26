#!/usr/bin/env bash
# v7.5: v7.4's diversified prog_v1 data + KL-to-v6 replay rows (see
# open_spark_jev/experimental/replay_kl.py for why -- it's decider-4b's v2->v2.1 fix, borrowed).
# Waits for v7.4 to clear the GPU first.
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

while pgrep -f "scripts/run_v7_4.sh" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') v7.4 finished, starting v7.5" | tee -a $LOG

step make_replay_v6 python -m open_spark_jev.experimental.replay_kl --model checkpoints/v6-4b \
  --source data/synthetic/v5_train.jsonl --n 2500 --out data/synthetic/replay_v6.jsonl
python scripts/tools/benchmark_overlap.py check data/synthetic/replay_v6.jsonl \
  --fp data/benchmarks/jevbench_fingerprints.json --fail \
  > runs/v7/overlap_replay_v6.log 2>&1 \
  && echo "$(date '+%F %T') OK    overlap_replay_v6" | tee -a $LOG \
  || { echo "$(date '+%F %T') FAIL  overlap_replay_v6 (see runs/v7/overlap_replay_v6.log)" | tee -a $LOG; exit 1; }

step train_v7.5 python -m open_spark_jev.train.sft --config configs/train/sft_v7_5.yaml
step eval_v7.5_prog python -m open_spark_jev.eval.osdg --model checkpoints/v7.5-4b \
  --name v7.5-4b-progsplits --data-prefix data/synthetic/prog_v1/ --perms 3
step eval_v7.5_v5 python -m open_spark_jev.eval.osdg --model checkpoints/v7.5-4b \
  --name v7.5-4b-v5splits --data-prefix data/benchmarks/v5_ --perms 3
step ext_v7.5 python -m open_spark_jev.eval.external --model checkpoints/v7.5-4b \
  --name v7.5-4b-jev-external --sources $JEV
step jevbench_v7.5 scripts/run_jevbench.sh v7.5-4b v7.5-4b
step soft_fidelity_v7.5 python scripts/analysis/soft_fidelity.py \
  --rows runs/osdg/v7.5-4b-progsplits/rows.jsonl --source data/synthetic/prog_v1 \
  --summary runs/osdg/v7.5-4b-progsplits/summary.json --out runs/v7/soft_fidelity_v7.5-4b-progsplits.json

echo "$(date '+%F %T') V7.5 DONE" | tee -a $LOG
