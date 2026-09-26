#!/usr/bin/env bash
# v7.x part 2: the transfer test, plus the two readout/depth experiments.
#
# Everything in part 1 was measured in-distribution (our own generator wrote both the training and the
# test rows), and in-distribution gains are the easy half. The question part 2 answers is whether the
# programmatic families transfer to items we did not write: JevBench public, where v6-4b scored
# easy/standard/hard 1.000/1.000/0.5946 with hard-tier ECE 0.265. Our families are easier than theirs
# (v6 scored 0.813 on our temporal_numeric against 0.07 on JevBench's), so transfer is genuinely open.
#
# Then A7 (depth truncation) and A10 (option-text readout) on the best checkpoint.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HOME="$PWD/.hf" HF_HUB_OFFLINE=1
mkdir -p runs/v7
LOG=runs/v7/v7x_part2.log
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a $LOG
  if "$@" > "runs/v7/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a $LOG
  else echo "$(date '+%F %T') FAIL  $name (see runs/v7/$name.log)" | tee -a $LOG; fi; }

# wait for part 1 (v7.3 training) to finish before touching the GPU
while pgrep -f "scripts/run_v7x.sh" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') part 1 finished, starting part 2" | tee -a $LOG

# --- transfer: JevBench public, the only out-of-distribution accuracy+calibration check we have ---
step jevbench_v7.2 scripts/run_jevbench.sh v7.2-4b v7.2-4b
step jevbench_v7.3 scripts/run_jevbench.sh v7.3-4b v7.3-4b

# --- v7.3 on the same splits as v7.2, so lambda_brier is the only difference ---------------------
step soft_fidelity_v7.3 python scripts/analysis/soft_fidelity.py \
  --rows runs/osdg/v7.3-4b-progsplits/rows.jsonl --source data/synthetic/prog_v1 \
  --summary runs/osdg/v7.3-4b-progsplits/summary.json --out runs/v7/soft_fidelity_v7.3-4b-progsplits.json

# --- A7 / v7.1: truncate the best checkpoint at the layer the probe said the answer settles at ---
step truncate_v7.1 python -m open_spark_jev.experimental.truncate \
  --model checkpoints/v7.2-4b --keep 24 --out checkpoints/v7.1-4b-L24
step eval_v7.1_prog python -m open_spark_jev.eval.osdg --model checkpoints/v7.1-4b-L24 \
  --name v7.1-4b-L24-progsplits --data-prefix data/synthetic/prog_v1/ --perms 3
step eval_v7.1_v5 python -m open_spark_jev.eval.osdg --model checkpoints/v7.1-4b-L24 \
  --name v7.1-4b-L24-v5splits --data-prefix data/benchmarks/v5_ --perms 3

# --- A10 / v7.5: option-text readout vs the letter readout, zero-shot on both checkpoints --------
step option_readout_v7.2_prog python -m open_spark_jev.experimental.option_readout \
  --model checkpoints/v7.2-4b --items data/synthetic/prog_v1/test_locked.jsonl --limit 400 \
  --out runs/v7/option_readout_v7.2_prog.json
step option_readout_v7.2_v5 python -m open_spark_jev.experimental.option_readout \
  --model checkpoints/v7.2-4b --items data/benchmarks/v5_test_locked.jsonl --limit 400 \
  --out runs/v7/option_readout_v7.2_v5.json

echo "$(date '+%F %T') V7X PART 2 DONE" | tee -a $LOG
