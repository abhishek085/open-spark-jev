#!/usr/bin/env bash
# v7.x ladder: what the programmatic families and a stronger proper-scoring-rule weight buy us.
#
# v7.0 is not a training run -- it is v6 measured on the new splits, so every later number has a
# before. v7.2 adds the data; v7.3 changes only lambda_brier on top of the same data, so the two
# are separately interpretable. Every stage writes per-row predictions (repo rule), and each model
# is scored on BOTH the new programmatic splits and the v5 splits, because the thing most likely to
# go wrong is buying hard-family accuracy at the cost of the Jev-style scope we already had.
#
# Kernels: flash-linear-attention 0.5.2 + causal_conv1d 1.7.0 are installed and GPU-verified, and
# TRITON_CACHE_DIR must be set or fla silently falls back to CPU (the root-owned ~/.triton/cache).
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

# --- v7.0: v6 on the new families, as a baseline (no training) -----------------------------
step eval_v7.0_v6_prog python -m open_spark_jev.eval.osdg --model checkpoints/v6-4b \
  --name v7.0-v6-4b-progsplits --data-prefix data/synthetic/prog_v1/ --perms 3

# --- v7.2: v5 data + programmatic families (soft targets exercised for the first time) ------
step train_v7.2 python -m open_spark_jev.train.sft --config configs/train/sft_v7_2.yaml
step eval_v7.2_prog python -m open_spark_jev.eval.osdg --model checkpoints/v7.2-4b \
  --name v7.2-4b-progsplits --data-prefix data/synthetic/prog_v1/ --perms 3
step eval_v7.2_v5 python -m open_spark_jev.eval.osdg --model checkpoints/v7.2-4b \
  --name v7.2-4b-v5splits --data-prefix data/benchmarks/v5_ --perms 3
step ext_v7.2 python -m open_spark_jev.eval.external --model checkpoints/v7.2-4b \
  --name v7.2-4b-jev-external --sources $JEV

# --- v7.3: same data, Brier weight 0.5 -> 3.0 ----------------------------------------------
step train_v7.3 python -m open_spark_jev.train.sft --config configs/train/sft_v7_3.yaml
step eval_v7.3_prog python -m open_spark_jev.eval.osdg --model checkpoints/v7.3-4b \
  --name v7.3-4b-progsplits --data-prefix data/synthetic/prog_v1/ --perms 3
step eval_v7.3_v5 python -m open_spark_jev.eval.osdg --model checkpoints/v7.3-4b \
  --name v7.3-4b-v5splits --data-prefix data/benchmarks/v5_ --perms 3
step ext_v7.3 python -m open_spark_jev.eval.external --model checkpoints/v7.3-4b \
  --name v7.3-4b-jev-external --sources $JEV

echo "$(date '+%F %T') V7X DONE" | tee -a $LOG
