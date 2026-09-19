#!/usr/bin/env bash
# Runs the whole A0-A6 ladder back-to-back on the GPU, after the main SFT/RLCD-direct chain is done.
# Each step logs to runs/variants/<name>.log; a failing step is recorded and the ladder continues.
# Smoke runs (tiny, minutes) go first so a plumbing bug in A2-A5 fails fast, not after an hour.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
mkdir -p runs/variants

# wait for the main training chain to release the GPU
while ps -eo cmd | grep -qE "[o]pen_spark_jev\.train\.|[t]rain_rlcd_direct|[t]rain_sft"; do sleep 30; done

step() { # name, cmd...
  local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/variants/ladder.log
  if "$@" > "runs/variants/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/variants/ladder.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/variants/$name.log)" | tee -a runs/variants/ladder.log; fi
}

# 1. main-model evals on the clean test sets (A0 = the trained checkpoints)
for m in sft-qwen3-1.7b rlcd-direct-qwen3-1.7b; do
  step "eval_$m" python -m open_spark_jev.eval.benchmark --backend hf --model checkpoints/$m --data data/benchmarks/sim_test.jsonl data/benchmarks/teacher_test.jsonl --out runs/eval_v2_$m.json
done
# 2. A1: parallel readout on the RLCD-direct checkpoint (no training)
step a1_parallel python -m open_spark_jev.experimental.parallel_readout --model checkpoints/rlcd-direct-qwen3-1.7b --data data/benchmarks/sim_test.jsonl --limit 200
# 3. smokes
for v in a2 a3 a4; do step smoke_$v python -m open_spark_jev.experimental.variants --variant $v --out runs/variants/smoke_$v --smoke; done
step smoke_a5 python -m open_spark_jev.experimental.a5_routed --out runs/variants/smoke_a5 --smoke
# 4. matched-budget runs
for v in a0 a2 a3 a4; do step full_$v python -m open_spark_jev.experimental.variants --variant $v --out runs/variants/$v; done
step full_a5 python -m open_spark_jev.experimental.a5_routed --out runs/variants/a5
# 5. A6: two arms via the standard SFT loop (LoRA), then joint eval
step a6_gen python -m open_spark_jev.experimental.a6_joint gen
for arm in independent joint; do
  step a6_train_$arm python -m open_spark_jev.train.sft --config configs/train/sft.yaml --set "data=[data/synthetic/a6/${arm}_train.jsonl]" "output_dir=checkpoints/a6-$arm" "lora={r: 16, alpha: 32}" epochs=1 lr=2.0e-4
done
step a6_eval python -m open_spark_jev.experimental.a6_joint eval --independent checkpoints/a6-independent --joint checkpoints/a6-joint
echo "$(date '+%F %T') LADDER DONE" | tee -a runs/variants/ladder.log
