#!/usr/bin/env bash
# Reordered A0-A6 ladder: architecture variants first on the finished SFT v2 checkpoint, RLCD-direct later.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
mkdir -p runs/variants
while ps -eo cmd | grep -qE "[o]pen_spark_jev\.train\.|[t]rain_rlcd_direct|[t]rain_sft"; do sleep 20; done
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/variants/ladder.log
  if "$@" > "runs/variants/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/variants/ladder.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/variants/$name.log)" | tee -a runs/variants/ladder.log; fi; }
V="python -m open_spark_jev.experimental.variants"
step eval_sft python -m open_spark_jev.eval.benchmark --backend hf --model checkpoints/sft-qwen3-1.7b --data data/benchmarks/sim_test.jsonl data/benchmarks/teacher_test.jsonl --out runs/eval_v2_sft-qwen3-1.7b.json
step a1_parallel python -m open_spark_jev.experimental.parallel_readout --model checkpoints/sft-qwen3-1.7b --data data/benchmarks/sim_test.jsonl --limit 200
step smoke_a2 $V --variant a2 --out runs/variants/smoke_a2 --smoke
step full_a0 $V --variant a0 --out runs/variants/a0
step full_a2 $V --variant a2 --out runs/variants/a2
for v in a3 a4; do step smoke_$v $V --variant $v --out runs/variants/smoke_$v --smoke; step full_$v $V --variant $v --out runs/variants/$v; done
step external_sft python -m open_spark_jev.eval.external --model checkpoints/sft-qwen3-1.7b --name sft-qwen3-1.7b
step train_rlcd_direct bash scripts/train_rlcd_direct.sh
step eval_direct python -m open_spark_jev.eval.benchmark --backend hf --model checkpoints/rlcd-direct-qwen3-1.7b --data data/benchmarks/sim_test.jsonl data/benchmarks/teacher_test.jsonl --out runs/eval_v2_rlcd-direct-qwen3-1.7b.json
step external_direct python -m open_spark_jev.eval.external --model checkpoints/rlcd-direct-qwen3-1.7b --name rlcd-direct-qwen3-1.7b
step smoke_a5 python -m open_spark_jev.experimental.a5_routed --out runs/variants/smoke_a5 --smoke
step full_a5 python -m open_spark_jev.experimental.a5_routed --out runs/variants/a5
step a6_gen python -m open_spark_jev.experimental.a6_joint gen
for arm in independent joint; do
  step a6_train_$arm python -m open_spark_jev.train.sft --config configs/train/sft.yaml --set "data=[data/synthetic/a6/${arm}_train.jsonl]" "output_dir=checkpoints/a6-$arm" "lora={r: 16, alpha: 32}" epochs=1 lr=2.0e-4
done
step a6_eval python -m open_spark_jev.experimental.a6_joint eval --independent checkpoints/a6-independent --joint checkpoints/a6-joint
echo "$(date '+%F %T') LADDER DONE" | tee -a runs/variants/ladder.log
