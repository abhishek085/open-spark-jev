#!/usr/bin/env bash
# Ladder v3: every variant SAVES a servable checkpoint (checkpoints/variants/<v>) and is scored on all external
# sources through the shared runner (runs/external/<name>/). Smokes first so plumbing bugs fail fast.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
mkdir -p runs/variants
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/variants/ladder.log
  if "$@" > "runs/variants/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/variants/ladder.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/variants/$name.log)" | tee -a runs/variants/ladder.log; fi; }
V="python -m open_spark_jev.experimental.variants"
for v in a2 a3 a4; do step smoke_$v $V --variant $v --out runs/variants/smoke_$v --smoke --save-dir runs/variants/smoke_save_$v; done
step smoke_load_a3 python -m open_spark_jev.eval.external --model runs/variants/smoke_save_a3 --name smoke_a3 --limit 6 --sources ext-toolcall-risk ext-jev-directory
step smoke_load_a4 python -m open_spark_jev.eval.external --model runs/variants/smoke_save_a4 --name smoke_a4 --limit 6 --sources ext-toolcall-risk ext-jev-directory
for v in a2 a0 a3 a4; do
  step full_$v $V --variant $v --out runs/variants/$v
  step external_$v python -m open_spark_jev.eval.external --model checkpoints/variants/$v --name variant-$v
done
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
