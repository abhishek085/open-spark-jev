#!/usr/bin/env bash
# spark-s1-v3: LoRA decision model trained on osdg_train with option-order augmentation, three backbone sizes,
# each scored on the osdg splits (raw + calibration-split temperature + option-permutation robustness).
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
mkdir -p runs/osdg
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/osdg/v3.log
  if "$@" > "runs/osdg/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/osdg/v3.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/osdg/$name.log)" | tee -a runs/osdg/v3.log; fi; }
for m in "Qwen/Qwen3-1.7B:models/Qwen3-1.7B:1.7b" "Qwen/Qwen3-0.6B:Qwen/Qwen3-0.6B:0.6b" "Qwen/Qwen3-4B:Qwen/Qwen3-4B:4b"; do
  IFS=: read -r _ base size <<< "$m"
  step train_v3_$size python -m open_spark_jev.train.sft --config configs/train/sft.yaml --set "model=$base" "output_dir=checkpoints/v3-$size" \
      "data=[data/synthetic/osdg_train_aug.jsonl]" "lora={r: 16, alpha: 32}" epochs=3 lr=2.0e-4
  step eval_v3_$size python -m open_spark_jev.eval.osdg --model checkpoints/v3-$size --name v3-$size-osdg --perms 3
done
echo "$(date '+%F %T') V3 DONE" | tee -a runs/osdg/v3.log
