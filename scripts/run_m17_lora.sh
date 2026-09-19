#!/usr/bin/env bash
# M17-LoRA: the A0 LoRA recipe (matched to variants/a0) plus real public text. Isolates the
# adaptation method: compare against variants/a0 (no public text) and sft-m17 (full FT + public text).
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/m17_lora.log
  if "$@" > "runs/m17_lora_$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/m17_lora.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/m17_lora_$name.log)" | tee -a runs/m17_lora.log; fi; }
step train python -m open_spark_jev.experimental.variants --variant a0 --n-train 16000 \
  --extra-data data/raw/public_train_split.jsonl --out runs/variants/a0-public --save-dir checkpoints/variants/a0-public
step external python -m open_spark_jev.eval.external --model checkpoints/variants/a0-public --name spark-s1-1.7b-lora-m17
echo "$(date '+%F %T') M17-LORA DONE" | tee -a runs/m17_lora.log
