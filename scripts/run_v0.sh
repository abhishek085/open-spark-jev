#!/usr/bin/env bash
# spark-s1-v0 / v1: frozen direct-logit baselines on the os-datagen splits, across base sizes/families.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
mkdir -p runs/osdg
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/osdg/v0.log
  if "$@" > "runs/osdg/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/osdg/v0.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/osdg/$name.log)" | tee -a runs/osdg/v0.log; fi; }
for m in "Qwen/Qwen3-0.6B v0-qwen3-0.6b" "models/Qwen3-1.7B v0-qwen3-1.7b" "Qwen/Qwen3-4B v0-qwen3-4b" "google/gemma-4-E4B-it v0-gemma4-e4b" "google/gemma-4-12B-it v0-gemma4-12b"; do
  set -- $m
  step $2 python -m open_spark_jev.eval.osdg --model $1 --name $2 --perms 3
done
echo "$(date '+%F %T') V0 DONE" | tee -a runs/osdg/v0.log
