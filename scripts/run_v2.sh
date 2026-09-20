#!/usr/bin/env bash
# spark-s1-v2: frozen backbone + MLP head, on the os-datagen splits.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
for m in "models/Qwen3-1.7B v2-1.7b" "Qwen/Qwen3-4B v2-4b"; do
  set -- $m
  echo "$(date '+%F %T') START $2" | tee -a runs/osdg/v2.log
  python -m open_spark_jev.experimental.frozen_head --base $1 --name $2 > runs/osdg/$2.log 2>&1 && echo "$(date '+%F %T') OK    $2" | tee -a runs/osdg/v2.log || echo "$(date '+%F %T') FAIL  $2" | tee -a runs/osdg/v2.log
done
echo "$(date '+%F %T') V2 DONE" | tee -a runs/osdg/v2.log
