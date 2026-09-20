#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
for m in v3-4b v3-1.7b; do
  python -m open_spark_jev.eval.external --model checkpoints/$m --name spark-s1-$m-osdg-trained > runs/osdg/ext_$m.log 2>&1 && echo "$(date '+%F %T') OK $m" | tee -a runs/osdg/ext_v3.log || echo "$(date '+%F %T') FAIL $m" | tee -a runs/osdg/ext_v3.log
done
echo "$(date '+%F %T') EXT DONE" | tee -a runs/osdg/ext_v3.log
