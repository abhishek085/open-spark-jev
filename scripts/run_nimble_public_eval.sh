#!/usr/bin/env bash
# Score released spark-s1 models on Bespoke Nimble's 13 public human-labelled subsets (each a separate source).
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
SRC=$(ls data/benchmarks/external | grep '^ext-nimble-' | sed 's/.jsonl//' | tr '\n' ' ')
for m in v3-4b v3-1.7b; do
  python -m open_spark_jev.eval.external --model checkpoints/$m --name nimble-public-$m --sources $SRC > runs/nimble_public_$m.log 2>&1 && echo "$(date '+%F %T') OK $m" | tee -a runs/nimble_public.log || echo "$(date '+%F %T') FAIL $m" | tee -a runs/nimble_public.log
done
echo "$(date '+%F %T') DONE" | tee -a runs/nimble_public.log
