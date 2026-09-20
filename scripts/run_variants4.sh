#!/usr/bin/env bash
# Ladder v4: A0/A3/A4 (variants.py) and A5 on the same matched budget, every arm including the in-domain
# tool-call-risk corpus in full (--always-data), each saved and scored on all external sources.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
mkdir -p runs/variants4
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/variants4/ladder.log
  if "$@" > "runs/variants4/$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/variants4/ladder.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/variants4/$name.log)" | tee -a runs/variants4/ladder.log; fi; }
V="python -m open_spark_jev.experimental.variants --n-train 12000 --always-data data/synthetic/toolcall_risk_train.jsonl"
for v in a0 a3 a4; do
  step train_$v $V --variant $v --out runs/variants4/$v --save-dir checkpoints/variants4/$v
  step external_$v python -m open_spark_jev.eval.external --model checkpoints/variants4/$v --name v4-$v
done
step a5 python -m open_spark_jev.experimental.a5_routed --n-train 12000 --always-data data/synthetic/toolcall_risk_train.jsonl --out runs/variants4/a5
echo "$(date '+%F %T') LADDER4 DONE" | tee -a runs/variants4/ladder.log
