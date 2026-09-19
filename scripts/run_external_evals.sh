#!/usr/bin/env bash
# Score the main checkpoints on every third-party source, separately. Waits for the A0-A6 ladder to finish
# (GPU is serial), then writes runs/external/<name>/{<source>.json,SUMMARY.md}.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
while ! grep -q "LADDER DONE" runs/variants/ladder.log 2>/dev/null; do sleep 60; done
for m in sft-qwen3-1.7b rlcd-direct-qwen3-1.7b; do
  python -m open_spark_jev.eval.external --model checkpoints/$m --name "$m" > "runs/external_$m.log" 2>&1 || echo "FAILED $m (see runs/external_$m.log)"
done
echo "EXTERNAL EVALS DONE"
