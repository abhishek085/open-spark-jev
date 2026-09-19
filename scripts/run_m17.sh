#!/usr/bin/env bash
# M17: SFT including real public text, then the numbers that matter -- the 60-row tool-call set
# (accuracy + speed vs Jev's own recorded run) and the full external sweep.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
mkdir -p runs/variants
step() { local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a runs/m17.log
  if "$@" > "runs/m17_$name.log" 2>&1; then echo "$(date '+%F %T') OK    $name" | tee -a runs/m17.log
  else echo "$(date '+%F %T') FAIL  $name (see runs/m17_$name.log)" | tee -a runs/m17.log; fi; }

step train python -m open_spark_jev.train.sft --config configs/train/sft_m17.yaml
step eval_internal python -m open_spark_jev.eval.benchmark --backend hf --model checkpoints/sft-m17-qwen3-1.7b \
  --data data/benchmarks/sim_test.jsonl data/benchmarks/teacher_test.jsonl data/benchmarks/public_test.jsonl \
  --out runs/eval_m17.json
step external python -m open_spark_jev.eval.external --model checkpoints/sft-m17-qwen3-1.7b --name spark-s1-1.7b-sft-m17
# headline: the 60-row set, accuracy AND speed, against Jev's committed run. Needs an idle GPU.
step speed60 python -m open_spark_jev.eval.speed_vs_generation --menu-model checkpoints/sft-m17-qwen3-1.7b \
  --base-model models/Qwen3-1.7B --data data/benchmarks/external/ext-toolcall-risk.jsonl --limit 60 \
  --out runs/speed60_m17.json
echo "$(date '+%F %T') M17 DONE" | tee -a runs/m17.log
