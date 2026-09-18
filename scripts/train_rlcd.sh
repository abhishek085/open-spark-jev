#!/usr/bin/env bash
# Full Phase 2: contrastive pairs -> reward model -> exact menu policy gradient.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
CFG=${1:-configs/train/rlcd.yaml}
python -m open_spark_jev.train.rlcd pairs  --config "$CFG"
python -m open_spark_jev.train.rlcd rm     --config "$CFG"
python -m open_spark_jev.train.rlcd policy --config "$CFG"
