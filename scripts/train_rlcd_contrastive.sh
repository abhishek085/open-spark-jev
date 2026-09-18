#!/usr/bin/env bash
# Phase 2, mechanism 1 (academic RLCD, Yang et al.): contrastive pairs -> reward model ->
# exact menu policy gradient. See open_spark_jev/train/rlcd.py module docstring.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
CFG=${1:-configs/train/rlcd_contrastive.yaml}
python -m open_spark_jev.train.rlcd pairs  --config "$CFG"
python -m open_spark_jev.train.rlcd rm     --config "$CFG"
python -m open_spark_jev.train.rlcd policy --config "$CFG"
