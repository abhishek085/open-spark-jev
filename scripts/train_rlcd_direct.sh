#!/usr/bin/env bash
# Phase 2, mechanism 2: direct calibration objective (no contrastive pairs, no reward model).
# See open_spark_jev/train/rlcd.py module docstring and configs/train/rlcd_direct.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
python -m open_spark_jev.train.rlcd policy --config "${1:-configs/train/rlcd_direct.yaml}"
