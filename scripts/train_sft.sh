#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
python -m open_spark_jev.train.sft --config "${1:-configs/train/sft.yaml}" "${@:2}"
