#!/usr/bin/env bash
# One tuning cycle, one model role at a time (2 parallel instances each):
#   generate-only (generator up) -> swap -> reverify (verifier up, incl. supportability audit).
# usage: scripts/dev_cycle.sh GEN_RUN VER_RUN COUNT_PER_PACK [pack ...]     (no packs = all)
set -uo pipefail
cd "$(dirname "$0")/.."; export PYTHONPATH=src
GEN="$1"; VER="$2"; N="$3"; shift 3
PACKS=(); if [ $# -gt 0 ]; then for p in "$@"; do PACKS+=(--task-pack "$p"); done; else PACKS=(--all); fi
PY=../.venv/bin/python
./scripts/serve.sh stop verifier >/dev/null 2>&1
OSJ_INSTANCES=2 OSJ_GPU_UTIL=0.25 OSJ_MAX_LEN=6144 ./scripts/serve.sh start generator || exit 1
$PY -m os_datagen generate-only "${PACKS[@]}" --models configs/models.dev.yaml --count "$N" --out "artifacts/$GEN" 2>&1 | grep -v INFO
./scripts/serve.sh stop generator >/dev/null 2>&1
OSJ_INSTANCES=2 OSJ_GPU_UTIL=0.26 OSJ_MAX_LEN=6144 OSJ_PORT=8100 ./scripts/serve.sh start verifier || exit 1
$PY -m os_datagen reverify --run "artifacts/$GEN" --out "artifacts/$VER" --models configs/models.dev.yaml 2>&1 | grep -E "accepted"
echo CYCLE_DONE
