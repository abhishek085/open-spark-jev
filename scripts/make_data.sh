#!/usr/bin/env bash
# Build the default training + benchmark corpora. Teacher-generated data is optional and needs
# a local OpenAI-compatible server (TEACHER_BASE_URL) hosting a larger open model.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
osj simulate --n 3000 --seed 0 --out data/synthetic/sim_train.jsonl
osj simulate --n 500  --seed 1 --out data/benchmarks/sim_test.jsonl
osj public --limit 2000 --seed 0 --out data/raw/public_train.jsonl || echo "public datasets skipped (offline?)"
if [ -n "${TEACHER_BASE_URL:-}" ]; then
  osj synth --n 300 --distill --out data/synthetic/teacher_train.jsonl
fi
