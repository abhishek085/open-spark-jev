#!/usr/bin/env bash
# Create a repo-local Python environment on DGX Spark (aarch64, CUDA 13 driver).
# PyPI's linux/aarch64 torch wheels ship with CUDA 13.0 support, which matches the GB10
# driver stack, so no custom index is needed. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
[ -d .venv ] || $PY -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip wheel
pip install -q "huggingface_hub[cli]>=1.0"
pip install -q torch
pip install -q -e ".[train,serve,dev]"
python - <<'PY'
import torch, transformers, trl, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__)
PY
