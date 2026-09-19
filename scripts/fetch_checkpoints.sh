#!/usr/bin/env bash
# Download released Spark-S1 checkpoints from a Hugging Face repo into checkpoints/ (no repo id is hard-coded).
#   scripts/fetch_checkpoints.sh <hf-repo-id> [checkpoint-dir-name ...]
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
repo="${1:?usage: fetch_checkpoints.sh <hf-repo-id> [name ...]}"; shift || true
python - "$repo" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, names = sys.argv[1], sys.argv[2:]
pat = [f"{n}/*" for n in names] if names else None
p = snapshot_download(repo_id=repo, local_dir="checkpoints", allow_patterns=pat)
print("downloaded to", p)
PY
python -c "from open_spark_jev.serve.gateway import discover_models as d; print('available:', list(d()))"
