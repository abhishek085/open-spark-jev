#!/usr/bin/env bash
# Download a released spark-s1 model from Hugging Face into checkpoints/<release-id>/ so the gateway, UI and Decision Lab find it.
#   scripts/fetch_checkpoints.sh abhishek085/spark-s1-4b-v3        # ~8 GB
#   scripts/fetch_checkpoints.sh abhishek085/spark-s1-1.7b-v3      # ~3.4 GB
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
repo="${1:?usage: fetch_checkpoints.sh <hf-repo-id>   e.g. abhishek085/spark-s1-4b-v3}"
python - "$repo" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
name = repo.split("/")[-1]
p = snapshot_download(repo_id=repo, local_dir=os.path.join("checkpoints", name))
print("downloaded to", p)
PY
python -c "from open_spark_jev.serve.gateway import discover_models as d; print('available:', list(d()))"
