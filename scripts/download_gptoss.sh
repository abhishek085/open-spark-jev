#!/usr/bin/env bash
# Download the gpt-oss-120b teacher. It is tooling for data generation only, not part of the
# shipped open-spark-Jev checkpoint, so it does not belong in this repo's models/ dir. It also
# can't go in the box's shared ~/.cache/huggingface/hub (root-owned, no write access for this
# user) -- unlike Qwen/Nemotron, which were already cached there by someone with write access
# before this session started. We use a second, admin-owned HF cache instead.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HOME="$HOME/.cache/huggingface-osj"  # ~/.cache/huggingface/hub is root-owned on this box; use our own cache instead of fighting for write access
mkdir -p "$HF_HOME"
hf download openai/gpt-oss-120b --exclude "metal/*" --exclude "original/*"
echo "done: $(du -sh "$HF_HOME/hub/models--openai--gpt-oss-120b" | cut -f1)"
