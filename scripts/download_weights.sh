#!/usr/bin/env bash
# Download the open backbone weights into ./models (git-ignored). Uses a repo-local HF cache
# so nothing depends on the machine's shared ~/.cache/huggingface.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HOME="$PWD/.hf"
MODEL=${1:-Qwen/Qwen3-1.7B}
DEST="models/$(basename "$MODEL")"
mkdir -p "$DEST"
hf download "$MODEL" --local-dir "$DEST" --exclude "*.gguf"
ls -la "$DEST"
