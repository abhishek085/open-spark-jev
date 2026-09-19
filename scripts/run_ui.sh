#!/usr/bin/env bash
# Start the local playground (web UI + Jev-compatible API).
#   scripts/run_ui.sh                    # http://127.0.0.1:8400, this machine only
#   scripts/run_ui.sh --tailscale        # bind this machine's Tailscale IP so your laptop can open it directly
#   scripts/run_ui.sh --host 0.0.0.0     # all interfaces (LAN) - only on a network you trust
#   scripts/run_ui.sh --model sft-qwen3-1.7b --port 8500
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache"
args=()
for a in "$@"; do
  if [ "$a" = "--tailscale" ]; then
    ip=$(tailscale ip -4 2>/dev/null | head -1 || true)
    [ -n "$ip" ] || { echo "tailscale not available; use --host <ip>" >&2; exit 1; }
    args+=(--host "$ip"); echo "Open http://$ip:8400 from any device on your tailnet"
  else args+=("$a"); fi
done
exec osj ui "${args[@]}"
