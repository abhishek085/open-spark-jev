#!/usr/bin/env bash
# Score a spark-s1 checkpoint on the public JevBench items (internal comparison only; the 24 private + 109 held-out hard items cannot be run locally).
#   scripts/run_jevbench.sh <checkpoint-dir-name under checkpoints/> [label]        e.g. scripts/run_jevbench.sh v5-1.7b
# Serves the checkpoint with the gateway (its /v1/systemone alias speaks the hosted-Jev wire format the harness expects), runs the harness, summarises.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
CK="${1:?checkpoint dir name under checkpoints/}"; LABEL="${2:-$CK}"
JB=/home/admin/llm-workspace/external/jevbench
OUT="runs/jevbench/$LABEL"; mkdir -p "$OUT"; PORT=8471
python -m open_spark_jev.serve.gateway --backend hf --default-model "$CK" --port $PORT > "$OUT/gateway.log" 2>&1 &
GW=$!; trap 'kill $GW 2>/dev/null' EXIT
until curl -s -m 3 localhost:$PORT/healthz >/dev/null 2>&1; do sleep 3; kill -0 $GW 2>/dev/null || { echo "gateway died"; tail -5 "$OUT/gateway.log"; exit 1; }; done
curl -s -m 60 -X POST localhost:$PORT/v1/systemone -H 'Content-Type: application/json' -d '{"state":"warm-up","model":"'"$CK"'","questions":{"decision":{"type":"noul","instructions":"Is this a warm-up?"}}}' >/dev/null
cd "$JB"
for T in original easy hard; do
  # local compute is free: own ledger per checkpoint, no cap, no per-token price (else the shared default ledger's $15 cap
  # gets split across every checkpoint run against it and can cut a later run's hard tier short).
  python -m jevbench.cli run --tasks datasets/public/$T.jsonl --adapter typesafe --endpoint http://127.0.0.1:$PORT --key-env '' --model "$CK" \
    --cost-basis local_no_provider_tariff --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1000000 --reserve-usd 0 \
    --ledger "$OLDPWD/$OUT/ledger.jsonl" --results "$OLDPWD/$OUT/$T.results.jsonl" --raw-dir "$OLDPWD/$OUT/raw_$T" --manifest "$OLDPWD/$OUT/$T.manifest.json" > "$OLDPWD/$OUT/$T.run.log" 2>&1
  python -m jevbench.cli summarize --tasks datasets/public/$T.jsonl --results "$OLDPWD/$OUT/$T.results.jsonl" > "$OLDPWD/$OUT/$T.summary.json" 2> "$OLDPWD/$OUT/$T.summary.err"
  echo "$T: $(python -c "import json;s=json.load(open('$OLDPWD/$OUT/$T.summary.json'));print({k:s.get(k) for k in ('accuracy','brier_mean','ece','n_valid','n_planned')})" 2>/dev/null)"
done
