#!/usr/bin/env bash
# Score a spark-s1 checkpoint on the public JevBench items (internal comparison only; the 24 private + 109 held-out hard items cannot be run locally).
#   scripts/run_jevbench.sh <checkpoint-dir-name under checkpoints/> [label]        e.g. scripts/run_jevbench.sh v5-1.7b
# Serves the checkpoint with the gateway (its /v1/systemone alias speaks the hosted-Jev wire format the harness expects), runs the harness, summarises.
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
CK="${1:?checkpoint dir name under checkpoints/}"; LABEL="${2:-$CK}"
JB=/home/admin/llm-workspace/external/jevbench
OUT="runs/jevbench/$LABEL"; mkdir -p "$OUT"
# a fixed port let a still-shutting-down previous gateway's socket collide with the next run's healthz
# probe (v7.3's first two JevBench attempts both died silently right after weight load, 0/111 valid);
# hash the label into the port so sequential runs in the same driver never share one.
PORT=$((8471 + $(echo -n "$LABEL" | cksum | cut -d' ' -f1) % 400))
python -m open_spark_jev.serve.gateway --backend hf --default-model "$CK" --port $PORT > "$OUT/gateway.log" 2>&1 &
GW=$!; trap 'kill $GW 2>/dev/null' EXIT
n=0
until curl -s -m 3 localhost:$PORT/healthz >/dev/null 2>&1; do
  sleep 3; n=$((n+1))
  kill -0 $GW 2>/dev/null || { echo "gateway died after $n probes"; tail -20 "$OUT/gateway.log"; exit 1; }
  [ $n -gt 100 ] && { echo "gateway never came healthy after $n probes"; tail -20 "$OUT/gateway.log"; exit 1; }
done
curl -s -m 60 -X POST localhost:$PORT/v1/systemone -H 'Content-Type: application/json' -d '{"state":"warm-up","model":"'"$CK"'","questions":{"decision":{"type":"noul","instructions":"Is this a warm-up?"}}}' >/dev/null
cd "$JB"
fail=0
for T in original easy hard; do
  # local compute is free: own ledger per checkpoint, no cap, no per-token price (else the shared default ledger's $15 cap
  # gets split across every checkpoint run against it and can cut a later run's hard tier short).
  python -m jevbench.cli run --tasks datasets/public/$T.jsonl --adapter typesafe --endpoint http://127.0.0.1:$PORT --key-env '' --model "$CK" \
    --cost-basis local_no_provider_tariff --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1000000 --reserve-usd 0 \
    --ledger "$OLDPWD/$OUT/ledger.jsonl" --results "$OLDPWD/$OUT/$T.results.jsonl" --raw-dir "$OLDPWD/$OUT/raw_$T" --manifest "$OLDPWD/$OUT/$T.manifest.json" > "$OLDPWD/$OUT/$T.run.log" 2>&1
  python -m jevbench.cli summarize --tasks datasets/public/$T.jsonl --results "$OLDPWD/$OUT/$T.results.jsonl" > "$OLDPWD/$OUT/$T.summary.json" 2> "$OLDPWD/$OUT/$T.summary.err"
  nv=$(python -c "import json;print(json.load(open('$OLDPWD/$OUT/$T.summary.json')).get('n_valid',0))" 2>/dev/null || echo 0)
  [ "$nv" = "0" ] && fail=1
  echo "$T: $(python -c "import json;s=json.load(open('$OLDPWD/$OUT/$T.summary.json'));print({k:s.get(k) for k in ('accuracy','brier_mean','ece','n_valid','n_planned')})" 2>/dev/null)"
done
exit $fail
