#!/usr/bin/env bash
# Score an NVFP4-quantized engine on the public JevBench items. Unlike run_jevbench.sh (HF backend, bf16 checkpoints),
# an NVFP4 engine must already be served by vLLM (weights aren't loadable through plain transformers); this starts
# our gateway in --backend openai mode in front of that vLLM endpoint, so JevBench's /v1/systemone calls get routed
# through to it with the checkpoint's own calibration.json applied.
#   scripts/run_jevbench_quant.sh <engine-dir under engines/> <vllm-upstream-url> [label]
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export TRITON_CACHE_DIR="$PWD/.triton_cache" HF_HUB_OFFLINE=1
ENGINE="${1:?engine dir name under engines/}"; UPSTREAM="${2:?vllm upstream, e.g. http://127.0.0.1:8355/v1}"; LABEL="${3:-$ENGINE}"
JB=/home/admin/llm-workspace/external/jevbench
OUT="runs/jevbench/$LABEL"; mkdir -p "$OUT"; PORT=8472
python -m open_spark_jev.serve.gateway --backend openai --upstream "$UPSTREAM" --model "engines/$ENGINE" --port $PORT > "$OUT/gateway.log" 2>&1 &
GW=$!; trap 'kill $GW 2>/dev/null' EXIT
until curl -s -m 3 localhost:$PORT/healthz >/dev/null 2>&1; do sleep 3; kill -0 $GW 2>/dev/null || { echo "gateway died"; tail -5 "$OUT/gateway.log"; exit 1; }; done
curl -s -m 60 -X POST localhost:$PORT/v1/systemone -H 'Content-Type: application/json' -d '{"state":"warm-up","model":"'"$ENGINE"'","questions":{"decision":{"type":"noul","instructions":"Is this a warm-up?"}}}' >/dev/null
cd "$JB"
for T in original easy hard; do
  python -m jevbench.cli run --tasks datasets/public/$T.jsonl --adapter typesafe --endpoint http://127.0.0.1:$PORT --key-env '' --model "$ENGINE" \
    --cost-basis local_no_provider_tariff --price-in-per-m 0 --price-out-per-m 0 --cap-usd 1000000 --reserve-usd 0 \
    --ledger "$OLDPWD/$OUT/ledger.jsonl" --results "$OLDPWD/$OUT/$T.results.jsonl" --raw-dir "$OLDPWD/$OUT/raw_$T" --manifest "$OLDPWD/$OUT/$T.manifest.json" > "$OLDPWD/$OUT/$T.run.log" 2>&1
  python -m jevbench.cli summarize --tasks datasets/public/$T.jsonl --results "$OLDPWD/$OUT/$T.results.jsonl" > "$OLDPWD/$OUT/$T.summary.json" 2> "$OLDPWD/$OUT/$T.summary.err"
  echo "$T: $(python -c "import json;s=json.load(open('$OLDPWD/$OUT/$T.summary.json'));print({k:s.get(k) for k in ('accuracy','brier_mean','ece','n_valid','n_planned')})" 2>/dev/null)"
done
