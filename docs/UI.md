# Playground: web UI and API

A local web UI and a Jev-compatible HTTP API over the Spark-S1 checkpoints. One process serves both.
Everything runs on your machine; no request leaves it.

## Run it

```bash
# once: environment + backbone weights (skip if already done)
scripts/setup_env.sh
scripts/download_weights.sh Qwen/Qwen3-1.7B        # -> models/Qwen3-1.7B (also the "zero-shot" comparison model)

# checkpoints: train them (docs/ROADMAP.md) or fetch released ones from a Hugging Face repo
scripts/fetch_checkpoints.sh <hf-repo-id>          # -> checkpoints/<name>/

scripts/run_ui.sh                                  # http://127.0.0.1:8400
```

The UI lists every model found in `checkpoints/` (plus the untrained base). Labels and notes come from
[`configs/serve/models.yaml`](../configs/serve/models.yaml); update that file when a checkpoint is retrained so
the UI never mislabels a model. Models load on first use (a few seconds) and up to `OSJ_MAX_LOADED` (default 2)
stay in memory.

## Open it from another machine (e.g. your laptop)

The server binds `127.0.0.1` by default. Two safe options:

* **Tailscale** (if both machines are on the same tailnet): `scripts/run_ui.sh --tailscale`, then open the URL it
  prints (`http://<tailscale-ip>:8400`) on the laptop.
* **SSH tunnel** (no extra software): on the laptop run
  `ssh -L 8400:localhost:8400 <user>@<spark-host>`, keep it open, and browse to `http://localhost:8400`.

`--host 0.0.0.0` exposes it to the whole LAN. There is no authentication, so only do that on a network you trust.

## Using the UI

1. Pick an **example task** (left). Each is one *state* plus one or more typed *questions*.
2. Edit the state or the questions, or press **+ question** to add your own. Types are `choice` (options as
   `key: description`), `score` (ordered levels, lowest first) and `boolean` (a claim; returns P(true)).
3. **Run** (Ctrl+Enter). Every question is answered in one call; bars show the probability of each option.
4. The **escalate** slider shows how a confidence threshold would gate automation: below it, the answer is
   flagged "escalate to a human". Moving it re-runs nothing on the server side beyond the same call.
5. **Compare with** runs a second model on the same input side by side. Comparing SFT against the zero-shot
   base is the clearest demo of what training changes: the untrained model is often 100% sure and wrong.
6. Expand **API request / response** to copy the exact `curl` for what you just ran.

URL parameters for demos and recording: `/?preset=sql&run=1` (auto-run an example), `&compare=1` (compare mode).
Preset ids: `support`, `retrieval`, `sql`, `injection`, `toolcall`, `email`, `rag`, `incident`, `testfail`, `form`.
Tip for screen recording: the page follows your OS light/dark setting; a 1500px-wide window shows all three panels.

## API

`GET /v1/models` lists models. `POST /v1/evaluate` is Jev-compatible (`boolean` is accepted as an alias of `noul`);
`POST /v1/decide` is the native schema. Both take an optional `model`.

```bash
curl -s localhost:8400/v1/evaluate -H 'Content-Type: application/json' -d '{
  "model": "sft-qwen3-1.7b",
  "state": "Subject: charged twice\nI was billed $49 twice and cannot find the invoice.",
  "questions": {
    "queue":   {"type": "choice",  "instructions": "Which queue?", "criteria": {"billing": "payments", "technical": "bugs", "account": "login"}},
    "urgency": {"type": "score",   "instructions": "How urgent?",  "criteria": ["low", "medium", "high", "critical"]},
    "double":  {"type": "boolean", "instructions": "The customer was charged more than once."}
  }}'
```

```python
import requests
r = requests.post("http://localhost:8400/v1/evaluate", json={
    "state": {"sql": "DELETE FROM orders WHERE created_at < '2024-01-01';", "target_db": "production"},
    "questions": {"safety": {"type": "choice", "instructions": "Is this safe for an agent to run?",
                             "criteria": {"safe": "read-only or tiny reversible write", "needs_approval": "large or irreversible", "unsafe": "never automatic"}}},
}).json()
a = r["answers"]["safety"]      # {"choice": ..., "probabilities": {...}, "confidence": ...}
if a["confidence"] < 0.7:       # calibrated confidence -> route to a human instead of acting
    ...
```

Responses: `choice` -> `{choice, probabilities, confidence}`; `score` -> `{score (expected level), probabilities, confidence}`;
`boolean`/`noul` -> `{probability}`. Every response also has `latency_ms`. More client examples: [`examples/`](../examples).

## What to expect (honest notes)

* Latency is about 130-200 ms for a multi-question call on the GB10 with the GPU shared with training; first use of a
  model adds its load time.
* The models were trained on simulators and synthetic data over 31 task types, **not on real-world text**. The example
  tasks are sensible, not cherry-picked to pass; some are answered wrongly or with low confidence. Observed on SFT v2 on
  2026-09-19: support triage, retrieval, RAG sufficiency, form status looked right; incident routing and email triage
  were plausible but low-confidence; the tool-call risk example picked "destructive" (0.53) where "exfiltration" is
  arguably right; **the prompt-injection example missed the injection** (P=0.15) while the trust question called the
  document untrusted data. Treat the confidence slider as a demonstration, and calibrate on your own data before relying on it.
* Not for consequential decisions (financial, legal, medical, access control) without human review.

## Troubleshooting

* `PermissionError ... /.triton/cache`: run through `scripts/run_ui.sh` (it sets the repo-local `TRITON_CACHE_DIR`).
* `no checkpoints found`: run `scripts/fetch_checkpoints.sh` or train one. The untrained base appears in the list
  whenever `models/Qwen3-1.7B` exists.
* Port in use: `scripts/run_ui.sh --port 8500`.
* GPU memory: lower `OSJ_MAX_LOADED` to 1.
* `osj ui` is intentionally not started with `python -m open_spark_jev...`, so the thermal guard
  ([DGX_SPARK.md](DGX_SPARK.md)) never freezes the UI mid-demo; inference bursts are short.
