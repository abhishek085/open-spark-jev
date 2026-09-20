# UI

Two local pages share one gateway: the **Decision Lab** (`/lab`, tool-call decisions, described first) and the original **playground** (`/`, general typed questions, described below). Both are for local use.

## Decision Lab (`osj lab`)

```bash
pip install -e ".[serve]"     # enough for the Lab: no torch, no weights
osj lab                       # http://127.0.0.1:8400/lab   (--port, --host, --model)
```

![Decision Lab](img/decision-lab.png)

**Areas.** A. *Agent state*: tool, proposed command (text only, never run), optional session context, and a selector of ten safe fixtures. B. *Typed decision contract*: the fixed options (`allow`, `ask`, `deny`) and their definitions, the auto-allow threshold, which question is asked (the benchmarked risk-posture question mapped to allow/ask/deny, or the unbenchmarked direct question) and the source of the model output. C. *Decision response*: selected choice, a probability bar for every option, the risk-posture distribution, confidence, latency and the final policy action. *Policy trace*: model output plus every rule match, and an explicit line when a rule overrode the model. D. *Workflow comparison*: an explanatory diagram of a generative loop versus this one (not a performance claim, nothing is simulated). E. *Benchmark view*: the runs in `open_spark_jev/serve/lab_data/benchmark_runs.json`, labelled diagnostic, with hardware and caveats.

**Sources of the model output, always labelled in the page.** *Recorded*: real `spark-s1` outputs saved for the ten fixtures (`recorded_outputs.json`, produced by `scripts/analysis/record_lab_outputs.py` on a DGX Spark), available without weights. *Live*: inference on this machine, when a checkpoint is present under `checkpoints/`. *No model*: deterministic rules only, which can never produce `allow`. A link such as `/lab#curl_exfil_shadow` opens a fixture and runs it.

**Data flow and security boundaries.** The browser calls `GET /v1/lab/fixtures`, `POST /v1/gate` (or `POST /v1/lab/resolve` for recorded outputs) and `GET /v1/lab/benchmarks` on the local process. Nothing in `policy.py`, `gate.py` or the gateway executes, spawns or evaluates a command; the text is tokenised and pattern-matched, or rendered into the model prompt. Request bodies are not logged by the application (the server's access log records the request line only). The server binds to `127.0.0.1` by default; if you expose it on a network, add authentication and TLS first, because anyone who can reach it can use the model.

**Screenshots and GIF.** `docs/img/decision-lab.png` is a headless-browser capture of `/lab#curl_exfil_shadow` in recorded mode; regenerate it with headless Chromium after starting `osj lab`. A screen recording placeholder and the recording checklist are in [LAUNCH.md](LAUNCH.md).

**Tests.** Policy behaviour (`tests/test_policy.py`), the gate and Lab endpoints, fixture loading and the no-execution guarantee (`tests/test_gate_lab.py`).

---

## Playground: web UI and API (`osj ui`)

A local web UI and a Jev-compatible HTTP API over the spark-s1 checkpoints (release ids look like
`spark-s1-1.7b-sft-v2`; "menu scoring" is the mechanism, not a model name). One process serves both.
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
