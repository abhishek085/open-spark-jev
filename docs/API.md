# API

The gateway (`osj lab` or `osj ui`, default `http://127.0.0.1:8400`) serves generated OpenAPI docs at `/docs` and the schema at `/openapi.json`. **No endpoint executes a tool or command**: proposed calls are text that the model reads and the rules pattern-match. Inputs are validated with typed schemas; malformed input gets a 4xx, and an evaluator failure inside the gate resolves to `ask`, never to a raised error or an automatic action.

| Endpoint | Purpose | Status |
|---|---|---|
| `POST /v1/gate` | Tool-call approval: spark-s1 distribution plus the deterministic policy, returns `allow` / `ask` / `deny` | New in 0.1; the documented surface for tool approval |
| `POST /v1/decide` | Typed questions over a state (choice / boolean / score) with a per-decision response; also accepts the native `DecisionRequest` shape | Stable |
| `POST /v1/evaluate` | Jev-style wire format (`questions` as a name-keyed map with `criteria`) | Compatibility layer; scope below |
| `GET /v1/models`, `GET /healthz` | Available checkpoints, liveness | Stable |
| `GET /lab`, `GET /v1/lab/*` | Decision Lab page and its fixture, benchmark and rule endpoints | Local demo only |

## `POST /v1/gate`

Request:

```json
{
  "state": {
    "tool": "bash",
    "command": "git status",
    "context": {"workspace": "/workspace"}
  },
  "policy": {"auto_allow_threshold": 0.995},
  "model": "spark-s1-4b-v3",
  "mode": "risk_posture"
}
```

* `state.tool` (required): the tool name. Shell tools (`bash`, `sh`, `zsh`, `shell`, `terminal`) get the deterministic rules; any other tool is treated as unsupported and resolves to at least `ask`.
* `state.command` or `state.arguments.command`: the proposed command, as text, at most 8,000 characters.
* `policy.auto_allow_threshold`: between 0.5 and 1.0, default 0.995. `allow` requires P(allow) at or above it.
* `model`: a model id from `/v1/models`; omit for the default; `"none"` runs the rules only.
* `mode`: `risk_posture` (default) asks the benchmarked four-way question (readonly / destructive / privileged / exfiltration) and maps it to allow / ask / deny (readonly to allow; destructive and privileged to ask; exfiltration to deny). `direct` asks allow / ask / deny outright and is **not benchmarked**.

Response:

```json
{
  "choice": "allow",
  "probabilities": {"allow": 0.885, "ask": 0.081, "deny": 0.033},
  "confidence": 0.885,
  "policy_action": "ask",
  "policy_trace": [
    "Model: allow (0.885); P(allow)=0.885, threshold 0.995",
    "Model allow is below the auto-allow threshold: escalate",
    "Final policy action: ask"
  ],
  "mode": "risk_posture",
  "posture_probabilities": {"destructive": 0.048, "exfiltration": 0.033, "privileged": 0.033, "readonly": 0.885},
  "detections": [],
  "model_version": "spark-s1-4b-v3",
  "model_mode": "live",
  "latency_ms": 66.8
}
```

`choice`, `probabilities`, `confidence` and `posture_probabilities` are `null` and `model_mode` is `policy_only` when no model ran. `policy_action` is what a harness should do; it is never less restrictive than the deterministic rules.

**Calibration.** Probabilities are the softmax of the option logits divided by a temperature stored with the checkpoint (`calibration.json`). Temperatures are per question type (Choice, Score, Boolean), global across tasks, and only the Choice temperature has been fitted (on the os-datagen calibration split). They are not task-specific and are not guaranteed to transfer: see [MODEL_CARD.md](../MODEL_CARD.md#calibration).

```bash
curl -s localhost:8400/v1/gate -H 'content-type: application/json' \
  -d '{"state": {"tool": "bash", "command": "rm -rf ./build-cache"}, "model": "spark-s1-4b-v3"}'
```

```python
import httpx
r = httpx.post("http://127.0.0.1:8400/v1/gate", json={"state": {"tool": "bash", "command": "aws s3 sync ./reports s3://company-backup/"}})
print(r.json()["policy_action"], r.json()["policy_trace"])
```

## `POST /v1/decide`

Typed questions over one state; every question reuses the state prefix.

```json
{
  "state": {"message": "I was charged twice for my subscription. Please reverse the duplicate charge."},
  "questions": [
    {"id": "department", "type": "choice", "instructions": "Which team should handle this request?",
     "options": [{"id": "billing", "definition": "Charges, invoices, refunds."}, {"id": "technical", "definition": "Bugs or outages."}]},
    {"id": "refund_requested", "type": "boolean", "instructions": "Does the customer explicitly request money back?"},
    {"id": "urgency", "type": "score", "instructions": "How urgent is this?", "levels": [{"value": 0, "definition": "Routine"}, {"value": 1, "definition": "Soon"}]}
  ],
  "model": "spark-s1-4b-v3"
}
```

Response: `{"model", "latency_ms", "decisions": {"<id>": {"selected", "probabilities", "confidence", "margin", "entropy", "latency_ms"}}}`. Boolean questions use the option ids `true` and `false`; Score options are the level values as strings. `latency_ms` per decision is the request time divided by the number of questions.

Supported: `choice` (2 to 26 options, definitions optional), `boolean`, `score` (2 to 26 ordered levels). Not supported: free text, ranking, multi-select, more than 26 options. The native `DecisionRequest` shape (`schema.py`) still works on the same path.

## `POST /v1/evaluate` (Jev-compatible)

Accepts `{"state", "questions": {name: {type, instructions, criteria}}}` and returns `{"answers": {name: {...}}}` so client code written for Jev's hosted API can point at a local server with a URL change. Scope: `noul`/`boolean`, `choice` and `score` only; response fields follow the same names but this is a re-implementation, not a guarantee of identical behaviour. See `examples/jev_compatible_client.py`.

## Python

```python
from open_spark_jev import classify_tool_call
r = classify_tool_call(tool="bash", command="git status", model="checkpoints/spark-s1-4b-v3")
print(r.choice, r.probabilities, r.policy_action)
```

For typed questions use `MenuScorer.decide(State(...), [Choice(...), Noul(...), Score(...)])` (`open_spark_jev.model`, `open_spark_jev.schema`); see `examples/routing.py`. Invalid input raises `ValueError`; nothing is executed.
