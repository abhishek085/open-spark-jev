# Open Spark Jev

**A small, fast, open model that makes yes/no/route decisions for AI agents, on your own machine.** The model is called **`spark-s1`**. Open Spark Jev is part of **Nokast**, an open-source AI community.

Give it what an agent is about to do (a shell command, a support ticket, a request) and a fixed list of choices. It answers with the choice and a probability for every option in about 30-70 ms, without writing any text.

> **Very early release.** `spark-s1` is trained on under a thousand examples. It is close to Jev on some tests and clearly behind on others (charts below), and we are adding more data and capabilities. Use it behind safety checks, never as the only gate.

## Quickstart

Download the model and run it as a local API in one command:

```bash
git clone https://github.com/abhishek085/open-spark-jev && cd open-spark-jev
python3 -m venv .venv && . .venv/bin/activate
pip install torch -e ".[infer]"
osj serve --pull spark-s1-4b-v3        # downloads from Hugging Face once (about 8 GB), then serves http://127.0.0.1:8400
```

In another terminal, ask it about a command (nothing is ever executed; the command is just text):

```bash
curl -s localhost:8400/v1/gate -H 'content-type: application/json' \
  -d '{"state": {"tool": "bash", "command": "kubectl get pods -n production"}}'
```

```json
{
  "choice": "allow",
  "probabilities": {"allow": 0.948, "ask": 0.036, "deny": 0.016},
  "confidence": 0.948,
  "policy_action": "ask",
  "policy_trace": ["Model: allow (0.948); P(allow)=0.948, threshold 0.995",
                   "Model allow is below the auto-allow threshold: escalate",
                   "Final policy action: ask"],
  "latency_ms": 65.0
}
```

*(Example output from a real run of `spark-s1-4b-v3`, trimmed.)* `policy_action` is what your agent should do. The default threshold is strict: `allow` needs 99.5% confidence, otherwise the agent asks a human. Lower it per request with `"policy": {"auto_allow_threshold": 0.9}`.

Use `spark-s1-1.7b-v3` instead for a smaller (3.4 GB) and faster model. From Python, no server:

```python
from open_spark_jev import classify_tool_call
r = classify_tool_call(tool="bash", command="kubectl get pods -n production", model="checkpoints/spark-s1-4b-v3")
print(r.policy_action, r.probabilities)
```

You need a GPU for the speeds below (tested on an NVIDIA DGX Spark; other GPUs should work but are untested). More: [docs/API.md](docs/API.md).

## Decision Lab (optional demo)

```bash
pip install -e ".[serve]" && osj lab       # open http://127.0.0.1:8400/lab   (no model download needed)
```

![spark-s1 Decision Lab](docs/img/decision-lab.png)

A demo page for trying commands: pick a sample or type your own, and see the model's probabilities next to the safety rules that can override them. Without a model it uses recorded outputs, clearly labelled. More: [docs/UI.md](docs/UI.md).

## How good is it?

On **Kev's out-of-domain test suite**, made by someone else on data we never trained on, `spark-s1-4b-v3` scores 0.749 (Kev-4B 0.806, Kev-8B 0.780). It is close on some sources and behind on others:

![spark-s1 vs Kev vs Jev](docs/img/kev-comparison.png)

On our **60-call tool-approval set**, directly against Jev's recorded run:

![spark-s1 vs Jev on the 60-call set](docs/img/toolcall-60-comparison.png)

| | Accuracy | Median time per decision |
|---|---:|---:|
| Jev (hosted, recorded) | 0.917 | 421.6 ms |
| `spark-s1-4b-v3` | 0.850 | 65.9 ms |
| `spark-s1-1.7b-v3` | 0.783 | 29.6 ms |

The 60-call set is small (about ±5 points) and was looked at during development, so it is a diagnostic, not a final test. Jev's time includes the network, so the speed gap is not a like-for-like comparison. Known weak spots: commands that weaken security settings, copy credentials out, or hide a cloud copy as a "backup". More detail: [MODEL_CARD.md](MODEL_CARD.md).

## Safety

> [!WARNING]
> Never use a model decision as the only control before running a tool. Keep permission checks, least-privilege credentials, logging and human approval for sensitive actions. Anything low-confidence or unrecognised should go to a human, not run.

The built-in rules (`open_spark_jev/policy.py`) are simple and incomplete. They block obvious cases, such as sending a secrets file over the network, and can never be overridden by the model.

## Learn more

[Model card](MODEL_CARD.md) · [API](docs/API.md) · [Use it in your agent](docs/INTEGRATIONS.md) · [How it works](docs/ARCHITECTURE.md) · [Benchmarks and method](docs/BENCHMARKS.md) · [Roadmap](docs/ROADMAP.md) · [All docs](docs/README.md)

## Contribute

We want integrations, benchmark results and examples of wrong decisions. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/CONTRIBUTION_IDEAS.md](docs/CONTRIBUTION_IDEAS.md).

## License

Apache-2.0 for our code, docs and data tools ([LICENSE](LICENSE)). Models, datasets and test suites from others keep their own licenses ([NOTICE](NOTICE)). Independent of and not endorsed by TypeSafe AI (Jev), Kev's author, or NVIDIA.
