# Integrations

Small, safe examples. In every one the model decides whether a proposed call should proceed; **it never executes a tool**. Keep deterministic policy, least privilege and human approval in front of anything with side effects ([SECURITY.md](../SECURITY.md)).

## 1. Generic Python agent loop

```python
from open_spark_jev import classify_tool_call

def run_tool(name, command):            # your executor: sandboxed, least privilege, audited
    ...

def step(tool, command, ask_human):
    r = classify_tool_call(tool=tool, command=command, model="checkpoints/spark-s1-4b-v3")
    if r.policy_action == "allow":
        return run_tool(tool, command)
    if r.policy_action == "ask" and ask_human(command, r.policy_trace):
        return run_tool(tool, command)
    return {"blocked": True, "why": r.policy_trace}      # deny, or ask refused
```

With `model="none"` the call uses the rules only, so `allow` is never returned and everything unsupported goes to a human.

## 2. `examples/tool_gate.py`

Runs the ten Decision Lab fixtures through the gate in rules-only mode (no weights, nothing executed): `python examples/tool_gate.py`. Pass `--model checkpoints/spark-s1-4b-v3` for live decisions.

## 3. `examples/agent_gate.py`

The earlier calibration-aware gate: a Boolean question ("executing the next call is unsafe or irreversible") turned into `ACT`, `REVIEW_WITH_LLM` or `ESCALATE_TO_HUMAN` by two thresholds. It loads the untrained Qwen3-1.7B path by default; point it at a spark-s1 checkpoint for meaningful numbers. Kept as is.

## 4. `examples/routing.py`

Routes a support ticket, scores urgency and checks a claim from one state (one cached prefix): `python examples/routing.py`.

## 5. `examples/jev_compatible_client.py`

Uses the Jev-style wire format against the local gateway (`POST /v1/evaluate`). See [API.md](API.md#post-v1evaluate-jev-compatible) for the compatibility scope and deviations.

## 6. llama.cpp and Ollama (GGUF builds)

The gateway can sit in front of a GGUF build instead of vLLM, without torch (`pip install -e ".[serve]"`):

```bash
llama-server -hf abhishek085/spark-s1-4b-v6-GGUF:Q8_0 -c 16384 --port 8080
python -m open_spark_jev.serve.gateway --backend openai --openai-mode completions --upstream http://localhost:8080/v1 --model <dir with calibration.json>

ollama pull hf.co/abhishek085/spark-s1-4b-v6-GGUF:Q8_0      # Ollama >= 0.12 (logprobs)
python -m open_spark_jev.serve.gateway --backend ollama --upstream http://localhost:11434 --upstream-model hf.co/abhishek085/spark-s1-4b-v6-GGUF:Q8_0 --model <dir with calibration.json>
```

Both read the top 20 first-token logprobs, so menus with more than 20 options need the HF backend. Use Q8_0: Q4_K_M keeps accuracy but
not calibration. Builds, parity numbers and listings: [release/local-runtimes/README.md](../release/local-runtimes/README.md).

## 7. MCP tool-approval (design sketch, not implemented)

A wrapper MCP server would intercept `tools/call` requests, send `{tool, arguments}` to `POST /v1/gate`, forward the call to the real server only when `policy_action == "allow"`, and return a structured refusal (`ask` or `deny`, with `policy_trace`) otherwise. The sketch is deliberately not shipped: an interceptor must fail closed on timeouts and unknown tools, log decisions, and never proxy on `ask` without an approval channel. Contributions welcome, see [CONTRIBUTION_IDEAS.md](CONTRIBUTION_IDEAS.md).
