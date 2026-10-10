"""Response parsing for the llama-server and Ollama backends (mocked HTTP; no model is run)."""

import asyncio
import json
import math

import httpx

from open_spark_jev.model import Calibration
from open_spark_jev.schema import Choice, State
from open_spark_jev.serve.client import OllamaBackend, OpenAICompletionsBackend

STATE = State(content={"tool": "bash", "command": "ls"})
Q = Choice(prompt="What should happen?", options=["allow", "ask", "deny"])
LOGPROBS = {"A": -0.1, "B": -2.5, "C": -4.0, "I": -3.0}  # "I" is a non-label token the server also returns


def _top(fmt: str) -> list[dict]:
    return [{"token": t, "logprob": v, "bytes": list(t.encode())} for t, v in LOGPROBS.items()]


def _llama_server(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/v1/completions"
    body = json.loads(request.content)
    assert body["max_tokens"] == 1 and body["prompt"].endswith("</think>\n\n")
    top = _top("llama")
    return httpx.Response(200, json={"choices": [{"text": "A", "logprobs": {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": top}]}}],
                                     "usage": {"prompt_tokens": 42}})


def _ollama(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/generate"
    body = json.loads(request.content)
    assert body["raw"] is True and body["options"]["num_predict"] == 1 and body["top_logprobs"] <= 20
    return httpx.Response(200, json={"response": "A", "prompt_eval_count": 42,
                                     "logprobs": [{"token": "A", "logprob": -0.1, "top_logprobs": _top("ollama")}]})


def _expected(t: float) -> list[float]:
    z = [LOGPROBS["A"], LOGPROBS["B"], LOGPROBS["C"]]
    e = [math.exp((x - max(z)) / t) for x in z]
    return [x / sum(e) for x in e]


def _check(backend) -> None:
    a = backend.decide(STATE, [Q])[0]
    assert a.selected == "allow"
    assert all(abs(p - q) < 1e-6 for p, q in zip(a.probs, _expected(1.5)))
    assert backend.last_prompt_tokens == 42
    answers, n = asyncio.run(backend.adecide(STATE, [Q, Q]))
    assert n == 84 and all(x.selected == "allow" for x in answers)


def _cal() -> Calibration:
    return Calibration(temperature={"choice": 1.5})


def test_llama_server_completions_mode():
    b = OpenAICompletionsBackend("http://x/v1", model="m", mode="completions", calibration=_cal())
    b.client = httpx.Client(base_url="http://x/v1", transport=httpx.MockTransport(_llama_server))
    b.aclient = httpx.AsyncClient(base_url="http://x/v1", transport=httpx.MockTransport(_llama_server))
    _check(b)


def test_legacy_completions_shape_still_parses():
    b = OpenAICompletionsBackend("http://x/v1", model="m", mode="completions")
    top = b._parse({"choices": [{"logprobs": {"top_logprobs": [{"A": -0.1, "B": -2.0}]}}]})
    assert top == {"A": -0.1, "B": -2.0}


def test_ollama_backend():
    b = OllamaBackend("http://x:11434/v1", model="spark-s1-4b-v6:q8_0", calibration=_cal(), top_logprobs=50)
    assert b.base_url == "http://x:11434" and b.top_logprobs == 20
    b.client = httpx.Client(base_url=b.base_url, transport=httpx.MockTransport(_ollama))
    b.aclient = httpx.AsyncClient(base_url=b.base_url, transport=httpx.MockTransport(_ollama))
    _check(b)
