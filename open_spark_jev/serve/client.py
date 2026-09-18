"""Backends and clients.

``OpenAICompletionsBackend`` reproduces menu scoring against *any* OpenAI-compatible server
(trtllm-serve, vLLM, SGLang) using ``/v1/completions`` with ``max_tokens=1`` and ``logprobs``:
the prompt is the exact same text HF sees, the server returns top-k next-token logprobs, we
pick the label tokens out of them and renormalise. Prefix caching on the server (TRT-LLM
``enable_block_reuse``) plays the role of our in-process state cache: all questions for one
state share the prefix, so sending them together is what makes it fast.

``GatewayClient`` talks to the open-spark-Jev gateway's ``/v1/decide``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import httpx

from ..model import Calibration
from ..prompting import label_texts, render_prompt
from ..schema import Answer, Question, State


class OpenAICompletionsBackend:
    def __init__(self, base_url: str, model: str | None = None, calibration: Calibration | None = None,
                 top_logprobs: int = 30, timeout: float = 60.0, api_key: str = "EMPTY", floor_logprob: float = -30.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self.model = model or self.client.get("/models").json()["data"][0]["id"]
        self.calibration = calibration or Calibration()
        self.top_logprobs = top_logprobs
        self.floor = floor_logprob
        self.name = self.model

    def _label_logits(self, prompt: str, labels: list[str]) -> list[float]:
        body = {"model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0, "logprobs": self.top_logprobs}
        r = self.client.post("/completions", json=body)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        lp = ch["logprobs"]
        top = (lp.get("top_logprobs") or [{}])[0]
        # Servers differ on whether keys are token strings or "token:id"; match on string.
        out = []
        for t in labels:
            v = top.get(t)
            if v is None:  # some servers strip the leading space in keys
                v = top.get(t.strip())
            out.append(float(v) if v is not None else self.floor)
        return out

    def decide(self, state: State, questions: Sequence[Question], temperature: float | None = None, return_logits: bool = False) -> list[Answer]:
        answers = []
        for q in questions:
            z = self._label_logits(render_prompt(state, q), label_texts(q))
            t = temperature if temperature is not None else self.calibration.t(q.type)
            m = max(z)
            e = [math.exp((x - m) / t) for x in z]
            s = sum(e)
            answers.append(Answer.from_probs(q, [x / s for x in e], raw_logits=z if return_logits else None))
        return answers


class GatewayClient:
    def __init__(self, base_url: str = "http://localhost:8400/v1", timeout: float = 60.0):
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.name = "gateway"

    def decide(self, state: State, questions: Sequence[Question], **kw) -> list[Answer]:
        body = {"state": state.model_dump(), "questions": [q.model_dump() for q in questions]}
        r = self.client.post("/decide", json=body)
        r.raise_for_status()
        return [Answer(**a) for a in r.json()["answers"]]
