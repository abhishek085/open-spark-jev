"""Backends and clients.

``OpenAICompletionsBackend`` reproduces menu scoring against *any* OpenAI-compatible server
(trtllm-serve, vLLM, SGLang). Default ``mode="chat"``: ``/v1/chat/completions`` with the
system+user messages from ``prompting.render_messages``, ``chat_template_kwargs=
{"enable_thinking": false}``, ``max_tokens=1``, ``logprobs=true``, ``top_logprobs=N``. The
rendered prompt is byte-identical to what HF sees, the server returns the top-N first-token
logprobs, we pick the label tokens out of them and renormalise. ``mode="completions"`` sends
the raw prompt to ``/v1/completions`` for servers that support ``logprobs`` there (vLLM,
SGLang; trtllm-serve 1.2.1 does not).

Caveat: OpenAI-style ``top_logprobs`` is capped (20 on trtllm-serve). Labels outside the top-N
get ``floor_logprob``; for menus with more than ~15 options or exact full-vocab gathering use
``serve/trtllm_backend.py`` (in-container TRT-LLM Python API) or the HF backend.

Prefix caching on the server (TRT-LLM ``enable_block_reuse``) plays the role of our in-process
state cache: all questions for one state share the prefix, so send them together.

``GatewayClient`` talks to the open-spark-Jev gateway's ``/v1/decide``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import httpx

from ..model import Calibration
from ..prompting import label_texts, render_messages, render_prompt
from ..schema import Answer, Question, State


class OpenAICompletionsBackend:
    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        calibration: Calibration | None = None,
        top_logprobs: int = 20,
        timeout: float = 60.0,
        api_key: str = "EMPTY",
        floor_logprob: float = -30.0,
        mode: str = "chat",
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})
        self.model = model or self.client.get("/models").json()["data"][0]["id"]
        self.calibration = calibration or Calibration()
        self.top_logprobs = top_logprobs
        self.floor = floor_logprob
        self.mode = mode
        self.name = self.model

    def _top_logprobs(self, state: State, q: Question) -> dict[str, float]:
        if self.mode == "chat":
            body = {
                "model": self.model,
                "messages": render_messages(state, q),
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": self.top_logprobs,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            r = self.client.post("/chat/completions", json=body)
            r.raise_for_status()
            content = r.json()["choices"][0]["logprobs"]["content"]
            return {t["token"]: float(t["logprob"]) for t in content[0]["top_logprobs"]} if content else {}
        body = {"model": self.model, "prompt": render_prompt(state, q), "max_tokens": 1, "temperature": 0.0, "logprobs": self.top_logprobs}
        r = self.client.post("/completions", json=body)
        r.raise_for_status()
        lp = r.json()["choices"][0]["logprobs"]
        return {k: float(v) for k, v in ((lp.get("top_logprobs") or [{}])[0]).items()}

    def _label_logits(self, state: State, q: Question, labels: list[str]) -> list[float]:
        top = self._top_logprobs(state, q)
        return [top[t] if t in top else self.floor for t in labels]

    def decide(
        self, state: State, questions: Sequence[Question], temperature: float | None = None, return_logits: bool = False
    ) -> list[Answer]:
        answers = []
        for q in questions:
            z = self._label_logits(state, q, label_texts(q))
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
