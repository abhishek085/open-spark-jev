"""Backends and clients.

``OpenAICompletionsBackend`` reproduces menu scoring against *any* OpenAI-compatible server
(trtllm-serve, vLLM, SGLang). Default ``mode="chat"``: ``/v1/chat/completions`` with the
system+user messages from ``prompting.render_messages``, ``chat_template_kwargs=
{"enable_thinking": false}``, ``max_tokens=1``, ``logprobs=true``, ``top_logprobs=N``. The
rendered prompt is byte-identical to what HF sees, the server returns the top-N first-token
logprobs, we pick the label tokens out of them and renormalise. ``mode="completions"`` sends
the raw prompt to ``/v1/completions`` for servers that support ``logprobs`` there (vLLM,
SGLang, llama.cpp ``llama-server``; trtllm-serve 1.2.1 does not). ``OllamaBackend`` does the
same over Ollama's native ``/api/generate`` (``raw`` prompt + ``top_logprobs``).

Caveat: OpenAI-style ``top_logprobs`` is capped (20 on trtllm-serve). Labels outside the top-N
get ``floor_logprob``; for menus with more than ~15 options or exact full-vocab gathering use
``serve/trtllm_backend.py`` (in-container TRT-LLM Python API) or the HF backend.

Prefix caching on the server (TRT-LLM ``enable_block_reuse``) plays the role of our in-process
state cache: all questions for one state share the prefix, so send them together.

``GatewayClient`` talks to the open-spark-Jev gateway's ``/v1/decide``.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

import httpx

from ..calibration import Calibration
from ..prompting import label_texts, render_messages, render_prompt
from ..schema import Answer, Question, State


class OpenAICompletionsBackend:
    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        calibration: Calibration | None = None,
        top_logprobs: int = 20,
        timeout: float = 900.0,
        api_key: str = "EMPTY",
        floor_logprob: float = -30.0,
        mode: str = "chat",
        max_state_chars: int = 12_000,
    ):
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)
        self.aclient = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, headers=headers,
                                         limits=httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=300))
        self.last_prompt_tokens = 0
        self.model = model or self.client.get("/models").json()["data"][0]["id"]
        self.calibration = calibration or Calibration()
        self.top_logprobs = top_logprobs
        self.floor = floor_logprob
        self.mode = mode
        self.max_state_chars = max_state_chars  # the model was trained with 12k-char states; raise only after checking calibration
        self.name = self.model

    def _request(self, state: State, q: Question) -> tuple[str, dict]:
        if self.mode == "chat":
            return "/chat/completions", {
                "model": self.model,
                "messages": render_messages(state, q, max_chars=self.max_state_chars),
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": self.top_logprobs,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        return "/completions", {"model": self.model, "prompt": render_prompt(state, q, max_chars=self.max_state_chars), "max_tokens": 1, "temperature": 0.0, "logprobs": self.top_logprobs}

    def _parse(self, data: dict) -> dict[str, float]:
        lp = data["choices"][0]["logprobs"]
        # llama-server returns the chat-style ``content[]`` shape on /v1/completions too
        if self.mode == "chat" or "content" in lp:
            content = lp["content"]
            return {t["token"]: float(t["logprob"]) for t in content[0]["top_logprobs"]} if content else {}
        return {k: float(v) for k, v in ((lp.get("top_logprobs") or [{}])[0]).items()}

    def _top_logprobs(self, state: State, q: Question) -> dict[str, float]:
        path, body = self._request(state, q)
        r = self.client.post(path, json=body)
        r.raise_for_status()
        data = r.json()
        self.last_prompt_tokens = int((data.get("usage") or {}).get("prompt_tokens") or 0)
        return self._parse(data)

    def _label_logits(self, state: State, q: Question, labels: list[str]) -> list[float]:
        top = self._top_logprobs(state, q)
        return [top[t] if t in top else self.floor for t in labels]

    # --- async path: one pooled keep-alive connection set, all questions of a state in flight at once ---
    async def _alabel_logits(self, state: State, q: Question) -> tuple[list[float], int]:
        path, body = self._request(state, q)
        r = await self.aclient.post(path, json=body)
        r.raise_for_status()
        data = r.json()
        top = self._parse(data)
        return [top.get(t, self.floor) for t in label_texts(q)], int((data.get("usage") or {}).get("prompt_tokens") or 0)

    async def adecide(
        self, state: State, questions: Sequence[Question], temperature: float | None = None, return_logits: bool = False
    ) -> tuple[list[Answer], int]:
        """Concurrent ``decide``. Returns (answers, total prompt tokens the server processed)."""
        res = await asyncio.gather(*(self._alabel_logits(state, q) for q in questions))
        answers = [self._answer(q, z, temperature, return_logits) for q, (z, _) in zip(questions, res)]
        return answers, sum(n for _, n in res)

    async def aclose(self) -> None:
        await self.aclient.aclose()

    def _answer(self, q: Question, z: list[float], temperature: float | None, return_logits: bool) -> Answer:
        t = temperature if temperature is not None else self.calibration.t(q.type)
        m = max(z)
        e = [math.exp((x - m) / t) for x in z]
        s = sum(e)
        return Answer.from_probs(q, [x / s for x in e], raw_logits=z if return_logits else None)

    def decide(
        self, state: State, questions: Sequence[Question], temperature: float | None = None, return_logits: bool = False
    ) -> list[Answer]:
        return [self._answer(q, self._label_logits(state, q, label_texts(q)), temperature, return_logits) for q in questions]


class OllamaBackend(OpenAICompletionsBackend):
    """Menu scoring against Ollama's native ``/api/generate`` (Ollama >= 0.12 for logprobs).

    ``raw=true`` sends the rendered prompt untouched (no Modelfile template), ``num_predict=1``,
    ``logprobs=true``, ``top_logprobs<=20`` (Ollama's cap). Same top-N caveat as the OpenAI path.
    ``base_url`` is the Ollama root (``http://localhost:11434``), not ``/v1``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "spark-s1-4b-v6:q8_0",
        calibration: Calibration | None = None,
        top_logprobs: int = 20,
        timeout: float = 900.0,
        floor_logprob: float = -30.0,
        max_state_chars: int = 12_000,
        num_ctx: int = 16_384,
    ):
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.aclient = httpx.AsyncClient(base_url=self.base_url, timeout=timeout,
                                         limits=httpx.Limits(max_connections=64, max_keepalive_connections=32, keepalive_expiry=300))
        self.last_prompt_tokens = 0
        self.model = model
        self.calibration = calibration or Calibration()
        self.top_logprobs = min(top_logprobs, 20)
        self.floor = floor_logprob
        self.mode = "ollama"
        self.max_state_chars = max_state_chars
        self.num_ctx = num_ctx  # Ollama's default context is small; a longer state would be silently truncated
        self.name = self.model

    def _request(self, state: State, q: Question) -> tuple[str, dict]:
        return "/api/generate", {
            "model": self.model,
            "prompt": render_prompt(state, q, max_chars=self.max_state_chars),
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "options": {"num_predict": 1, "temperature": 0, "num_ctx": self.num_ctx},
        }

    def _parse(self, data: dict) -> dict[str, float]:
        lp = data.get("logprobs")
        if not lp:
            raise RuntimeError(f"Ollama returned no logprobs (needs Ollama >= 0.12): {str(data)[:200]}")
        return {t["token"]: float(t["logprob"]) for t in lp[0]["top_logprobs"]}

    def _top_logprobs(self, state: State, q: Question) -> dict[str, float]:
        path, body = self._request(state, q)
        r = self.client.post(path, json=body)
        r.raise_for_status()
        data = r.json()
        self.last_prompt_tokens = int(data.get("prompt_eval_count") or 0)
        return self._parse(data)

    async def _alabel_logits(self, state: State, q: Question) -> tuple[list[float], int]:
        path, body = self._request(state, q)
        r = await self.aclient.post(path, json=body)
        r.raise_for_status()
        data = r.json()
        top = self._parse(data)
        return [top.get(t, self.floor) for t in label_texts(q)], int(data.get("prompt_eval_count") or 0)


class GatewayClient:
    def __init__(self, base_url: str = "http://localhost:8400/v1", timeout: float = 60.0):
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.name = "gateway"

    def decide(self, state: State, questions: Sequence[Question], **kw) -> list[Answer]:
        body = {"state": state.model_dump(), "questions": [q.model_dump() for q in questions]}
        r = self.client.post("/decide", json=body)
        r.raise_for_status()
        return [Answer(**a) for a in r.json()["answers"]]
