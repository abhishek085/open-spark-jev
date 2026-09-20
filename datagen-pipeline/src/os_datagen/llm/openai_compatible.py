from __future__ import annotations

import itertools
import threading
import time
from typing import Any

import httpx

from ..config import ModelConfig
from .client import LLMRequest, LLMResponse


class LLMError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Minimal /v1/chat/completions client. Works with vLLM, SGLang, TRT-LLM serve, llama.cpp server."""

    def __init__(self, cfg: ModelConfig, retries: int = 2):
        self.cfg = cfg
        self.model_name = cfg.model
        self.urls = [u.rstrip("/") for u in (cfg.base_urls or [cfg.base_url])]
        self.base_url = self.urls[0]
        self._rr = itertools.count()
        self._lock = threading.Lock()
        self.retries = retries
        self._http = httpx.Client(timeout=cfg.timeout_s, headers={"Authorization": f"Bearer {cfg.api_key()}"})

    def _payload(self, req: LLMRequest, mode: str) -> dict[str, Any]:
        p: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
        }
        if req.seed is not None:
            p["seed"] = req.seed
        if self.cfg.disable_thinking:
            p["chat_template_kwargs"] = {"enable_thinking": False}
        if self.cfg.json_mode:
            if mode == "json_schema" and req.json_schema:
                p["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": req.json_schema, "strict": True},
                }
            else:
                p["response_format"] = {"type": "json_object"}
        return p

    def chat(self, req: LLMRequest) -> LLMResponse:
        with self._lock:
            base = self.urls[next(self._rr) % len(self.urls)]  # round-robin over instances of the same model
        mode = "json_schema" if (self.cfg.json_schema_mode and req.json_schema) else "json_object"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                r = self._http.post(f"{base}/chat/completions", json=self._payload(req, mode))
                if r.status_code == 400 and mode == "json_schema":
                    mode = "json_object"  # endpoint rejected guided schema; degrade once
                    continue
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage") or {}
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"] or "",
                    model=data.get("model", self.cfg.model),
                    latency_ms=int((time.time() - t0) * 1000),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
            except (httpx.HTTPError, KeyError, ValueError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"{self.base_url}: {last}")

    def healthy(self) -> tuple[bool, str]:
        """True only when EVERY configured instance is up and serving the model."""
        info = []
        for u in self.urls:
            try:
                r = self._http.get(f"{u}/models", timeout=5)
                r.raise_for_status()
                ids = [m.get("id") for m in r.json().get("data", [])]
                if ids and self.cfg.model not in ids:
                    return False, f"{u} serves {ids}"
                info.append(f"{u} ok")
            except Exception as e:  # noqa: BLE001
                return False, f"{u}: {e}"
        return True, "; ".join(info)
