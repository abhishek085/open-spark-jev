from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    messages: list[dict[str, str]]
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1000
    seed: int | None = None
    json_schema: dict[str, Any] | None = None  # optional guided-decoding schema
    # Local-only side channel for fake clients; NEVER serialized onto the wire.
    metadata: dict[str, Any] = Field(default_factory=dict, exclude=True)

    model_config = {"arbitrary_types_allowed": True}


class LLMResponse(BaseModel):
    text: str
    model: str
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    model_name: str
    base_url: str

    def chat(self, req: LLMRequest) -> LLMResponse: ...
