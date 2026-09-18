"""The System One gateway: a typed JSON decision API in front of the backbone.

Endpoints
---------
POST /v1/decide      native schema (DecisionRequest -> DecisionResponse)
POST /v1/evaluate    Jev-compatible wire format: {"state", "questions": {name: {type, instructions,
                     criteria}}} -> {"answers": {name: {...}}, "usage": {...}}. Lets code written for
                     TypeSafe's hosted API point at a local open-spark-Jev with a URL change.
GET  /v1/models, /healthz

Backends (``--backend``):
  hf       in-process MenuScorer (dev / small-batch)
  openai   an OpenAI-compatible chat server, normally trtllm-serve on the same Spark (production)
  trtllm   in-process TensorRT-LLM Python API (run inside the TRT-LLM container; full-vocab logits)

Run:
  python -m open_spark_jev.serve.gateway --backend openai --upstream http://localhost:8355/v1 --port 8400
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..schema import Answer, Choice, DecisionRequest, DecisionResponse, Noul, Score, State

log = logging.getLogger("osj.gateway")
app = FastAPI(title="open-spark-Jev", version="0.1.0")
BACKEND: Any = None
BACKEND_KIND = "none"


class JevQuestion(BaseModel):
    type: str
    instructions: str
    criteria: Any = None  # noul: {"true":..,"false":..} | choice: {opt: desc} | score: [desc, ...]


class JevRequest(BaseModel):
    state: Any
    questions: dict[str, JevQuestion]
    model: str | None = None


def _from_jev(name: str, jq: JevQuestion):
    if jq.type == "noul":
        return Noul(id=name, prompt=jq.instructions)
    if jq.type == "choice":
        if not isinstance(jq.criteria, dict):
            raise HTTPException(400, f"{name}: choice needs criteria map")
        return Choice(id=name, prompt=jq.instructions + "\n" + "\n".join(f"- {k}: {v}" for k, v in jq.criteria.items()), options=list(jq.criteria))
    if jq.type == "score":
        if not isinstance(jq.criteria, list):
            raise HTTPException(400, f"{name}: score needs criteria list")
        return Score(id=name, prompt=jq.instructions, levels=[str(i) for i in range(len(jq.criteria))],
                     rubric="\n".join(f"{i}: {d}" for i, d in enumerate(jq.criteria)))
    raise HTTPException(400, f"{name}: unknown type {jq.type}")


def _to_jev(a: Answer, q) -> dict[str, Any]:
    if a.type == "noul":
        return {"type": "noul", "noul": a.probability}
    if a.type == "choice":
        return {"type": "choice", "choice": a.selected, "probabilities": dict(zip(a.labels, a.probs)), "confidence": a.confidence}
    legend = {str(i): lvl for i, lvl in enumerate(q.labels)}
    return {"type": "score", "score": a.expected_value, "legend": legend,
            "probabilities": dict(zip(a.labels, a.probs)), "confidence": a.confidence}


@app.get("/healthz")
def healthz():
    return {"ok": BACKEND is not None, "backend": BACKEND_KIND}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": getattr(BACKEND, "name", "open-spark-jev"), "object": "model"}]}


@app.post("/v1/decide", response_model=DecisionResponse)
def decide(req: DecisionRequest):
    if BACKEND is None:
        raise HTTPException(503, "backend not loaded")
    qs = req.parsed_questions()
    t0 = time.perf_counter()
    answers = BACKEND.decide(req.state, qs, temperature=req.temperature, return_logits=req.return_logits)
    ms = (time.perf_counter() - t0) * 1000
    return DecisionResponse(answers=answers, model=getattr(BACKEND, "name", "?"), latency_ms=ms, state_tokens=-1, backend=BACKEND_KIND)


@app.post("/v1/evaluate")
def evaluate(req: JevRequest):
    if BACKEND is None:
        raise HTTPException(503, "backend not loaded")
    state = State(content=req.state)
    names = list(req.questions)
    qs = [_from_jev(n, req.questions[n]) for n in names]
    answers = BACKEND.decide(state, qs)
    return {"model": getattr(BACKEND, "name", "open-spark-jev"),
            "answers": {n: _to_jev(a, q) for n, a, q in zip(names, answers, qs)},
            "usage": {"input_tokens": None, "output_tokens": 0}}


def build_backend(kind: str, model: str | None, upstream: str | None):
    global BACKEND, BACKEND_KIND
    if kind == "hf":
        from ..model import MenuScorer

        BACKEND = MenuScorer(model or os.environ.get("OSJ_MODEL", "models/Qwen3-1.7B"))
    elif kind == "trtllm":
        from ..model import Calibration
        from .trtllm_backend import TRTLLMBackend

        path = model or os.environ.get("OSJ_MODEL", "models/Qwen3-1.7B")
        cal = Calibration.load(os.path.join(path, "calibration.json")) if os.path.exists(os.path.join(path, "calibration.json")) else None
        BACKEND = TRTLLMBackend(path, calibration=cal)
    else:
        from ..model import Calibration
        from .client import OpenAICompletionsBackend

        cal = None
        if model and os.path.exists(os.path.join(model, "calibration.json")):
            cal = Calibration.load(os.path.join(model, "calibration.json"))
        BACKEND = OpenAICompletionsBackend(upstream or "http://localhost:8355/v1", calibration=cal)
    BACKEND_KIND = kind


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai", "trtllm"], default="openai")
    ap.add_argument("--model", help="HF path (also used to load calibration.json for openai backend)")
    ap.add_argument("--upstream", default="http://localhost:8355/v1")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8400)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_backend(a.backend, a.model, a.upstream)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
