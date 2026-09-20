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
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..gate import GateRequest
from ..schema import Answer, Choice, DecisionRequest, DecisionResponse, Noul, Score, State

log = logging.getLogger("osj.gateway")
app = FastAPI(title="open-spark-Jev", version="0.1.0")
BACKEND: Any = None
BACKEND_KIND = "none"

# --- multi-model support (hf backend): several checkpoints selectable per request, loaded lazily ---
UI_DIR = Path(__file__).parent / "ui"
CHECKPOINT_ROOT = "checkpoints"
REGISTRY_FILE = "configs/serve/models.yaml"
DEFAULT_MODEL: str | None = None
MAX_LOADED = int(os.environ.get("OSJ_MAX_LOADED", "2"))
_LOADED: OrderedDict[str, Any] = OrderedDict()
_LOAD_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()


def discover_models() -> dict[str, dict]:
    """id -> {path, name, note}. Registry entries whose weights exist, plus any other checkpoint dir."""
    meta: dict[str, dict] = {}
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE) as f:
            meta = (yaml.safe_load(f) or {}).get("models", {})
    found: dict[str, dict] = {}
    for mid, m in meta.items():
        path = m.get("path") or os.path.join(CHECKPOINT_ROOT, mid)
        if os.path.exists(os.path.join(path, "config.json")):
            found[mid] = {"path": path, "name": m.get("name", mid), "note": m.get("note", "")}
    vroot = os.path.join(CHECKPOINT_ROOT, "variants")
    if os.path.isdir(vroot):  # saved architecture variants (experimental/variants.py)
        for d in sorted(os.listdir(vroot)):
            p = os.path.join(vroot, d)
            if os.path.exists(os.path.join(p, "variant.json")):
                mid = f"variants/{d}"
                m = meta.get(mid, {})
                found[mid] = {"path": p, "name": m.get("name", f"Experimental architecture {d.upper()}"), "note": m.get("note", "")}
    if os.path.isdir(CHECKPOINT_ROOT):
        for d in sorted(os.listdir(CHECKPOINT_ROOT)):
            if d == "variants":
                continue
            p = os.path.join(CHECKPOINT_ROOT, d)
            if d not in found and not d.startswith("rm-") and os.path.exists(os.path.join(p, "config.json")):
                found[d] = {"path": p, "name": d, "note": ""}
    return found


def get_backend(model: str | None):
    """Resolve a request's model to a backend. Non-hf backends serve one fixed model."""
    if BACKEND_KIND != "hf":
        return BACKEND
    avail = discover_models()
    mid = model or DEFAULT_MODEL
    if mid not in avail:
        raise HTTPException(404, f"unknown model {mid!r}; available: {list(avail)}")
    with _LOAD_LOCK:
        if mid in _LOADED:
            _LOADED.move_to_end(mid)
            return _LOADED[mid]
        from ..model import MenuScorer

        log.info("loading %s from %s", mid, avail[mid]["path"])
        while len(_LOADED) >= MAX_LOADED:
            old, sc = _LOADED.popitem(last=False)
            log.info("unloading %s", old)
            del sc
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        from ..experimental.variant_scorer import VariantScorer, is_variant

        _LOADED[mid] = VariantScorer(avail[mid]["path"]) if is_variant(avail[mid]["path"]) else MenuScorer(avail[mid]["path"])
        _LOADED[mid].model_id = mid
        _LOADED[mid].release_id = avail[mid].get("name", mid)
        return _LOADED[mid]


class JevQuestion(BaseModel):
    type: str
    instructions: str
    criteria: Any = None  # noul: {"true":..,"false":..} | choice: {opt: desc} | score: [desc, ...]


class JevRequest(BaseModel):
    state: Any
    questions: dict[str, JevQuestion]
    model: str | None = None


def _from_jev(name: str, jq: JevQuestion):
    if jq.type in ("noul", "boolean"):  # "boolean" is the AI SDK / hosted-Jev spelling
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
        return {"type": "noul", "noul": a.probability, "probability": a.probability}
    if a.type == "choice":
        return {"type": "choice", "choice": a.selected, "probabilities": dict(zip(a.labels, a.probs)), "confidence": a.confidence}
    legend = {str(i): lvl for i, lvl in enumerate(q.labels)}
    return {"type": "score", "score": a.expected_value, "legend": legend,
            "probabilities": dict(zip(a.labels, a.probs)), "confidence": a.confidence}


@app.get("/healthz")
def healthz():
    return {"ok": BACKEND is not None or bool(_LOADED), "backend": BACKEND_KIND, "loaded": list(_LOADED)}


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(UI_DIR / "index.html")


@app.get("/v1/models")
def models():
    if BACKEND_KIND != "hf":
        return {"object": "list", "default": None, "data": [{"id": getattr(BACKEND, "name", "open-spark-jev"), "object": "model", "name": "served model", "note": "", "loaded": True}]}
    return {"object": "list", "default": DEFAULT_MODEL,
            "data": [{"id": k, "object": "model", "name": v["name"], "note": v["note"], "loaded": k in _LOADED} for k, v in discover_models().items()]}


@app.post("/v1/decide")
def decide(body: dict[str, Any] = Body(...)):
    from . import contract

    if contract.is_contract(body):
        backend = get_backend(body.get("model"))
        try:
            state, qs, ids = contract.parse(body)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        t0 = time.perf_counter()
        with _GPU_LOCK:
            answers = backend.decide(state, qs)
        ms = (time.perf_counter() - t0) * 1000
        return {"model": getattr(backend, "release_id", getattr(backend, "model_id", getattr(backend, "name", "?"))),
                "decisions": {i: contract.format_answer(a, ms / len(qs)) for i, a in zip(ids, answers)}, "latency_ms": round(ms, 2)}
    req = DecisionRequest(**body)
    backend = get_backend(req.model)
    qs = req.parsed_questions()
    t0 = time.perf_counter()
    with _GPU_LOCK:
        answers = backend.decide(req.state, qs, temperature=req.temperature, return_logits=req.return_logits)
    ms = (time.perf_counter() - t0) * 1000
    return DecisionResponse(answers=answers, model=getattr(backend, "model_id", getattr(backend, "name", "?")), latency_ms=ms, state_tokens=-1, backend=BACKEND_KIND).model_dump()


@app.post("/v1/evaluate")
def evaluate(req: JevRequest):
    backend = get_backend(req.model)
    state = State(content=req.state)
    names = list(req.questions)
    qs = [_from_jev(n, req.questions[n]) for n in names]
    t0 = time.perf_counter()
    with _GPU_LOCK:
        answers = backend.decide(state, qs)
    ms = (time.perf_counter() - t0) * 1000
    return {"model": getattr(backend, "model_id", getattr(backend, "name", "open-spark-jev")),
            "answers": {n: _to_jev(a, q) for n, a, q in zip(names, answers, qs)},
            "usage": {"input_tokens": None, "output_tokens": 0}, "latency_ms": round(ms, 1)}


LAB_DATA = Path(__file__).parent / "lab_data"
LAB_MODE = False
PREFERRED = ["spark-s1-4b-v3", "v3-4b", "spark-s1-1.7b-v3", "v3-1.7b", "sft-qwen3-1.7b"]


def _lab_json(name: str):
    import json

    p = LAB_DATA / name
    return json.loads(p.read_text()) if p.exists() else None


@app.get("/lab", include_in_schema=False)
def lab_page():
    return FileResponse(UI_DIR / "lab.html")


@app.get("/v1/lab/fixtures")
def lab_fixtures():
    """Safe fixtures (proposed calls only, never executed), real recorded model outputs for them, and the runtime mode."""
    from ..gate import DEFAULT_THRESHOLD, DESCRIPTIONS, OPTIONS

    avail = discover_models() if BACKEND_KIND == "hf" else {}
    return {"fixtures": _lab_json("tool_calls.json") or [], "recorded": _lab_json("recorded_outputs.json") or {},
            "live_models": [{"id": k, "name": v["name"]} for k, v in avail.items()], "default_model": DEFAULT_MODEL,
            "options": OPTIONS, "descriptions": DESCRIPTIONS, "default_threshold": DEFAULT_THRESHOLD}


@app.get("/v1/lab/comparison")
def lab_comparison():
    """Recorded real measurements: spark-s1 versus the same-size untrained model generating JSON."""
    return _lab_json("slm_comparison.json") or {}


@app.get("/v1/lab/benchmarks")
def lab_benchmarks():
    return {"runs": _lab_json("benchmark_runs.json") or []}


class ResolveRequest(BaseModel):
    tool: str = "bash"
    command: str
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    threshold: float = 0.995


@app.post("/v1/lab/resolve")
def lab_resolve(req: ResolveRequest):
    """Decision Lab helper: apply the deterministic policy to a model output supplied by the caller (used to show recorded outputs). Executes nothing."""
    from .. import policy as P

    if len(req.command) > 8000 or not 0.5 <= req.threshold <= 1.0:
        raise HTTPException(400, "command too long or threshold out of range")
    f = P.analyze(req.tool, req.command)
    action, trace = P.resolve(f, req.choice, req.probabilities, req.threshold)
    return {"policy_action": action, "policy_trace": trace, "detections": [{"rule": m.rule, "floor": m.floor, "detail": m.detail} for m in f.matches]}


@app.post("/v1/gate")
def gate_endpoint(req: GateRequest):
    """Tool-call gate: spark-s1 allow/ask/deny distribution + deterministic policy. Never executes anything. model="none" = policy rules only."""
    from ..gate import gate as run_gate

    scorer, version = None, None
    if req.model != "none":
        try:
            scorer = get_backend(req.model)
            version = getattr(scorer, "release_id", getattr(scorer, "model_id", getattr(scorer, "name", None)))
        except HTTPException:
            if req.model:
                raise
            scorer = None  # no model available: policy-only, which never returns allow
    with _GPU_LOCK:
        res = run_gate(req.state, scorer, req.policy.auto_allow_threshold, version, req.mode)
    return res.model_dump()


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


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai", "trtllm"], default="openai")
    ap.add_argument("--model", help="HF path (also used to load calibration.json for openai backend)")
    ap.add_argument("--upstream", default="http://localhost:8355/v1")
    ap.add_argument("--default-model", help="hf backend: default model id (see configs/serve/models.yaml); the UI can switch per request")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 or your Tailscale IP to reach it from another machine")
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--lab", action="store_true", help="Decision Lab: start even with no checkpoints (recorded/policy-only fixture mode)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    global DEFAULT_MODEL, BACKEND_KIND
    if a.backend == "hf":
        BACKEND_KIND = "hf"
        avail = discover_models()
        DEFAULT_MODEL = a.default_model or next((m for m in PREFERRED if m in avail), next(iter(avail), None))
        if DEFAULT_MODEL is None and not a.lab:
            raise SystemExit("no checkpoints found under checkpoints/ (see docs/UI.md; `osj lab` runs without weights)")
        if DEFAULT_MODEL is not None and not a.lab:
            _b = get_backend(DEFAULT_MODEL)  # load now so the first request is fast (in lab mode it loads on first live use)
            try:  # one throwaway decision so the first real request is not slowed by kernel warm-up
                _b.decide(State(content="warm-up"), [Choice(prompt="warm-up", options=["a", "b"])])
            except Exception:  # noqa: BLE001
                pass
    else:
        build_backend(a.backend, a.model, a.upstream)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
