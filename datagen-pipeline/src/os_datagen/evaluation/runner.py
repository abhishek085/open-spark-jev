from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

import httpx

from ..config import policies
from ..training.render import option_ids, render_prompt, truth_sets
from ..utils.jsonl import read_jsonl
from .calibration import fit_temperature, softmax_from_scores
from .metrics import (
    brier,
    cost_quality_frontier,
    ece,
    nll,
    normalize,
    per_option_pr,
    selective_risk,
    summarize_rows,
    top1,
)
from .reports import write_eval
from .slice_analysis import length_bucket, slices


class Predictor:
    """Returns (raw_output, raw_scores|None, probabilities, source)."""

    mode = "base"

    def predict(self, prompt: str, options: list[str], record: dict[str, Any]) -> tuple[str, dict[str, float] | None, dict[str, float], str]:
        raise NotImplementedError


class UniformPredictor(Predictor):
    mode = "baseline_uniform"

    def predict(self, prompt, options, record):  # type: ignore[no-untyped-def]
        return options[0], None, {o: 1 / len(options) for o in options}, "uniform_baseline"


class PriorPredictor(Predictor):
    """Label-frequency baseline fitted on a train file (a sanity floor for any candidate model)."""

    mode = "baseline_prior"

    def __init__(self, train_path: Path | None = None):
        self.freq: dict[str, dict[str, float]] = {}
        if train_path:
            for r in read_jsonl(train_path):
                pref, _ = truth_sets(r)
                self.freq.setdefault(r["task_pack"], {}).setdefault(pref, 0)
                self.freq[r["task_pack"]][pref] += 1

    def predict(self, prompt, options, record):  # type: ignore[no-untyped-def]
        f = self.freq.get(record["task_pack"], {})
        p = normalize({o: f.get(o, 0) + 1 for o in options})
        return top1(p)[0], {o: math.log(v) for o, v in p.items()}, p, "train_label_prior"


class ChatPredictor(Predictor):
    """constrained_generation: temperature-0 chat completion parsed to an option id. Without token logprobs the
    probability is a smoothed one-hot and says so; use CandidateLogprobPredictor for scoring-based probabilities."""

    mode = "constrained_generation"

    def __init__(self, endpoint: str, model: str, api_key: str = "local-not-a-secret", timeout: float = 120):
        self.url, self.model = endpoint.rstrip("/"), model
        self.http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def predict(self, prompt, options, record):  # type: ignore[no-untyped-def]
        r = self.http.post(f"{self.url}/chat/completions", json={
            "model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 24,
            "chat_template_kwargs": {"enable_thinking": False}, "guided_choice": options})
        if r.status_code == 400:  # endpoint without guided_choice: free generation, parsed strictly below
            r = self.http.post(f"{self.url}/chat/completions", json={
                "model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 24,
                "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        raw = (r.json()["choices"][0]["message"]["content"] or "").strip()
        sel = next((o for o in options if raw.strip("`'\". ").lower() == o.lower()), None) or \
            next((o for o in sorted(options, key=len, reverse=True) if re.search(rf"(?<![A-Za-z0-9_]){re.escape(o)}(?![A-Za-z0-9_])", raw, re.I)), None)
        eps = 0.02
        p = {o: (1 - eps) if o == sel else eps / max(1, len(options) - 1) for o in options} if sel else {o: 1 / len(options) for o in options}
        return raw, None, p, "smoothed_one_hot_from_generation" if sel else "uniform_unparseable_output"


class CandidateLogprobPredictor(Predictor):
    """candidate_logprob: score each option as the continuation of the prompt via /v1/completions (echo + logprobs)."""

    mode = "candidate_logprob"

    def __init__(self, endpoint: str, model: str, api_key: str = "local-not-a-secret", timeout: float = 120):
        self.url, self.model = endpoint.rstrip("/"), model
        self.http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def predict(self, prompt, options, record):  # type: ignore[no-untyped-def]
        base = prompt + "\nAnswer: "
        scores: dict[str, float] = {}
        for o in options:
            r = self.http.post(f"{self.url}/completions", json={"model": self.model, "prompt": base + o, "max_tokens": 0, "echo": True, "logprobs": 0})
            r.raise_for_status()
            lp = r.json()["choices"][0]["logprobs"]
            scores[o] = sum(x for x, off in zip(lp["token_logprobs"], lp["text_offset"]) if off >= len(base) and x is not None)
        p = softmax_from_scores(scores)
        return top1(p)[0], scores, p, "candidate_logprob_softmax"


def make_predictor(endpoint: str, model: str, mode: str, train_path: Path | None = None) -> Predictor:
    if endpoint == "baseline://uniform":
        return UniformPredictor()
    if endpoint.startswith("baseline://prior"):
        return PriorPredictor(train_path)
    return CandidateLogprobPredictor(endpoint, model) if mode == "candidate_logprob" else ChatPredictor(endpoint, model)


def _slice(rec: dict[str, Any], prompt: str) -> dict[str, Any]:
    m = rec.get("meta", {})
    return {"namespace": m.get("namespace"), "task_pack": rec["task_pack"], "scenario_family": m.get("scenario_family"),
            "difficulty": rec["quality"]["difficulty"], "option_count": len(option_ids(rec)), "length_bucket": length_bucket(len(prompt)),
            "truth_tier": rec["truth"]["label_quality"], "label_source": rec["truth"]["label_source"]}


def run_evaluation(dataset: Path, endpoint: str, model: str, out: Path, mode: str = "constrained_generation", limit: int | None = None,
                   calibration_dataset: Path | None = None, train_path: Path | None = None, predictor: Predictor | None = None) -> dict[str, Any]:
    pred = predictor or make_predictor(endpoint, model, mode, train_path)
    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(read_jsonl(dataset)):
        if limit and i >= limit:
            break
        prompt, opts = render_prompt(rec), option_ids(rec)
        t0 = time.time()
        raw, scores, probs, src = pred.predict(prompt, opts, rec)
        lat = int((time.time() - t0) * 1000)
        pref, acc = truth_sets(rec)
        sel, conf = top1(probs)
        rows.append({"record_id": rec["record_id"], "model": model, "inference_mode": pred.mode, "raw_output": raw, "raw_scores": scores,
                     "probabilities": probs, "probabilities_source": src, "selected": sel, "confidence": conf,
                     "preferred": pref, "acceptable": acc, "truth_match": sel == pref, "correct": sel in acc,
                     "nll": nll(probs, acc), "brier": brier(probs, pref), "latency_ms": lat, "split": rec["split"], "slice": _slice(rec, prompt)})
    calib: dict[str, Any] = {"fitted_on": None, "temperature": None}
    if calibration_dataset and rows and all(r["raw_scores"] for r in rows):
        # temperature is fit on the calibration split only, then applied to this (test) dataset
        cal_rows = []
        for rec in read_jsonl(calibration_dataset):
            _, sc, _, _ = pred.predict(render_prompt(rec), option_ids(rec), rec)
            cal_rows.append({"raw_scores": sc, "acceptable": truth_sets(rec)[1]})
        T = fit_temperature(cal_rows)
        for r in rows:
            p = softmax_from_scores(r["raw_scores"], T)
            sel, conf = top1(p)
            r.update(probabilities=p, selected=sel, confidence=conf, truth_match=sel == r["preferred"], correct=sel in r["acceptable"],
                     nll=nll(p, r["acceptable"]), brier=brier(p, r["preferred"]), probabilities_source="temperature_scaled")
        calib = {"fitted_on": str(calibration_dataset), "temperature": T}
    overall = summarize_rows(rows)
    e, reliability = ece(rows)
    opts_all = sorted({o for r in rows for o in r["probabilities"]})
    frontier = None
    if any(r["slice"]["task_pack"] == "harness_next_action_router_v1" for r in rows):
        rr = [r for r in rows if r["slice"]["task_pack"] == "harness_next_action_router_v1"]
        frontier = cost_quality_frontier(rr, policies()["router"]["cost_units"])
    metrics = {"model": model, "inference_mode": pred.mode, "overall": overall, "per_option": per_option_pr(rows, opts_all),
               "calibration": calib, "probabilities_source": sorted({r["probabilities_source"] for r in rows}), "dataset": str(dataset),
               "splits": sorted({r["split"] for r in rows})}
    write_eval(out, rows, metrics, slices(rows), reliability, selective_risk(rows), frontier)
    return metrics
