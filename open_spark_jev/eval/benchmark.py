"""Run a decision corpus through any backend and report accuracy + calibration per slice.

Backends:
  hf        in-process MenuScorer (HF Transformers)
  openai    any OpenAI-compatible /v1/completions endpoint (trtllm-serve, vLLM, SGLang)
  gateway   the open-spark-Jev gateway (/v1/decide), i.e. the production path

Metrics per (question type, domain): accuracy, macro-F1, Brier, NLL, ECE. When a record has a
known posterior (``target.dist`` from the simulators), we additionally report **soft Brier**
against it - calibration against the true posterior rather than a single realised label -
and, for injected states, the **injection flip rate** (argmax differs from the clean twin).

Usage:
  python -m open_spark_jev.eval.benchmark --backend hf --model models/Qwen3-1.7B --data data/benchmarks/sim_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from typing import Any

from ..calibration import brier_noul, brier_soft, ece_noul, summary
from ..data.corpus import Record, read_jsonl
from ..schema import Answer


def _pad(rows: list[list[float]]) -> list[list[float]]:
    K = max(len(r) for r in rows)
    return [r + [0.0] * (K - len(r)) for r in rows]


def run(backend, records: list[Record], batch: int = 16) -> tuple[list[Answer], float]:
    """backend must expose decide(state, questions) -> list[Answer]. Records are grouped by
    identical state so shared-state questions reuse the KV cache."""
    answers: list[Answer] = [None] * len(records)  # type: ignore[list-item]
    t0 = time.perf_counter()
    by_state: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_state[json.dumps(r.state, sort_keys=True)].append(i)
    for idxs in by_state.values():
        for j in range(0, len(idxs), batch):
            chunk = idxs[j : j + batch]
            qs = [records[i].question_obj() for i in chunk]
            out = backend.decide(records[chunk[0]].state_obj(), qs)
            for i, a in zip(chunk, out):
                answers[i] = a
    return answers, time.perf_counter() - t0


def report(records: list[Record], answers: list[Answer]) -> dict[str, Any]:
    slices: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r, a in zip(records, answers):
        key = (r.question["type"], r.domain)
        s = slices[key]
        s["probs"].append(a.probs)
        s["labels"].append(r.label_index())
        td = r.target_dist()
        if td is not None:
            s["soft_brier"].append(brier_soft([a.probs], [td]))
        if r.question["type"] == "noul":
            s["p_true"].append(a.probability)
            s["truth"].append(1 if r.target["label"] == "yes" else 0)
        if r.meta.get("injected"):
            s["injected_conf"].append(a.confidence)
    out: dict[str, Any] = {}
    all_probs, all_labels = [], []
    for (qt, dom), s in sorted(slices.items()):
        rep = summary(_pad(s["probs"]), s["labels"])
        if s.get("soft_brier"):
            rep["soft_brier_vs_posterior"] = float(sum(s["soft_brier"]) / len(s["soft_brier"]))
        if s.get("p_true"):
            rep["noul_brier"] = brier_noul(s["p_true"], s["truth"])
            rep["noul_ece"] = ece_noul(s["p_true"], s["truth"])
        if s.get("injected_conf"):
            rep["n_injected"] = len(s["injected_conf"])
        out[f"{qt}/{dom}"] = rep
        all_probs.extend(_pad(s["probs"]) if False else s["probs"])
        all_labels.extend(s["labels"])
    out["overall"] = summary(_pad(all_probs), all_labels)
    return out


def injection_flip_rate(records: list[Record], answers: list[Answer]) -> float | None:
    """Fraction of injected records whose argmax differs from the *posterior* argmax."""
    flips = n = 0
    for r, a in zip(records, answers):
        if not r.meta.get("injected"):
            continue
        td = r.target_dist()
        if td is None:
            continue
        n += 1
        flips += int(max(range(len(td)), key=lambda i: td[i]) != a.probs.index(max(a.probs)))
    return flips / n if n else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai", "gateway"], default="hf")
    ap.add_argument("--model", help="HF path (hf) or model name (openai)")
    ap.add_argument("--base-url", default="http://localhost:8355/v1")
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()
    recs: list[Record] = []
    for p in a.data:
        recs.extend(read_jsonl(p))
    if a.limit:
        recs = recs[: a.limit]
    if a.backend == "hf":
        from ..model import MenuScorer

        backend = MenuScorer(a.model)
    elif a.backend == "openai":
        from ..serve.client import OpenAICompletionsBackend

        backend = OpenAICompletionsBackend(a.base_url, a.model)
    else:
        from ..serve.client import GatewayClient

        backend = GatewayClient(a.base_url)
    answers, secs = run(backend, recs, a.batch)
    rep = report(recs, answers)
    rep["throughput_decisions_per_s"] = len(recs) / secs
    fr = injection_flip_rate(recs, answers)
    if fr is not None:
        rep["injection_flip_rate"] = fr
    print(json.dumps(rep, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
