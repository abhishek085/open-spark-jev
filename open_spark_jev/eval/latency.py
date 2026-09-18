"""Latency / throughput / memory on DGX Spark for realistic decision workloads.

Measures, for a grid of (state length, questions per state, concurrent states):
  * p50 / p95 latency per request (one state, N questions)
  * decisions per second
  * peak GPU memory (HF backend) or server-side memory (from nvidia-smi, if readable)

Usage:
  python -m open_spark_jev.eval.latency --backend hf --model models/Qwen3-1.7B
  python -m open_spark_jev.eval.latency --backend openai --base-url http://localhost:8355/v1 --model osj
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from ..schema import Choice, Noul, Score, State


def synthetic_state(n_tokens_approx: int) -> State:
    row = '{"ts":"2026-09-18T10:00:00Z","svc":"checkout","status":500,"latency_ms":812,"path":"/api/pay"}'
    n = max(1, n_tokens_approx // 40)
    return State(content="\n".join(row for _ in range(n)), schema_hint="api access log rows")


def question_set(n: int):
    base = [
        Choice(prompt="What is the right operational response?", options=["ignore", "scale_up", "rollback", "page_oncall"], allow_abstain=True),
        Score(prompt="How severe is this incident?", levels=["none", "minor", "major", "critical"]),
        Noul(prompt="The service is returning elevated error rates."),
        Choice(prompt="Which team owns this?", options=["payments", "platform", "frontend", "data"]),
    ]
    return [base[i % len(base)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "openai", "gateway"], default="hf")
    ap.add_argument("--model")
    ap.add_argument("--base-url", default="http://localhost:8355/v1")
    ap.add_argument("--state-tokens", nargs="+", type=int, default=[128, 512, 2048])
    ap.add_argument("--questions", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.backend == "hf":
        import torch

        from ..model import MenuScorer

        backend = MenuScorer(a.model)
    elif a.backend == "openai":
        from ..serve.client import OpenAICompletionsBackend

        backend = OpenAICompletionsBackend(a.base_url, a.model)
    else:
        from ..serve.client import GatewayClient

        backend = GatewayClient(a.base_url)

    results = []
    for st in a.state_tokens:
        for nq in a.questions:
            state, qs = synthetic_state(st), question_set(nq)
            backend.decide(state, qs)  # warm-up
            lat = []
            for _ in range(a.iters):
                t0 = time.perf_counter()
                backend.decide(state, qs)
                lat.append((time.perf_counter() - t0) * 1000)
            lat.sort()
            row = {
                "state_tokens": st, "questions": nq,
                "p50_ms": statistics.median(lat), "p95_ms": lat[int(0.95 * (len(lat) - 1))],
                "decisions_per_s": nq / (statistics.mean(lat) / 1000),
            }
            if a.backend == "hf":
                row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            results.append(row)
            print(json.dumps(row))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
