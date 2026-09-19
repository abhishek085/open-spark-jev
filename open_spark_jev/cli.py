"""``osj`` command line: quick decisions, data generation, eval, serve."""

from __future__ import annotations

import argparse
import json
import sys


def _decide(a):
    from .model import MenuScorer
    from .schema import State, parse_question

    scorer = MenuScorer(a.model)
    state = State(content=a.state if not a.json else json.loads(a.state))
    qs = [parse_question(json.loads(q)) for q in a.question]
    answers, ms, n = scorer.decide_timed(state, qs)
    print(json.dumps({"answers": [x.model_dump(exclude_none=True) for x in answers], "latency_ms": ms, "state_tokens": n}, indent=2))


def _simulate(a):
    from .data.corpus import write_jsonl
    from .data.simulators import SIMULATORS, generate_all

    recs = generate_all(a.n, seed=a.seed, hard_label=a.hard_label, abstain_frac=a.abstain_frac, inject_frac=a.inject_frac)
    print(f"wrote {write_jsonl(a.out, recs)} records ({', '.join(SIMULATORS)}) to {a.out}", file=sys.stderr)


def _public(a):
    from .data.corpus import write_jsonl
    from .data.public import REGISTRY, build

    names = a.datasets or list(REGISTRY)
    print(f"wrote {write_jsonl(a.out, build(names, a.limit, a.seed))} records to {a.out}", file=sys.stderr)


def _synth(a):
    from .data.corpus import write_jsonl
    from .data.synth import DOMAIN_BRIEFS, Teacher, distill, generate_scenarios

    t = Teacher(a.base_url, a.model)
    recs = []
    for d in a.domains or list(DOMAIN_BRIEFS):
        r = generate_scenarios(t, d, a.n, a.seed, concurrency=a.concurrency)
        if a.distill:
            r = distill(t, r, samples=a.samples, concurrency=a.concurrency)
        recs.extend(r)
        print(f"{d}: {len(r)} records", file=sys.stderr)
    write_jsonl(a.out, recs)


def _ui(a):
    """Start the local playground (UI + API). Deliberately a light `osj ui` process, not
    `python -m open_spark_jev...`, so the thermal guard's process pattern never SIGSTOPs it mid-demo."""
    from .serve.gateway import main as gateway_main

    argv = ["--backend", "hf", "--host", a.host, "--port", str(a.port)]
    if a.model:
        argv += ["--default-model", a.model]
    gateway_main(argv)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="osj")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decide", help="one state, N questions, in-process HF backend")
    d.add_argument("--model", required=True)
    d.add_argument("--state", required=True)
    d.add_argument("--json", action="store_true", help="parse --state as JSON")
    d.add_argument("--question", "-q", action="append", required=True, help='JSON question, repeatable')
    d.set_defaults(fn=_decide)

    s = sub.add_parser("simulate", help="known-posterior simulator corpus")
    s.add_argument("--n", type=int, default=2000, help="records per domain")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--hard-label", default="sample", choices=["sample", "argmax", "latent"])
    s.add_argument("--abstain-frac", type=float, default=0.3)
    s.add_argument("--inject-frac", type=float, default=0.1)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=_simulate)

    p = sub.add_parser("public", help="public HF datasets -> decision records")
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_public)

    u = sub.add_parser("ui", help="local playground: web UI + Jev-compatible API over the checkpoints in checkpoints/")
    u.add_argument("--host", default="127.0.0.1", help="0.0.0.0 or your Tailscale IP to reach it from another machine")
    u.add_argument("--port", type=int, default=8400)
    u.add_argument("--model", help="default model id (see configs/serve/models.yaml)")
    u.set_defaults(fn=_ui)

    y = sub.add_parser("synth", help="teacher-generated scenarios (+ optional distillation)")
    y.add_argument("--base-url")
    y.add_argument("--model")
    y.add_argument("--domains", nargs="*")
    y.add_argument("--n", type=int, default=200)
    y.add_argument("--seed", type=int, default=0)
    y.add_argument("--distill", action="store_true")
    y.add_argument("--samples", type=int, default=1)
    y.add_argument("--concurrency", type=int, default=6, help="teacher calls in flight at once (sequential=1 "
                    "projects to 10+ hours across many domains; see data/synth.py generate_scenarios docstring)")
    y.add_argument("--out", required=True)
    y.set_defaults(fn=_synth)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
