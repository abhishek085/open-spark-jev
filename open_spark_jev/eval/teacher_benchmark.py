"""Ground-truth-anchored comparison of candidate teacher models.

Unlike ordinary LLM-judge comparisons (which only measure inter-model *agreement*), this
benchmark asks each teacher to grade states from ``data/simulators.py``, whose posteriors are
known exactly by construction. That turns "which teacher looks better" into a measurable
question: soft Brier and ECE against the true posterior, per teacher, per domain - the same
metrics ``eval/benchmark.py`` uses to score the student. An equal-weight ensemble of the
teachers is scored the same way, since it is the natural target distribution for distillation
if no single teacher dominates.

Usage:
  python -m open_spark_jev.eval.teacher_benchmark --teachers qwen27b nemotron120b gptoss120b \
      --data data/benchmarks/sim_test.jsonl --limit-per-domain 40 --out runs/teacher_benchmark.json

Unreachable teachers (not launched yet) are skipped with a warning rather than failing the run,
so this can be re-run as each teacher comes online (see scripts/teachers/).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
import yaml

from ..calibration import brier_soft
from ..data.corpus import Record, read_jsonl
from ..data.synth import Teacher, label_distribution

log = logging.getLogger("osj.teacher_benchmark")


def load_registry(path: str = "configs/teachers.yaml") -> dict[str, dict]:
    with open(path) as f:
        return yaml.safe_load(f)["teachers"]


def make_teacher(name: str, cfg: dict) -> Teacher | None:
    try:
        t = Teacher(base_url=cfg["base_url"], model=cfg["hf_id"])
        t.client.get("/models", timeout=5).raise_for_status()
        return t
    except (httpx.HTTPError, OSError) as e:
        log.warning("teacher %s unreachable at %s (%s) - skipping", name, cfg["base_url"], e)
        return None


def grade_all(teacher: Teacher, records: list[Record], samples: int, concurrency: int) -> dict[str, dict[str, float] | None]:
    """record.id -> label->prob dict, or None if grading failed for that record."""
    out: dict[str, dict[str, float] | None] = {}

    def _one(r: Record):
        try:
            return r.id, label_distribution(teacher, r, samples)
        except Exception as e:  # noqa: BLE001
            log.warning("grade failed for %s: %s", r.id, e)
            return r.id, None

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_one, r) for r in records]
        for fut in as_completed(futs):
            rid, dist = fut.result()
            out[rid] = dist
    return out


def aligned_probs(rec: Record, dist: dict[str, float]) -> list[float]:
    labels = rec.question_obj().labels
    v = [float(dist.get(lab, 0.0)) for lab in labels]
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


def score(records: list[Record], grades: dict[str, dict[str, float] | None]) -> dict[str, Any]:
    by_domain: dict[str, list[float]] = defaultdict(list)
    hits = defaultdict(int)
    n = defaultdict(int)
    for r in records:
        g = grades.get(r.id)
        if g is None:
            continue
        p = aligned_probs(r, g)
        td = r.target_dist()
        if td is None:
            continue
        by_domain[r.domain].append(brier_soft([p], [td]))
        n[r.domain] += 1
        labels = r.question_obj().labels
        pred = labels[max(range(len(p)), key=lambda i: p[i])]
        true_argmax = labels[max(range(len(td)), key=lambda i: td[i])]
        hits[r.domain] += int(pred == true_argmax)
    out = {}
    for d, vals in by_domain.items():
        out[d] = {"n": n[d], "soft_brier_vs_posterior": sum(vals) / len(vals), "argmax_agreement": hits[d] / n[d]}
    all_vals = [v for vs in by_domain.values() for v in vs]
    all_hits = sum(hits.values())
    all_n = sum(n.values())
    out["overall"] = {
        "n": all_n,
        "soft_brier_vs_posterior": sum(all_vals) / len(all_vals) if all_vals else None,
        "argmax_agreement": all_hits / all_n if all_n else None,
        "coverage": all_n / len(records) if records else 0.0,
    }
    return out


def ensemble(records: list[Record], per_teacher: dict[str, dict[str, dict[str, float] | None]]) -> dict[str, dict[str, float] | None]:
    out = {}
    for r in records:
        dists = [aligned_probs(r, per_teacher[t][r.id]) for t in per_teacher if per_teacher[t].get(r.id) is not None]
        if not dists:
            out[r.id] = None
            continue
        labels = r.question_obj().labels
        mean = [sum(d[i] for d in dists) / len(dists) for i in range(len(labels))]
        out[r.id] = dict(zip(labels, mean))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", nargs="*", default=None, help="names from configs/teachers.yaml (default: all)")
    ap.add_argument("--registry", default="configs/teachers.yaml")
    ap.add_argument("--data", default="data/benchmarks/sim_test.jsonl")
    ap.add_argument("--limit-per-domain", type=int, default=40)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="runs/teacher_benchmark.json")
    a = ap.parse_args()

    registry = load_registry(a.registry)
    names = a.teachers or list(registry)

    recs = read_jsonl(a.data)
    by_domain: dict[str, list[Record]] = defaultdict(list)
    for r in recs:
        by_domain[r.domain].append(r)
    subset = [r for rs in by_domain.values() for r in rs[: a.limit_per_domain]]
    log.info("scoring %d records (%d per domain x %d domains)", len(subset), a.limit_per_domain, len(by_domain))

    per_teacher_grades: dict[str, dict[str, dict[str, float] | None]] = {}
    report: dict[str, Any] = {"n_records": len(subset), "teachers": {}}
    for name in names:
        cfg = registry[name]
        teacher = make_teacher(name, cfg)
        if teacher is None:
            report["teachers"][name] = {"status": "unreachable", "base_url": cfg["base_url"], "hf_id": cfg["hf_id"]}
            continue
        conc = min(a.concurrency, cfg.get("max_concurrency", a.concurrency))
        log.info("grading with %s (%s) at concurrency %d", name, cfg["hf_id"], conc)
        grades = grade_all(teacher, subset, a.samples, conc)
        per_teacher_grades[name] = grades
        report["teachers"][name] = {"status": "ok", "hf_id": cfg["hf_id"], **score(subset, grades)}

    if len(per_teacher_grades) >= 2:
        ens = ensemble(subset, per_teacher_grades)
        report["ensemble"] = {"members": list(per_teacher_grades), **score(subset, ens)}

    print(json.dumps(report, indent=2))
    import os

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
