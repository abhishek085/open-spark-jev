from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..schemas.decision import DatasetRecord
from ..utils.paths import project_path
from ..validation.deduplication import visible_text


def _csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def reason_category(r: str) -> str:
    return ":".join(r.split(":")[:2]) if r.startswith(("schema", "label_leak:gold")) else r.split(":")[0]


def composition(records: list[DatasetRecord]) -> dict[str, Any]:
    p = project_path("configs", "mixtures", "general_decision_v1.yaml")
    targets = (yaml.safe_load(p.read_text()) or {}).get("targets", {}) if p.exists() else {}
    fam = Counter(r.meta.get("composition_family", "?") for r in records)
    total = sum(fam.values()) or 1
    return {f: {"count": fam.get(f, 0), "share": round(fam.get(f, 0) / total, 4), "target": t,
                "gap": round(fam.get(f, 0) / total - t, 4)} for f, t in targets.items()} | \
           {f: {"count": c, "share": round(c / total, 4), "target": 0.0, "gap": round(c / total, 4)}
            for f, c in fam.items() if f not in targets}


def build_reports(out: Path, *, cands: list[Any], records: list[DatasetRecord], clusters: list[dict[str, Any]],
                  isolation: dict[str, Any], packs: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    rep = out / "reports"
    rep.mkdir(exist_ok=True)
    tot: Counter[str] = Counter(c.world.task_pack for c in cands)
    acc: Counter[str] = Counter(r.task_pack for r in records)
    reasons: Counter[str] = Counter()
    for c in cands:
        for r in c.reasons:
            reasons[reason_category(r)] += 1
    by_pack: dict[str, Any] = {}
    for n in tot:
        recs = [r for r in records if r.task_pack == n]
        labels = Counter(_label(r) for r in recs)
        by_pack[n] = {
            "candidates": tot[n], "accepted": acc[n], "acceptance_rate": round(acc[n] / tot[n], 4) if tot[n] else 0,
            "label_distribution": dict(labels), "difficulty": dict(Counter(r.quality.difficulty for r in recs)),
            "families": dict(Counter(r.meta.get("scenario_family") for r in recs)),
            "truth_tier": dict(Counter(r.truth.label_quality for r in recs)),
            "avg_visible_chars": round(sum(len(visible_text(r)) for r in recs) / max(1, len(recs)), 1),
            "option_position_of_truth": dict(Counter(_pos(r) for r in recs)),
            "distractor_tags": dict(Counter(t for r in recs for t in r.quality.tags)),
        }
    summary = {
        "candidates": len(cands), "accepted": len(records), "rejected": len(cands) - len(records),
        "namespaces": dict(Counter(r.meta.get("namespace") for r in records)),
        "splits": dict(Counter(r.split for r in records)), "by_pack": by_pack,
        "composition_vs_target": composition(records), "split_isolation_ok": isolation["ok"], **extra,
    }
    (rep / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _csv(rep / "acceptance_by_task.csv", ["task_pack", "namespace", "candidates", "accepted", "acceptance_rate"],
         [[n, packs[n].namespace, tot[n], acc[n], by_pack[n]["acceptance_rate"]] for n in sorted(tot)])
    _csv(rep / "rejection_reasons.csv", ["reason", "count"], sorted(reasons.items(), key=lambda kv: -kv[1]))
    cov: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        cov[(r.task_pack, r.meta.get("scenario_family"))] += 1
    _csv(rep / "coverage_matrix.csv", ["task_pack", "scenario_family", "accepted"], [[a, b, n] for (a, b), n in sorted(cov.items())])
    (rep / "split_isolation.json").write_text(json.dumps(isolation, indent=2, sort_keys=True))
    (rep / "duplicate_clusters.json").write_text(json.dumps(clusters, indent=2))
    lines = ["# Sample audit\n"]
    for n in sorted(tot):
        lines.append(f"\n## {n}\n\n### Accepted (up to 10)\n")
        for r in [x for x in records if x.task_pack == n][:10]:
            lines.append(f"- `{r.record_id}` [{r.split}/{r.meta.get('scenario_family')}] truth=`{_label(r)}` ({r.truth.reason_code})\n  ```\n  {json.dumps(r.decision.state, ensure_ascii=False)[:700]}\n  ```\n")
        lines.append("\n### Rejected (up to 10)\n")
        for c in [x for x in cands if x.world.task_pack == n and not x.alive][:10]:
            lines.append(f"- `{c.candidate_id}` reasons={c.reasons}\n")
    (rep / "sample_audit.md").write_text("".join(lines))
    return summary


def _label(r: DatasetRecord) -> str:
    if r.truth.preferred_option is not None:
        return r.truth.preferred_option
    if r.truth.preferred_level is not None:
        return str(r.truth.preferred_level)
    return str(r.truth.truth).lower()


def _pos(r: DatasetRecord) -> str:
    opts = [o["id"] for o in r.decision.question.get("options", [])]
    return str(opts.index(r.truth.preferred_option)) if r.truth.preferred_option in opts else "n/a"
