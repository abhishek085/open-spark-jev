"""Training exports. Only the train split is exported unless you explicitly allow others (calibration/locked_test/challenge must never
leak into weight training)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.jsonl import read_jsonl, write_jsonl
from .render import option_ids, render_prompt, truth_sets

TRAIN_ONLY = {"train"}
FORMATS = ("prompt_completion", "chat", "jev")


def _dist(r: dict[str, Any], ids: list[str]) -> dict[str, float]:
    pref, acc = truth_sets(r)
    d = r["truth"].get("distribution")
    if d:
        return {i: float(d.get(i, 0.0)) for i in ids}
    return {i: (1.0 if i == pref else 0.0) for i in ids}  # deterministic point mass on the preferred option


def _row_prompt_completion(r: dict[str, Any]) -> dict[str, Any]:
    pref, acc = truth_sets(r)
    ids = option_ids(r)
    return {"record_id": r["record_id"], "task_pack": r["task_pack"], "decision_type": r["decision"]["type"], "prompt": render_prompt(r), "target": pref,
            "acceptable": acc, "options": ids, "target_distribution": _dist(r, ids), "label_quality": r["truth"]["label_quality"],
            "probability_source": "declared_distribution" if r["truth"].get("distribution") else "deterministic_point_mass", "split": r["split"]}


def _row_chat(r: dict[str, Any]) -> dict[str, Any]:
    base = _row_prompt_completion(r)
    return {**{k: base[k] for k in ("record_id", "task_pack", "acceptable", "label_quality", "split")},
            "messages": [{"role": "user", "content": base["prompt"]}, {"role": "assistant", "content": base["target"]}]}


def _jev_question(r: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    """Map to open-spark-Jev's typed question (Choice / Score / Noul). Returns (question dict, labels in order, label-id -> label map)."""
    d = r["decision"]
    q = d["question"]
    if d["type"] == "choice":
        opts = [o["id"] for o in q["options"]]
        defs = "\n".join(f"- {o['id']}: {o['definition']}" for o in q["options"])
        return {"type": "choice", "prompt": f"{q['instructions']}\nOption definitions:\n{defs}", "options": opts}, opts, {o: o for o in opts}
    if d["type"] == "score":
        levels = [str(x["value"]) for x in q["levels"]]
        return {"type": "score", "prompt": q["instructions"], "levels": levels,
                "rubric": "\n".join(f"{x['value']}: {x['definition']}" for x in q["levels"])}, levels, {lv: lv for lv in levels}
    return {"type": "noul", "prompt": q["instructions"]}, ["yes", "no"], {"true": "yes", "false": "no"}


def _row_jev(r: dict[str, Any]) -> dict[str, Any]:
    """open_spark_jev.data.corpus.Record JSON: id/domain/source/state/question/target/meta (loadable by its SFT/RLCD trainers)."""
    question, labels, to_label = _jev_question(r)
    pref, acc = truth_sets(r)
    ids = option_ids(r)
    dist = _dist(r, ids)
    return {
        "id": r["record_id"], "domain": r["task_pack"], "source": "os-datagen:synthetic",
        "state": {"content": r["decision"]["state"], "schema_hint": f"{r['task_pack']} decision state (JSON)", "domain": r["task_pack"]},
        "question": question,
        "target": {"label": to_label[pref], "dist": {to_label[i]: dist[i] for i in ids}},
        "meta": {"task_pack": r["task_pack"], "namespace": r["meta"].get("namespace"), "scenario_family": r["meta"].get("scenario_family"),
                 "split": r["split"], "difficulty": r["quality"]["difficulty"], "label_quality": r["truth"]["label_quality"],
                 "acceptable": [to_label[a] for a in acc if a in to_label], "supportability": r["quality"].get("supportability")},
    }


def export_training(dataset: Path | list[Path], out: Path, *, fmt: str = "prompt_completion", allow_splits: set[str] | None = None,
                    min_label_quality_exclude: tuple[str, ...] = ("model_consensus",)) -> int:
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    allow = allow_splits or TRAIN_ONLY
    fn = {"prompt_completion": _row_prompt_completion, "chat": _row_chat, "jev": _row_jev}[fmt]
    paths = [dataset] if isinstance(dataset, Path) else dataset
    rows = [fn(r) for p in paths for r in read_jsonl(p) if r["split"] in allow and r["truth"]["label_quality"] not in min_label_quality_exclude]
    return write_jsonl(out, rows)


def export_sft(dataset: Path, out: Path, *, allow_splits: set[str] | None = None) -> int:
    """Backward-compatible prompt/completion export."""
    return export_training(dataset, out, fmt="prompt_completion", allow_splits=allow_splits)


_ = json
