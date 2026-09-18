from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..schema import Question, State, parse_question


@dataclass
class Record:
    id: str
    domain: str
    source: str
    state: dict[str, Any]
    question: dict[str, Any]
    target: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    # -- typed views -------------------------------------------------------------
    def state_obj(self) -> State:
        return State(**self.state)

    def question_obj(self) -> Question:
        return parse_question(self.question)

    def label_index(self) -> int:
        labels = self.question_obj().labels
        lab = self.target["label"]
        if lab not in labels:
            raise ValueError(f"{self.id}: target label {lab!r} not in {labels}")
        return labels.index(lab)

    def target_dist(self) -> list[float] | None:
        """Soft target aligned to question.labels, or None if only a hard label exists."""
        d = self.target.get("dist")
        if not d:
            return None
        labels = self.question_obj().labels
        v = [float(d.get(lab, 0.0)) for lab in labels]
        s = sum(v)
        return [x / s for x in v] if s > 0 else None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "source": self.source,
            "state": self.state,
            "question": self.question,
            "target": self.target,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Record:
        return cls(
            id=d["id"],
            domain=d.get("domain", "unknown"),
            source=d.get("source", "unknown"),
            state=d["state"],
            question=d["question"],
            target=d["target"],
            meta=d.get("meta", {}),
        )


def iter_records(path: str) -> Iterator[Record]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield Record.from_json(json.loads(line))


def read_jsonl(path: str) -> list[Record]:
    return list(iter_records(path))


def write_jsonl(path: str, records: Iterable[Record], append: bool = False) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "a" if append else "w") as f:
        for r in records:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def split_records(records: list[Record], val_frac: float = 0.1, seed: int = 0) -> tuple[list[Record], list[Record]]:
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val = {i for i in idx[:n_val]}
    return [r for i, r in enumerate(records) if i not in val], [r for i, r in enumerate(records) if i in val]
