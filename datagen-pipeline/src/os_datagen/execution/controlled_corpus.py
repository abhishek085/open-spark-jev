"""Controlled corpus / knowledge world: documents with known fact coverage, dates and authority."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Doc:
    doc_id: str
    entity: str
    covers: dict[str, str]  # attribute -> value the document states
    period: str
    published: date
    authority: int  # 2 official, 1 informal
    title: str = ""


@dataclass
class Corpus:
    docs: list[Doc] = field(default_factory=list)

    def coverage(self, doc: Doc, entity: str, period: str, required: list[str]) -> float:
        if doc.entity != entity or doc.period != period:
            return 0.0
        return sum(a in doc.covers for a in required) / max(1, len(required))

    def is_stale(self, doc: Doc, today: date, max_age_days: int) -> bool:
        return (today - doc.published).days > max_age_days
