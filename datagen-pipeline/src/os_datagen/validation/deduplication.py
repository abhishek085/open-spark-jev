from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ..schemas.decision import DatasetRecord
from ..utils.hashing import sha256_text
from .leakage import flatten

DIFF_RANK = {"easy": 0, "medium": 1, "hard": 2}
Embedder = Callable[[str], list[float]]


def visible_text(record: DatasetRecord) -> str:
    """Model-visible text used for dedupe/isolation; constant code-rendered policy text is excluded so it cannot dominate similarity."""
    from ..taskpacks.registry import get_pack

    try:
        state = get_pack(record.task_pack).leak_view(record.decision.state)
    except KeyError:
        state = record.decision.state
    return "\n".join(flatten(state, keys=False))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def shingles(text: str, n: int = 4) -> set[int]:
    w = normalize(text).split()
    if len(w) <= n:
        return {int(sha256_text(" ".join(w))[:8], 16)} if w else set()
    return {int(sha256_text(" ".join(w[i:i + n]))[:8], 16) for i in range(len(w) - n + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def dedupe(records: list[DatasetRecord], threshold: float = 0.90, embedder: Embedder | None = None
           ) -> tuple[list[DatasetRecord], list[tuple[DatasetRecord, str, str]], list[dict[str, Any]]]:
    """Exact-hash then near-duplicate removal per task pack. Harder (then lower record_id) records are kept.
    `embedder` is injectable (local embedding model or a test mock); default is shingle-Jaccard.
    Returns (kept, dropped[(record, status, duplicate_of)], clusters)."""
    order = sorted(records, key=lambda r: (-DIFF_RANK.get(r.quality.difficulty, 0), r.record_id))
    kept: list[DatasetRecord] = []
    dropped: list[tuple[DatasetRecord, str, str]] = []
    exact: dict[tuple[str, str], str] = {}
    sketches: dict[str, list[tuple[str, Any]]] = defaultdict(list)  # pack -> [(record_id, features)]
    for r in order:
        text = visible_text(r)
        key = (r.task_pack, sha256_text(normalize(text)))
        if key in exact:
            dropped.append((r, "exact_duplicate", exact[key]))
            continue
        feat: Any = embedder(text) if embedder else shingles(text)
        dup_of = None
        for rid, other in sketches[r.task_pack]:
            sim = cosine(feat, other) if embedder else jaccard(feat, other)
            if sim >= threshold:
                dup_of = rid
                break
        if dup_of:
            dropped.append((r, "near_duplicate", dup_of))
            continue
        exact[key] = r.record_id
        sketches[r.task_pack].append((r.record_id, feat))
        kept.append(r)
    clusters: dict[str, list[str]] = defaultdict(list)
    for r, status, of in dropped:
        clusters[of].append(f"{r.record_id}:{status}")
    return kept, dropped, [{"kept": k, "dropped": v} for k, v in clusters.items()]


def diversity_select(records: list[DatasetRecord], max_easy_share: float = 0.6, min_n: int = 20, min_hard_share: float = 0.25
                     ) -> tuple[list[DatasetRecord], list[DatasetRecord]]:
    """Cap the easy share so a split is not dominated by easy templates (per pack). Returns (kept, dropped).
    A pack that is structurally easy (fewer than `min_hard_share` non-easy rows; its hard cases live in the challenge split) is left
    alone: trimming easy rows there would only shrink the data without adding diversity."""
    by_pack: dict[str, list[DatasetRecord]] = defaultdict(list)
    for r in records:
        by_pack[r.task_pack].append(r)
    kept: list[DatasetRecord] = []
    dropped: list[DatasetRecord] = []
    for rs in by_pack.values():
        easy = sorted([r for r in rs if r.quality.difficulty == "easy"], key=lambda r: r.record_id)
        rest = [r for r in rs if r.quality.difficulty != "easy"]
        if len(rs) < min_n or len(rest) < min_hard_share * len(rs):
            kept += rs
            continue
        # allowed easy count e satisfies e <= share * (e + len(rest))
        cap = int(max_easy_share * len(rest) / (1 - max_easy_share)) if max_easy_share < 1 else len(easy)
        cap = max(cap, int(max_easy_share * len(rs) * 0.5))
        kept += rest + easy[:cap]
        dropped += easy[cap:]
    return sorted(kept, key=lambda r: r.record_id), dropped
