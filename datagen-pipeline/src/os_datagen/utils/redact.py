"""Artifact cleanup: scrub secret-looking strings from raw generations and optionally delete raw files."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from ..config import RetentionConfig


def redact_text(text: str, patterns: list[str]) -> tuple[str, int]:
    n = 0
    for p in patterns:
        text, k = re.subn(p, "[REDACTED]", text)
        n += k
    return text, n


def redact_run(run: Path, cfg: RetentionConfig | None = None, delete_raw: bool = False) -> dict[str, int]:
    """Redact `raw/` generations in place (patterns from retention.redact_patterns), truncate to max_raw_bytes,
    or delete the whole raw directory. Accepted/rejected JSONL rows only hold artifact-relative paths, so they stay valid
    references (the file simply no longer exists when deleted)."""
    cfg = cfg or RetentionConfig()
    raw = run / "raw"
    stats = {"files": 0, "redactions": 0, "deleted": 0}
    if not raw.exists():
        return stats
    if delete_raw:
        stats["deleted"] = sum(1 for _ in raw.rglob("*") if _.is_file())
        shutil.rmtree(raw)
        return stats
    for f in raw.rglob("*.json"):
        text, n = redact_text(f.read_text(), cfg.redact_patterns)
        f.write_text(text[: cfg.max_raw_bytes * 4])
        stats["files"] += 1
        stats["redactions"] += n
    return stats
