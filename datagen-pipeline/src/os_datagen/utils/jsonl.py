from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            obj = r.model_dump(mode="json") if isinstance(r, BaseModel) else r
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
