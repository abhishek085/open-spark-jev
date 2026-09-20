"""Deterministic, fictional fixtures for the sandbox (CSV, SQLite, JSON)."""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from ..utils.seeds import make_rng


def make_csv(dir_: Path, name: str, column: str, seed: int, n_rows: int = 40) -> tuple[Path, float]:
    """Write a CSV; returns (path, ground-truth column total computed while generating, independent of the sandbox)."""
    rng = make_rng("csv", seed)
    total = 0.0
    path = dir_ / name
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "region", column])
        for i in range(n_rows):
            v = round(rng.uniform(10, 900), 2)
            total += v
            w.writerow([i, rng.choice(["north", "south", "east", "west"]), v])
    return path, round(total, 2)


def make_sqlite(dir_: Path, seed: int) -> Path:
    rng = make_rng("sqlite", seed)
    path = dir_ / "fixture.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, amount REAL)")
    con.executemany("INSERT INTO orders VALUES (?,?,?)",
                    [(i, rng.choice(["acme", "birch", "cedar"]), round(rng.uniform(5, 500), 2)) for i in range(30)])
    con.commit()
    con.close()
    return path


def make_json(dir_: Path, name: str, obj: dict[str, Any]) -> Path:
    p = dir_ / name
    p.write_text(json.dumps(obj))
    return p
