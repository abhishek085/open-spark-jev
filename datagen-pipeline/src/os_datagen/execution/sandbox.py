"""Restricted local sandbox. It NEVER executes text taken from generated data: only operations registered in
ALLOWED_OPS (or code snippets whose sha256 is in the fixture allowlist) can run, inside a temp directory,
with a timeout and resource limits. No network is opened by any operation."""
from __future__ import annotations

import csv
import json
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..schemas.execution import ExecutionRecord, SandboxOutcome, SandboxResult
from ..utils.hashing import sha256_text
from .validators import json_schema_valid


class SandboxViolation(RuntimeError):
    pass


def _csv_sum(root: Path, file: str, column: str) -> dict[str, Any]:
    with open(_safe(root, file)) as f:
        return {"total": round(sum(float(r[column]) for r in csv.DictReader(f)), 2)}


def _csv_count(root: Path, file: str, column: str, equals: str) -> dict[str, Any]:
    with open(_safe(root, file)) as f:
        return {"count": sum(1 for r in csv.DictReader(f) if r[column] == equals)}


def _json_validate(root: Path, file: str, schema: dict[str, Any]) -> dict[str, Any]:
    errs = json_schema_valid(json.loads(_safe(root, file).read_text()), schema)
    return {"valid": not errs, "errors": errs}


def _sqlite_query(root: Path, file: str, sql: str) -> dict[str, Any]:
    if not sql.lstrip().lower().startswith("select") or ";" in sql.strip().rstrip(";"):
        raise SandboxViolation("only a single read-only SELECT is allowed")
    con = sqlite3.connect(f"file:{_safe(root, file)}?mode=ro", uri=True)
    try:
        return {"rows": con.execute(sql).fetchall()}
    finally:
        con.close()


def _file_check(root: Path, file: str, min_bytes: int = 1) -> dict[str, Any]:
    p = root / file
    return {"exists": p.exists(), "nonempty": p.exists() and p.stat().st_size >= min_bytes}


def _safe(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise SandboxViolation(f"path escapes sandbox root: {rel}")
    return p


ALLOWED_OPS: dict[str, Callable[..., dict[str, Any]]] = {
    "csv_sum": _csv_sum, "csv_count": _csv_count, "json_validate": _json_validate,
    "sqlite_query_readonly": _sqlite_query, "file_check": _file_check,
}
# sha256 of fixture-owned snippets that may run in a subprocess (unit-test style checks)
ALLOWED_SNIPPETS: dict[str, str] = {}


def register_snippet(code: str) -> str:
    h = sha256_text(code)
    ALLOWED_SNIPPETS[h] = code
    return h


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))


class Sandbox:
    def __init__(self, root: Path | None = None):
        self._tmp = None if root else tempfile.TemporaryDirectory(prefix="osj-sbx-")
        self.root = root or Path(self._tmp.name)  # type: ignore[union-attr]

    def close(self) -> None:
        if self._tmp:
            self._tmp.cleanup()

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()

    def run_op(self, op: str, **params: Any) -> SandboxResult:
        if op not in ALLOWED_OPS:
            raise SandboxViolation(f"operation not allowlisted: {op}")
        t0 = time.time()
        try:
            out = ALLOWED_OPS[op](self.root, **params)
            return SandboxResult(ok=True, returncode=0, artifacts={**out, "runtime_ms": int((time.time() - t0) * 1000)})
        except SandboxViolation:
            raise
        except Exception as e:  # noqa: BLE001
            return SandboxResult(ok=False, returncode=1, stderr=f"{type(e).__name__}: {e}")

    def run_snippet(self, code: str, timeout_s: float = 5.0) -> SandboxResult:
        if sha256_text(code) not in ALLOWED_SNIPPETS:
            raise SandboxViolation("snippet is not in the fixture allowlist")
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], cwd=self.root, capture_output=True, text=True,
                               timeout=timeout_s, preexec_fn=_limits, env={"PATH": "/usr/bin"})
            return SandboxResult(ok=p.returncode == 0, returncode=p.returncode, stdout=p.stdout[-2000:], stderr=p.stderr[-2000:])
        except subprocess.TimeoutExpired:
            return SandboxResult(ok=False, returncode=-9, timed_out=True)


_counter = 0


def execution_record(scenario_id: str, action: str, res: SandboxResult, *, goal: bool, policy_ok: bool = True,
                     artifact_valid: bool = True, tests: tuple[int, int] = (0, 0), tool_calls: int = 1,
                     large_calls: int = 0, detail: dict[str, Any] | None = None) -> ExecutionRecord:
    global _counter
    _counter += 1
    return ExecutionRecord(
        execution_id=f"exec-{_counter:06d}", scenario_id=scenario_id, candidate_action=action,
        outcome=SandboxOutcome(exit_code=res.returncode, goal_completed=goal, policy_compliant=policy_ok,
                               artifact_valid=artifact_valid, tests_passed=tests[0], tests_failed=tests[1],
                               runtime_ms=int(res.artifacts.get("runtime_ms", 0)), tool_calls=tool_calls,
                               large_model_calls=large_calls), detail=detail or {})
