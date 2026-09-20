from __future__ import annotations

import subprocess
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]  # datagen-pipeline/


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def git_commit() -> str:
    """HEAD SHA, suffixed with +dirty_worktree when tracked files under this project are modified."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return sha + ("+dirty_worktree" if dirty else "")
    except Exception:
        return "unknown"


def rel_to(path: Path, base: Path) -> str:
    """Artifact-relative path only (never absolute) for provenance."""
    return str(path.resolve().relative_to(base.resolve()))
