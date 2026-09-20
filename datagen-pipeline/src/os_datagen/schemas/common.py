from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

DecisionType = Literal["choice", "score", "boolean"]
SPLITS: tuple[str, ...] = ("train", "calibration", "locked_test", "challenge")
Namespace = Literal["foundation", "harness", "domain"]
TRUTH_TIERS = ("deterministic", "executable", "controlled_world", "rollout", "adjudicated", "model_consensus")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
