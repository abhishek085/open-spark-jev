from __future__ import annotations

from ..schemas.common import SPLITS

FILE_FOR_SPLIT = {"train": "accepted_train.jsonl", "calibration": "accepted_calibration.jsonl",
                  "locked_test": "accepted_test_locked.jsonl", "challenge": "accepted_challenge.jsonl"}


def split_counts(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Per-split world counts from a total (largest-remainder; every split with share>0 gets >=1 when total allows)."""
    raw = {s: total * shares.get(s, 0) for s in SPLITS}
    out = {s: int(v) for s, v in raw.items()}
    rem = total - sum(out.values())
    for s in sorted(SPLITS, key=lambda s: raw[s] - out[s], reverse=True)[:rem]:
        out[s] += 1
    for s in SPLITS:
        if shares.get(s, 0) > 0 and out[s] == 0 and total >= len(SPLITS):
            donor = max(out, key=lambda k: out[k])
            out[donor] -= 1
            out[s] = 1
    return out
