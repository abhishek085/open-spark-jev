"""The control families' gold must be recomputable, and their rows must match the decision schema.

The schema half of this matters more than it looks: os_datagen.programmatic's first training launch
died on row loading because ``noul`` rows were written with true/false labels while ``schema.Noul``
renders yes/no. That is a class of bug the gold verifier cannot catch, because the gold was right --
it was the label space that was wrong. Both halves are checked here.
"""
from __future__ import annotations

import pytest

from os_datagen.control import FAMILIES, SPLITS, generate, verify

N = 40


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_gold_recomputes_independently(family: str) -> None:
    rows = [r for s in SPLITS for r in generate(family, s, N)]
    ok, bad = verify(rows)
    assert not bad, f"{family}: gold verification failed for {bad[:5]}"
    assert ok == len(rows)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_rows_are_deterministic(family: str) -> None:
    assert generate(family, "train", N) == generate(family, "train", N)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_splits_do_not_share_states(family: str) -> None:
    """A world may not appear in two splits: the seed namespace includes the split name."""
    seen: dict[str, str] = {}
    for split in SPLITS:
        for r in generate(family, split, N):
            state = r["state"]["content"]
            assert state not in seen, f"{family}: state shared by {seen.get(state)} and {split}"
            seen[state] = split


def test_options_are_small() -> None:
    """The point of these families is JevControl's actual decision-site shape: a handful of options,
    not JevBench's up-to-26-option choice questions."""
    for f in FAMILIES:
        for r in generate(f, "train", N):
            if r["question"]["type"] == "choice":
                assert len(r["question"]["options"]) <= 5, f"{f}: too many options for a decision-site call"


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_rows_load_through_the_decision_schema(family: str) -> None:
    """Round-trip every row through open_spark_jev's Record, the loader training actually uses."""
    corpus = pytest.importorskip("open_spark_jev.data.corpus")
    for row in generate(family, "train", N):
        rec = corpus.Record.from_json(row)
        q = rec.question_obj()
        assert rec.target["label"] in q.labels, f"{rec.id}: gold {rec.target['label']!r} not in {q.labels}"
        dist = rec.target_dist()
        assert dist is not None and abs(sum(dist) - 1.0) < 1e-6
        assert rec.state_obj().content
        assert rec.label_index() == q.labels.index(rec.target["label"])


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_gold_label_is_not_leaked_into_the_state(family: str) -> None:
    """The rationale explains the answer, so it must stay in meta and never reach the visible state."""
    for row in generate(family, "train", N):
        state = row["state"]["content"].lower()
        for banned in ("rationale", "gold", "correct answer", "expected answer"):
            assert banned not in state, f"{row['id']}: {banned!r} leaked into the state"
