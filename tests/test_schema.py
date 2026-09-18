import pytest

from open_spark_jev.schema import Answer, Choice, Noul, Score, parse_question


def test_choice_labels_and_abstain():
    q = Choice(prompt="route?", options=["a", "b"], allow_abstain=True)
    assert q.labels == ["a", "b", "abstain"]
    with pytest.raises(ValueError):
        Choice(prompt="x", options=["a", "a"])


def test_score_levels_vs_band():
    q = Score(prompt="risk?", levels=["low", "high"])
    assert q.level_values == [0.0, 1.0]
    b = Score(prompt="risk?", band={"min": 0, "max": 100, "steps": 5})
    assert b.labels == ["0", "25", "50", "75", "100"]
    with pytest.raises(ValueError):
        Score(prompt="x")


def test_answer_from_probs_noul_ignores_abstain_mass():
    q = Noul(prompt="claim", allow_abstain=True)
    a = Answer.from_probs(q, [0.3, 0.1, 0.6])
    assert a.abstained and a.selected == "abstain"
    assert abs(a.probability - 0.75) < 1e-9


def test_answer_score_expected_value():
    q = Score(prompt="s", band={"min": 0, "max": 10, "steps": 3})
    a = Answer.from_probs(q, [0.25, 0.5, 0.25])
    assert abs(a.expected_value - 5.0) < 1e-9


def test_parse_question_roundtrip():
    q = parse_question({"type": "choice", "prompt": "p", "options": ["x", "y"]})
    assert isinstance(q, Choice)
    with pytest.raises(ValueError):
        parse_question({"type": "nope", "prompt": "p"})
