"""The public /v1/decide contract: request parsing and response shape (CPU only, no model)."""
import pytest

from open_spark_jev.schema import Answer, Choice, Noul, Score
from open_spark_jev.serve import contract

BODY = {
    "state": {"message": "I was charged twice."},
    "questions": [
        {"id": "dept", "type": "choice", "instructions": "Which team?",
         "options": [{"id": "billing", "definition": "Charges."}, {"id": "technical", "definition": "Bugs."}]},
        {"id": "refund", "type": "boolean", "instructions": "Explicit refund request?"},
        {"id": "urgency", "type": "score", "instructions": "How urgent?",
         "levels": [{"value": 0, "definition": "Routine"}, {"value": 1, "definition": "Soon"}]},
    ],
}


def test_detects_contract_shape_but_not_native():
    assert contract.is_contract(BODY)
    native = {"state": {"content": "x"}, "questions": [{"type": "noul", "prompt": "p"}]}
    assert not contract.is_contract(native)


def test_parse_maps_types_and_appends_definitions():
    state, qs, ids = contract.parse(BODY)
    assert ids == ["dept", "refund", "urgency"]
    assert isinstance(qs[0], Choice) and qs[0].options == ["billing", "technical"]
    assert "Option definitions:" in qs[0].prompt and "- billing: Charges." in qs[0].prompt
    assert isinstance(qs[1], Noul)
    assert isinstance(qs[2], Score) and qs[2].levels == ["0", "1"] and qs[2].rubric.startswith("0: Routine")
    assert state.content == {"message": "I was charged twice."}


def test_unknown_type_rejected():
    bad = {"state": {"a": 1}, "questions": [{"id": "q", "type": "ranking", "instructions": "x"}]}
    with pytest.raises(ValueError):
        contract.parse(bad)


def test_format_answer_shape_and_boolean_labels():
    _, qs, _ = contract.parse(BODY)
    a = Answer.from_probs(qs[0], [0.7, 0.3])
    out = contract.format_answer(a, 12.345)
    assert out["selected"] == "billing" and out["probabilities"] == {"billing": 0.7, "technical": 0.3}
    assert out["confidence"] == 0.7 and out["margin"] == 0.4 and out["latency_ms"] == 12.35 and out["entropy"] > 0
    b = contract.format_answer(Answer.from_probs(qs[1], [0.2, 0.8]), 1.0)
    assert set(b["probabilities"]) == {"true", "false"} and b["selected"] == "false"
    s = contract.format_answer(Answer.from_probs(qs[2], [0.1, 0.9]), 1.0)
    assert s["selected"] == "1"
