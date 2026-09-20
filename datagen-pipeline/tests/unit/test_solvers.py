from __future__ import annotations

from datetime import date

import pytest

from os_datagen.execution.validators import has_path_traversal, json_schema_valid, validate_arguments
from os_datagen.generation.symbolic import closure, entailment_verdict, literal_status
from os_datagen.taskpacks.foundation.authorization_gate import authorize
from os_datagen.taskpacks.foundation.document_type import classify
from os_datagen.taskpacks.foundation.extraction_validation import judge_extraction
from os_datagen.taskpacks.foundation.temporal_reasoning import relate
from os_datagen.taskpacks.harness.next_action_router import route
from os_datagen.taskpacks.harness.tool_action_gate import gate


def F(pred, ent, pos=True, auth=1):
    return {"pred": pred, "ent": ent, "pos": pos, "auth": auth}


def R(quant, p, q):
    return {"quant": quant, "body": [[p, True]], "head": [q, quant != "no"]}


def test_spec_example_multi_hop():
    # "All red objects are fragile. Every fragile object requires careful handling. The vase is red."
    facts = [F("red", "vase")]
    rules = [R("all", "red", "fragile"), R("all", "fragile", "careful")]
    assert entailment_verdict(facts, rules, {"pred": "careful", "ent": "vase", "pos": True}) == "entailed"


def test_entailment_verdicts():
    rules = [R("no", "red", "fragile")]
    assert entailment_verdict([F("red", "v")], rules, {"pred": "fragile", "ent": "v", "pos": True}) == "contradicted"
    assert entailment_verdict([F("blue", "v")], rules, {"pred": "fragile", "ent": "v", "pos": True}) == "unknown"
    # 'some' licenses nothing
    assert entailment_verdict([F("red", "v")], [R("some", "red", "fragile")], {"pred": "fragile", "ent": "v", "pos": True}) == "unknown"
    # source authority resolves conflicts; ties are unknown
    assert entailment_verdict([F("p", "e", True, 2), F("p", "e", False, 1)], [], {"pred": "p", "ent": "e", "pos": True}) == "entailed"
    assert entailment_verdict([F("p", "e", True, 1), F("p", "e", False, 1)], [], {"pred": "p", "ent": "e", "pos": True}) == "unknown"
    assert literal_status(closure([F("p", "e", True, 1), F("p", "e", False, 1)], []), "p", "e", True) == "conflict"


def test_exception_rules_are_order_independent():
    facts = [F('a', 'e'), F('b', 'e')]
    rules = [{'quant': 'all', 'body': [['a', True]], 'head': ['c', True], 'unless': ['d']}, R('all', 'b', 'd')]
    for order in (rules, rules[::-1]):
        assert literal_status(closure(facts, order), 'c', 'e', True) == 'unknown'


def test_document_type_policy():
    assert classify({"invoice_number", "amount_due", "payment_terms"})[0] == "invoice"
    assert classify({"problem_description", "help_request", "sender_contact", "mentions_invoice_reference"})[0] == "support_request"
    assert classify({"receipt_number", "amount_paid", "payment_method", "refund_language"})[0] == "receipt"
    assert classify({"free_text_paragraph"})[0] == "other"
    assert classify({"invoice_number"})[0] == "other"  # coverage 1/3 < 0.6


def _ex(src, prop, req=None, amb=None):
    return judge_extraction({"source_fields": src, "proposed": prop, "required_fields": req or list(src), "ambiguous_field": amb})[0]


def test_extraction_comparison():
    src = {"event": "Meeting", "date": "October 14", "time": "2:00 PM"}
    assert _ex(src, dict(src)) == "correct"
    assert _ex(src, {**src, "date": "October 15"}) == "incorrect"
    assert _ex(src, {"event": "Meeting", "date": "October 14"}) == "incomplete"
    assert _ex(src, dict(src), amb="date") == "ambiguous"
    assert _ex(src, {"event": "Meeting", "date": "2:00 PM", "time": "October 14"}) == "incorrect"  # swapped roles


def _ev(when, tz="America/New_York", prec="timestamp", dur=None, auth=1):
    return {"when": when, "tz": tz, "precision": prec, "duration_min": dur, "authority": auth}


def _rel(a, b):
    return relate({"events": {"A": {"sources": a}, "B": {"sources": b}}, "question": {"a": "A", "b": "B"}})[0]


def test_temporal_relations():
    assert _rel([_ev("2026-03-08T01:30", dur=30)], [_ev("2026-03-08T03:30", dur=30)]) == "before"  # DST gap
    assert _rel([_ev("2026-03-08T20:00", "Asia/Tokyo", dur=60)], [_ev("2026-03-07T23:00", "America/New_York", dur=60)]) == "after"  # 11:00Z vs 04:00Z
    assert _rel([_ev("2026-03-08T09:00", "Asia/Tokyo", dur=60)], [_ev("2026-03-07T23:00", "America/New_York", dur=60)]) == "before"  # 00:00Z vs 04:00Z
    assert _rel([_ev("2026-03-08T10:00", dur=30)], [_ev("2026-03-08T09:00", dur=180)]) == "during"
    assert _rel([_ev(None)], [_ev("2026-03-08T09:00")]) == "unknown"
    assert _rel([_ev("2026-03-08", prec="date")], [_ev("2026-03-08T12:00", dur=60)]) == "unknown"
    assert _rel([_ev("2026-03-08T09:00", auth=2), _ev("2026-03-09T09:00", auth=2)], [_ev("2026-04-08T09:00")]) == "conflicting"
    assert _rel([_ev("2026-03-08T09:00", auth=2), _ev("2026-03-09T09:00", auth=1)], [_ev("2026-04-08T09:00")]) == "before"


def _auth(**over):
    f = {"requester": {"role": "analyst", "permissions": ["read", "read_aggregates"], "tenant": "t1", "external_collaborator": False},
         "resource": {"name": "payroll_export", "classification": "internal", "tenant": "t1", "allowed_purposes": ["monthly reporting"],
                      "includes_prohibited_field": False, "shared_with_partners": False},
         "action": "read", "purpose": "monthly reporting", "access": {"today": "2026-09-19", "expires": None, "owner_approval": False, "delegation": None}}
    for k, v in over.items():
        f[k] = {**f[k], **v} if isinstance(f[k], dict) else v
    return authorize(f)[0]


def test_authorization_policy():
    assert _auth() == "allow"
    # spec example: analyst with aggregate-only permission asks for a restricted export -> deny
    assert _auth(action="export", resource={"classification": "restricted"}) == "deny"
    assert _auth(resource={"classification": "restricted"}) == "require_approval"
    assert _auth(resource={"classification": "restricted"}, access={"owner_approval": True}) == "allow"
    assert _auth(resource={"tenant": "t2"}) == "deny"
    assert _auth(access={"expires": "2026-09-01"}) == "deny"
    assert _auth(access={"expires": "2026-12-01"}) == "allow"
    assert _auth(purpose="curiosity") == "deny"
    assert _auth(requester={"external_collaborator": True}, resource={"classification": "confidential"}) == "require_approval"


def test_validators():
    schema = {"required": ["query"], "types": {"query": "string", "limit": "integer"}}
    assert validate_arguments(schema, {"query": "x", "limit": 3}) == []
    assert validate_arguments(schema, {"limit": 3}) == ["missing_required:query"]
    assert validate_arguments(schema, {"query": "x", "limit": "3"}) == ["wrong_type:limit"]
    assert validate_arguments(schema, {"_raw_arguments": '{"query": }'}) == ["malformed_json"]
    assert has_path_traversal({"path": "../../etc/passwd"}) and not has_path_traversal({"path": "reports/a.txt"})
    assert json_schema_valid({"a": 1}, {"type": "object", "required": ["a", "b"]}) == ["required:b"]


def _tool(**over):
    f = {"tool": {"name": "search_notes", "schema": {"required": ["query"], "types": {"query": "string"}},
                  "metadata": {"side_effect": "none", "available": True}},
         "arguments": {"query": "q"}, "target": {"scope": "internal", "tenant": "a", "session_tenant": "a"}, "approval": "none",
         "data_sensitivity": "none", "prior_identical_failure": False, "alternative_tool": None}
    for k, v in over.items():
        f[k] = v if k == "arguments" else ({**f[k], **v} if isinstance(f[k], dict) and isinstance(v, dict) else v)
    return gate(f)[0]


def test_tool_gate_policy():
    assert _tool() == "allow"
    assert _tool(arguments={}) == "repair_arguments"
    assert _tool(tool={"metadata": {"side_effect": "external_write", "available": True}}) == "require_confirmation"
    assert _tool(tool={"metadata": {"side_effect": "external_write", "available": True}}, approval="granted") == "allow"
    assert _tool(target={"scope": "external"}, data_sensitivity="sensitive") == "deny"
    assert _tool(tool={"metadata": {"side_effect": "none", "available": False}}, alternative_tool="list_documents") == "choose_alternative"
    assert _tool(arguments={"query": "../../etc/passwd"}) == "deny"
    assert _tool(target={"session_tenant": "b"}) == "deny"


def _router(**over):
    f = {"cache": {"available": False, "key_matches": False, "age_days": 0, "ttl_days": 30, "artifact_validated": False},
         "network_allowed": True, "must_be_exact": True, "user_information_complete": True, "task_complexity": "low",
         "needs_fresh_info": False, "local_context_contains_answer": False, "disallowed_actions": [], "prior_failed": None,
         "available_files": ["sales.csv"]}
    for k, v in over.items():
        f[k] = v
    return route(f)[0]


def test_router_spec_example_and_fallbacks():
    assert _router() == "use_python"  # the brief's exact-CSV example
    assert _router(cache={"available": True, "key_matches": True, "age_days": 3, "ttl_days": 30, "artifact_validated": True}) == "use_cache"
    assert _router(cache={"available": True, "key_matches": True, "age_days": 90, "ttl_days": 30, "artifact_validated": True}) == "use_python"
    assert _router(must_be_exact=False, available_files=[], task_complexity="high", disallowed_actions=["call_large_model"]) == "call_small_model"
    assert _router(needs_fresh_info=True, network_allowed=False) == "ask_user"


@pytest.mark.parametrize("a,b", [(date(2026, 1, 1), 1)])
def test_noop(a, b):
    assert a and b


def test_rule_notation_drift_is_canonicalised():
    from os_datagen.taskpacks.foundation.rule_application import RuleApplication

    c = RuleApplication._canon_rule
    assert c("flammable|unless:fragile>waterproof") == c("flammable>waterproof|unless:fragile") == "flammable>waterproof|unless:fragile"
    assert c("Shiny & red > heavy") == "red&shiny>heavy" and c("score>=70>eligible") == "score>=70>eligible"


def test_temporal_verifier_notation_is_canonicalised():
    from os_datagen.schemas.scenario import ScenarioWorld
    from os_datagen.taskpacks.foundation.temporal_reasoning import TemporalReasoning
    from os_datagen.utils.seeds import make_rng

    p = TemporalReasoning()
    w = p.sample_world(make_rng(3), "train", families=["missing_time"])
    exp, model = p.expected_extraction(w), p.verifier_schema()
    got = {"event_times": [x.replace("launch|unknown", "launch event|unknown|Asia/Tokyo") for x in exp["event_times"]], "question_events": "the launch|the review"}
    assert p.compare_facts(w, model.model_validate(got)).ok
    assert not p.compare_facts(w, model.model_validate({**got, "question_events": "review|launch"})).ok  # direction still matters
    assert isinstance(w, ScenarioWorld)
