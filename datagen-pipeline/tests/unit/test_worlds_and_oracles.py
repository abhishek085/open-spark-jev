from __future__ import annotations

import pytest

from os_datagen.generation.oracle import solve
from os_datagen.schemas.common import SPLITS
from os_datagen.taskpacks.registry import get_pack
from os_datagen.utils.seeds import make_rng


def test_fixed_seed_identical_worlds(pack):
    for split in SPLITS:
        a = pack.sample_world(make_rng(42, pack.name, split), split)
        b = pack.sample_world(make_rng(42, pack.name, split), split)
        assert a.model_dump() == b.model_dump()


def test_every_family_has_a_legal_label(pack):
    """Oracle produces a label for every scenario family (incl. challenge) and it is a legal option."""
    seen: set[str] = set()
    for split in ("train", "challenge"):
        fams = pack.challenge_families if split == "challenge" else pack.families
        for i in range(400):
            w = pack.sample_world(make_rng("fam", pack.name, split, i), split)
            t = solve(pack, w)
            seen.add(w.scenario_family)
            assert pack.label_key(t) in pack.option_ids()
            if all(f in seen for f in fams) and i > 40:
                break
    assert seen >= set(pack.families) | set(pack.challenge_families)


def test_oracle_is_pure(pack):
    w = pack.sample_world(make_rng(1, pack.name), "train")
    assert solve(pack, w).model_dump() == solve(pack, w).model_dump()


def test_hidden_fields_never_in_render_request(pack):
    for split in SPLITS:
        w = pack.sample_world(make_rng(9, pack.name, split), split)
        t = solve(pack, w)
        req = pack.render_context(w, t)
        blob = str(req.template_vars) + str(req.world_facts)
        for secret in ("preferred_option", "acceptable_options", pack.oracle_version, "label_source"):
            if secret:
                assert str(secret) not in blob, (pack.name, secret)
        assert "hidden" not in req.world_facts and "doc_class" not in blob  # class is derived from features, never passed


EXPECT = {
    ("foundation_semantic_entailment_v1", "direct_fact"): {"entailed"},
    ("foundation_semantic_entailment_v1", "missing_evidence"): {"unknown"},
    ("foundation_semantic_entailment_v1", "ambiguous_wording"): {"unknown"},
    ("foundation_semantic_entailment_v1", "long_chain"): {"entailed"},
    ("foundation_semantic_entailment_v1", "double_negation"): {"entailed"},
    ("foundation_semantic_entailment_v1", "similar_entity"): {"unknown", "entailed"},
    ("foundation_authorization_gate_v1", "tenant_boundary"): {"deny"},
    ("foundation_authorization_gate_v1", "expired_access"): {"deny"},
    ("foundation_authorization_gate_v1", "purpose_limitation"): {"deny"},
    ("foundation_authorization_gate_v1", "restricted_needs_approval"): {"require_approval"},
    ("foundation_authorization_gate_v1", "owner_approval_given"): {"allow"},
    ("harness_tool_action_gate_v1", "malformed_json_argument"): {"repair_arguments"},
    ("harness_tool_action_gate_v1", "external_side_effect_no_approval"): {"require_confirmation"},
    ("harness_tool_action_gate_v1", "external_side_effect_approved"): {"allow"},
    ("harness_tool_action_gate_v1", "sensitive_exfiltration"): {"deny"},
    ("harness_tool_action_gate_v1", "path_traversal"): {"deny"},
    ("harness_tool_action_gate_v1", "tenant_boundary"): {"deny"},
    ("harness_tool_action_gate_v1", "hidden_side_effect"): {"require_confirmation"},
    ("harness_next_action_router_v1", "exact_local_calculation"): {"use_python"},
    ("harness_next_action_router_v1", "valid_cache"): {"use_cache"},
    ("harness_next_action_router_v1", "stale_cache"): {"use_python"},
    ("harness_next_action_router_v1", "cache_collision"): {"use_python"},
    ("harness_next_action_router_v1", "search_bait"): {"use_python"},
    ("harness_next_action_router_v1", "network_disallowed"): {"ask_user"},
    ("harness_next_action_router_v1", "repeated_failed_tool"): {"ask_user"},
    ("harness_next_action_router_v1", "missing_essential_information"): {"ask_user"},
    ("harness_termination_gate_v1", "complete_valid"): {"finish"},
    ("harness_termination_gate_v1", "hidden_failed_test"): {"repair"},
    ("harness_termination_gate_v1", "retry_budget_exhausted"): {"escalate"},
    ("harness_termination_gate_v1", "transient_within_budget"): {"retry"},
    ("harness_termination_gate_v1", "tool_success_but_artifact_absent"): {"retry"},
    ("harness_answer_sufficiency_v1", "complete_supported"): {"return"},
    ("harness_answer_sufficiency_v1", "injection_in_retrieval"): {"return"},
    ("harness_prompt_injection_gate_v1", "benign_document"): {"safe_data"},
    ("harness_prompt_injection_gate_v1", "exfiltration_request"): {"block_and_escalate"},
    ("harness_prompt_injection_gate_v1", "quoted_security_discussion"): {"safe_data"},
    ("harness_prompt_injection_gate_v1", "trusted_tool_command"): {"safe_data"},
    ("harness_prompt_injection_gate_v1", "trusted_side_effect_instruction"): {"require_confirmation"},
    ("harness_retrieval_gate_v1", "answer_in_context"): {"no_retrieval"},
    ("harness_retrieval_gate_v1", "answer_in_local_file"): {"read_local_context"},
    ("harness_retrieval_gate_v1", "stale_local_source"): {"search_web", "ask_user", "search_local_index"},
    ("harness_retrieval_gate_v1", "internal_target_web_available"): {"ask_user"},
    ("harness_retrieval_gate_v1", "current_public_fact"): {"search_web"},
    ("harness_retrieval_gate_v1", "source_unreadable"): {"search_local_index", "ask_user", "search_web"},
    ("harness_next_action_router_v1", "generic_guidance_request"): {"call_small_model"},
    ("harness_retrieval_gate_v1", "identifier_missing"): {"ask_user"},
    ("harness_retrieval_gate_v1", "web_instruction_inside_data"): {"read_local_context"},
    ("foundation_document_relevance_v1", "direct_answer"): {"directly_relevant"},
    ("foundation_document_relevance_v1", "partial_answer"): {"partially_relevant"},
    ("foundation_document_relevance_v1", "wrong_time_period"): {"irrelevant"},
    ("foundation_document_relevance_v1", "conflicting_authoritative"): {"conflicting"},
    ("foundation_extraction_validation_v1", "omitted_required"): {"incomplete"},
    ("foundation_extraction_validation_v1", "swapped_roles"): {"incorrect"},
    ("foundation_extraction_validation_v1", "source_ambiguity"): {"ambiguous"},
    ("foundation_extraction_validation_v1", "invalid_normalization"): {"incorrect"},
    ("foundation_temporal_reasoning_v1", "missing_time"): {"unknown"},
    ("foundation_temporal_reasoning_v1", "conflicting_sources"): {"conflicting"},
    ("foundation_temporal_reasoning_v1", "policy_window"): {"before", "during", "after"},
    ("foundation_rule_application_v1", "incomplete_premise"): {"false"},
    ("foundation_rule_application_v1", "exception_chain"): {"false"},
    ("foundation_communication_urgency_v1", "safety_report"): {"3"},
    ("foundation_communication_urgency_v1", "false_deadline"): {"0", "1"},
    ("foundation_communication_intent_v1", "quoted_other_sender"): {"request_information", "request_action", "report_problem", "provide_update",
                                                                     "schedule_or_reschedule", "cancel_or_decline", "feedback_or_complaint"},
}


@pytest.mark.parametrize("key", sorted(EXPECT))
def test_family_golden_labels(key):
    name, family = key
    p = get_pack(name)
    n = 0
    for i in range(3000):
        split = "challenge" if family in p.challenge_families else "train"
        w = p.sample_world(make_rng("gold", name, family, i), split, families=[family])
        if w.scenario_family != family:
            continue
        n += 1
        assert p.label_key(solve(p, w)) in EXPECT[key], (key, w.facts)
        if n >= 25:
            break
    assert n >= 5, f"family {family} never sampled"


def test_oracle_invariant_to_json_roundtrip(pack):
    """Worlds are stored as (sorted-key) JSON and reloaded: the oracle must give the identical TruthRecord."""
    import json

    from os_datagen.schemas.scenario import ScenarioWorld

    for split in ("train", "challenge"):
        for i in range(60):
            w = pack.sample_world(make_rng("rt", pack.name, split, i), split)
            w2 = ScenarioWorld.model_validate(json.loads(json.dumps(w.model_dump(mode="json"), sort_keys=True)))
            assert solve(pack, w).model_dump() == solve(pack, w2).model_dump(), (pack.name, w.scenario_family)
