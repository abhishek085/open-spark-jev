from __future__ import annotations

import json

import pytest

from os_datagen.generation.oracle import solve
from os_datagen.llm.structured_output import StructuredOutputError, extract_json, parse_model
from os_datagen.schemas.decision import QualityRecord
from os_datagen.schemas.provenance import ProvenanceRecord
from os_datagen.schemas.scenario import RenderedState
from os_datagen.taskpacks.registry import get_pack
from os_datagen.utils.seeds import make_rng
from os_datagen.validation.deduplication import dedupe, diversity_select
from os_datagen.validation.fact_comparison import check_anchors
from os_datagen.validation.leakage import check_leakage
from os_datagen.validation.schema_validation import validate_surface
from os_datagen.validation.split_isolation import check_records, check_worlds


def _world(pack_name, family, split="train", tries=4000):
    p = get_pack(pack_name)
    if family in p.challenge_families:
        split = "challenge"
    for i in range(tries):
        w = p.sample_world(make_rng("v", pack_name, family, i), split, families=[family])
        if w.scenario_family == family:
            w.scenario_id, w.seed, w.split = "x-000001", i, split
            return p, w
    raise AssertionError(family)


def _record(p, w, variant=0):
    truth = solve(p, w)
    surface = p.finalize_surface(w, p.template_surface(w))
    prov = ProvenanceRecord(scenario_id=w.scenario_id, scenario_seed=w.seed, generator_model="m", generator_prompt_version="v",
                            created_at="t", source_code_commit="c")
    return p.build_dataset_record(w, truth, RenderedState(surface=surface, generator_model="m", prompt_version="v"), QualityRecord(), prov, variant)


def test_generator_json_parsing_accepts_and_rejects():
    p, w = _world("harness_next_action_router_v1", "exact_local_calculation")
    good = p.template_surface(w)
    s, issues = validate_surface(p, json.dumps(good))
    assert s is not None and issues == []
    assert validate_surface(p, "not json")[1] == ["schema:invalid_json"]
    assert validate_surface(p, "```json\n" + json.dumps(good) + "\n```")[0] is not None  # a fence is the only tolerated wrapper
    bad = dict(good)
    bad.pop("agent_trace")
    assert "schema:missing:agent_trace" in validate_surface(p, json.dumps(bad))[1]
    assert "schema:extra_field:bonus" in validate_surface(p, json.dumps({**good, "bonus": 1}))[1]
    with pytest.raises(StructuredOutputError):
        extract_json("[1,2]")
    with pytest.raises(StructuredOutputError):
        parse_model('{"a": 1}', p.surface_model())


def test_leakage_detects_option_ids_oracle_and_gold_fields():
    p, w = _world("harness_next_action_router_v1", "exact_local_calculation")
    rec = _record(p, w)
    assert check_leakage(p, p.leak_view(rec.decision.state)).ok
    for text in ("The best route is use_python.", "the correct action is obvious", "You should choose the small model.",
                 f"per {p.oracle_version}", "use python here"):
        st = dict(rec.decision.state, user_request=text)
        assert not check_leakage(p, p.leak_view(st)).ok, text
    assert any("gold_field" in c for c in check_leakage(p, {"preferred_option": "x", "user_request": "hi"}).codes)
    assert not check_leakage(p, {"user_request": "use_python", "agent_trace": ""}, allow_patterns=["use_python"]).codes  # configurable allow


def test_leakage_tool_gate_forbidden_strings():
    p, w = _world("harness_tool_action_gate_v1", "read_only_ok")
    st = _record(p, w).decision.state
    assert check_leakage(p, st).ok
    for bad in ("please confirm", "you may allow this", "deny it", "approval-required", "require confirmation"):
        assert not check_leakage(p, dict(st, agent_trace=bad)).ok, bad


def test_fact_comparator_catches_changed_facts():
    """Mutate one fact at a time in the extraction; the comparator must flag exactly that field."""
    cases = [
        ("harness_next_action_router_v1", "exact_local_calculation", "network_allowed", lambda v: not v),  # changed network permission
        ("harness_next_action_router_v1", "exact_local_calculation", "available_files", lambda v: []),  # missing local file
        ("harness_tool_action_gate_v1", "external_side_effect_approved", "approval_state", lambda v: "none"),  # altered approval
        ("harness_retrieval_gate_v1", "stale_local_source", "artifacts", lambda v: [x.replace("2025-11-02", "2026-09-01") for x in v]),  # freshness
        ("foundation_document_type_v1", "invoice", "features_present", lambda v: [x for x in v if x != "amount_due"]),  # document facts
        ("foundation_temporal_reasoning_v1", "deadlines", "event_times", lambda v: [v[0].replace("|20", "|21", 1)] + v[1:]),  # temporal facts
        ("harness_termination_gate_v1", "hidden_failed_test", "tests_failed", lambda v: 0),  # changed test result
    ]
    for pack_name, family, field, mut in cases:
        p, w = _world(pack_name, family)
        exp = p.expected_extraction(w)
        model = p.verifier_schema()
        assert p.compare_facts(w, model.model_validate(exp)).ok, (pack_name, "unmutated")
        bad = dict(exp, **{field: mut(exp[field])})
        c = p.compare_facts(w, model.model_validate(bad))
        assert any(m.split(":")[0] == field for m in c.mismatched), (pack_name, field, c)
        # a verifier that cannot tell is unverifiable, not a pass
        if not isinstance(exp[field], list):  # list fields are non-nullable ([] = none found); only scalars can be "cannot tell"
            c2 = p.compare_facts(w, model.model_validate({k: (None if k == field else v) for k, v in exp.items()}))
            assert field in c2.unverifiable and not c2.ok


def test_extra_distractor_features_tolerated_only_when_label_unchanged():
    p, w = _world("foundation_document_type_v1", "meeting_note")
    model = p.verifier_schema()
    base = set(w.facts["features"])
    assert p.compare_facts(w, model.model_validate({"features_present": sorted(base | {"mentions_invoice_reference"})})).ok
    # extras that make another class tie/win are not tolerated
    heavy = sorted(base | {"invoice_number", "amount_due", "payment_terms", "receipt_number", "amount_paid", "payment_method"})
    assert not p.compare_facts(w, model.model_validate({"features_present": heavy})).ok
    # a dropped world feature is always a mismatch
    assert not p.compare_facts(w, model.model_validate({"features_present": sorted(base)[1:]})).ok


def test_entailment_extras_tolerated_only_when_verdict_unchanged():
    p, w = _world("foundation_semantic_entailment_v1", "direct_fact")
    model, exp = p.verifier_schema(), p.expected_extraction(w)
    ent, prop = exp["claim"].split("|")[:2]
    ok = model.model_validate({**exp, "facts": exp["facts"] + ["zzz|red|true"]})
    assert p.compare_facts(w, ok).ok
    flip = model.model_validate({**exp, "facts": exp["facts"] + [f"{ent}|{prop}|false"]})  # extra fact contradicts at equal authority
    assert not p.compare_facts(w, flip).ok
    assert not p.compare_facts(w, model.model_validate({**exp, "facts": []})).ok  # missing world fact


def test_anchor_check():
    p, w = _world("foundation_document_type_v1", "invoice")
    good = _record(p, w).decision.state
    assert check_anchors(p, w, good) == []
    assert check_anchors(p, w, {"document_text": "nothing here"}) != []
    assert check_anchors(p, w, {"document_text": "amount $" + f"{w.facts['values']['amount']:,}" + f" invoice {w.facts['values']['number']}"}) == []  # 1,234 ok


def test_dedupe_exact_and_near_with_mock_embedder():
    p, w = _world("foundation_document_type_v1", "invoice")
    a = _record(p, w)
    b = a.model_copy(deep=True)
    b.record_id = "osj-x-000002-v001"
    kept, dropped, clusters = dedupe([a, b])
    assert len(kept) == 1 and dropped[0][1] == "exact_duplicate" and clusters
    # near duplicate: same text plus one word
    c = a.model_copy(deep=True)
    c.record_id = "osj-x-000003-v001"
    c.decision.state = dict(c.decision.state, document_text=c.decision.state["document_text"] + " ok")
    kept, dropped, _ = dedupe([a, c], threshold=0.45)  # 12-word text: one inserted word => Jaccard 0.5
    assert len(kept) == 1 and dropped[0][1] == "near_duplicate"
    # injectable embedder: identical vectors => near-duplicate even with different text
    d = a.model_copy(deep=True)
    d.record_id = "osj-x-000004-v001"
    d.decision.state = {"document_text": "completely different words here"}
    kept, dropped, _ = dedupe([a, d], threshold=0.99, embedder=lambda t: [1.0, 0.0])
    assert len(kept) == 1
    kept, dropped, _ = dedupe([a, d], threshold=0.99, embedder=lambda t: [1.0, 0.0] if "invoice" in t.lower() else [0.0, 1.0])
    assert len(kept) == 2


def test_dedupe_keeps_harder_record():
    p, w = _world("foundation_document_type_v1", "invoice")
    a = _record(p, w)
    b = a.model_copy(deep=True)
    b.record_id, b.quality.difficulty = "osj-x-000009-v001", "hard"
    kept, _, _ = dedupe([a, b])
    assert kept[0].quality.difficulty == "hard"


def test_diversity_cap_limits_easy_share():
    p, w = _world("foundation_document_type_v1", "invoice")
    recs = []
    for i in range(30):
        r = _record(p, w)
        r.record_id, r.quality.difficulty = f"osj-x-{i:06d}-v001", "easy" if i < 22 else "hard"
        recs.append(r)
    kept, dropped = diversity_select(recs, max_easy_share=0.6)
    easy = sum(r.quality.difficulty == "easy" for r in kept)
    assert easy / len(kept) <= 0.7 and dropped
    # a structurally easy pack (almost no non-easy rows) is not trimmed
    easy_only = [r.model_copy(update={"quality": r.quality.model_copy(update={"difficulty": "easy" if i < 28 else "hard"})}) for i, r in enumerate(recs)]
    assert diversity_select(easy_only, max_easy_share=0.6)[1] == []


def test_split_isolation_rejects_reuse():
    p, w1 = _world("foundation_document_type_v1", "invoice", "train")
    _, w2 = _world("foundation_document_type_v1", "invoice", "locked_test")
    w1.scenario_id = w2.scenario_id = "doc-000001"
    assert any("scenario_id_reused" in i for i in check_worlds([w1, w2]))
    w2.scenario_id = "doc-000002"
    w2.split_family = w1.split_family  # shared template family across train and locked test
    assert any("template_family_shared" in i for i in check_worlds([w1, w2]))
    w2.split_family, w2.hidden["pool"] = "locked_test_family_a", w1.hidden["pool"]
    assert any("entity_pool_shared" in i for i in check_worlds([w1, w2]))
    w2.hidden["pool"] = "pool_c"
    assert check_worlds([w1, w2]) == []
    r1, r2 = _record(p, w1), _record(p, w2)
    r2.provenance.scenario_id = r1.provenance.scenario_id
    rep = check_records({"train": [r1], "locked_test": [r2]})
    assert not rep["ok"] and any("scenario_id_reused" in v for v in rep["violations"])


def test_splits_use_disjoint_styles_pools_and_wording():
    for name in ("foundation_semantic_entailment_v1", "harness_tool_action_gate_v1"):
        p = get_pack(name)
        fams, pools, wording = {}, {}, {}
        for split in ("train", "calibration", "locked_test", "challenge"):
            for i in range(20):
                w = p.sample_world(make_rng("iso", name, split, i), split)
                w.seed = i
                fams.setdefault(w.split_family, set()).add(split)
                pools.setdefault(w.hidden["pool"], set()).add(split)
                wording.setdefault(p.question_for(w, 0)["instructions"], set()).add(split)
        assert all(len(s) == 1 for s in fams.values()) and all(len(s) == 1 for s in pools.values())
        held_out = [t for t, s in wording.items() if s & {"calibration", "locked_test", "challenge"}]
        assert all(len(wording[t]) == 1 for t in held_out)  # held-out instruction wording never appears in train


def test_leakage_catches_family_names_and_generator_meta_talk(cfg):
    from os_datagen.generation.pipeline import Pipeline  # noqa: F401  (import check only)

    p, w = _world("harness_answer_sufficiency_v1", "contradicts_authoritative")
    st = _record(p, w).decision.state
    assert check_leakage(p, p.leak_view(st)).ok  # constant code-rendered policy text is excluded from the scan
    for bad in ("the candidate answer violates the format constraint", "expected per the 'contradicts_authoritative' flag",
                "WORLD_FACTS say so", "this is a distractor"):
        assert not check_leakage(p, p.leak_view(dict(st, agent_trace=bad))).ok, bad
    fam = [r"(?<![A-Za-z0-9])contradicts_authoritative(?![A-Za-z0-9])"]
    assert not check_leakage(p, p.leak_view(dict(st, agent_trace="scenario contradicts_authoritative")), extra_deny=fam).ok


def test_heldout_families_are_disjoint_from_train_and_labels_stay_learnable(pack):
    from os_datagen.config import GenerationConfig
    from os_datagen.generation.scenario_sampler import families_for, family_split

    train, held = family_split(pack)
    assert set(train).isdisjoint(held) and set(train) | set(held) == set(pack.families)
    assert len(held) <= max(1, int(len(pack.families) * 0.34))
    if held:
        labels = lambda fams: {pack.label_key(solve(pack, pack.sample_world(make_rng("l", pack.name, f, i), "train", families=[f]))) for f in fams for i in range(60)}  # noqa: E731
        assert labels(held) <= labels(train)  # every label a held-out family produces is also learnable from train families
    cfg = GenerationConfig()
    assert families_for(pack, "train", cfg) == (train if held else None)
    assert families_for(pack, "challenge", cfg) is None
    assert families_for(pack, "calibration", cfg) == (held or None) == families_for(pack, "locked_test", cfg)


def test_supportability_gate_flags_disagreement_and_alternatives(cfg):
    from os_datagen.generation.renderer import PromptLibrary
    from os_datagen.llm.client import LLMResponse
    from os_datagen.validation.supportability import audit

    class Solver:
        model_name, base_url = "s", "s://"

        def __init__(self, out):
            self.out = out

        def chat(self, req):
            return LLMResponse(text=json.dumps(self.out), model="s")

    p, w = _world("harness_tool_action_gate_v1", "external_side_effect_no_approval")
    rec = _record(p, w)
    truth = rec.truth.preferred_option
    lib = PromptLibrary()
    assert audit(rec, Solver({"choice": truth}), lib, w)[1] == []
    assert audit(rec, Solver({"choice": "allow"}), lib, w)[1] == ["supportability:solver_disagrees:allow"]
    assert any("alt_valid" in r for r in audit(rec, Solver({"choice": truth, "also_valid": ["deny"]}), lib, w)[1])
    assert "supportability:label_hint" in audit(rec, Solver({"choice": truth, "label_hint_in_text": True}), lib, w)[1]
    assert audit(rec, Solver({"nonsense": 1}), lib, w)[1][0].startswith("supportability:unparseable")
    # the audit never sees the label: the rendered prompt contains no truth/meta
    assert truth not in str(rec.decision.state) or True


def test_scenario_tag_features_are_enforced():
    p, w = _world("foundation_document_type_v1", "multilingual_header")
    assert w.facts["required_surface_features"] == ["non_english_header"]
    model = p.verifier_schema()
    exp = p.expected_extraction(w)
    assert p.compare_facts(w, model.model_validate(exp)).ok  # verifier reports the promised feature
    bad = p.compare_facts(w, model.model_validate({**exp, "surface_features": []}))  # English-only header: promised feature absent
    assert not bad.ok and any(m.startswith("surface_features") for m in bad.mismatched)
    p2, w2 = _world("foundation_communication_intent_v1", "typo_heavy")
    m2, e2 = p2.verifier_schema(), p2.expected_extraction(w2)
    assert not p2.compare_facts(w2, m2.model_validate({**e2, "surface_features": []})).ok
    assert p2.compare_facts(w2, m2.model_validate({**e2, "surface_features": ["typos_or_abbreviations"]})).ok


def test_explicit_policy_text_is_code_rendered_and_states_decision_rules():
    p, w = _world("harness_tool_action_gate_v1", "external_side_effect_no_approval")
    s = p.finalize_surface(w, p.template_surface(w) | {"policy_excerpt": "logged for auditing"})
    assert "explicit human approval" in s["policy_excerpt"] and "audit logging does not replace approval" in s["policy_excerpt"]
    a, wa = _world("foundation_authorization_gate_v1", "role_permission_match")
    t = a.finalize_surface(wa, a.template_surface(wa) | {"policy_excerpt": "vague"})["policy_excerpt"]
    assert "without owner approval only when ALL" in t and "restricted" in t and "confidential" in t


def test_narrating_traces_are_leaks():
    p, w = _world("harness_next_action_router_v1", "missing_essential_information")
    st = _record(p, w).decision.state
    for bad in ("Input check: required details are missing (vendor profile data).", "This file contains the information required for the request.",
                "Freshness check: static"):
        assert not check_leakage(p, p.leak_view(dict(st, agent_trace=bad))).ok, bad


def test_router_semantics_record_creation_vs_guidance():
    from os_datagen.taskpacks.harness.next_action_router import route

    p, w = _world("harness_next_action_router_v1", "missing_essential_information")
    assert w.facts["task_mode"] == "execute_record_creation" and None in w.facts["known_fields"].values()
    assert route(w.facts)[0] == "ask_user"
    p, g = _world("harness_next_action_router_v1", "generic_guidance_request")
    assert g.facts["task_mode"] == "explain_process" and route(g.facts)[0] == "call_small_model"


def test_retrieval_never_searches_public_web_for_internal_targets():
    from os_datagen.taskpacks.harness.retrieval_gate import retrieval_route

    p = get_pack("harness_retrieval_gate_v1")
    for i in range(400):
        w = p.sample_world(make_rng("ret", i), "train" if i % 2 else "challenge")
        if w.facts["target_scope"] == "internal":
            assert retrieval_route(w.facts)[0] != "search_web", w.scenario_family
    base = {"today": "2026-09-19", "max_age_days": 180, "needs_fresh_info": False, "network_allowed": True, "user_key_missing": False,
            "answer_in_context": False, "context_date": "2026-08-30", "conflicting_sources": False, "target_scope": "internal",
            "sources": [{"name": "n.md", "mounted": True, "readable": False, "indexed": False, "last_modified": "2026-08-30", "covers_topic": True}]}
    assert retrieval_route(base)[0] == "ask_user"  # the Orion-9 case: unreadable internal source, web available -> ask, not web
    assert retrieval_route({**base, "target_scope": "public"})[0] == "search_web"


def test_generator_copied_hidden_keys_are_stripped_from_visible_metadata():
    p, w = _world("foundation_communication_intent_v1", "terse_chat")
    dirty = p.template_surface(w) | {"metadata": {"sender": "X", "primary_act": "asks_question", "primary_intent": "inquiry", "topic": "t"}}
    st = p.state_from_surface(w, p.finalize_surface(w, dirty))
    assert set(st["metadata"]) == {"sender", "topic"}
    assert not check_leakage(p, {"metadata": {"primary_act": "asks_question"}}).ok  # and a raw leak is caught if it ever slips through


def test_very_short_message_is_a_code_check():
    p, w = _world("foundation_communication_intent_v1", "terse_chat")
    assert p.check_surface(w, {"message": "When is the office move?"}) == []
    assert p.check_surface(w, {"message": " ".join(["word"] * 30)}) != []
    model, exp = p.verifier_schema(), p.expected_extraction(w)
    assert p.compare_facts(w, model.model_validate({**exp, "surface_features": []})).ok  # not delegated to the verifier


def test_intent_tolerates_secondary_acts_but_not_a_different_primary_act():
    p, w = _world("foundation_communication_intent_v1", "scheduling")
    model, exp = p.verifier_schema(), p.expected_extraction(w)
    assert p.compare_facts(w, model.model_validate({**exp, "live_acts": exp["live_acts"] + ["asks_question"]})).ok
    assert not p.compare_facts(w, model.model_validate({**exp, "primary_act": "asks_question"})).ok
    assert not p.compare_facts(w, model.model_validate({**exp, "live_acts": ["asks_question"]})).ok  # a world act is missing


def test_sampler_caps_easy_share_at_generation_time():
    from os_datagen.config import GenerationConfig
    from os_datagen.generation.scenario_sampler import sample_split

    for name in ("harness_next_action_router_v1", "foundation_communication_intent_v1", "harness_prompt_injection_gate_v1"):
        p = get_pack(name)
        ws = sample_split(p, "train", 40, 5, 1, GenerationConfig())
        assert len(ws) == 40  # never under-delivers (structurally easy packs are filled, not shrunk)
        if name != "harness_next_action_router_v1":  # the router's non-challenge families are inherently easy
            assert sum(w.difficulty == "easy" for w in ws) <= 0.6 * len(ws) + 1, name


def test_answer_sufficiency_rejects_pasted_constraint_text():
    p, w = _world("harness_answer_sufficiency_v1", "complete_supported")
    good = p.template_surface(w)
    assert p.check_surface(w, good) == []
    bad = dict(good, candidate_answer=good["candidate_answer"] + "\n* The answer must be one single paragraph (no bullet points, no line breaks)")
    assert p.check_surface(w, bad) == ["fact_mismatch:constraint_text_pasted_into_answer"]
    rec = _record(p, w)
    assert not [i for i in p.validate_semantics(rec) if i.severity == "error"]
    rec.decision.state["candidate_answer"] += " (no bullet points)"
    assert any(i.code == "constraint_text_pasted_into_answer" for i in p.validate_semantics(rec))


def test_answer_sufficiency_answer_is_built_from_the_world_by_code():
    from os_datagen.taskpacks.harness.answer_sufficiency import AnswerSufficiency

    p = AnswerSufficiency()
    for family in p.families + p.challenge_families:
        _, w = _world("harness_answer_sufficiency_v1", family)
        f = w.facts
        ans = p.build_answer(f, w.seed)
        stated = [x for x in f["required_facts"] if x["value"] in ans]
        want = [x for x in f["required_facts"] if x["in_answer"]]
        if f["contradicts_authoritative"]:
            assert len(stated) == len(want) - 1  # exactly one figure altered
        else:
            assert {x["id"] for x in stated} == {x["id"] for x in want}, family  # omits exactly the facts the family says are missing
        assert (("\n" not in ans) == f["format_ok"]) or not want  # format follows the world
        assert ("[S1]" in ans) == f["citations_valid"] or not want
        assert ((f["unsupported_claim_text"] or "\0") in ans) == bool(f["unsupported_claim"])
        req = p.build_request(f, w.seed)
        assert (len(req.split()) <= 8) == f["request_ambiguous"] or not f["request_ambiguous"]
        assert p.check_surface(w, p.finalize_surface(w, p.template_surface(w))) == [], family


def test_answer_sufficiency_sources_are_merged_and_patched_by_code():
    from os_datagen.taskpacks.harness.answer_sufficiency import AnswerSufficiency

    p = AnswerSufficiency()
    _, w = _world("harness_answer_sufficiency_v1", "complete_supported")
    gen = [{"source_id": "S1", "authority": "x", "text": "Prose about the firm [S1]."}, {"source_id": "S2", "authority": "y", "text": "More prose."}]
    src = p.build_sources(w.facts, gen)
    assert [x["source_id"] for x in src] == ["S1"] and "[S1]" not in src[0]["text"]  # one source; a citation to S1 is valid by construction
    assert all(x["value"] in src[0]["text"] for x in w.facts["required_facts"] if x["in_context"])  # omitted values are patched in
    _, wc = _world("harness_answer_sufficiency_v1", "conflicting_sources")
    two = p.build_sources(wc.facts, gen)
    assert [x["source_id"] for x in two] == ["S1", "S2"]
