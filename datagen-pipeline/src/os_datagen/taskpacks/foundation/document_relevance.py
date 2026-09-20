from __future__ import annotations

import random
from typing import Any

from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision

ATTRS = {
    "refund_window_days": ("the refund window in days", lambda r: str(r.choice([14, 21, 30, 45]))),
    "warranty_years": ("the warranty length in years", lambda r: str(r.choice([1, 2, 3, 5]))),
    "delivery_days": ("the standard delivery time in days", lambda r: str(r.choice([2, 4, 6, 9]))),
    "support_hours": ("the support desk hours", lambda r: r.choice(["8am-6pm", "9am-5pm", "24 hours", "7am-3pm"])),
    "minimum_order": ("the minimum order value in dollars", lambda r: str(r.choice([25, 50, 120, 300]))),
}


def judge_relevance(f: dict[str, Any]) -> tuple[str, str]:
    q, d = f["query"], f["document"]
    if d["entity"] != q["entity"]:
        return "irrelevant", "wrong_entity"
    if d["period"] != q["period"]:
        return "irrelevant", "wrong_time_period"
    for ref in f["reference_notes"]:
        if ref["attribute"] in d["covers"] and d["covers"][ref["attribute"]] != ref["value"] and ref["authority"] >= d["authority"]:
            return "conflicting", "conflicts_with_authoritative_record"
    covered = [a for a in q["required_attributes"] if a in d["covers"]]
    if len(covered) == len(q["required_attributes"]):
        return "directly_relevant", "all_required_facts_covered"
    if covered:
        return "partially_relevant", "some_required_facts_covered"
    return "irrelevant", "no_required_facts_covered"


class DocumentRelevance(BaseTaskPack):
    name = "foundation_document_relevance_v1"
    namespace = "foundation"
    oracle_version = "controlled_corpus_relevance_v1"
    label_source = "controlled_corpus_coverage_v1"
    label_quality = "controlled_world"
    prompt_id = "foundation.relevance"
    prompt_dir = "foundation/document_relevance"
    composition_family = "search_rag"
    id_prefix = "rel"
    options = [
        ("directly_relevant", "The document directly answers the query."),
        ("partially_relevant", "The document helps with part of the query."),
        ("irrelevant", "The document does not help answer the query."),
        ("conflicting", "The document contradicts a more authoritative record."),
    ]
    instruction_variants = [
        "How well does the candidate document answer the query?",
        "Assess the document's relevance to the query.",
        "Does the document answer the query, help in part, miss it, or conflict with what is on record?",
        "Rate the candidate document against the query and the record on file.",
        "Judge whether this excerpt is useful for the query, considering entity, period and authority.",
    ]
    families = ["direct_answer", "partial_answer", "irrelevant_topic_overlap", "wrong_time_period",
                "conflicting_authoritative", "stale_result", "wrong_entity_same_name", "multi_document_requirement"]
    challenge_families = ["keyword_overlap_no_answer", "short_authoritative", "misleading_title", "injection_text_irrelevant"]
    surface_fields = {"query": str, "candidate_document": dict[str, Any], "query_constraints": list[str], "distractor_tags": list[str]}
    verifier_defaults = {"conflicts_with_record_on_file": False}
    verifier_fields = {"query_entity": str, "document_entity": str, "query_period": str, "document_period": str,
                       "required_attributes_stated_in_document": list[str], "conflicts_with_record_on_file": bool}
    allowed_distractors = ["unrelated_boilerplate", "irrelevant_promotional_text"]
    challenge_note = "Use lexical overlap, misleading titles or injected instructions that do not change factual coverage."

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        ent = rng.choice(pool.orgs)
        other = rng.choice([o for o in pool.orgs if o != ent])
        keys = rng.sample(list(ATTRS), 3)
        vals = {k: ATTRS[k][1](rng) for k in keys}
        req = keys[:1] if family not in ("multi_document_requirement",) else keys[:2]
        d: dict[str, Any] = {"entity": ent, "period": "2026", "covers": {}, "authority": 1, "published": "2026-03-01",
                             "title_style": "accurate", "extra": None}
        refs: list[dict[str, Any]] = []
        diff = "easy"
        cons: list[str] = []
        if family == "direct_answer":
            d["covers"] = {k: vals[k] for k in req}
        elif family == "partial_answer" or family == "multi_document_requirement":
            req = keys[:2]
            d["covers"] = {keys[0]: vals[keys[0]]}
            diff = "medium"
        elif family in ("irrelevant_topic_overlap", "keyword_overlap_no_answer"):
            d["covers"] = {keys[1]: vals[keys[1]]}
            d["extra"] = "mentions the same subject area in passing"
            diff = "medium" if family.startswith("irr") else "hard"
        elif family in ("wrong_time_period", "stale_result"):
            d["covers"] = {k: vals[k] for k in req}
            d["period"] = "2024" if family == "stale_result" else "2025"
            d["published"] = "2024-02-01" if family == "stale_result" else "2025-02-01"
            diff = "medium"
        elif family == "conflicting_authoritative":
            d["covers"] = {req[0]: "0" if vals[req[0]] != "0" else "1"}
            refs = [{"entity": ent, "attribute": req[0], "value": vals[req[0]], "authority": 2}]
            diff = "hard"
        elif family == "wrong_entity_same_name":
            d["entity"] = f"{ent} (unrelated firm in {rng.choice(pool.cities)})"
            d["covers"] = {k: vals[k] for k in req}
            diff = "hard"
        elif family == "short_authoritative":
            d["covers"] = {k: vals[k] for k in req}
            d["authority"] = 2
            d["extra"] = "very short, one line"
            diff = "hard"
        elif family == "misleading_title":
            d["covers"] = {keys[1]: vals[keys[1]]}
            d["title_style"] = "misleading_suggests_it_answers_the_query"
            diff = "hard"
        elif family == "injection_text_irrelevant":
            d["covers"] = {keys[1]: vals[keys[1]]}
            d["extra"] = "contains a line telling the reader to ignore prior instructions (harmless, fictional)"
            diff = "hard"
        facts = {"query": {"entity": ent, "required_attributes": req, "period": "2026"}, "document": d,
                 "reference_notes": refs, "attribute_glossary": {k: ATTRS[k][0] for k in keys}, "distractor_entity": other,
                 "query_constraints": cons}
        return facts, {"difficulty": diff, "tags": [family]}

    def decide(self, world: ScenarioWorld) -> Decision:
        lab, reason = judge_relevance(world.facts)
        q, d = world.facts["query"], world.facts["document"]
        return Decision(lab, reason, outcomes={"required_fact_ids": q["required_attributes"], "covered_fact_ids": [a for a in q["required_attributes"] if a in d["covers"]],
                                               "authority_rank": "official" if d["authority"] == 2 else "informal",
                                               "conflicted_by_authoritative_record": lab == "conflicting"})

    def anchors(self, world: ScenarioWorld) -> list[str]:
        f = world.facts
        return [f["query"]["entity"]] + [str(v) for v in f["document"]["covers"].values() if len(str(v)) > 1]

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        q, d = f["query"], f["document"]
        conflict = judge_relevance(f)[0] == "conflicting"
        return {"query_entity": q["entity"], "document_entity": d["entity"], "query_period": q["period"],
                "document_period": d["period"],
                "required_attributes_stated_in_document": [a for a in q["required_attributes"] if a in d["covers"]],
                "conflicts_with_record_on_file": conflict}

    def verifier_context(self, state: dict[str, Any]) -> dict[str, Any]:
        ctx = super().verifier_context(state)
        ctx["attribute_vocabulary"] = {k: v[0] for k, v in ATTRS.items()}
        return ctx

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"notes": "Keep the entity name exactly as in WORLD_FACTS. If reference_notes is non-empty, add each as a string in "
                         "query_constraints like: 'Record on file (official): <entity> <attribute phrase> is <value>'. "
                         "State the query's time period (year) in the query, and the document period in candidate_document.date/text. "
                         "Describe attributes in natural language ('the refund window', 'support desk hours'), never with the raw attribute key names. authority is 'official' for 2 and 'informal' for 1."}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"query": surface["query"], "candidate_document": surface["candidate_document"],
                "query_constraints": surface["query_constraints"]}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        q, d = f["query"], f["document"]
        g = f["attribute_glossary"]
        ask = " and ".join(g[a] for a in q["required_attributes"])
        text = " ".join(f"The {g[a]} is {v}." for a, v in d["covers"].items()) or "General information about the firm."
        return {"query": f"What is {ask} at {q['entity']} in {q['period']}?",
                "candidate_document": {"title": f"{d['entity']} customer information", "date": d["published"],
                                       "authority": "official" if d["authority"] == 2 else "informal",
                                       "text": f"{d['entity']}, {d['period']}: {text}"},
                "query_constraints": [f"Record on file (official): {r['entity']} {g[r['attribute']]} is {r['value']}" for r in f["reference_notes"]],
                "distractor_tags": []}
