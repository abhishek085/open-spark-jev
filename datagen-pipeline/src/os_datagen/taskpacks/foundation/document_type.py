from __future__ import annotations

import random
from typing import Any

from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision

# Required-feature signatures (the "required-feature policy"). The generator sees *features*, never the class.
SIGNATURES: dict[str, set[str]] = {
    "invoice": {"invoice_number", "amount_due", "payment_terms"},
    "receipt": {"receipt_number", "amount_paid", "payment_method"},
    "contract": {"named_parties", "effective_date", "obligations_clause", "signature_block"},
    "support_request": {"problem_description", "help_request", "sender_contact"},
    "report": {"findings_section", "period_covered", "summary_section"},
    "meeting_note": {"attendee_list", "agenda_items", "action_items"},
}
MISLEADING = {"invoice": "notes.txt", "receipt": "statement_final.txt", "contract": "scan_004.txt",
              "support_request": "invoice_copy.txt", "report": "contract_v2.txt", "meeting_note": "receipt.txt",
              "other": "invoice_2024.txt"}
DISTRACTORS = {"mentions_invoice_reference": "the text mentions an invoice number belonging to another document",
               "refund_language": "the text discusses a refund",
               "payment_table": "the text includes a small table of payment amounts",
               "amends_prior_contract": "the text says it amends an earlier agreement",
               "multilingual_header": "the header is written in another language",
               "ocr_noise": "some characters look like OCR errors"}
MIN_COVERAGE = 0.6
FEATURE_VALUE_KEYS = {
    "invoice_number": ["number"], "amount_due": ["amount"], "payment_terms": ["terms"], "receipt_number": ["number"], "amount_paid": ["amount"],
    "payment_method": ["method"], "named_parties": ["org", "person", "other_person"], "effective_date": ["date"], "obligations_clause": ["org"],
    "signature_block": ["person", "other_person"], "problem_description": ["topic"], "sender_contact": ["person"], "findings_section": ["topic"],
    "attendee_list": ["person", "other_person"], "agenda_items": ["topic"], "action_items": ["person", "topic"],
    "mentions_invoice_reference": ["number"], "payment_table": ["amount"], "free_text_paragraph": ["topic", "city"],
}


def classify(features: set[str]) -> tuple[str, str]:
    """Highest required-feature coverage wins; coverage < MIN_COVERAGE (or a tie) => other."""
    cov = {c: len(features & sig) / len(sig) for c, sig in SIGNATURES.items()}
    best = max(cov.values())
    winners = [c for c, v in cov.items() if v == best]
    if best < MIN_COVERAGE or len(winners) != 1:
        return "other", "insufficient_or_tied_feature_coverage"
    return winners[0], f"feature_coverage_{best:.2f}"


class DocumentType(BaseTaskPack):
    name = "foundation_document_type_v1"
    namespace = "foundation"
    oracle_version = "structured_document_type_v1"
    label_source = "structured_document_type_v1"
    prompt_id = "foundation.document_type"
    prompt_dir = "foundation/document_type"
    composition_family = "document_intelligence"
    id_prefix = "doc"
    options = [
        ("invoice", "A bill requesting payment for goods or services."),
        ("contract", "A binding agreement between named parties."),
        ("support_request", "A message asking for help with a problem."),
        ("report", "A document presenting findings for a period or topic."),
        ("meeting_note", "A record of a meeting with attendees, agenda and actions."),
        ("receipt", "Proof that a payment was already made."),
        ("other", "None of the other types."),
    ]
    instruction_variants = [
        "What type of document is this?",
        "Which document type best describes the text?",
        "Classify the document into one of the allowed types.",
        "Based on its content, what kind of document is this?",
        "Pick the document type that fits the text and metadata best.",
    ]
    leak_exclude = {"invoice", "contract", "support_request", "report", "meeting_note", "receipt", "other"}  # class words occur naturally in documents
    extra_leak_patterns = [r"\bdocument type\b"]
    families = ["invoice", "receipt", "contract", "support_request", "report", "meeting_note", "other",
                "mixed_partial", "misleading_filename", "ocr_noise", "multilingual_header", "distractor_language"]
    challenge_families = ["invoice_ref_in_email", "contract_amendment", "receipt_with_refund",
                          "report_with_payment_table", "truncated_scan"]
    surface_fields = {"document_text": str, "metadata": dict[str, Any], "noise_tags": list[str]}
    verifier_fields = {"features_present": list[str], "surface_features": list[str]}
    allowed_distractors = list(DISTRACTORS)
    challenge_note = "Make surface cues misleading while keeping the required features intact."

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        vals = {"org": rng.choice(pool.orgs), "person": pool.person(rng), "other_person": pool.person(rng),
                "number": rng.randint(1000, 9999), "amount": rng.randint(40, 4800), "city": rng.choice(pool.cities),
                "topic": rng.choice(pool.topics), "terms": rng.choice(["Net 30", "Net 15", "Due on receipt"]),
                "method": rng.choice(["card", "bank transfer", "cash"]), "date": f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"}
        base = {"invoice", "receipt", "contract", "support_request", "report", "meeting_note", "other"}
        cls = family if family in base else rng.choice(sorted(SIGNATURES))
        feats: set[str] = set(SIGNATURES.get(cls, {"free_text_paragraph"}))
        dist: list[str] = []
        misleading = False
        diff = "easy"
        if family == "mixed_partial":
            feats -= {sorted(feats)[0]}
            diff = "medium"
        elif family == "misleading_filename":
            misleading = True
            diff = "medium"
        elif family in ("ocr_noise", "multilingual_header", "distractor_language"):
            dist = {"ocr_noise": ["ocr_noise"], "multilingual_header": ["multilingual_header"],
                    "distractor_language": [rng.choice(["refund_language", "mentions_invoice_reference"])]}[family]
            diff = "medium"
        elif family == "invoice_ref_in_email":
            cls = "support_request"
            feats = set(SIGNATURES[cls]) | {"mentions_invoice_reference"}
            dist, diff = ["mentions_invoice_reference"], "hard"
        elif family == "contract_amendment":
            cls, feats = "contract", set(SIGNATURES["contract"]) | {"amends_prior_contract"}
            dist, diff = ["amends_prior_contract"], "hard"
        elif family == "receipt_with_refund":
            cls, feats = "receipt", set(SIGNATURES["receipt"]) | {"refund_language"}
            dist, diff = ["refund_language"], "hard"
        elif family == "report_with_payment_table":
            cls, feats = "report", set(SIGNATURES["report"]) | {"payment_table"}
            dist, diff = ["payment_table"], "hard"
        elif family == "truncated_scan":
            cls = rng.choice(["invoice", "contract", "report"])
            feats = set(SIGNATURES[cls]) - {sorted(SIGNATURES[cls])[-1]}
            dist, diff = ["ocr_noise"], "hard"
        derived, _ = classify(feats)
        assert derived == cls, (family, cls, derived, feats)
        req_sf = {"multilingual_header": ["non_english_header"], "ocr_noise": ["ocr_noise"], "truncated_scan": ["ocr_noise"]}.get(family, [])
        keep = {"topic", "city"} | {k for f in feats for k in FEATURE_VALUE_KEYS.get(f, [])}
        vals = {k: v for k, v in vals.items() if k in keep}  # hand the generator only values the features call for (an 'other' doc must not get invoice figures)
        facts = {"required_surface_features": req_sf, "features": sorted(feats), "values": vals, "distractors": {d: DISTRACTORS[d] for d in dist},
                 "filename_style": "misleading" if misleading else rng.choice(["descriptive", "generic"]),
                 "features_glossary": "Features describe what the document contains; render each one naturally."}
        return facts, {"doc_class": cls, "difficulty": diff, "tags": [family, "document"]}

    def decide(self, world: ScenarioWorld) -> Decision:
        cls, reason = classify(set(world.facts["features"]))
        assert cls == world.hidden["doc_class"], "sampler/oracle disagree"
        return Decision(cls, reason, outcomes={"coverage_policy": "max_required_feature_coverage"})

    def anchors(self, world: ScenarioWorld) -> list[str]:
        v = world.facts["values"]
        f = set(world.facts["features"])
        out = []
        if "amount_due" in f or "amount_paid" in f:
            out.append(str(v["amount"]))
        if "invoice_number" in f or "receipt_number" in f:
            out.append(str(v["number"]))
        return out

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"features_present": sorted(world.facts["features"]), "surface_features": world.facts["required_surface_features"]}

    def compare_facts(self, world: ScenarioWorld, extracted: Any) -> Any:
        """Every world feature must be present. Extra features (distractors the generator added) are tolerated only if
        the required-feature policy, re-run on world features + extras, still returns the same class."""
        from ...schemas.validation import FactComparison

        got = set(extracted.model_dump().get("features_present") or [])
        want = set(world.facts["features"])
        c = FactComparison()
        if want - got:
            c.mismatched.append("features_present:missing_" + sorted(want - got)[0])
            return c
        if classify(want | got)[0] != classify(want)[0]:
            c.mismatched.append("features_present:extra_features_change_class")
            return c
        c.matched.append("features_present")
        self.check_surface_features(world, extracted.model_dump(), c)
        return c

    def verifier_context(self, state: dict[str, Any]) -> dict[str, Any]:
        ctx = super().verifier_context(state)
        ctx["surface_vocabulary"] = ["non_english_header", "ocr_noise"]
        ctx["feature_vocabulary"] = sorted({f for s in SIGNATURES.values() for f in s} | set(DISTRACTORS) | {"free_text_paragraph"})
        return ctx

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"notes": "Render EXACTLY the listed features and use only the listed values. Do not add any other document element: no invoice or receipt "
                         "numbers, amounts, payment terms, parties, signatures, agendas, findings or attendee lists unless a listed feature calls for it. "
                         "A document whose only feature is free_text_paragraph is a plain informal note with none of those elements. Add at most one "
                         "distractor, and only the ones listed under distractors."}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"document_text": surface["document_text"], "metadata": surface["metadata"]}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = set(world.facts["features"])
        v = world.facts["values"]
        L: list[str] = []
        if "invoice_number" in f:
            L.append(f"Invoice No. {v['number']}")
        if "receipt_number" in f:
            L.append(f"Receipt #{v['number']}")
        if "named_parties" in f:
            L.append(f"This agreement is between {v['org']} and {v['person']}.")
        if "effective_date" in f:
            L.append(f"Effective {v['date']}.")
        if "obligations_clause" in f:
            L.append(f"{v['org']} shall provide the services described herein.")
        if "signature_block" in f:
            L.append(f"Signed: {v['person']} / {v['other_person']}")
        if "amount_due" in f:
            L.append(f"Amount due: ${v['amount']}")
        if "amount_paid" in f:
            L.append(f"Amount paid: ${v['amount']}")
        if "payment_terms" in f:
            L.append(f"Payment terms: {v['terms']}")
        if "payment_method" in f:
            L.append(f"Paid by {v['method']}")
        if "problem_description" in f:
            L.append(f"The {v['topic']} tool keeps failing when I open it.")
        if "help_request" in f:
            L.append("Could someone please help me sort this out?")
        if "sender_contact" in f:
            L.append(f"Contact: {v['person'].split()[0].lower()}@example.test")
        if "findings_section" in f:
            L.append(f"Findings: the {v['topic']} showed steady progress.")
        if "period_covered" in f:
            L.append("Period covered: Q2")
        if "summary_section" in f:
            L.append("Summary: results were within expectations.")
        if "attendee_list" in f:
            L.append(f"Attendees: {v['person']}, {v['other_person']}")
        if "agenda_items" in f:
            L.append(f"Agenda: {v['topic']} status")
        if "action_items" in f:
            L.append(f"Action items: {v['person']} to follow up")
        if "mentions_invoice_reference" in f:
            L.append(f"(Ref: earlier invoice {v['number'] + 1})")
        if "refund_language" in f:
            L.append("Refund policy: returns accepted within 14 days.")
        if "payment_table" in f:
            L.append("Payments: | Q1 | $120 | Q2 | $150 |")
        if "amends_prior_contract" in f:
            L.append("This document amends the earlier agreement.")
        if "free_text_paragraph" in f:
            L.append(f"A short note about the {v['topic']} in {v['city']}.")
        name = "notes.txt" if world.facts["filename_style"] == "misleading" else "document.txt"
        return {"document_text": "\n".join(L), "metadata": {"filename": name, "source": "scan", "date": None}, "noise_tags": []}
