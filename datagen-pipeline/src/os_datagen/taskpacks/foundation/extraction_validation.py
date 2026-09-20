from __future__ import annotations

import random
import re
from typing import Any

from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def norm(v: Any) -> str:
    return re.sub(r"[\s,$]+", "", str(v).lower())


def judge_extraction(f: dict[str, Any]) -> tuple[str, str]:
    src, prop = f["source_fields"], f["proposed"]
    amb = f.get("ambiguous_field")
    if amb:
        return "ambiguous", f"source_ambiguous:{amb}"
    for k, v in sorted(prop.items()):  # sorted: the reported reason must not depend on dict insertion order
        if k in src and norm(v) != norm(src[k]):
            return "incorrect", f"value_mismatch:{k}"
        if k not in src:
            return "incorrect", f"unsupported_field:{k}"
    missing = [k for k in f["required_fields"] if k not in prop or prop[k] in (None, "")]
    if missing:
        return "incomplete", f"missing_required:{missing[0]}"
    return "correct", "all_required_fields_match_source"


class ExtractionValidation(BaseTaskPack):
    name = "foundation_extraction_validation_v1"
    namespace = "foundation"
    oracle_version = "structured_fact_comparison_v1"
    label_source = "structured_fact_comparison_v1"
    label_quality = "controlled_world"
    prompt_id = "foundation.extraction"
    prompt_dir = "foundation/extraction_validation"
    composition_family = "document_intelligence"
    id_prefix = "ext"
    options = [
        ("correct", "Every required field is present and supported by the document."),
        ("incorrect", "At least one extracted value is not supported by the document."),
        ("incomplete", "Extracted values are supported but a required field is missing."),
        ("ambiguous", "The document does not settle one of the values."),
    ]
    instruction_variants = [
        "Is the proposed structured extraction supported by the document?",
        "Check the proposed extraction against the source document.",
        "Decide whether the extraction is right, wrong, missing something, or cannot be settled from the text.",
        "Compare each extracted field with the document and judge the extraction as a whole.",
        "Evaluate the proposed extraction for correctness and completeness against the document.",
    ]
    leak_exclude = {"correct", "incorrect", "incomplete", "ambiguous"}
    families = ["dates_times", "names_roles", "amounts", "addresses", "identifiers", "relations", "omitted_required",
                "swapped_roles", "invalid_normalization", "source_ambiguity"]
    challenge_families = ["multiple_dates", "similar_names", "quoted_prior_message", "wrong_entity_value", "ocr_errors"]
    surface_fields = {"document": str, "proposed_extraction": dict[str, Any], "required_fields": list[str], "source_metadata": dict[str, Any]}
    verifier_fields = {"document_values": dict[str, str], "ambiguous_fields": list[str]}
    allowed_distractors = ["unrelated_sentence", "signature_line"]
    challenge_note = "Add distractor dates/amounts/names or OCR noise; the source facts and proposed extraction must stay exactly as given."

    SCHEMAS = {
        "meeting": ["event", "date", "time", "organizer"],
        "payment": ["payer", "payee", "amount", "due_date"],
        "shipment": ["recipient", "city", "tracking_id"],
        "role": ["person", "role", "organization"],
    }

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = {"dates_times": "meeting", "names_roles": "role", "amounts": "payment", "addresses": "shipment",
                  "identifiers": "shipment", "relations": "role", "multiple_dates": "meeting", "similar_names": "role",
                  "quoted_prior_message": "meeting", "wrong_entity_value": "payment", "ocr_errors": "payment"}.get(family, rng.choice(list(self.SCHEMAS)))
        p1, p2 = pool.people(rng, 2)
        m1, m2 = rng.sample(MONTHS, 2)
        d1, d2 = rng.randint(2, 27), rng.randint(2, 27)
        src = {
            "meeting": {"event": f"{rng.choice(pool.topics).title()} review", "date": f"{m1} {d1}", "time": f"{rng.randint(1, 11)}:{rng.choice(['00', '30'])} PM", "organizer": p1},
            "payment": {"payer": rng.choice(pool.orgs), "payee": p1, "amount": f"${rng.randint(50, 9000)}", "due_date": f"{m1} {d1}"},
            "shipment": {"recipient": p1, "city": rng.choice(pool.cities), "tracking_id": f"TRK-{rng.randint(10000, 99999)}"},
            "role": {"person": p1, "role": rng.choice(["treasurer", "site lead", "auditor", "coordinator"]), "organization": rng.choice(pool.orgs)},
        }[schema]
        required = list(src)
        prop: dict[str, Any] = dict(src)
        f: dict[str, Any] = {"schema_name": schema, "source_fields": src, "required_fields": required, "proposed": prop,
                             "ambiguous_field": None, "distractors": [], "quoted_prior": None, "ocr_noise": False, "other_person": p2}
        diff = "easy"
        keys = list(src)
        if family == "omitted_required":
            prop.pop(rng.choice(keys))
            diff = "medium"
        elif family == "swapped_roles":
            a, b = ("payer", "payee") if schema == "payment" else (keys[0], keys[1])
            if schema in ("payment", "role"):
                prop[a], prop[b] = src[b], src[a]
            else:
                prop[keys[0]], prop[keys[1]] = src[keys[1]], src[keys[0]]
            diff = "medium"
        elif family == "invalid_normalization":
            k = next((k for k in keys if "date" in k or "amount" in k or "tracking" in k), keys[0])
            prop[k] = f"{d2}/{m2}/26" if "date" in k else (src[k].replace("$", "") + "0" if "amount" in k else src[k] + "X")
            diff = "medium"
        elif family == "source_ambiguity":
            k = next((k for k in keys if k in ("date", "due_date", "amount", "city")), keys[1])
            alt = {"date": f"{m2} {d2}", "due_date": f"{m2} {d2}", "amount": f"${rng.randint(50, 9000)}",
                   "city": rng.choice([c for c in pool.cities if c != src.get("city")])}.get(k, "another value")
            if alt == src[k]:
                alt = alt + " (alternate)"
            f["ambiguous_field"], f["ambiguous_alternative"] = k, alt
            diff = "hard"
        elif family == "multiple_dates":
            f["distractors"] = [{"kind": "other_date", "value": f"{m2} {d2}", "context": "unrelated deadline"}]
            if rng.random() < 0.5:
                k = next(k for k in keys if "date" in k)
                prop[k] = f"{m2} {d2}"
            diff = "hard"
        elif family == "similar_names":
            f["distractors"] = [{"kind": "similar_name", "value": p1.split()[0] + " " + p2.split()[1]}]
            if rng.random() < 0.5:
                k = keys[0] if schema == "role" else next((k for k in keys if k in ("organizer", "payee", "recipient")), keys[0])
                prop[k] = f["distractors"][0]["value"]
            diff = "hard"
        elif family == "quoted_prior_message":
            k = next(k for k in keys if "date" in k)
            f["quoted_prior"] = {"field": k, "value": f"{m2} {d2}"}
            if rng.random() < 0.5:
                prop[k] = f"{m2} {d2}"
            diff = "hard"
        elif family == "wrong_entity_value":
            prop["payer"], prop["payee"] = src["payee"], src["payer"]
            diff = "hard"
        elif family == "ocr_errors":
            f["ocr_noise"] = True
            diff = "hard"
        return f, {"difficulty": diff, "tags": [family]}

    def decide(self, world: ScenarioWorld) -> Decision:
        lab, reason = judge_extraction(world.facts)
        return Decision(lab, reason)

    def finalize_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        """Structured, truth-bearing content is injected by code, never re-typed by the LLM (which tends to 'fix' it)."""
        return {**surface, "proposed_extraction": dict(world.facts["proposed"]), "required_fields": list(world.facts["required_fields"])}

    def check_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> list[str]:
        f = world.facts
        out = []
        if surface.get("proposed_extraction") != f["proposed"]:
            out.append("fact_mismatch:proposed_extraction")
        if sorted(surface.get("required_fields", [])) != sorted(f["required_fields"]):
            out.append("fact_mismatch:required_fields")
        return out

    def anchors(self, world: ScenarioWorld) -> list[str]:
        f = world.facts
        if f["ocr_noise"]:
            return []
        return [str(v) for k, v in f["source_fields"].items() if k != f.get("ambiguous_field")]

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        amb = f.get("ambiguous_field")
        return {"document_values": {k: v for k, v in f["source_fields"].items() if k != amb},
                "ambiguous_fields": [amb] if amb else []}

    def compare_facts(self, world: ScenarioWorld, extracted: Any) -> Any:
        from ...schemas.validation import FactComparison

        exp, got = self.expected_extraction(world), extracted.model_dump()
        out = FactComparison()
        dv = got.get("document_values")
        if dv is None:
            out.unverifiable.append("document_values")
        else:
            bad = [k for k, v in exp["document_values"].items() if norm(dv.get(k, "")) != norm(v)]
            (out.mismatched if bad else out.matched).append("document_values" if not bad else f"document_values:{bad[0]}")
        af = got.get("ambiguous_fields")
        if af is None:
            out.unverifiable.append("ambiguous_fields")
        else:
            (out.matched if sorted(map(norm, af)) == sorted(map(norm, exp["ambiguous_fields"])) else out.mismatched).append("ambiguous_fields")
        return out

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"notes": "proposed_extraction must equal WORLD_FACTS.proposed exactly (same keys and values, even if wrong or incomplete); "
                         "required_fields must equal WORLD_FACTS.required_fields. Render every source_fields value verbatim in the document. "
                         "If ambiguous_field is set, the document must state both ambiguous_alternative and the source value for that field "
                         "without saying which one applies. Do not comment on whether the extraction is right."}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"document": surface["document"], "proposed_extraction": surface["proposed_extraction"],
                "required_fields": surface["required_fields"], "source_metadata": surface["source_metadata"]}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        s = dict(f["source_fields"])
        amb = f.get("ambiguous_field")
        parts = []
        for k, v in s.items():
            if k == amb:
                parts.append(f"{k.replace('_', ' ')}: {v} or {f['ambiguous_alternative']} (to be confirmed)")
            else:
                parts.append(f"{k.replace('_', ' ')}: {v}")
        doc = "; ".join(parts) + "."
        for d in f["distractors"]:
            doc += f" Separately, {d['value']} was mentioned in an unrelated note."
        if f["quoted_prior"]:
            doc += f" > Earlier message said {f['quoted_prior']['value']}."
        return {"document": doc, "proposed_extraction": f["proposed"], "required_fields": f["required_fields"], "source_metadata": {"kind": f["schema_name"]}}
