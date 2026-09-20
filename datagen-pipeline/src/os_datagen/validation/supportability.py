"""Supportability audit: is the expected option uniquely justified by the VISIBLE state, policy and option definitions?

A blind solve by an independent model (it never sees the label). It complements, and never replaces, the code oracle:
the oracle establishes truth, this gate asks whether a reader could *derive* that truth from what is shown. A disagreement
means the row is underspecified, ambiguous or leaky and is rejected (or only tagged for the challenge split)."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ..generation.renderer import PromptLibrary
from ..llm.client import LLMClient, LLMRequest
from ..llm.structured_output import StructuredOutputError, parse_model
from ..schemas.decision import DatasetRecord
from ..training.render import option_ids, render_prompt, truth_sets


class SupportVerdict(BaseModel):
    analysis: str = ""  # brief reasoning FIRST (schema order): a solver that reasons before answering is a fairer test of derivability
    choice: str
    also_valid: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    label_hint_in_text: bool = False


def audit(record: DatasetRecord, client: LLMClient, prompts: PromptLibrary, world: Any, max_tokens: int = 900
          ) -> tuple[SupportVerdict | None, list[str]]:
    rec = record.model_dump(mode="json")
    pref, acceptable = truth_sets(rec)
    opts = option_ids(rec)
    user = prompts.render("system/supportability_user.j2", {"decision_prompt": render_prompt(rec),
                                                            "schema_json": json.dumps(SupportVerdict.model_json_schema())})
    req = LLMRequest(messages=[{"role": "system", "content": prompts.render("system/supportability_system.j2", {})},
                               {"role": "user", "content": user}],
                     temperature=0.0, max_tokens=max_tokens, json_schema=SupportVerdict.model_json_schema(),
                     metadata={"world": world, "mode": "support", "oracle_label": pref})  # metadata is a local channel for fakes only
    try:
        v = parse_model(client.chat(req).text, SupportVerdict)
    except (StructuredOutputError, Exception) as e:  # noqa: BLE001
        return None, [f"supportability:unparseable:{type(e).__name__}"]
    reasons: list[str] = []
    if v.choice not in opts:
        reasons.append("supportability:invalid_choice")
    elif v.choice not in acceptable:
        reasons.append(f"supportability:solver_disagrees:{v.choice}")
    extra = [o for o in v.also_valid if o in opts and o not in acceptable]
    if extra:
        reasons.append("supportability:alt_valid:" + ",".join(sorted(extra)))
    if v.label_hint_in_text:
        reasons.append("supportability:label_hint")
    return v, reasons
