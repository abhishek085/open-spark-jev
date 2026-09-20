from __future__ import annotations

import importlib

from .base import BaseTaskPack

# module path -> class name; imported lazily so a broken pack cannot break the CLI.
_PACKS: dict[str, tuple[str, str]] = {
    "foundation_semantic_entailment_v1": ("foundation.semantic_entailment", "SemanticEntailment"),
    "foundation_document_type_v1": ("foundation.document_type", "DocumentType"),
    "foundation_document_relevance_v1": ("foundation.document_relevance", "DocumentRelevance"),
    "foundation_extraction_validation_v1": ("foundation.extraction_validation", "ExtractionValidation"),
    "foundation_rule_application_v1": ("foundation.rule_application", "RuleApplication"),
    "foundation_temporal_reasoning_v1": ("foundation.temporal_reasoning", "TemporalReasoning"),
    "foundation_communication_intent_v1": ("foundation.communication_intent", "CommunicationIntent"),
    "foundation_communication_urgency_v1": ("foundation.communication_urgency", "CommunicationUrgency"),
    "foundation_authorization_gate_v1": ("foundation.authorization_gate", "AuthorizationGate"),
    "harness_next_action_router_v1": ("harness.next_action_router", "NextActionRouter"),
    "harness_tool_action_gate_v1": ("harness.tool_action_gate", "ToolActionGate"),
    "harness_retrieval_gate_v1": ("harness.retrieval_gate", "RetrievalGate"),
    "harness_termination_gate_v1": ("harness.termination_gate", "TerminationGate"),
    "harness_answer_sufficiency_v1": ("harness.answer_sufficiency", "AnswerSufficiency"),
    "harness_prompt_injection_gate_v1": ("harness.prompt_injection_gate", "PromptInjectionGate"),
}
_CACHE: dict[str, BaseTaskPack] = {}


def pack_names() -> list[str]:
    return list(_PACKS)


def get_pack(name: str) -> BaseTaskPack:
    if name not in _PACKS:
        raise KeyError(f"unknown task pack {name!r}; known: {', '.join(_PACKS)}")
    if name not in _CACHE:
        mod, cls = _PACKS[name]
        module = importlib.import_module(f"os_datagen.taskpacks.{mod}")
        _CACHE[name] = getattr(module, cls)()
    return _CACHE[name]


def implemented_packs() -> list[str]:
    out = []
    for n in _PACKS:
        try:
            get_pack(n)
            out.append(n)
        except (ImportError, ModuleNotFoundError):
            pass
    return out
