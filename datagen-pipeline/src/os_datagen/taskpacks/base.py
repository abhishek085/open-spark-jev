from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..generation.controlled_worlds import Pool, style_for
from ..schemas.decision import (
    DatasetRecord,
    DecisionSpec,
    LevelDef,
    OptionDef,
    ProbabilitySemantics,
    PublicTruth,
    QualityRecord,
    TruthRecord,
)
from ..schemas.provenance import ProvenanceRecord
from ..schemas.scenario import RenderedState, RenderRequest, ScenarioWorld
from ..schemas.validation import FactComparison, ValidationIssue
from ..utils.seeds import make_rng

GENERIC_LEAK_PATTERNS = [
    r"\bthe (?:correct|right|best|expected|preferred) (?:action|answer|option|label|choice|route|decision)\b",
    r"\b(?:gold|ground)[ -_]?(?:label|truth)\b",
    r"\bshould (?:use|choose|select|pick)\b",
    r"\bpreferred[ _]option\b",
    r"\bacceptable[ _]options?\b",
    r"\breason[ _]code\b",
    r"\boracle\b",
    r"\blabel[ _]source\b",
    r"\bworld[ _]?facts?\b",                                   # generator talking about its own instructions
    r"\bexpected per\b|\bper the ['\"`]?[\w ]+['\"`]? flag\b",
    r"\bthe (?:candidate )?answer (?:violates|fails|satisfies|meets|contradicts)\b",  # evaluating the answer in visible text
    r"\b(?:self[_ ]check|facts[_ ]preserved|contains[_ ]label[_ ]hint)\b",
    r"\bdistractors?\b",
    # oracle-style narration of conclusions inside visible traces/descriptions
    r"\b(?:input|freshness|cache|completeness|safety) check:",
    r"\b(?:required|needed|necessary) (?:details|information|inputs?|fields?) (?:are|is) (?:missing|absent|lacking)\b",
    r"\b(?:file|document|source|entry) (?:already )?contains the (?:information|answer|details|data)",
    r"\bcontains the information (?:required|requested|needed)\b",
]


@dataclass
class Decision:
    """What an oracle returns. `preferred` is an option id (choice), level int (score) or bool (boolean)."""

    preferred: Any
    reason: str
    acceptable: list[Any] | None = None
    outcomes: dict[str, Any] = field(default_factory=dict)
    quality: str | None = None  # overrides pack.label_quality


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v.strip().lower())
    if isinstance(v, (list, tuple, set)):
        return tuple(sorted(_norm(x) for x in v))
    if isinstance(v, dict):
        return tuple(sorted((k, _norm(x)) for k, x in v.items()))
    return v


class BaseTaskPack:
    """Shared machinery. A concrete pack supplies: families, sample(), decide(), surface fields,
    verifier fields, expected_extraction(), state_from_surface() (+ template_surface() for --dry-run)."""

    name: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    namespace: ClassVar[str]
    decision_type: ClassVar[str] = "choice"
    oracle_version: ClassVar[str]
    label_source: ClassVar[str]
    prompt_id: ClassVar[str]  # e.g. "harness.next_action"
    prompt_dir: ClassVar[str]  # e.g. "harness/next_action_router"
    composition_family: ClassVar[str]
    id_prefix: ClassVar[str]
    instruction_variants: ClassVar[list[str]]  # >= 5: train uses 0-1, calibration 2, locked 3, challenge 4
    options: ClassVar[list[tuple[str, str]]] = []
    levels: ClassVar[list[tuple[int, str]]] = []
    families: ClassVar[list[str]]
    challenge_families: ClassVar[list[str]]
    label_quality: ClassVar[str] = "deterministic"
    leak_exclude: ClassVar[set[str]] = set()  # option ids too common as words to police
    extra_leak_patterns: ClassVar[list[str]] = []
    surface_fields: ClassVar[dict[str, Any]] = {}
    verifier_fields: ClassVar[dict[str, Any]] = {}
    verifier_defaults: ClassVar[dict[str, Any]] = {}  # value implied when the text simply does not mention a fact
    allowed_distractors: ClassVar[list[str]] = []
    challenge_note: ClassVar[str] = ""
    code_checked_features: ClassVar[tuple[str, ...]] = ()  # surface features verified deterministically in check_surface
    code_rendered_keys: ClassVar[tuple[str, ...]] = ()  # state keys whose text is constant, code-rendered decision rules
    supportability: ClassVar[str] = "tag"  # 'reject': a blind-solver disagreement drops the row; 'tag': recorded only

    # ------------------------------------------------------------ properties
    @property
    def prompt_version(self) -> str:
        return f"{self.prompt_id}.surface.v1"

    @property
    def verifier_prompt_version(self) -> str:
        return f"{self.prompt_id}.verify.v1"

    def option_ids(self) -> list[str]:
        if self.decision_type == "choice":
            return [o for o, _ in self.options]
        if self.decision_type == "score":
            return [str(v) for v, _ in self.levels]
        return ["true", "false"]

    # ------------------------------------------------------------ sampling
    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (facts, hidden) for a scenario family. `facts` must not contain the answer."""
        raise NotImplementedError

    def sample_world(self, rng: random.Random, split: str, families: list[str] | None = None) -> ScenarioWorld:
        split_family, pool, tone, _ = style_for(rng, split)
        challenge = split == "challenge"
        family = rng.choice(families or (self.challenge_families if challenge else self.families))
        facts, hidden = self.sample(rng, family, pool, tone)
        facts["style"] = {"tone": tone}
        hidden["pool"] = pool.pool_id
        return ScenarioWorld(
            scenario_id=f"{self.id_prefix}-000000",
            task_pack=self.name,
            scenario_family=family,
            split_family=split_family,
            split=split,
            seed=0,
            facts=facts,
            hidden=hidden,
            candidate_actions=self.option_ids() if self.decision_type == "choice" else [],
            difficulty=hidden.pop("difficulty", "medium"),
            tags=hidden.pop("tags", []),
            is_challenge=challenge,
        )

    def generate_challenge_world(self, rng: random.Random) -> ScenarioWorld:
        return self.sample_world(rng, "challenge")

    # ------------------------------------------------------------ oracle
    def decide(self, world: ScenarioWorld) -> Decision:
        raise NotImplementedError

    def solve_oracle(self, world: ScenarioWorld) -> TruthRecord:
        d = self.decide(world)
        q = d.quality or self.label_quality
        base: dict[str, Any] = dict(
            oracle_version=self.oracle_version, decision_type=self.decision_type, reason_code=d.reason,
            expected_outcomes=d.outcomes, label_quality=q, label_source=self.label_source,
        )
        if self.decision_type == "choice":
            acc = d.acceptable or [d.preferred]
            return TruthRecord(preferred_option=d.preferred, acceptable_options=list(acc), **base)
        if self.decision_type == "score":
            acc = d.acceptable or [d.preferred]
            return TruthRecord(preferred_level=d.preferred, acceptable_levels=list(acc), **base)
        return TruthRecord(boolean_truth=bool(d.preferred), **base)

    def label_key(self, truth: TruthRecord) -> str:
        if self.decision_type == "choice":
            return str(truth.preferred_option)
        if self.decision_type == "score":
            return str(truth.preferred_level)
        return str(truth.boolean_truth).lower()

    def probability_semantics(self) -> ProbabilitySemantics:
        return ProbabilitySemantics(
            kind="deterministic_point_mass",
            description="One code-derived target; all mass on the preferred option (acceptable options scored as a set).",
        )

    # ------------------------------------------------------------ rendering
    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        return {}

    def render_context(self, world: ScenarioWorld, truth: TruthRecord) -> RenderRequest:
        # `truth` is accepted per the interface but deliberately never read here.
        facts = world.facts
        vars_: dict[str, Any] = {
            "task_family": self.name,
            "world_facts_json": json.dumps(facts, indent=2, ensure_ascii=False, sort_keys=True),
            "allowed_distractors_json": json.dumps(self.allowed_distractors),
            **self.extra_template_vars(world),
        }
        return RenderRequest(
            task_pack=self.name, prompt_version=self.prompt_version,
            template_path=f"{self.prompt_dir}/surface.j2", template_vars=vars_, world_facts=facts,
            seed=world.seed, challenge_note=self.challenge_note if world.is_challenge else None,
        )

    def surface_model(self) -> type[BaseModel]:
        fields = {k: (t, ...) for k, t in self.surface_fields.items()}
        return create_model(f"{self.name}_surface", __config__=ConfigDict(extra="forbid"), **fields)  # type: ignore[call-overload]

    def surface_json_schema(self) -> dict[str, Any]:
        return self.surface_model().model_json_schema()

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} has no offline template (real generator required)")

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def finalize_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        """Hook to overwrite structured fields of a schema-valid surface with the world's exact values."""
        return surface

    def check_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> list[str]:
        """Deterministic structured-content checks on the generated surface (fact_mismatch:* codes)."""
        return []

    def anchors(self, world: ScenarioWorld) -> list[str]:
        """Strings that must appear (case-insensitive) in the visible state text."""
        return []

    # ------------------------------------------------------------ verification
    def verifier_schema(self) -> type[BaseModel]:
        from typing import get_origin

        # list fields are non-nullable (empty list = none found); scalars stay Optional (null = cannot tell)
        fields: dict[str, Any] = {k: ((t, Field(default_factory=list)) if get_origin(t) is list else (t | None, None))
                                  for k, t in self.verifier_fields.items()}
        return create_model(f"{self.name}_extraction", __config__=ConfigDict(extra="forbid"), **fields)  # type: ignore[call-overload]

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        raise NotImplementedError

    def verifier_context(self, state: dict[str, Any]) -> dict[str, Any]:
        return {"candidate_json": json.dumps(state, indent=2, ensure_ascii=False),
                "defaults": json.dumps(self.verifier_defaults) if self.verifier_defaults else ""}

    def compare_facts(self, world: ScenarioWorld, extracted: BaseModel) -> FactComparison:
        exp = self.expected_extraction(world)
        got = extracted.model_dump()
        out = FactComparison()
        for k, v in exp.items():
            g = got.get(k)
            if g is None and k in self.verifier_defaults:
                g = self.verifier_defaults[k]  # silence about a fact means its default (e.g. no revenue impact stated)
            if g is None or g == "unknown":
                out.unverifiable.append(k)
            elif _norm(g) == _norm(v):
                out.matched.append(k)
            else:
                out.mismatched.append(k)
        return out

    # ------------------------------------------------------------ record building
    def visible_options(self, world: ScenarioWorld) -> list[tuple[str, str]]:
        return list(self.options)

    def question_for(self, world: ScenarioWorld, variant: int) -> dict[str, Any]:
        # instruction wording is tied to the split so held-out splits use unseen phrasings
        split_idx = {"train": (0, 1), "calibration": (2,), "locked_test": (3,), "challenge": (4,)}[world.split]
        rng = make_rng(world.seed, variant, "question")
        instr = self.instruction_variants[rng.choice(split_idx) % len(self.instruction_variants)]
        q: dict[str, Any] = {"instructions": instr}
        if self.decision_type == "choice":
            opts = self.visible_options(world)
            rng.shuffle(opts)
            q["options"] = [OptionDef(id=i, definition=d).model_dump() for i, d in opts]
        elif self.decision_type == "score":
            q["levels"] = [LevelDef(value=v, definition=d).model_dump() for v, d in self.levels]
        return q

    def build_dataset_record(
        self, world: ScenarioWorld, truth: TruthRecord, rendered: RenderedState,
        quality: QualityRecord, provenance: ProvenanceRecord, variant: int = 0,
    ) -> DatasetRecord:
        state = self.state_from_surface(world, rendered.surface)
        pub = PublicTruth(
            preferred_option=truth.preferred_option, acceptable_options=truth.acceptable_options,
            preferred_level=truth.preferred_level, acceptable_levels=truth.acceptable_levels,
            truth=truth.boolean_truth, label_source=truth.label_source, oracle_version=truth.oracle_version,
            reason_code=truth.reason_code, label_quality=truth.label_quality, distribution=truth.distribution,
        )
        quality = quality.model_copy(update={"difficulty": world.difficulty, "tags": world.tags,
                                             "is_challenge": world.is_challenge})
        return DatasetRecord(
            record_id=f"osj-{world.scenario_id}-v{variant + 1:03d}", task_pack=self.name,
            task_version=self.version, split=world.split,
            decision=DecisionSpec(type=self.decision_type, state=state, question=self.question_for(world, variant)),  # type: ignore[arg-type]
            truth=pub, quality=quality, provenance=provenance,
            meta={"namespace": self.namespace, "scenario_family": world.scenario_family,
                  "split_family": world.split_family, "composition_family": self.composition_family,
                  "pool": world.hidden.get("pool"), "variant": variant},
        )

    # ------------------------------------------------------------ leakage / semantics
    def leak_patterns(self) -> list[str]:
        pats = list(GENERIC_LEAK_PATTERNS) + list(self.extra_leak_patterns)
        pats.append(re.escape(self.oracle_version))
        for oid in self.option_ids() if self.decision_type == "choice" else []:
            if oid in self.leak_exclude:
                continue
            pats.append(r"(?<![A-Za-z0-9])" + re.escape(oid).replace(r"_", r"[ _-]") + r"(?![A-Za-z0-9])")
        return pats

    def required_surface_features(self, world: ScenarioWorld) -> list[str]:
        """Style/scenario features the visible text must actually instantiate (checked by the verifier)."""
        return list(world.facts.get("required_surface_features", []))

    def check_surface_features(self, world: ScenarioWorld, got: dict[str, Any], c: FactComparison) -> None:
        """Scenario tags are first-class: a row whose family promises a feature (e.g. a non-English header) but whose text
        lacks it is rejected, otherwise per-slice metrics are meaningless."""
        need = [f for f in self.required_surface_features(world) if f not in self.code_checked_features]  # some are checked by code instead
        if not need:
            return
        have = set(got.get("surface_features") or [])
        missing = [f for f in need if f not in have]
        (c.mismatched if missing else c.matched).append("surface_features:" + missing[0] if missing else "surface_features")

    def leak_view(self, state: dict[str, Any]) -> dict[str, Any]:
        """State as seen by the leakage scan. Override where option ids are legitimately visible
        (e.g. the router's tool inventory, whose tool names ARE the option ids in the spec's example)."""
        return {k: v for k, v in state.items() if k not in self.code_rendered_keys}

    def validate_semantics(self, record: DatasetRecord) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        t = record.truth
        if self.decision_type == "choice":
            ids = {o["id"] for o in record.decision.question.get("options", [])}
            if ids != {o for o, _ in self.options} and ids != set(record.decision.question.get("_subset", ids)):
                issues.append(ValidationIssue(code="options_mismatch"))
            if t.preferred_option not in ids or not set(t.acceptable_options) <= ids:
                issues.append(ValidationIssue(code="truth_not_in_options"))
            if t.preferred_option not in t.acceptable_options:
                issues.append(ValidationIssue(code="preferred_not_acceptable"))
        elif self.decision_type == "score":
            vals = {lv["value"] for lv in record.decision.question.get("levels", [])}
            if t.preferred_level not in vals:
                issues.append(ValidationIssue(code="truth_not_in_levels"))
        elif t.truth is None:
            issues.append(ValidationIssue(code="boolean_truth_missing"))
        if not record.decision.state:
            issues.append(ValidationIssue(code="empty_state"))
        return issues
