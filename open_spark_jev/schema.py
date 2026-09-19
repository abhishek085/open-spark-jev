"""Typed schemas for states, questions and answers.

These are the public API surface of open-spark-Jev. They are deliberately close to the
public shape of TypeSafe's Jev primitives (Choice / Score / Noul) so that code written
against a hosted Jev can be pointed at a local open-spark-Jev with minimal changes,
while staying fully open and self-hosted.

Design rules
------------
* A ``State`` is *data*, never instructions. It is rendered inside hard delimiters and
  the model is told to treat it as untrusted (see ``prompting.py``).
* A ``Question`` has a finite answer space. Every primitive is reduced to
  "one of N labels" so it can be scored with a single restricted-vocabulary softmax.
* An ``Answer`` always carries the full probability distribution plus a scalar
  confidence, so callers can threshold, abstain or escalate.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------


class State(BaseModel):
    """Arbitrary situation description: free text, a JSON object, or a list of records.

    ``content`` may be a string, a dict, or a list. Non-string content is serialised as
    compact JSON when rendered. ``schema_hint`` is an optional one-line description of
    the fields (e.g. "HTTP access log rows") that helps the model interpret raw JSON.
    """

    content: str | dict[str, Any] | list[Any]
    schema_hint: str | None = None
    domain: str | None = Field(
        default=None,
        description="Free-form domain tag (routing, moderation, security, game, ...). Used for "
        "evaluation slicing and domain-conditional rewards; not required at inference.",
    )

    def as_text(self, max_chars: int = 12_000) -> str:
        if isinstance(self.content, str):
            text = self.content
        else:
            text = json.dumps(self.content, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(text) > max_chars:
            # Keep head and tail; the middle of long logs is usually least informative.
            head = text[: max_chars * 2 // 3]
            tail = text[-(max_chars // 3) :]
            text = f"{head}\n...[truncated {len(text) - max_chars} chars]...\n{tail}"
        return text


# --------------------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------------------

MAX_OPTIONS = 26  # one single-token label per option (A..Z); see prompting.LabelSpace


class _QuestionBase(BaseModel):
    id: str | None = Field(default=None, description="Caller-supplied id echoed in the answer.")
    prompt: str = Field(..., description="The question, phrased about the state.")
    allow_abstain: bool = Field(
        default=False,
        description="Append an explicit 'insufficient information / abstain' option. Recommended "
        "for control policies where a wrong confident answer is costlier than escalation.",
    )

    @property
    def labels(self) -> list[str]:  # human-readable label per answer slot, in order
        raise NotImplementedError


class Choice(_QuestionBase):
    """Pick exactly one option from a finite list."""

    type: Literal["choice"] = "choice"
    options: list[str] = Field(..., min_length=2, max_length=MAX_OPTIONS)

    @field_validator("options")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("Choice options must be unique")
        return v

    @property
    def labels(self) -> list[str]:
        return list(self.options) + (["abstain"] if self.allow_abstain else [])


class Score(_QuestionBase):
    """Place the state on an ordered rubric.

    Either give explicit ordered ``levels`` (e.g. ["negligible","low","medium","high","critical"])
    or a numeric ``band`` ``{"min": 0, "max": 100, "steps": 11}`` that is discretised into
    evenly spaced levels. The answer's ``expected_value`` is the probability-weighted mean
    over level indices (or band values), which gives a continuous score for free.
    """

    type: Literal["score"] = "score"
    levels: list[str] | None = Field(default=None, min_length=2, max_length=MAX_OPTIONS)
    band: dict[str, float] | None = None
    rubric: str | None = Field(default=None, description="Optional text describing each level.")

    @model_validator(mode="after")
    def _levels_or_band(self) -> Score:
        if (self.levels is None) == (self.band is None):
            raise ValueError("Score needs exactly one of `levels` or `band`")
        if self.band is not None:
            for k in ("min", "max", "steps"):
                if k not in self.band:
                    raise ValueError(f"band needs key {k!r}")
            steps = int(self.band["steps"])
            if not 2 <= steps <= MAX_OPTIONS:
                raise ValueError(f"band.steps must be in [2, {MAX_OPTIONS}]")
        return self

    @property
    def level_values(self) -> list[float]:
        """Numeric value per level: band values, or 0..n-1 for named levels."""
        if self.band is not None:
            lo, hi, n = self.band["min"], self.band["max"], int(self.band["steps"])
            return [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        return [float(i) for i in range(len(self.levels or []))]

    @property
    def labels(self) -> list[str]:
        if self.levels is not None:
            base = list(self.levels)
        else:
            base = [f"{v:g}" for v in self.level_values]
        return base + (["abstain"] if self.allow_abstain else [])


class Noul(_QuestionBase):
    """Calibrated probability that ``claim`` is true of the state.

    ``prompt`` holds the claim. The answer space is {yes, no} (+ abstain).
    """

    type: Literal["noul"] = "noul"

    @property
    def claim(self) -> str:
        return self.prompt

    @property
    def labels(self) -> list[str]:
        return ["yes", "no"] + (["abstain"] if self.allow_abstain else [])


Question = Union[Choice, Score, Noul]


def parse_question(obj: dict[str, Any]) -> Question:
    t = obj.get("type")
    if t == "choice":
        return Choice(**obj)
    if t == "score":
        return Score(**obj)
    if t == "noul":
        return Noul(**obj)
    raise ValueError(f"unknown question type {t!r}")


# --------------------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------------------


class Answer(BaseModel):
    id: str | None = None
    type: Literal["choice", "score", "noul"]
    labels: list[str]
    probs: list[float] = Field(..., description="Softmax over `labels`, same order, sums to 1.")
    selected: str = Field(..., description="argmax label")
    confidence: float = Field(..., description="Probability mass on `selected`.")
    # Primitive-specific conveniences
    probability: float | None = Field(default=None, description="Noul only: P(claim is true).")
    expected_value: float | None = Field(default=None, description="Score only: E[level value].")
    entropy: float = Field(..., description="Shannon entropy (nats) of `probs`; high = uncertain.")
    abstained: bool = False
    raw_logits: list[float] | None = Field(default=None, description="Pre-softmax label logits.")

    @classmethod
    def from_probs(
        cls, q: Question, probs: list[float], raw_logits: list[float] | None = None
    ) -> Answer:
        import math

        labels = q.labels
        assert len(labels) == len(probs), (labels, probs)
        s = float(sum(probs))
        probs = [p / s for p in probs]
        idx = max(range(len(probs)), key=lambda i: probs[i])
        ent = -sum(p * math.log(p) for p in probs if p > 0)
        selected = labels[idx]
        ans: dict[str, Any] = dict(
            id=q.id,
            type=q.type,
            labels=labels,
            probs=probs,
            selected=selected,
            confidence=probs[idx],
            entropy=ent,
            abstained=(selected == "abstain"),
            raw_logits=raw_logits,
        )
        if isinstance(q, Noul):
            # Renormalise over {yes, no} so P(true) ignores abstain mass.
            py, pn = probs[0], probs[1]
            ans["probability"] = py / (py + pn) if (py + pn) > 0 else 0.5
        elif isinstance(q, Score):
            vals = q.level_values
            non_abstain = probs[: len(vals)]
            z = sum(non_abstain) or 1.0
            ans["expected_value"] = sum(p * v for p, v in zip(non_abstain, vals)) / z
        return cls(**ans)


# --------------------------------------------------------------------------------------
# Request / response envelopes for the HTTP API
# --------------------------------------------------------------------------------------


class DecisionRequest(BaseModel):
    """One state, many questions. All questions share the state's KV cache."""

    state: State
    questions: list[dict[str, Any]] = Field(..., min_length=1)
    model: str | None = Field(default=None, description="Model id from GET /v1/models (multi-model gateway); default if omitted.")
    temperature: float | None = Field(
        default=None, description="Override the calibrated temperature for this request."
    )
    return_logits: bool = False

    def parsed_questions(self) -> list[Question]:
        return [parse_question(q) for q in self.questions]


class DecisionResponse(BaseModel):
    answers: list[Answer]
    model: str
    latency_ms: float
    state_tokens: int
    backend: str
