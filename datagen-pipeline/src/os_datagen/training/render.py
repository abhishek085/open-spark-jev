from __future__ import annotations

import json
from typing import Any

from jinja2 import Environment

_ENV = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)

CHOICE_TMPL = _ENV.from_string("""STATE
{{ state }}

DECISION QUESTION
{{ instructions }}

ALLOWED OPTIONS
{% for option in options %}
- {{ option.id }}: {{ option.definition }}
{% endfor %}

Return only one option ID.
""")

SCORE_TMPL = _ENV.from_string("""STATE
{{ state }}

DECISION QUESTION
{{ instructions }}

SCORE LEVELS
{% for level in levels %}
- {{ level.value }}: {{ level.definition }}
{% endfor %}

Return only one level value.
""")

BOOLEAN_TMPL = _ENV.from_string("""STATE
{{ state }}

DECISION QUESTION
{{ instructions }}

Return only true or false.
""")


def render_state(state: dict[str, Any]) -> str:
    return json.dumps(state, indent=2, ensure_ascii=False)


def render_prompt(record: dict[str, Any]) -> str:
    """Model-facing prompt for a DatasetRecord (dict). Never includes truth or meta."""
    d = record["decision"]
    st, q = render_state(d["state"]), d["question"]
    if d["type"] == "choice":
        return CHOICE_TMPL.render(state=st, instructions=q["instructions"], options=q["options"])
    if d["type"] == "score":
        return SCORE_TMPL.render(state=st, instructions=q["instructions"], levels=q["levels"])
    return BOOLEAN_TMPL.render(state=st, instructions=q["instructions"])


def option_ids(record: dict[str, Any]) -> list[str]:
    d = record["decision"]
    if d["type"] == "choice":
        return [o["id"] for o in d["question"]["options"]]
    if d["type"] == "score":
        return [str(x["value"]) for x in d["question"]["levels"]]
    return ["true", "false"]


def truth_sets(record: dict[str, Any]) -> tuple[str, list[str]]:
    """(preferred id, acceptable ids) as strings."""
    t = record["truth"]
    if record["decision"]["type"] == "choice":
        return t["preferred_option"], list(t["acceptable_options"])
    if record["decision"]["type"] == "score":
        return str(t["preferred_level"]), [str(x) for x in t["acceptable_levels"]]
    v = str(bool(t["truth"])).lower()
    return v, [v]
