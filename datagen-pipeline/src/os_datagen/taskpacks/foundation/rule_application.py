from __future__ import annotations

import random
from typing import Any

from ...generation.controlled_worlds import Pool
from ...generation.symbolic import closure, literal_status
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision


def _rule(body: list[str], head: str, unless: list[str] | None = None) -> dict[str, Any]:
    return {"if_all": body, "then": head, "unless": unless or []}


def solve(f: dict[str, Any]) -> bool:
    facts = [{"pred": x["property"], "ent": x["entity"], "pos": True} for x in f["facts"] if "property" in x]
    for n in f["numbers"]:
        for th in f["threshold_rules"]:
            if th["attribute"] == n["attribute"] and n["value"] >= th["at_least"]:
                facts.append({"pred": th["then"], "ent": n["entity"], "pos": True})
    rules = [{"quant": "all", "body": [[b, True] for b in r["if_all"]], "head": [r["then"], True], "unless": r["unless"]} for r in f["rules"]]
    q = f["query"]
    return literal_status(closure(facts, rules), q["property"], q["entity"], True) == "true"


class RuleApplication(BaseTaskPack):
    name = "foundation_rule_application_v1"
    namespace = "foundation"
    decision_type = "boolean"
    oracle_version = "rule_engine_boolean_v1"
    label_source = "rule_engine_v1"
    prompt_id = "foundation.rule"
    prompt_dir = "foundation/rule_application"
    composition_family = "general_reasoning"
    id_prefix = "rul"
    instruction_variants = [
        "Do the rules and facts imply the proposition in the query?",
        "Under the stated rules, does the information establish the query?",
        "Is the query a consequence of the rules and facts (nothing else may be assumed)?",
        "Using only the given rules and facts, is the proposition true?",
        "Decide whether the rules and facts together entail the proposition; anything not derivable counts as not implied.",
    ]
    families = ["implication_chain", "conjunction", "disjunction", "exception_override", "category_membership",
                "numeric_threshold", "nested_conditions", "incomplete_premise"]
    challenge_families = ["distractor_rules", "exception_chain", "rule_order_variation", "incomplete_premise_hard"]
    surface_fields = {"rules": list[str], "facts": list[str], "query": str, "distractor_tags": list[str]}
    verifier_fields = {"query": str, "facts": list[str], "rules": list[str], "numbers": list[str]}
    allowed_distractors = ["irrelevant_rule", "irrelevant_fact"]
    challenge_note = "Add distractor rules or reorder rules; meaning must be unchanged."

    def probability_semantics(self):  # type: ignore[no-untyped-def]
        s = super().probability_semantics()
        s.description += " 'False' means 'not implied under closed-world rule application', not 'the proposition is disproved'."
        return s

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        e, e2 = rng.sample(list(pool.objects), 2)
        p = rng.sample(list(pool.adjectives), 6)
        F = lambda pred, ent=e: {"entity": ent, "property": pred}  # noqa: E731
        f: dict[str, Any] = {"rules": [], "facts": [], "numbers": [], "threshold_rules": [], "query": {"entity": e, "property": p[3]}}
        diff = "easy"
        if family == "implication_chain":
            f["facts"] = [F(p[0])]
            f["rules"] = [_rule([p[0]], p[1]), _rule([p[1]], p[2]), _rule([p[2]], p[3])]
            if rng.random() < 0.4:
                f["facts"] = [F(p[4])]
            diff = "medium"
        elif family == "conjunction":
            f["facts"] = [F(p[0])] + ([F(p[1])] if rng.random() < 0.5 else [])
            f["rules"] = [_rule([p[0], p[1]], p[3])]
        elif family == "disjunction":
            f["facts"] = [F(p[rng.choice([0, 1, 4])])]
            f["rules"] = [_rule([p[0]], p[3]), _rule([p[1]], p[3])]
        elif family == "exception_override":
            f["facts"] = [F(p[0])] + ([F(p[4])] if rng.random() < 0.5 else [])
            f["rules"] = [_rule([p[0]], p[3], unless=[p[4]])]
            diff = "medium"
        elif family == "category_membership":
            f["facts"] = [F(p[0])]
            f["rules"] = [_rule([p[0]], p[1]), _rule([p[1]], p[3])] if rng.random() < 0.7 else [_rule([p[0]], p[1])]
        elif family == "numeric_threshold":
            v = rng.randint(40, 95)
            f["numbers"] = [{"entity": e, "attribute": "score", "value": v}]
            f["threshold_rules"] = [{"attribute": "score", "at_least": 70, "then": p[2]}]
            f["rules"] = [_rule([p[2]], p[3])]
            diff = "medium"
        elif family == "nested_conditions":
            f["facts"] = [F(p[0]), F(p[1])] if rng.random() < 0.6 else [F(p[0])]
            f["rules"] = [_rule([p[0], p[1]], p[2]), _rule([p[2]], p[3])]
            diff = "hard"
        elif family == "incomplete_premise":
            f["facts"] = [F(p[0])]
            f["rules"] = [_rule([p[0], p[1]], p[3])]
        elif family == "distractor_rules":
            f["facts"] = [F(p[0])]
            f["rules"] = [_rule([p[0]], p[3]), _rule([p[4]], p[5]), _rule([p[5]], p[1])] if rng.random() < 0.6 else [_rule([p[4]], p[3]), _rule([p[4]], p[5])]
            f["facts"] += [F(p[5], e2)]
            diff = "hard"
        elif family == "exception_chain":
            f["facts"] = [F(p[0]), F(p[1])]
            f["rules"] = [_rule([p[0]], p[3], unless=[p[4]]), _rule([p[1]], p[4])]
            diff = "hard"
        elif family == "rule_order_variation":
            f["facts"] = [F(p[0])]
            f["rules"] = [_rule([p[2]], p[3]), _rule([p[1]], p[2]), _rule([p[0]], p[1])]
            diff = "hard"
        elif family == "incomplete_premise_hard":
            f["facts"] = [F(p[0], e2), F(p[1])]
            f["rules"] = [_rule([p[0]], p[3]), _rule([p[1]], p[2])]
            diff = "hard"
        rng.shuffle(f["rules"])
        return f, {"difficulty": diff, "tags": [family]}

    def decide(self, world: ScenarioWorld) -> Decision:
        v = solve(world.facts)
        return Decision(v, f"{world.scenario_family}:{'derived' if v else 'not_derivable'}")

    def anchors(self, world: ScenarioWorld) -> list[str]:
        f = world.facts
        out = [f["query"]["entity"], f["query"]["property"]]
        for r in f["rules"]:
            out += r["if_all"] + [r["then"]] + r["unless"]
        return sorted(set(out))

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        rules = [("&".join(sorted(r["if_all"])) + ">" + r["then"] + (("|unless:" + ",".join(sorted(r["unless"]))) if r["unless"] else "")) for r in f["rules"]]
        rules += [f"{t['attribute']}>={t['at_least']}>{t['then']}" for t in f["threshold_rules"]]
        rules = [self._canon_rule(r) for r in rules]
        return {"query": f"{f['query']['entity']}|{f['query']['property']}", "facts": [f"{x['entity']}|{x['property']}" for x in f["facts"]],
                "rules": rules, "numbers": [f"{n['entity']}|{n['attribute']}={n['value']}" for n in f["numbers"]]}

    @staticmethod
    def _canon_rule(x: str) -> str:
        """Canonical form of a rule string, tolerant of notation drift: 'a&b>c|unless:d' regardless of where the
        unless-clause was written, order of conditions/exceptions, spacing and case."""
        import re

        x = x.lower().replace(" ", "")
        unl = sorted({u for m in re.findall(r"\|?unless:([a-z_,]+)", x) for u in m.split(",") if u})
        core = re.sub(r"\|?unless:[a-z_,]+", "", x).strip("|")
        if ">" not in core:
            return core
        body, head = core.rsplit(">", 1)
        if ">=" in core:  # numeric threshold rule 'attr>=n>head'
            return core
        return "&".join(sorted(body.split("&"))) + ">" + head + (("|unless:" + ",".join(unl)) if unl else "")

    def compare_facts(self, world: ScenarioWorld, extracted: Any) -> Any:
        """Canonicalise the verifier's notation (both sides are canonical); no shared mutable state (runs in threads)."""
        d = extracted.model_dump()
        d["rules"] = [self._canon_rule(r) for r in d.get("rules") or []]
        nums = {t["attribute"] for t in world.facts["threshold_rules"]}  # "satchel|score" is a number statement, not a property fact
        d["facts"] = [f for f in d.get("facts") or [] if f.split("|")[-1] not in nums]
        return super().compare_facts(world, type(extracted).model_validate(d))

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"notes": "rules[]: one natural-language rule per WORLD_FACTS rule/threshold rule; an 'unless' entry is an exception "
                         "('...unless X'). facts[]: one sentence per fact/number. Use each entity and property word exactly as given. "
                         "query: ask whether the proposition holds for the entity ('Does it follow that the <entity> is <property>?')."}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"rules": surface["rules"], "facts": surface["facts"], "query": surface["query"]}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        rules = []
        for r in f["rules"]:
            s = "If something is " + " and ".join(r["if_all"]) + f", then it is {r['then']}"
            rules.append(s + (f", unless it is {' or '.join(r['unless'])}." if r["unless"] else "."))
        rules += [f"Anything whose {t['attribute']} is at least {t['at_least']} is {t['then']}." for t in f["threshold_rules"]]
        facts = [f"The {x['entity']} is {x['property']}." for x in f["facts"]] + [f"The {n['entity']} has a {n['attribute']} of {n['value']}." for n in f["numbers"]]
        return {"rules": rules, "facts": facts, "query": f"Does it follow that the {f['query']['entity']} is {f['query']['property']}?", "distractor_tags": []}
