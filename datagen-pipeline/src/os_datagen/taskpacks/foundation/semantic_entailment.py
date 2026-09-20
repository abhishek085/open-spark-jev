from __future__ import annotations

import random
from typing import Any

from ...generation.controlled_worlds import Pool
from ...generation.symbolic import entailment_verdict
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision

AUTH = {"official": 2, "informal": 1}


def _fact(pred: str, ent: str, holds: bool = True, source: str = "S1", **kw: Any) -> dict[str, Any]:
    return {"type": "fact", "entity": ent, "property": pred, "holds": holds, "source": source, **kw}


def _rule(quant: str, p: str, q: str, source: str = "S1") -> dict[str, Any]:
    return {"type": "rule", "quantifier": quant, "if_property": p, "then_property": q, "source": source}


class SemanticEntailment(BaseTaskPack):
    name = "foundation_semantic_entailment_v1"
    namespace = "foundation"
    decision_type = "choice"
    oracle_version = "rule_engine_v1"
    label_source = "rule_engine_v1"
    prompt_id = "foundation.entailment"
    prompt_dir = "foundation/semantic_entailment"
    composition_family = "general_semantic"
    id_prefix = "sem"
    options = [
        ("entailed", "The evidence supports the claim."),
        ("contradicted", "The evidence conflicts with the claim."),
        ("unknown", "The evidence does not establish the claim."),
    ]
    instruction_variants = [
        "How is the claim supported by the evidence?",
        "Given only the evidence, what is the status of the claim?",
        "Decide whether the evidence establishes the claim, conflicts with it, or leaves it open.",
        "Based strictly on the evidence, classify the claim.",
        "Judge the claim against the evidence provided; do not use outside knowledge.",
    ]
    families = ["direct_fact", "direct_contradiction", "missing_evidence", "multi_hop", "distractor_facts",
                "quantifier_set", "source_conflict", "ambiguous_wording"]
    challenge_families = ["double_negation", "similar_entity", "long_chain", "salient_distractor"]
    allowed_distractors = ["unrelated_entity_fact", "irrelevant_rule", "emotionally_salient_unrelated_event"]
    surface_fields = {"evidence": list[str], "claim": str, "source_metadata": list[dict[str, Any]],
                      "distractor_tags": list[str]}
    verifier_fields = {"claim": str, "facts": list[str], "rules": list[str]}
    challenge_note = "Stress negation scope, similar entity names, long chains and salient irrelevant text."

    # ---------------------------------------------------------------- sampling
    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        ents = rng.sample(list(pool.objects), 4)
        props = rng.sample(list(pool.adjectives), 6)
        e, e2, e3 = ents[0], ents[1], ents[2]
        p = props
        sources = {"S1": "official"}
        ev: list[dict[str, Any]] = []
        claim: dict[str, Any]
        tags: list[str] = [family]
        diff = "easy"
        hidden: dict[str, Any] = {}

        def distract(n: int) -> list[dict[str, Any]]:
            out = [_fact(p[3], e3, True, distractor=True)] if n >= 1 else []
            if n >= 2:
                out.append(_rule("all", p[4], p[5]))
            return out

        if family == "direct_fact":
            ev = [_fact(p[0], e)] + distract(1)
            claim = {"entity": e, "property": p[0], "holds": True}
        elif family == "direct_contradiction":
            neg = rng.random() < 0.5
            ev = [_fact(p[0], e, not neg)] + distract(1)
            claim = {"entity": e, "property": p[0], "holds": neg}
        elif family == "missing_evidence":
            ev = [_fact(p[1], e), _fact(p[0], e2)]
            claim = {"entity": e, "property": p[0], "holds": rng.random() < 0.7}
        elif family in ("multi_hop", "long_chain"):
            n = 2 if family == "multi_hop" else 4
            ev = [_fact(p[0], e)] + [_rule("all", p[i], p[i + 1]) for i in range(n)]
            last = "no" if rng.random() < 0.3 and family == "multi_hop" else "all"
            ev[-1] = _rule(last, p[n - 1], p[n])
            claim = {"entity": e, "property": p[n], "holds": True}
            diff = "medium" if family == "multi_hop" else "hard"
            rng.shuffle(ev)
        elif family == "distractor_facts":
            ev = [_fact(p[0], e), _rule("all", p[0], p[1])] + distract(2) + [_fact(p[2], e2)]
            claim = {"entity": e, "property": p[1], "holds": True}
            rng.shuffle(ev)
            diff = "medium"
        elif family == "quantifier_set":
            q = rng.choice(["all", "no", "some"])
            ev = [_rule(q, p[0], p[1]), _fact(p[0], e)]
            claim = {"entity": e, "property": p[1], "holds": True}
        elif family == "source_conflict":
            sources = {"S1": "official", "S2": rng.choice(["official", "informal"])}
            ev = [_fact(p[0], e, True, "S1"), _fact(p[0], e, False, "S2")]
            claim = {"entity": e, "property": p[0], "holds": True}
            diff = "hard"
        elif family == "ambiguous_wording":
            ev = [_rule(rng.choice(["most", "several"]), p[0], p[1]), _fact(p[0], e)]
            claim = {"entity": e, "property": p[1], "holds": True}
            tags.append("ambiguity_permitted")
            hidden["adjudicated"] = True
            diff = "hard"
        elif family == "double_negation":
            ev = [_fact(p[0], e, True, phrase_as_double_negation=True)]
            claim = {"entity": e, "property": p[0], "holds": True}
            diff = "hard"
        elif family == "similar_entity":
            ea, eb = f"{e} A", f"{e} B"
            ev = [_fact(p[0], ea), _rule("all", p[0], p[1])]
            claim = {"entity": eb if rng.random() < 0.6 else ea, "property": p[1], "holds": True}
            diff = "hard"
        elif family == "salient_distractor":
            neg = rng.random() < 0.5
            ev = [_fact(p[0], e, not neg), {"type": "noise", "kind": "emotionally_salient_unrelated_event"}]
            claim = {"entity": e, "property": p[0], "holds": True}
            diff = "medium"
        else:
            raise ValueError(family)
        facts = {"evidence": ev, "claim": claim, "sources": sources}
        hidden.update({"tags": tags, "difficulty": diff})
        return facts, hidden

    # ---------------------------------------------------------------- oracle
    @staticmethod
    def _solver_inputs(facts: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        auth = {sid: AUTH[a] for sid, a in facts["sources"].items()}
        fl, rl = [], []
        for it in facts["evidence"]:
            if it["type"] == "fact":
                fl.append({"pred": it["property"], "ent": it["entity"].lower(), "pos": it["holds"], "auth": auth[it["source"]]})
            elif it["type"] == "rule":
                q = it["quantifier"]
                rl.append({"quant": q, "body": [[it["if_property"], True]], "head": [it["then_property"], q != "no"]})
        c = facts["claim"]
        return fl, rl, {"pred": c["property"], "ent": c["entity"].lower(), "pos": c["holds"]}

    def decide(self, world: ScenarioWorld) -> Decision:
        fl, rl, claim = self._solver_inputs(world.facts)
        v = entailment_verdict(fl, rl, claim)
        adj = bool(world.hidden.get("adjudicated"))
        return Decision(v, f"{world.scenario_family}:{v}", quality="adjudicated" if adj else None,
                        outcomes={"symbolic_verdict": v})

    # ---------------------------------------------------------------- rendering helpers
    def anchors(self, world: ScenarioWorld) -> list[str]:
        f = world.facts
        out = [f["claim"]["entity"], f["claim"]["property"]]
        for it in f["evidence"]:
            if it["type"] == "fact":
                out += [it["entity"], it["property"]]
            elif it["type"] == "rule":
                out += [it["if_property"], it["then_property"]]
        return sorted(set(out))

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        c = f["claim"]
        facts_ = [f"{i['entity']}|{i['property']}|{'true' if i['holds'] else 'false'}" for i in f["evidence"] if i["type"] == "fact"]
        rules_ = [f"{i['quantifier']}:{i['if_property']}>{i['then_property']}" for i in f["evidence"] if i["type"] == "rule"]
        return {"claim": f"{c['entity']}|{c['property']}|{'true' if c['holds'] else 'false'}", "facts": facts_, "rules": rules_}

    def compare_facts(self, world: ScenarioWorld, extracted: Any) -> Any:
        """Claim must match exactly; every world fact and rule must be present. Extra facts/rules are tolerated only if
        the solver, run on world items + extras, still returns the same verdict (extras cannot change the label)."""
        from ...schemas.validation import FactComparison

        exp, got = self.expected_extraction(world), extracted.model_dump()
        c = FactComparison()
        if got.get("claim") is None:
            c.unverifiable.append("claim")
        elif str(got["claim"]).lower().strip() == exp["claim"]:
            c.matched.append("claim")
        else:
            c.mismatched.append("claim")
        norm = lambda xs: {str(x).lower().strip() for x in (xs or [])}  # noqa: E731
        fl, rl, claim = self._solver_inputs(world.facts)
        for key in ("facts", "rules"):
            missing = norm(exp[key]) - norm(got.get(key))
            extras = norm(got.get(key)) - norm(exp[key])
            if missing:
                c.mismatched.append(f"{key}:missing")
                continue
            for x in extras:  # fold extras into the solver input
                try:
                    if key == "facts":
                        e, p_, v = x.split("|")
                        fl.append({"pred": p_, "ent": e, "pos": v == "true", "auth": max(AUTH.values())})  # unknown source: assume the strongest
                    else:
                        q, rest = x.split(":")
                        a, b = rest.split(">")
                        rl.append({"quant": q, "body": [[a, True]], "head": [b, q != "no"]})
                except ValueError:
                    c.mismatched.append(f"{key}:unparseable")
            if not any(m.startswith(key) for m in c.mismatched):
                c.matched.append(key)
        if not c.mismatched and entailment_verdict(fl, rl, claim) != entailment_verdict(*self._solver_inputs(world.facts)):
            c.mismatched.append("facts:extras_change_verdict")
        return c

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"evidence": surface["evidence"], "claim": surface["claim"], "source_metadata": surface["source_metadata"]}

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        n = ("Use each entity and property word exactly as given in WORLD_FACTS. Do not add sources, entities or statements "
             "beyond WORLD_FACTS (other than a noise item if one is listed). source_metadata must list exactly the sources in WORLD_FACTS.")
        if len(world.facts["sources"]) > 1:
            n += " Prefix each statement with its source id in square brackets, e.g. [S2]."
        return {"notes": n}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        multi = len(f["sources"]) > 1
        ev: list[str] = []
        for it in f["evidence"]:
            pre = f"[{it['source']}] " if multi else ""
            if it["type"] == "fact":
                e, pr = it["entity"], it["property"]
                if it.get("phrase_as_double_negation"):
                    s = f"It is not the case that the {e} is not {pr}."
                else:
                    s = f"The {e} is {pr}." if it["holds"] else f"The {e} is not {pr}."
                ev.append(pre + s)
            elif it["type"] == "rule":
                q, a, b = it["quantifier"], it["if_property"], it["then_property"]
                ev.append(pre + {"all": f"All {a} things are {b}.", "no": f"No {a} thing is {b}.",
                                 "some": f"Some {a} things are {b}.", "most": f"Most {a} things are {b}.",
                                 "several": f"Several {a} things are {b}."}[q])
            else:
                ev.append("Unrelated: the whole neighborhood was devastated by a flood last spring.")
        c = f["claim"]
        claim = f"The {c['entity']} is {c['property']}." if c["holds"] else f"The {c['entity']} is not {c['property']}."
        return {"evidence": ev, "claim": claim,
                "source_metadata": [{"source_id": s, "authority": a} for s, a in f["sources"].items()],
                "distractor_tags": []}
