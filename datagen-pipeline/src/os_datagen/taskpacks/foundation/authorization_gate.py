from __future__ import annotations

import random
from datetime import date
from typing import Any

from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision


def authorize(f: dict[str, Any]) -> tuple[str, str]:
    r, res, a = f["requester"], f["resource"], f["access"]
    perms = set(r["permissions"]) | (set(a["delegation"]["delegator_permissions"]) if a["delegation"] else set())
    if r["tenant"] != res["tenant"] and not (r["external_collaborator"] and res["shared_with_partners"]):
        return "deny", "tenant_boundary"
    if a["expires"] and date.fromisoformat(a["expires"]) < date.fromisoformat(a["today"]):
        return "deny", "access_expired"
    if f["action"] not in perms:
        return "deny", "least_privilege_violation"
    if f["purpose"] not in res["allowed_purposes"]:
        return "deny", "purpose_limitation"
    if res["includes_prohibited_field"]:
        return "deny", "prohibited_field_included"
    if res["classification"] == "restricted":
        return ("allow", "restricted_with_owner_approval") if a["owner_approval"] else ("require_approval", "restricted_requires_owner_approval")
    if res["classification"] == "confidential" and r["external_collaborator"]:
        return "require_approval", "confidential_external_collaborator"
    return "allow", "permitted_by_policy"


POLICY_CLAUSES = [
    "Access is granted without owner approval only when ALL of these hold: the requester's tenant matches the resource's tenant "
    "(or the resource is shared with partners and the requester is an external collaborator); the requester holds the permission for the "
    "requested action (permissions delegated to the requester count); the stated purpose is one of the resource's allowed purposes; "
    "the resource is classified public or internal; and no prohibited field is included.",
    "Owner approval is required, and the request is held until the owner approves, when the resource is restricted, or when it is "
    "confidential and the requester is an external collaborator. If the owner has already approved, the request may proceed.",
    "Access is refused when the tenants differ (outside partner sharing), the requester lacks the permission for the action, the purpose "
    "is not allowed, a prohibited field is included, or the time-limited access has expired.",
    "Order of evaluation: if any refusal condition applies, refuse; owner approval is considered only when no refusal condition applies; "
    "otherwise access is granted.",
]


class AuthorizationGate(BaseTaskPack):
    name = "foundation_authorization_gate_v1"
    namespace = "foundation"
    oracle_version = "rbac_abac_policy_v1"
    label_source = "rbac_policy_engine_v1"
    prompt_id = "foundation.authorization"
    prompt_dir = "foundation/authorization_gate"
    composition_family = "security_policy"
    code_rendered_keys = ("policy_excerpt",)
    supportability = "reject"
    id_prefix = "aut"
    options = [
        ("allow", "Grant the requested access."),
        ("deny", "Refuse the requested access."),
        ("require_approval", "Hold the request until an owner approves it."),
    ]
    instruction_variants = [
        "How should this access request be handled?",
        "What should the access system do with this request under the policy?",
        "Apply the policy excerpt and context to decide how to treat the request.",
        "Decide the outcome of this access request.",
        "Given the requester, resource and policy, how should access be handled?",
    ]
    extra_leak_patterns = [r"\bshould be (granted|denied)\b"]
    families = ["role_permission_match", "least_privilege_violation", "owner_approval_given", "tenant_boundary", "restricted_needs_approval",
                "temporary_access", "expired_access", "delegated_access", "purpose_limitation"]
    challenge_families = ["conflicting_role_claims", "aggregate_with_prohibited_field", "external_collaborator", "inherited_permission", "approval_exception"]
    surface_fields = {"access_request": str, "requester_context": dict[str, Any], "resource_context": dict[str, Any],
                      "policy_excerpt": str, "distractor_tags": list[str]}
    verifier_defaults = {"access_expired": False, "external_collaborator": False, "prohibited_field_included": False, "owner_approval": False}
    verifier_fields = {"requester_permissions": list[str], "requested_action": str, "classification": str, "same_tenant": bool,
                       "access_expired": bool, "purpose_permitted": bool, "owner_approval": bool, "external_collaborator": bool,
                       "prohibited_field_included": bool}
    challenge_note = "Add conflicting claims or lookalike-permitted requests; permissions, dates and approvals must stay exactly as given."

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        role = rng.choice(["analyst", "engineer", "support_agent", "coordinator"])
        tenant = f"tenant-{rng.choice('abcdef')}{rng.randint(1, 9)}"
        res_name = rng.choice(["payroll_export", "customer_ledger", "site_plans", "incident_archive", "training_records"])
        purposes = ["monthly reporting", "incident review", "audit support"]
        f: dict[str, Any] = {
            "requester": {"role": role, "permissions": ["read", "read_aggregates"], "tenant": tenant, "external_collaborator": False, "person": pool.person(rng)},
            "resource": {"name": res_name, "classification": "internal", "tenant": tenant, "allowed_purposes": purposes,
                         "includes_prohibited_field": False, "shared_with_partners": False, "owner": pool.person(rng)},
            "action": "read", "purpose": purposes[0],
            "access": {"today": "2026-09-19", "expires": None, "owner_approval": False, "delegation": None},
        }
        R, res, a = f["requester"], f["resource"], f["access"]
        diff = "easy"
        if family == "role_permission_match":
            pass
        elif family == "least_privilege_violation":
            f["action"] = rng.choice(["export", "write"])
            res["classification"] = rng.choice(["restricted", "confidential"])
        elif family == "owner_approval_given":
            res["classification"], a["owner_approval"] = "restricted", True
            diff = "medium"
        elif family == "restricted_needs_approval":
            res["classification"] = "restricted"
            diff = "medium"
        elif family == "tenant_boundary":
            res["tenant"] = "tenant-z9"
        elif family == "temporary_access":
            a["expires"] = "2026-12-31"
        elif family == "expired_access":
            a["expires"] = rng.choice(["2026-08-01", "2026-09-18"])
        elif family == "delegated_access":
            R["permissions"] = ["read_aggregates"]
            f["action"] = "export"
            a["delegation"] = {"delegator": pool.person(rng), "delegator_permissions": ["export"] if rng.random() < 0.7 else ["read"]}
            diff = "medium"
        elif family == "purpose_limitation":
            f["purpose"] = rng.choice(["marketing research", "personal curiosity"])
        elif family == "conflicting_role_claims":
            R["claimed_roles"] = [role, "administrator (self-declared, unverified)"]
            f["action"] = "export"
            diff = "hard"
        elif family == "aggregate_with_prohibited_field":
            f["action"] = "read_aggregates"
            res["includes_prohibited_field"] = True
            diff = "hard"
        elif family == "external_collaborator":
            R["external_collaborator"], R["tenant"] = True, "tenant-ext1"
            res["classification"], res["shared_with_partners"] = rng.choice(["confidential", "internal"]), True
            diff = "hard"
        elif family == "inherited_permission":
            R["permissions"], R["inherited_from_group"] = ["read_aggregates"], "regional-analysts"
            f["action"] = "read_aggregates"
            R["permissions"].append("read")
            diff = "hard"
        elif family == "approval_exception":
            res["classification"] = "restricted"
            a["owner_approval"] = rng.random() < 0.5
            f["action"] = "read_aggregates"
            R["permissions"] = ["read", "read_aggregates"]
            diff = "hard"
        f["policy_clauses"] = POLICY_CLAUSES
        return f, {"difficulty": diff, "tags": [family]}

    def decide(self, world: ScenarioWorld) -> Decision:
        a, why = authorize(world.facts)
        return Decision(a, why)

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        r, res, ac = f["requester"], f["resource"], f["access"]
        perms = set(r["permissions"]) | (set(ac["delegation"]["delegator_permissions"]) if ac["delegation"] else set())
        exp = bool(ac["expires"]) and date.fromisoformat(ac["expires"]) < date.fromisoformat(ac["today"])
        return {"requester_permissions": sorted(perms), "requested_action": f["action"], "classification": res["classification"],
                "same_tenant": r["tenant"] == res["tenant"], "access_expired": exp, "purpose_permitted": f["purpose"] in res["allowed_purposes"],
                "owner_approval": ac["owner_approval"], "external_collaborator": r["external_collaborator"],
                "prohibited_field_included": res["includes_prohibited_field"]}

    def compare_facts(self, world: ScenarioWorld, extracted: Any) -> Any:
        import re

        d = extracted.model_dump()
        d["requester_permissions"] = sorted({re.sub(r"\s*\(.*?\)", "", str(x)).strip().lower() for x in d.get("requester_permissions") or []})  # 'export (delegated from X)' -> 'export'
        return super().compare_facts(world, type(extracted).model_validate(d))

    def finalize_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        """The policy carries the decision rules, so it is rendered by code (verbatim clauses), never paraphrased by the LLM."""
        return {**surface, "policy_excerpt": " ".join(world.facts["policy_clauses"])}

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"notes": "requester_context.permissions: list every permission the requester effectively holds (include the delegator's permissions "
                         "when access is delegated and say it is delegated). resource_context must include classification, tenant, allowed purposes, "
                         "and whether the export includes a prohibited field. State today's date and any expiry date, and whether the resource owner "
                         "has approved. policy_excerpt: write any short placeholder sentence (it is replaced by the official policy text); never say what should happen. "
                         "access_request: one or two sentences with the action and stated purpose."}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {k: surface[k] for k in ("access_request", "requester_context", "resource_context", "policy_excerpt")}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        r, res, a = f["requester"], f["resource"], f["access"]
        return {
            "access_request": f"{r['person']} ({r['role']}) requests to {f['action'].replace('_', ' ')} {res['name']} for {f['purpose']}.",
            "requester_context": {"role": r["role"], "permissions": sorted(set(r["permissions"]) | (set(a["delegation"]["delegator_permissions"]) if a["delegation"] else set())),
                                  "tenant": r["tenant"], "external_collaborator": r["external_collaborator"], "delegated": bool(a["delegation"])},
            "resource_context": {"name": res["name"], "classification": res["classification"], "tenant": res["tenant"],
                                 "allowed_purposes": res["allowed_purposes"], "includes_prohibited_field": res["includes_prohibited_field"],
                                 "owner_approval": a["owner_approval"], "today": a["today"], "access_expires": a["expires"]},
            "policy_excerpt": " ".join(f["policy_clauses"]), "distractor_tags": []}
