"""v2 family: ambiguous_routing -- the evidence may or may not pick one option; the honest answer is a distribution.

Replaces prog_v1 underdetermined: four hard-coded ticket queues with four fixed signal sentences, always "several unrelated
issues of equal weight". Here the options and signal phrasings come from 5 skins (support tickets, HR requests, incident
triage | held out: sales leads, legal intake), and the weighting RULE is stated in the document and varies:
  equal    -- every distinct issue counts the same          -> uniform over the issues raised
  urgency  -- issues carry urgency tags (urgent x3, normal x1) -> proportional to the tagged weights
  first    -- only the first-mentioned issue decides         -> one-hot (a fully determined case)
  single   -- one issue only (no ambiguity)                  -> one-hot
Mixing determined and ambiguous rows matters: a model must learn to spread mass only when the stated rule says to, not
always. Gold dist is computed from the structured world and re-derived independently by verify_row().
"""

from __future__ import annotations

import random

from .v2 import _org, _person, record_v2

FAMILY = "ambiguous_routing"
TRAIN_SKINS = ["support_ticket", "hr_request", "incident_triage"]
SPLIT_CFG = {
    "train": dict(
        skins=TRAIN_SKINS,
        layouts=["email", "form"],
        rules=["equal", "urgency", "first", "single"],
        phrase=(0, 1, 2),
    ),
    "calibration": dict(
        skins=TRAIN_SKINS, layouts=["chat"], rules=["equal", "urgency", "first", "single"], phrase=(3,)
    ),
    "test_locked": dict(
        skins=["sales_lead"],
        layouts=["email", "form"],
        rules=["equal", "urgency", "first", "single"],
        phrase=(3, 4),
    ),
    "challenge": dict(
        skins=["legal_intake"], layouts=["chat"], rules=["equal", "urgency", "first", "single"], phrase=(3, 4)
    ),
}
SKINS = {
    "support_ticket": dict(
        subject="ticket",
        dest="queue",
        head="inbound message",
        opts={
            "billing": [
                "an unexpected charge on the last invoice",
                "a refund that never arrived",
                "being billed twice this month",
                "a pricing error on the renewal",
            ],
            "technical": [
                "the device power-cycling during use",
                "an app that crashes on launch",
                "intermittent connection drops",
                "error 500 after the update",
            ],
            "account_access": [
                "being unable to sign in since Tuesday",
                "a password reset email that never comes",
                "a locked account after travelling",
                "two-factor codes that are rejected",
            ],
            "shipping": [
                "a parcel that has not moved in nine days",
                "a delivery left at the wrong address",
                "tracking that shows no updates",
                "a damaged box on arrival",
            ],
            "privacy": [
                "a request to delete personal data",
                "an unknown party emailing from our list",
                "a question about data retention",
                "a wish to export stored records",
            ],
        },
    ),
    "hr_request": dict(
        subject="request",
        dest="team",
        head="employee request",
        opts={
            "payroll": [
                "a missing overtime payment",
                "an incorrect tax code on the payslip",
                "a bonus that was not paid",
                "a deduction nobody explained",
            ],
            "benefits": [
                "adding a dependant to the health plan",
                "a pension contribution question",
                "a dental claim that was declined",
                "the enrolment deadline",
            ],
            "leave": [
                "carrying over unused holiday",
                "parental leave dates",
                "an unpaid leave application",
                "a sick-note submission",
            ],
            "it_equipment": [
                "a laptop that will not boot",
                "a replacement headset",
                "access to the shared drive",
                "a second monitor",
            ],
            "relocation": [
                "moving to another office",
                "a visa extension",
                "shipping household goods",
                "temporary housing support",
            ],
        },
    ),
    "incident_triage": dict(
        subject="alert",
        dest="on-call group",
        head="monitoring alert",
        opts={
            "network": [
                "packet loss between two regions",
                "a VPN tunnel that keeps flapping",
                "DNS answers timing out",
                "a saturated uplink",
            ],
            "database": [
                "replication lag growing for an hour",
                "connections exhausted on the primary",
                "a slow query locking a table",
                "disk nearly full on the data volume",
            ],
            "security": [
                "repeated failed logins from one address",
                "an unexpected admin role grant",
                "a leaked key in a public repository",
                "outbound traffic to an unknown host",
            ],
            "application": [
                "a deployment stuck at 60 percent",
                "memory climbing steadily in one service",
                "queue consumers falling behind",
                "a feature flag misfiring",
            ],
            "hardware": [
                "a fan failure warning on a rack",
                "an ECC error count rising",
                "a power supply reporting degraded",
                "temperature over threshold in row C",
            ],
        },
    ),
    "sales_lead": dict(
        subject="lead",
        dest="team",
        head="inbound lead",
        opts={
            "enterprise": [
                "a 5,000-seat rollout across three regions",
                "a request for a security questionnaire and SSO",
                "a multi-year procurement process",
                "a board-level sponsor asking for a roadmap",
            ],
            "smb": [
                "a ten-person team wanting to start this week",
                "a card-on-file signup with a question",
                "a small shop asking about monthly pricing",
                "a founder comparing two plans",
            ],
            "partner": [
                "an agency asking about reseller margins",
                "an integration vendor proposing a co-sell",
                "a consultancy wanting certification",
                "a marketplace listing request",
            ],
            "existing_customer": [
                "an expansion of a current contract",
                "a renewal with a new cost centre",
                "a seat transfer between departments",
                "an upgrade from the trial tier",
            ],
            "support_redirect": [
                "a bug report sent to sales by mistake",
                "a login problem on the pricing page",
                "a complaint about an invoice",
                "a how-to question about exports",
            ],
        },
    ),
    "legal_intake": dict(
        subject="matter",
        dest="practice group",
        head="intake note",
        opts={
            "contracts": [
                "a dispute over an auto-renewal clause",
                "a draft supply agreement to review",
                "a missed delivery milestone",
                "a request to add a liability cap",
            ],
            "employment": [
                "a wrongful dismissal allegation",
                "a non-compete question for a new hire",
                "unpaid commission owed to a rep",
                "a workplace complaint under investigation",
            ],
            "ip": [
                "a trademark opposition deadline",
                "a suspected copyright copy on the website",
                "a patent licence renewal",
                "an open-source licence conflict",
            ],
            "compliance": [
                "a regulator's information request",
                "a data breach notification decision",
                "an export-control screening hit",
                "an anti-bribery policy question",
            ],
            "litigation": [
                "a claim form received this morning",
                "a preservation notice from opposing counsel",
                "a settlement offer with a short fuse",
                "an injunction application",
            ],
        },
    ),
}
RULES_T = {
    "equal": [
        "Routing rule: every distinct issue raised counts equally. A {subject} raising several issues of equal weight has no single correct {dest}; do not guess.",
        "Every issue mentioned has the same weight. When a {subject} raises more than one, there is no single right {dest} and the router must not pretend otherwise.",
        "Treat all issues raised as equally important. If there are several, no one {dest} is correct.",
        "Policy: issues are not ranked by the order of mention; each counts once. More than one issue means the {dest} is genuinely undetermined.",
        "All issues in a {subject} carry equal weight; with several, the decision between {dest}s is open.",
    ],
    "urgency": [
        "Routing rule: issues tagged [URGENT] count three times as much as untagged issues. The {dest} is undetermined to the extent the weights are split.",
        "Weights: an [URGENT] issue counts 3, any other issue counts 1. Where the weights are split across {dest}s, the choice is correspondingly uncertain.",
        "Each [URGENT] issue is worth triple an ordinary one. Spread your confidence across {dest}s in proportion to the summed weights.",
        "Policy: urgency multiplies an issue's weight by three. The evidence for each {dest} is the total weight of the issues that point to it.",
        "Count [URGENT] issues as 3 and ordinary issues as 1; the {dest} with the most weight is favoured, in proportion.",
    ],
    "first": [
        "Routing rule: only the first issue mentioned decides the {dest}; later issues are ignored.",
        "The {dest} is decided solely by the issue raised first in the {subject}.",
        "Go by the first issue stated and disregard any others.",
        "Policy: the opening issue governs; anything after it does not change the {dest}.",
        "Only the first-mentioned issue counts when choosing the {dest}.",
    ],
    "single": [
        "Routing rule: the {subject} goes to the {dest} matching its issue.",
        "Send the {subject} to the {dest} that handles the issue it raises.",
        "The {dest} is the one matching the issue in the {subject}.",
        "Policy: match the issue to its {dest}.",
        "Choose the {dest} that owns the issue described.",
    ],
}
PROMPTS = [
    "Which {dest} should this {subject} go to?",
    "Route this {subject} to the correct {dest}.",
    "Select the {dest} this {subject} belongs with.",
    "Given only what the {subject} says, which {dest} applies?",
    "Assign this {subject} to a {dest}.",
]
OPENERS = [
    "Hello, I have a couple of separate problems and I am not sure who to ask.",
    "Hi team, several things need attention.",
    "Writing about a few different matters.",
    "Good morning. Here is what is going on.",
    "I need help with the following.",
]
OPENERS_ONE = [
    "Hello, I need some help with something.",
    "Hi, quick question for you.",
    "Writing about one thing.",
    "Good morning. Here is the situation.",
    "Could you point me in the right direction?",
]


def _weights(issues, rule):
    """issues: list of (option, urgent) in the order mentioned. Returns {option: weight} per the stated rule."""
    w: dict[str, float] = {}
    if rule == "first":
        w[issues[0][0]] = 1.0
    else:
        for o, urgent in issues:
            w[o] = w.get(o, 0.0) + (3.0 if (rule == "urgency" and urgent) else 1.0)
    return w


def gen(split: str, i: int) -> dict:
    cfg = SPLIT_CFG[split]
    rng = random.Random(f"os-prog-v2|{FAMILY}|{split}|{i}")
    skin_name = rng.choice(cfg["skins"])
    sk = SKINS[skin_name]
    layout, rule = rng.choice(cfg["layouts"]), rng.choice(cfg["rules"])
    org, who = _org(rng), _person(rng)
    all_opts = list(sk["opts"])
    k = 1 if rule == "single" else rng.choice([2, 2, 3, 4]) if rule != "first" else rng.choice([2, 3])
    chosen = rng.sample(all_opts, k)
    issues = []
    for o in chosen:
        issues.append((o, rule == "urgency" and rng.random() < 0.5, rng.choice(sk["opts"][o])))
    if rule == "urgency" and not any(u for _, u, _ in issues) and rng.random() < 0.5:
        j = rng.randrange(len(issues))
        issues[j] = (issues[j][0], True, issues[j][2])
    rng.shuffle(issues)
    w = _weights([(o, u) for o, u, _ in issues], rule)
    tot = sum(w.values())
    dist = {o: w.get(o, 0.0) / tot for o in all_opts}
    top = max(dist.values())
    acceptable = sorted(o for o, v in dist.items() if abs(v - top) < 1e-9)
    gold = acceptable[0]
    phrase_i = rng.choice(cfg["phrase"])
    bullets = [f"{p}{' [URGENT]' if u else ''}" for _, u, p in issues]
    if layout == "email":
        body = (
            [
                f"{org} — {sk['head']}",
                f"From: {who}",
                "",
                "Message:",
                f"  {rng.choice(OPENERS_ONE if len(issues) == 1 else OPENERS)}",
            ]
            + [f"  - {b}" for b in bullets]
            + ["  Please sort it out."]
        )
    elif layout == "form":
        body = [f"{org} — {sk['head']} form", f"Submitted by: {who}", "Issues reported:"] + [
            f"  {n}. {b}" for n, b in enumerate(bullets, 1)
        ]
    else:
        body = [
            f"[{org} chat] {who}: "
            + rng.choice(OPENERS_ONE if len(issues) == 1 else OPENERS)
            + " "
            + ("; ".join(f"({n}) {b}" for n, b in enumerate(bullets, 1)) if len(issues) > 1 else bullets[0])
            + "."
        ]
    body += ["", RULES_T[rule][phrase_i].format(subject=sk["subject"], dest=sk["dest"])]
    state = "\n".join(body)
    options = list(all_opts)
    rng.shuffle(options)
    world = {
        "skin": skin_name,
        "layout": layout,
        "rule": rule,
        "issues": [[o, bool(u), p] for o, u, p in issues],
        "all_options": all_opts,
    }
    kind = "determined" if len(acceptable) == 1 and top > 0.999 else "spread"
    return record_v2(
        FAMILY,
        split,
        i,
        state=state,
        prompt=rng.choice(PROMPTS).format(subject=sk["subject"], dest=sk["dest"]),
        qtype="choice",
        options=options,
        label=gold,
        dist=dist,
        difficulty="hard" if kind == "spread" else "medium",
        rationale=f"rule={rule}; weights { {o: round(v, 3) for o, v in w.items()} } -> {kind}",
        world=world,
        extra={"acceptable": acceptable, "kind": kind},
    )


def verify_row(r: dict) -> str | None:
    w, text = r["meta"]["world"], r["state"]["content"]
    issues = w["issues"]
    if w["rule"] == "first":
        exp = {issues[0][0]: 1.0}
    else:
        exp = {}
        for o, urgent, _ in issues:
            exp[o] = exp.get(o, 0.0) + (3.0 if (w["rule"] == "urgency" and urgent) else 1.0)
    s = sum(exp.values())
    for o in w["all_options"]:
        if abs(r["target"]["dist"].get(o, 0.0) - round(exp.get(o, 0.0) / s, 6)) > 2e-6:
            return f"dist mismatch on {o}"
    if sorted(r["question"]["options"]) != sorted(w["all_options"]):
        return "options changed"
    acc = r["meta"]["acceptable"]
    if r["target"]["label"] not in acc or any(
        abs(r["target"]["dist"][a] - max(r["target"]["dist"].values())) > 1e-6 for a in acc
    ):
        return "label not in argmax set"
    for _, urgent, phrase in issues:
        line = next((ln for ln in text.splitlines() if phrase in ln), None)
        if line is None:
            return "render lost an issue"
        if w["layout"] != "chat" and ((" [URGENT]" in line.split(phrase, 1)[1][:10]) != urgent):
            return "urgency tag mismatch on an issue line"
        if w["layout"] == "chat" and urgent and f"{phrase} [URGENT]" not in text:
            return "render lost urgency tag"
        if w["layout"] == "chat" and not urgent and f"{phrase} [URGENT]" in text:
            return "spurious urgency tag"
    return None
