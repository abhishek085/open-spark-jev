"""v2 family: policy_precedence -- read a long policy, apply the stated precedence, pick the disposition.

Replaces prog_v1 long_policy, which had fixed constants (vacancy limit always 60 days, seepage always 2 weeks, same four
clauses, same precedence), so the rule could be memorised without reading the document. Here the clause SET, the
thresholds, the facts, and the precedence rule (3 different statements of it) all change per row, so the answer depends on
reading. 5 skins (property claim, expense policy, SaaS refund, procurement approval | held out: permit review, warranty
return), 3 layouts of the facts, compositional filler clauses (no fixed boilerplate), generated names.

Splits (see v2.py): eval rule phrasings 3-4 never appear in train (0-2); calibration holds out the prose layout and the
'chain' precedence statement; test_locked holds out a whole skin (permit_review); challenge holds out a skin
(warranty_return) plus prose layout plus 'chain'. Gold comes from meta.world and is re-derived independently by verify().
"""

from __future__ import annotations

import operator
import random

from .v2 import _org, _person, _site, record_v2

FAMILY = "policy_precedence"
TRAIN_SKINS = ["property_claim", "expense_policy", "saas_refund", "procurement_approval"]
SPLIT_CFG = {
    "train": dict(
        skins=TRAIN_SKINS,
        layouts=["lines", "table"],
        prec=["numbered", "category"],
        phrase=(0, 1, 2),
        filler=(8, 36),
    ),
    "calibration": dict(skins=TRAIN_SKINS, layouts=["prose"], prec=["chain"], phrase=(3,), filler=(8, 36)),
    "test_locked": dict(
        skins=["permit_review"],
        layouts=["lines", "table"],
        prec=["numbered", "category", "chain"],
        phrase=(3, 4),
        filler=(8, 36),
    ),
    "challenge": dict(
        skins=["warranty_return"], layouts=["prose"], prec=["chain"], phrase=(3, 4), filler=(24, 36)
    ),
}
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt}
OP_WORDS = {
    ">": ["exceeds", "is more than", "is greater than"],
    ">=": ["is at least", "reaches", "is no less than"],
    "<": ["is below", "is less than", "falls short of"],
}


def _c(cid, heading, kind, fact, op, lo, hi, unit, outcome, out, step=1):
    return dict(
        id=cid,
        heading=heading,
        kind=kind,
        fact=fact,
        op=op,
        lo=lo,
        hi=hi,
        unit=unit,
        outcome=outcome,
        out=out,
        step=step,
    )


SKINS = {
    "property_claim": dict(
        subject="claim",
        title=["PROPERTY WORDING (extract)", "PROPERTY POLICY EXTRACT", "EXTRACT FROM PROPERTY COVER"],
        facts_h="LOSS FACTS",
        default=("pay_in_full", "paid in full"),
        clauses=[
            _c(
                "vac",
                "Vacancy",
                "exclusion",
                "days the premises were vacant before the loss",
                ">",
                14,
                90,
                "days",
                "deny_vacancy",
                "declined for vacancy",
            ),
            _c(
                "seep",
                "Seepage",
                "exclusion",
                "weeks the water was escaping before discovery",
                ">=",
                1,
                6,
                "weeks",
                "deny_seepage",
                "declined as gradual seepage",
            ),
            _c(
                "late",
                "Late notice",
                "exclusion",
                "days between discovery and notice to the insurer",
                ">",
                7,
                60,
                "days",
                "deny_late_notice",
                "declined for late notice",
            ),
            _c(
                "sub",
                "Unattended sublimit",
                "limit",
                "repair estimate",
                ">",
                10000,
                40000,
                "USD",
                "pay_capped",
                "paid only up to the sublimit",
                2500,
            ),
        ],
    ),
    "expense_policy": dict(
        subject="expense claim",
        title=[
            "TRAVEL AND EXPENSE POLICY",
            "EXPENSE REIMBURSEMENT RULES",
            "EMPLOYEE EXPENSE POLICY (extract)",
        ],
        facts_h="CLAIM FACTS",
        default=("approve_full", "approved in full"),
        clauses=[
            _c(
                "cap",
                "Daily cap",
                "limit",
                "claimed amount",
                ">",
                50,
                400,
                "USD",
                "approve_capped",
                "approved only up to the cap",
                10,
            ),
            _c(
                "rcpt",
                "Receipts",
                "exclusion",
                "receipts missing from the claim",
                ">=",
                1,
                3,
                "",
                "reject_no_receipt",
                "rejected for missing receipts",
            ),
            _c(
                "late",
                "Submission window",
                "exclusion",
                "days between the trip ending and submission",
                ">",
                30,
                90,
                "days",
                "reject_late",
                "rejected as late",
            ),
            _c(
                "proh",
                "Prohibited items",
                "exclusion",
                "prohibited items listed on the claim",
                ">=",
                1,
                2,
                "",
                "escalate_compliance",
                "escalated to compliance",
            ),
        ],
    ),
    "saas_refund": dict(
        subject="refund request",
        title=["REFUND POLICY", "SUBSCRIPTION REFUND TERMS", "TERMS OF SERVICE: REFUNDS"],
        facts_h="ACCOUNT FACTS",
        default=("refund_full", "refunded in full"),
        clauses=[
            _c(
                "use",
                "Usage",
                "exclusion",
                "hours of service used since purchase",
                ">",
                10,
                200,
                "hours",
                "deny_usage",
                "refused because usage exceeded the limit",
                5,
            ),
            _c(
                "win",
                "Refund window",
                "exclusion",
                "days since purchase",
                ">",
                14,
                60,
                "days",
                "deny_window",
                "refused because the window has closed",
            ),
            _c(
                "abuse",
                "Acceptable use",
                "exclusion",
                "acceptable-use violations logged",
                ">=",
                1,
                3,
                "",
                "deny_abuse",
                "refused for acceptable-use violations",
            ),
            _c(
                "prorata",
                "Prepaid term",
                "limit",
                "months remaining on the prepaid term",
                "<",
                2,
                8,
                "months",
                "refund_pro_rata",
                "refunded pro rata only",
            ),
        ],
    ),
    "procurement_approval": dict(
        subject="purchase request",
        title=["PROCUREMENT POLICY", "PURCHASE APPROVAL RULES", "SPENDING AUTHORITY MATRIX (extract)"],
        facts_h="REQUEST FACTS",
        default=("auto_approve", "approved automatically"),
        clauses=[
            _c(
                "val",
                "Value threshold",
                "limit",
                "order value",
                ">",
                5000,
                50000,
                "USD",
                "needs_director",
                "routed to the director",
                500,
            ),
            _c(
                "quote",
                "Competitive quotes",
                "exclusion",
                "quotes obtained",
                "<",
                2,
                4,
                "",
                "reject_quotes",
                "rejected for too few quotes",
            ),
            _c(
                "risk",
                "Vendor risk",
                "exclusion",
                "vendor risk score",
                ">=",
                60,
                90,
                "points",
                "escalate_risk",
                "escalated to vendor risk",
                5,
            ),
            _c(
                "bud",
                "Budget control",
                "exclusion",
                "percent of budget already used",
                ">",
                70,
                95,
                "percent",
                "hold_budget",
                "placed on hold for budget",
            ),
        ],
    ),
    "permit_review": dict(
        subject="permit application",
        title=[
            "PLANNING PERMIT RULES",
            "BUILDING PERMIT REVIEW CRITERIA",
            "LOCAL PLANNING REGULATIONS (extract)",
        ],
        facts_h="APPLICATION FACTS",
        default=("approve_permit", "approved"),
        clauses=[
            _c(
                "h",
                "Height limit",
                "exclusion",
                "proposed building height",
                ">",
                10,
                40,
                "metres",
                "reject_height",
                "refused for excess height",
            ),
            _c(
                "sb",
                "Boundary setback",
                "exclusion",
                "setback from the boundary",
                "<",
                1,
                6,
                "metres",
                "reject_setback",
                "refused for insufficient setback",
            ),
            _c(
                "obj",
                "Public objections",
                "exclusion",
                "written objections received",
                ">=",
                3,
                12,
                "",
                "refer_committee",
                "referred to committee",
            ),
            _c(
                "area",
                "Floor area",
                "limit",
                "gross floor area",
                ">",
                100,
                400,
                "sqm",
                "approve_conditional",
                "approved subject to conditions",
                10,
            ),
        ],
    ),
    "warranty_return": dict(
        subject="return request",
        title=[
            "WARRANTY AND RETURNS POLICY",
            "PRODUCT RETURN CONDITIONS",
            "CUSTOMER RETURNS TERMS (extract)",
        ],
        facts_h="RETURN FACTS",
        default=("repair_free", "repaired free of charge"),
        clauses=[
            _c(
                "age",
                "Coverage period",
                "exclusion",
                "months since purchase",
                ">",
                6,
                24,
                "months",
                "deny_expired",
                "declined as out of warranty",
            ),
            _c(
                "dmg",
                "Misuse",
                "exclusion",
                "damage reports marked customer-caused",
                ">=",
                1,
                2,
                "",
                "deny_misuse",
                "declined for misuse",
            ),
            _c(
                "rep",
                "Repeat repairs",
                "exclusion",
                "prior repair attempts",
                ">=",
                2,
                4,
                "",
                "replace_unit",
                "resolved by replacing the unit",
            ),
            _c(
                "price",
                "Refund cap",
                "limit",
                "purchase price",
                ">",
                200,
                900,
                "USD",
                "refund_capped",
                "refunded only up to the cap",
                50,
            ),
        ],
    ),
}

CLAUSE_T = [
    "Where {cond}, the {subject} is {out}.",
    "The {subject} is {out} if {cond}.",
    "A {subject} for which {cond} is {out}.",
    "Whenever {cond}, the outcome is: the {subject} is {out}.",
    "If {cond}, treat the {subject} as {out}.",
]
PROMPTS = [
    "What is the outcome of this {subject}?",
    "Apply the policy: how should this {subject} be resolved?",
    "Which disposition does the policy require for this {subject}?",
    "Read the clauses and the facts, then decide the outcome of the {subject}.",
]
_HEAD = [
    "Notice",
    "Records",
    "Inspection",
    "Assignment",
    "Currency",
    "Jurisdiction",
    "Communications",
    "Amendments",
    "Waiver",
    "Severability",
    "Definitions",
    "Confidentiality",
    "Audit",
    "Interpretation",
    "Governing law",
    "Data handling",
    "Headings",
    "Counterparts",
    "Entire agreement",
]
_PARTY = [
    "applicant",
    "claimant",
    "customer",
    "requesting party",
    "account holder",
    "supplier",
    "owner",
    "employee",
]
_ACT = [
    "notify us of",
    "retain records of",
    "provide evidence of",
    "confirm in writing",
    "allow inspection of",
    "disclose",
    "update",
    "return",
]
_OBJ = [
    "any change of circumstances",
    "the relevant documents",
    "the supporting receipts",
    "the affected items",
    "contact details",
    "related correspondence",
    "all material facts",
]
_FILL = [
    "The {party} shall {act} {obj} within {n} {unit}.",
    "Nothing in this clause requires the {party} to {act} {obj} earlier than {n} {unit} after a request.",
    "Failure to {act} {obj} does not of itself alter any other term, but may be noted on the file for {n} {unit}.",
    "The {party} may {act} {obj} by any reasonable means; a period of {n} {unit} is allowed for this.",
    "References in this document to {obj} include copies held by the {party} for at least {n} {unit}.",
]


def _num(n) -> str:
    return f"{n:,}"


def _val(n, unit) -> str:
    return f"{_num(n)} {unit}".strip()


def _holds(c, v) -> bool:
    return OPS[c["op"]](v, c["thr"])


def _precedence(clauses, mode, order, numbering) -> list[str]:
    """Independent re-derivation of the stated precedence (used by both gen and verify)."""
    if mode == "category":
        return [
            c["id"] for c in sorted(clauses, key=lambda c: (c["kind"] != "exclusion", numbering[c["id"]]))
        ]
    return list(order)


def gen(split: str, i: int) -> dict:
    cfg = SPLIT_CFG[split]
    rng = random.Random(f"os-prog-v2|{FAMILY}|{split}|{i}")
    skin_name = rng.choice(cfg["skins"])
    sk = SKINS[skin_name]
    layout, mode = rng.choice(cfg["layouts"]), rng.choice(cfg["prec"])
    org, who, site = _org(rng), _person(rng), _site(rng)
    k = rng.randrange(3, 5)
    types = rng.sample(sk["clauses"], k)
    clauses = []
    for t in types:
        steps = range(t["lo"], t["hi"] + 1, t["step"])
        thr = rng.choice(list(steps))
        holds = rng.random() < 0.6
        if t["op"] == "<":
            fact = (
                thr - rng.randrange(1, max(2, thr - t["lo"] + 2))
                if holds
                else thr + rng.randrange(0, 4) * t["step"]
            )
        else:
            fact = (
                thr + rng.randrange(1, 4) * t["step"]
                if holds
                else max(0, thr - rng.randrange(1, 4) * t["step"])
            )
        if t["op"] == ">=" and not holds:
            fact = max(0, thr - 1)
        if t["op"] == ">" and not holds:
            fact = max(0, thr - rng.randrange(0, 3) * t["step"])
        c = dict(t)
        c["thr"] = thr
        c["value"] = fact
        c["holds"] = _holds(c, fact)
        clauses.append(c)
    n_fill = rng.randrange(*cfg["filler"])
    body = [("op", c) for c in clauses] + [("fill", None)] * n_fill
    rng.shuffle(body)
    numbering, lines_c, used = {}, [], set()
    phrase_i = rng.choice(cfg["phrase"])
    for n, (kind, c) in enumerate(body, 1):
        if kind == "op":
            numbering[c["id"]] = n
            cond = f"the {c['fact']} {rng.choice(OP_WORDS[c['op']]) if phrase_i < 3 else OP_WORDS[c['op']][-1]} {_val(c['thr'], c['unit'])}"
            tag = f" [{'Exclusion' if c['kind'] == 'exclusion' else 'Limit'}]" if mode == "category" else ""
            lines_c.append(
                f"{n}. {c['heading']}{tag}. "
                + CLAUSE_T[phrase_i].format(cond=cond, subject=sk["subject"], out=c["out"])
            )
        else:
            h = rng.choice([x for x in _HEAD if x not in used] or _HEAD)
            used.add(h)
            lines_c.append(
                f"{n}. {h}. "
                + rng.choice(_FILL).format(
                    party=rng.choice(_PARTY),
                    act=rng.choice(_ACT),
                    obj=rng.choice(_OBJ),
                    n=rng.choice([5, 7, 10, 14, 21, 30, 45, 60]),
                    unit=rng.choice(["days", "weeks", "months"]),
                )
            )
    order = [c["id"] for c in clauses]
    rng.shuffle(order)
    by_id = {c["id"]: c for c in clauses}
    default_label, default_out = sk["default"]
    if mode == "numbered":
        prec = (
            "Order of precedence, highest first: "
            + "; ".join(f"{j + 1}) {by_id[x]['heading']}" for j, x in enumerate(order))
            + f". Where more than one of these clauses applies, the highest-ranked applicable clause decides the outcome; if none applies, the {sk['subject']} is {default_out}."
        )
    elif mode == "category":
        prec = (
            f"Clauses marked [Exclusion] prevail over clauses marked [Limit]. Among clauses of the same kind, the lower-numbered clause prevails. "
            f"If no such clause applies, the {sk['subject']} is {default_out}."
        )
        order = _precedence(clauses, "category", None, numbering)
    else:
        prec = (
            " ".join(
                f"The {by_id[a]['heading']} clause prevails over the {by_id[b]['heading']} clause."
                for a, b in zip(order, order[1:])
            )
            + f" If no such clause applies, the {sk['subject']} is {default_out}."
        )
    gold_id = next((x for x in _precedence(clauses, mode, order, numbering) if by_id[x]["holds"]), None)
    gold = by_id[gold_id]["outcome"] if gold_id else default_label
    n_hold = sum(c["holds"] for c in clauses)

    facts = [(c["fact"], _val(c["value"], c["unit"])) for c in clauses] + [("handled by", who)]
    rng.shuffle(facts)
    if layout == "lines":
        fb = [sk["facts_h"]] + [f"  {a[0].upper() + a[1:]}: {b}" for a, b in facts]
    elif layout == "table":
        fb = [sk["facts_h"], "| Fact | Value |", "|---|---|"] + [
            f"| {a[0].upper() + a[1:]} | {b} |" for a, b in facts
        ]
    else:
        fb = [sk["facts_h"], "The file records that " + "; ".join(f"the {a} was {b}" for a, b in facts) + "."]
    blocks = [[f"{org} — {rng.choice(sk['title'])} ({site})"], ["OPERATIVE CLAUSES"] + lines_c, [prec], fb]
    state = "\n\n".join("\n".join(b) for b in blocks)
    outs = [c["outcome"] for c in clauses] + [default_label]
    extra_out = [c["outcome"] for c in sk["clauses"] if c["outcome"] not in outs]
    if extra_out and rng.random() < 0.5:
        outs.append(rng.choice(extra_out))
    rng.shuffle(outs)
    world = {
        "skin": skin_name,
        "layout": layout,
        "prec_mode": mode,
        "phrase": phrase_i,
        "order": order,
        "numbering": numbering,
        "default": default_label,
        "clauses": [
            {k_: c[k_] for k_ in ("id", "heading", "kind", "fact", "op", "thr", "value", "unit", "outcome")}
            for c in clauses
        ],
    }
    return record_v2(
        FAMILY,
        split,
        i,
        state=state,
        prompt=rng.choice(PROMPTS).format(subject=sk["subject"]),
        qtype="choice",
        options=outs,
        label=gold,
        dist={o: (1.0 if o == gold else 0.0) for o in outs},
        difficulty="hard" if n_hold >= 2 else "medium",
        rationale=f"{n_hold} clause(s) apply; precedence {mode}: {gold_id or 'none'} decides -> {gold}",
        world=world,
        extra={"n_applicable": n_hold},
    )


def verify_row(r: dict) -> str | None:
    w, text = r["meta"]["world"], r["state"]["content"]
    cl = w["clauses"]
    by = {c["id"]: c for c in cl}
    for c in cl:
        ok = OPS[c["op"]](c["value"], c["thr"])
        c["_h"] = ok
    order = _precedence(cl, w["prec_mode"], w["order"], w["numbering"])
    if sorted(order) != sorted(by):
        return "precedence order is not a permutation of the clauses"
    gold = next((by[x]["outcome"] for x in order if by[x]["_h"]), w["default"])
    if gold != r["target"]["label"]:
        return "gold mismatch"
    if r["target"]["label"] not in r["question"]["options"] or len(set(r["question"]["options"])) != len(
        r["question"]["options"]
    ):
        return "bad options"
    for c in cl:
        if (
            c["heading"] not in text
            or _val(c["thr"], c["unit"]) not in text
            or _val(c["value"], c["unit"]) not in text
        ):
            return f"render lost clause/fact for {c['id']}"
        if c["out"] if False else False:
            pass
    if w["prec_mode"] == "numbered" and not all(
        by[x]["heading"] in text.split("Order of precedence")[1] for x in order
    ):
        return "numbered precedence lost"
    if w["prec_mode"] == "chain" and text.count("prevails over") != len(order) - 1:
        return "chain precedence lost"
    return None
