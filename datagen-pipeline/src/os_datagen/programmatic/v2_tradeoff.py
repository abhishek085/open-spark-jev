"""v2 family: weighted_tradeoff -- score options against stated weights; a near-tie is honestly a near-tie.

Replaces prog_v1 tradeoff_weighted: always three criteria with weights 5/3/2 over the same four supplier names, "higher is
better" everywhere, always a within-1-point tie rule. Here the criteria set (3-5 of a per-skin pool), the weights (integers
or percentages), the direction of each criterion ("lower is better" => scored as 11 minus the value), the option count
(3-5) and the tie tolerance (0-3 points, or 0/10/20/30 for percentage weights) all vary and are stated in the document.
5 skins (supplier bids, candidate hiring, tool selection | held out: site selection, loan offers), 4 layouts.
Gold dist is uniform over the options within the stated tolerance of the top total; re-derived independently by verify_row().
"""

from __future__ import annotations

import random

from .v2 import _org, _person, record_v2

FAMILY = "weighted_tradeoff"
TRAIN_SKINS = ["supplier_bids", "candidate_hiring", "tool_selection"]
SPLIT_CFG = {
    "train": dict(skins=TRAIN_SKINS, layouts=["lines", "md_table"], phrase=(0, 1, 2)),
    "calibration": dict(skins=TRAIN_SKINS, layouts=["prose"], phrase=(3,)),
    "test_locked": dict(skins=["site_selection"], layouts=["lines", "md_table", "prose"], phrase=(3, 4)),
    "challenge": dict(skins=["loan_offers"], layouts=["memo"], phrase=(3, 4)),
}
H, L = "higher", "lower"
SKINS = {
    "supplier_bids": dict(
        noun="supplier",
        head="BID EVALUATION",
        crit=[
            ("unit price", L),
            ("lead time", L),
            ("defect rate", L),
            ("warranty length", H),
            ("sustainability rating", H),
        ],
    ),
    "candidate_hiring": dict(
        noun="candidate",
        head="CANDIDATE SCORECARD",
        crit=[
            ("technical skill", H),
            ("relevant experience", H),
            ("salary expectation", L),
            ("notice period", L),
            ("team fit", H),
        ],
    ),
    "tool_selection": dict(
        noun="tool",
        head="TOOL COMPARISON",
        crit=[
            ("licence cost", L),
            ("integrations", H),
            ("support quality", H),
            ("setup time", L),
            ("security posture", H),
        ],
    ),
    "site_selection": dict(
        noun="site",
        head="SITE SHORTLIST SCORING",
        crit=[
            ("monthly rent", L),
            ("commute time", L),
            ("foot traffic", H),
            ("parking", H),
            ("street visibility", H),
        ],
    ),
    "loan_offers": dict(
        noun="loan offer",
        head="LOAN OFFER REVIEW",
        crit=[
            ("annual interest rate", L),
            ("upfront fees", L),
            ("repayment flexibility", H),
            ("approval speed", H),
            ("lender reputation", H),
        ],
    ),
}
SYL = [
    "brik",
    "cor",
    "dun",
    "eld",
    "fal",
    "gor",
    "hal",
    "ist",
    "jor",
    "kel",
    "lum",
    "mar",
    "nor",
    "ost",
    "pel",
    "quin",
    "ros",
    "sel",
    "tav",
    "urn",
    "vel",
    "wyn",
]
PROMPTS = [
    "Which {noun} should be selected?",
    "Apply the scoring table and pick the {noun}.",
    "Score the options as specified and select the winner.",
    "Using the stated weights, which {noun} wins?",
    "Select the {noun} with the highest weighted score.",
]
TIE_T = [
    "Totals within {d} point{s} of the highest total are not meaningfully different and count as tied.",
    "Treat any option whose total is {d} point{s} or less below the top total as tied with it.",
    "A gap of {d} point{s} or less to the leader is within measurement error and counts as a tie.",
    "Options trailing the top total by no more than {d} point{s} are considered level with it.",
    "Anything within {d} point{s} of the best total is a tie for practical purposes.",
]
TIE0 = [
    "Only options with exactly equal top totals are tied.",
    "Ties arise only when totals are identical.",
    "A tie requires identical totals; any gap decides.",
    "There is no rounding allowance: only identical totals tie.",
    "Only an exact match on the top total counts as a tie.",
]


def _name(rng, skin):
    return _person(rng) if skin == "candidate_hiring" else rng.choice(SYL).capitalize() + rng.choice(SYL)


def _totals(crit, opts):
    return {
        o["name"]: sum(c["w"] * ((11 - s) if c["dir"] == L else s) for c, s in zip(crit, o["scores"]))
        for o in opts
    }


def gen(split: str, i: int) -> dict:
    cfg = SPLIT_CFG[split]
    rng = random.Random(f"os-prog-v2|{FAMILY}|{split}|{i}")
    skin_name = rng.choice(cfg["skins"])
    sk = SKINS[skin_name]
    layout = rng.choice(cfg["layouts"])
    pct = rng.random() < 0.4
    n_c, n_o = rng.randrange(3, 6), rng.randrange(3, 6)
    cr = rng.sample(sk["crit"], n_c)
    if pct:
        cuts = sorted(rng.sample(range(1, 20), n_c - 1))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [20])]
        weights = [5 * p for p in parts]
        delta = rng.choice([0, 10, 20, 30])
    else:
        weights = [rng.randrange(1, 6) for _ in range(n_c)]
        delta = rng.choice([0, 1, 2, 3])
    crit = [{"name": n, "dir": d, "w": w} for (n, d), w in zip(cr, weights)]
    names = []
    while len(names) < n_o:
        nm = _name(rng, skin_name)
        if nm not in names:
            names.append(nm)
    opts = [{"name": nm, "scores": [rng.randrange(1, 11) for _ in crit]} for nm in names]
    tot = _totals(crit, opts)
    if (
        rng.random() < 0.4
    ):  # force a genuine near-tie: nudge one non-leading option (single score change) to land within the tolerance of the leader
        lead = max(tot, key=tot.get)
        cands = []
        for o in opts:
            if o["name"] == lead:
                continue
            for ci, c in enumerate(crit):
                for ns in range(1, 11):
                    if ns == o["scores"][ci]:
                        continue
                    t2 = tot[o["name"]] + c["w"] * (
                        ((11 - ns) - (11 - o["scores"][ci])) if c["dir"] == L else (ns - o["scores"][ci])
                    )
                    gap = tot[lead] - t2
                    if 0 <= gap <= delta:
                        cands.append((o, ci, ns))
        if cands:
            o, ci, ns = rng.choice(cands)
            o["scores"][ci] = ns
            tot = _totals(crit, opts)
    best = max(tot.values())
    near = sorted(n for n, v in tot.items() if best - v <= delta)
    gold = sorted(n for n, v in tot.items() if v == best)[0]
    phrase_i = rng.choice(cfg["phrase"])

    def wtxt(c):
        return f"{c['w']}%" if pct else f"x{c['w']}"

    def ctxt(c):
        return f"{c['name']} {wtxt(c)}" + (" (lower is better)" if c["dir"] == L else "")

    org, who = _org(rng), _person(rng)
    rule = "Each option's total is the weighted sum of its criterion scores." + (
        " Criteria marked (lower is better) are scored as 11 minus the value shown."
        if any(c["dir"] == L for c in crit)
        else ""
    )
    tie = TIE0[phrase_i] if delta == 0 else TIE_T[phrase_i].format(d=delta, s="" if delta == 1 else "s")
    head = [f"{org} — {sk['head']}"]
    if layout == "lines":
        body = ["Criteria and weights: " + ", ".join(ctxt(c) for c in crit), ""] + [
            f"  {o['name']:<14}" + "  ".join(f"{c['name']}={s}" for c, s in zip(crit, o["scores"]))
            for o in opts
        ]
    elif layout == "md_table":
        body = [
            "Criteria and weights: " + ", ".join(ctxt(c) for c in crit),
            "",
            "| Option | " + " | ".join(c["name"] for c in crit) + " |",
            "|---|" + "---|" * len(crit),
        ] + [f"| {o['name']} | " + " | ".join(str(s) for s in o["scores"]) + " |" for o in opts]
    elif layout == "prose":
        body = ["The criteria, with their weights, are " + "; ".join(ctxt(c) for c in crit) + "."] + [
            f"{o['name']} scored " + ", ".join(f"{s} on {c['name']}" for c, s in zip(crit, o["scores"])) + "."
            for o in opts
        ]
    else:
        body = [
            f"MEMO from {who}: we compared {n_o} {sk['noun']}s. Weights — "
            + "; ".join(ctxt(c) for c in crit)
            + "."
        ] + [
            f"• {o['name']}: " + "; ".join(f"{c['name']} {s}" for c, s in zip(crit, o["scores"]))
            for o in opts
        ]
    state = "\n".join(head + [""] + body + ["", rule, tie, f"Prepared by {who}."])
    world = {"skin": skin_name, "layout": layout, "pct": pct, "delta": delta, "crit": crit, "opts": opts}
    options = sorted(names)
    return record_v2(
        FAMILY,
        split,
        i,
        state=state,
        prompt=rng.choice(PROMPTS).format(noun=sk["noun"]),
        qtype="choice",
        options=options,
        label=gold,
        dist={n: 1.0 for n in near},
        difficulty="hard" if len(near) > 1 else "medium",
        rationale=f"totals {tot}; within-{delta} set {near}",
        world=world,
        extra={"totals": tot, "near_tie": near, "acceptable": near},
    )


def verify_row(r: dict) -> str | None:
    w, text = r["meta"]["world"], r["state"]["content"]
    crit, opts, delta = w["crit"], w["opts"], w["delta"]
    tot = {}
    for o in opts:  # recompute totals with an explicit loop (generator used a comprehension)
        t = 0
        for c, s in zip(crit, o["scores"]):
            t += c["w"] * (11 - s if c["dir"] == "lower" else s)
        tot[o["name"]] = t
    best = max(tot.values())
    near = sorted(n for n, v in tot.items() if best - v <= delta)
    if near != sorted(r["meta"]["near_tie"]):
        return "near set mismatch"
    d = r["target"]["dist"]
    if any(abs(d[n] - 1 / len(near)) > 2e-6 for n in near) or any(d[n] > 1e-9 for n in d if n not in near):
        return "dist mismatch"
    if r["target"]["label"] not in near:
        return "label outside near set"
    for o in opts:
        if o["name"] not in text:
            return "render lost an option"
    for c in crit:
        if c["name"] not in text:
            return "render lost a criterion"
    if delta == 0 and not any(x in text for x in TIE0):
        return "render lost the tie rule"
    if delta > 0 and str(delta) not in text.split("Prepared by")[0].split("\n")[-2]:
        return "render lost the tie tolerance"
    if any(c["dir"] == "lower" for c in crit) != ("lower is better" in text):
        return "direction note mismatch"
    return None
