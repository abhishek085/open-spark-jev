"""v2 family: sampling_probability -- the honest answer is an exact probability strictly between 0 and 1.

Replaces prog_v1 probability_exact: one inspection world, one fixed rule ("reject if at least one sampled unit is
defective"), one question. Here the rule threshold K (1-3) is stated in the document in two logical forms ("K or more"
vs "more than K-1"), sampling is with or without replacement (stated), the event asked can be rejection OR acceptance,
the governing sampling plan is sometimes one of several dated revisions (latest governs), and there are 5 skins
(incoming inspection, invoice audit, vial QA | held out: ticket review, instrument check) and 4 layouts.
Gold distribution {yes: p, no: 1-p} is exact; verify() recomputes p by BRUTE-FORCE enumeration of subsets / sequences,
independent of the closed-form used to generate.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from fractions import Fraction
from itertools import combinations, product
from math import comb

from .v2 import _fmt_date, _org, _person, _site, record_v2

FAMILY = "sampling_probability"
TRAIN_SKINS = ["incoming_inspection", "invoice_audit", "vial_qa"]
SPLIT_CFG = {
    "train": dict(
        skins=TRAIN_SKINS, layouts=["prose", "kv", "bullets"], rule=["atleast"], phrase=(0, 1, 2), ds=(0, 1)
    ),
    "calibration": dict(skins=TRAIN_SKINS, layouts=["memo"], rule=["more_than"], phrase=(3,), ds=(2,)),
    "test_locked": dict(
        skins=["ticket_review"],
        layouts=["prose", "kv", "bullets"],
        rule=["atleast", "more_than"],
        phrase=(3, 4),
        ds=(0, 1),
    ),
    "challenge": dict(
        skins=["instrument_check"], layouts=["memo"], rule=["more_than"], phrase=(3, 4), ds=(2,)
    ),
}
SKINS = {
    "incoming_inspection": dict(
        items="units",
        item="unit",
        bad="defective",
        lot="lot",
        reject="the lot is rejected",
        head="INCOMING INSPECTION",
        prod=["flow regulators", "pressure sleeves", "drive couplings", "filter cartridges", "bearing races"],
    ),
    "invoice_audit": dict(
        items="invoices",
        item="invoice",
        bad="erroneous",
        lot="batch",
        reject="the batch is sent for full audit",
        head="INVOICE SAMPLING AUDIT",
        prod=["supplier invoices", "expense invoices", "freight invoices"],
    ),
    "vial_qa": dict(
        items="vials",
        item="vial",
        bad="contaminated",
        lot="batch",
        reject="the batch is quarantined",
        head="BATCH RELEASE TESTING",
        prod=["saline vials", "reagent vials", "buffer vials"],
    ),
    "ticket_review": dict(
        items="tickets",
        item="ticket",
        bad="misclassified",
        lot="queue",
        reject="the queue is sent back for re-triage",
        head="TRIAGE QUALITY REVIEW",
        prod=["support tickets", "access requests", "billing tickets"],
    ),
    "instrument_check": dict(
        items="instruments",
        item="instrument",
        bad="out of tolerance",
        lot="calibration run",
        reject="the calibration run is failed",
        head="CALIBRATION SPOT CHECK",
        prod=["torque wrenches", "pressure gauges", "digital scales"],
    ),
}
NUM_WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
RULE_T = {
    "atleast": [
        "The {lot} is rejected if {kw} or more of the sampled {items} are {bad}.",
        "Reject the {lot} when {kw} or more of the sampled {items} turn out {bad}.",
        "The {lot} fails inspection if the sample holds {kw} or more {bad} {items}.",
        "Rejection criterion: {kw} or more {bad} {items} in the sample.",
        "The {lot} is turned away whenever the sample contains {kw} or more {bad} {items}.",
    ],
    "more_than": [
        "The {lot} is rejected if more than {mw} of the sampled {items} are {bad}.",
        "Reject the {lot} when more than {mw} of the sampled {items} turn out {bad}.",
        "The {lot} fails inspection if the sample holds more than {mw} {bad} {items}.",
        "Rejection criterion: strictly more than {mw} {bad} {items} in the sample.",
        "The {lot} is turned away whenever the sample contains more than {mw} {bad} {items}.",
    ],
}
ASK = {
    "reject": [
        "Will {reject}? Give probabilities that reflect the evidence.",
        "Judge the probability that {reject}, using only the evidence in the state.",
        "Give a calibrated probability that {reject}.",
        "State how likely it is that {reject}.",
        "Weigh the evidence and give the probability that {reject}.",
    ],
    "accept": [
        "Will the {lot} pass? Give probabilities that reflect the evidence.",
        "Judge the probability that the {lot} passes, using only the evidence in the state.",
        "Give a calibrated probability that the {lot} is accepted.",
        "State how likely it is that the {lot} passes.",
        "Weigh the evidence and give the probability that the {lot} is accepted.",
    ],
}


def _p_event(lot, bad, draws, K, replace) -> Fraction:
    """Closed form (generator side): P(number of bad in sample >= K)."""
    if replace:
        q = Fraction(bad, lot)
        return sum(comb(draws, j) * q**j * (1 - q) ** (draws - j) for j in range(K, draws + 1))
    tot = comb(lot, draws)
    return Fraction(
        sum(comb(bad, j) * comb(lot - bad, draws - j) for j in range(K, min(bad, draws) + 1)), tot
    )


def gen(split: str, i: int) -> dict:
    cfg = SPLIT_CFG[split]
    rng = random.Random(f"os-prog-v2|{FAMILY}|{split}|{i}")
    skin_name = rng.choice(cfg["skins"])
    sk = SKINS[skin_name]
    layout, rule_form, ds = rng.choice(cfg["layouts"]), rng.choice(cfg["rule"]), rng.choice(cfg["ds"])
    replace = rng.random() < 0.3
    lot = rng.randrange(8, 13) if replace else rng.randrange(8, 17)
    bad = rng.randrange(1, max(2, lot // 2 + 1))
    draws = rng.randrange(2, 4) if replace else rng.randrange(2, 5)
    K = rng.randrange(1, min(draws, 3) + 1)
    asked = rng.choice(["reject", "accept"])
    org, who, site, prod = _org(rng), _person(rng), _site(rng), rng.choice(sk["prod"])
    n_plans = rng.choice([1, 1, 2, 3])
    base = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 300))
    plan_draws = (
        [draws]
        if n_plans == 1
        else rng.sample([d for d in range(2, 5) if d != draws], min(n_plans - 1, 2)) + [draws]
    )
    plans = [(base + timedelta(days=120 * k), d) for k, d in enumerate(plan_draws)]
    inspect_on = plans[-1][0] + timedelta(days=rng.randrange(1, 100))
    p_rej = _p_event(lot, bad, draws, K, replace)
    p = p_rej if asked == "reject" else 1 - p_rej
    gold = "yes" if p > Fraction(1, 2) else "no"

    phrase_i = rng.choice(cfg["phrase"])
    rule = RULE_T[rule_form][phrase_i].format(
        lot=sk["lot"], items=sk["items"], bad=sk["bad"], kw=NUM_WORD[K], mw=NUM_WORD[K - 1]
    )
    samp = (
        f"Each draw is made at random and the {sk['item']} is returned before the next draw."
        if replace
        else f"The sampled {sk['items']} are drawn at random without replacement."
    )

    def D(d):
        return _fmt_date(d, ds)

    truth = f"A full check found exactly {bad} of the {lot} {prod} {sk['bad']}; they cannot be told apart from the rest without that check."
    if n_plans > 1:
        plan_lines = [
            f"Sampling plan v{k + 1}: effective {D(d)}, draw {n} {sk['item'] if n == 1 else sk['items']}."
            for k, (d, n) in enumerate(plans)
        ]
        plan_txt = (
            "Sampling plan history (each version supersedes the previous one): "
            + " ".join(plan_lines)
            + f" Inspection date: {D(inspect_on)}."
        )
    else:
        plan_txt = f"Sampling plan: draw {draws} {sk['item'] if draws == 1 else sk['items']}. Inspection date: {D(inspect_on)}."
    note = ""
    if n_plans > 1 and rng.random() < 0.7:
        note = f'Note from {_person(rng)}: "Under the previous plan we drew {plans[-2][1]}, so I\'d expect this to be a coin flip."'
    if layout == "prose":
        parts = [
            f"{org} — {sk['head']}, {sk['lot']} of {lot} {prod} ({site})",
            truth,
            plan_txt,
            samp,
            rule,
            note,
        ]
    elif layout == "kv":
        parts = [
            f"{org} — {sk['head']} ({site})",
            f"  Population: {lot} {prod}",
            f"  Exactly {sk['bad']}: {bad}",
            f"  {plan_txt}",
            f"  Sampling: {samp}",
            f"  Rule: {rule}",
        ] + ([f"  {note}"] if note else [])
    elif layout == "bullets":
        parts = [
            f"{org} — {sk['head']} ({site})",
            f"- Population: {lot} {prod}.",
            f"- {truth}",
            f"- {plan_txt}",
            f"- {samp}",
            f"- {rule}",
        ] + ([f"- {note}"] if note else [])
    else:
        parts = [
            f"MEMO to {who}: {sk['head']} at {site}",
            f"We are testing a {sk['lot']} of {lot} {prod} for {org}. {truth} {plan_txt} {samp} {rule}"
            + (f" {note}" if note else ""),
        ]
    state = (
        "\n\n".join(x for x in parts if x)
        if layout in ("prose", "memo")
        else "\n".join(x for x in parts if x)
    )
    prompt = ASK[asked][phrase_i].format(reject=sk["reject"], lot=sk["lot"])
    world = {
        "skin": skin_name,
        "layout": layout,
        "rule_form": rule_form,
        "asked": asked,
        "lot": lot,
        "bad": bad,
        "draws": draws,
        "K": K,
        "replace": replace,
        "plans": [[d.isoformat(), n] for d, n in plans],
        "date_style": ds,
    }
    return record_v2(
        FAMILY,
        split,
        i,
        state=state,
        prompt=prompt,
        qtype="noul",
        options=None,
        label=gold,
        dist={"yes": float(p), "no": float(1 - p)},
        difficulty="hard",
        rationale=f"P(sample has >={K} {sk['bad']}) = {float(p_rej):.4f}; asked={asked} -> p={float(p):.4f}",
        world=world,
        extra={"exact_p_true": float(p)},
    )


def verify_row(r: dict) -> str | None:
    w, text = r["meta"]["world"], r["state"]["content"]
    lot, bad, d, K = w["lot"], w["bad"], w["draws"], w["K"]
    if w["plans"][-1][1] != d:
        return "latest plan does not match draws"
    units = [True] * bad + [False] * (lot - bad)
    if w["replace"]:
        seqs = list(product(range(lot), repeat=d))
        ok = sum(sum(units[j] for j in s) >= K for s in seqs)
        tot = len(seqs)
    else:
        subs = list(combinations(range(lot), d))
        ok = sum(sum(units[j] for j in s) >= K for s in subs)
        tot = len(subs)
    p_rej = ok / tot
    p = p_rej if w["asked"] == "reject" else 1 - p_rej
    if abs(p - r["meta"]["exact_p_true"]) > 1e-9:
        return "probability mismatch vs brute force"
    if abs(r["target"]["dist"]["yes"] - round(p, 6)) > 2e-6:
        return "target dist mismatch"
    if (r["target"]["label"] == "yes") != (p > 0.5):
        return "label mismatch"
    for needle in (str(lot), str(bad)):
        if needle not in text:
            return f"render lost {needle}"
    for _, n in w["plans"]:
        if f"draw {n} " not in text:
            return "render lost a plan draw count"
    if w["replace"] != ("returned before the next draw" in text):
        return "replacement statement mismatch"
    return None
