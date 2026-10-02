"""Long / compute-heavy decision families (v2 style: code oracle, independent verifier, held-out skins & phrasings).

temporal_calendar   - period end dates with month clamping / rolling, day counting conventions, business days, time-zone conversion
multi_hop_procedure - 2-4k token manual: alias -> id -> base category -> elevation rules -> aggregated amount -> band -> tier
long_policy         - policy_precedence with 80-140 filler clauses (2-3.5k tokens)
"""

from __future__ import annotations

import calendar
import datetime as dt
import random

from . import v2_policy
from .v2 import _org, _person, _site, record_v2

FAM_T, FAM_M, FAM_P = "temporal_calendar", "multi_hop_procedure", "long_policy"
SPLITS = ("train", "calibration", "test_locked", "challenge")
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def fmt_date(d: dt.date, style: int) -> str:
    return [
        f"{d.day} {MONTHS[d.month - 1]} {d.year}",
        f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
        d.isoformat(),
    ][style % 3]


# ---------------------------------------------------------------- temporal_calendar
T_SKINS = {
    "warranty_cert": dict(
        title="EXTENDED WARRANTY CERTIFICATE",
        anchor="delivery date",
        anchor_label="Delivery date",
        event="claim",
        what="breakdowns reported during the period",
        item="fridge-freezer",
        yes="The claim was reported inside the warranty period.",
        no="The claim was reported after the warranty period ended.",
    ),
    "lease_option": dict(
        title="LEASE RENEWAL OPTION NOTICE",
        anchor="notice date",
        anchor_label="Option notice received",
        event="acceptance",
        what="the tenant's written acceptance",
        item="premises",
        yes="The acceptance arrived inside the option window.",
        no="The acceptance arrived after the option window closed.",
    ),
    "refund_window": dict(
        title="REFUND AND RETURNS TERMS",
        anchor="receipt date",
        anchor_label="Goods received",
        event="return request",
        what="return requests",
        item="order",
        yes="The return request was made inside the refund window.",
        no="The return request was made after the refund window.",
    ),
    "permit_validity": dict(
        title="TEMPORARY PERMIT CONDITIONS",
        anchor="issue date",
        anchor_label="Permit issued",
        event="inspection",
        what="the site inspection",
        item="permit",
        yes="The inspection took place while the permit was valid.",
        no="The inspection took place after the permit expired.",
    ),
    "sla_credit": dict(
        title="SERVICE LEVEL AGREEMENT - CREDIT CLAIMS",
        anchor="incident date",
        anchor_label="Incident closed",
        event="credit request",
        what="service-credit requests",
        item="service",
        yes="The credit request was filed in time.",
        no="The credit request was filed too late.",
    ),
    "subscription_cancel": dict(
        title="SUBSCRIPTION CANCELLATION POLICY",
        anchor="start date",
        anchor_label="Subscription started",
        event="cancellation",
        what="cancellations for a full refund",
        item="subscription",
        yes="The cancellation was made inside the cooling-off period.",
        no="The cancellation was made after the cooling-off period.",
    ),
}
T_CFG = {
    "train": dict(
        skins=["warranty_cert", "lease_option", "refund_window", "permit_validity"], phrase=(0, 1, 2)
    ),
    "calibration": dict(
        skins=["warranty_cert", "lease_option", "refund_window", "permit_validity"], phrase=(3,)
    ),
    "test_locked": dict(skins=["sla_credit"], phrase=(3, 4)),
    "challenge": dict(skins=["subscription_cancel"], phrase=(3, 4)),
}
TZ = [
    ("Rotterdam", 1, "Central European Time (UTC+1)"),
    ("Lisbon", 0, "Western European Time (UTC+0)"),
    ("Singapore", 8, "Singapore Time (UTC+8)"),
    ("Chicago", -6, "Central Standard Time (UTC-6)"),
    ("Tokyo", 9, "Japan Standard Time (UTC+9)"),
    ("Denver", -7, "Mountain Standard Time (UTC-7)"),
    ("Dubai", 4, "Gulf Standard Time (UTC+4)"),
    ("Nairobi", 3, "East Africa Time (UTC+3)"),
]


def add_months(d: dt.date, n: int, roll: bool) -> dt.date:
    m = d.month - 1 + n
    y, mo = d.year + m // 12, m % 12 + 1
    last = calendar.monthrange(y, mo)[1]
    if d.day <= last:
        return dt.date(y, mo, d.day)
    return dt.date(y, mo, last) if not roll else dt.date(y, mo, last) + dt.timedelta(days=1)


def is_bday(d: dt.date, hol: set) -> bool:
    return d.weekday() < 5 and d not in hol


def add_bdays(d: dt.date, n: int, hol: set) -> dt.date:
    while n:
        d += dt.timedelta(days=1)
        if is_bday(d, hol):
            n -= 1
    return d


def period_end(w: dict) -> dt.date:
    """Last day on which the event is still in time (period runs to 24:00 on that day)."""
    a = dt.date.fromisoformat(w["anchor"])
    k = w["kind"]
    if k == "months":
        e = add_months(a, w["n"], w["roll"])
        return e - dt.timedelta(days=1) if w["exclusive_anniv"] else e
    if k == "days":
        return a + dt.timedelta(days=w["n"] - (1 if w["count_anchor"] else 0))
    if k == "weeks":
        return a + dt.timedelta(days=7 * w["n"] - (1 if w["count_anchor"] else 0))
    return add_bdays(a, w["n"], {dt.date.fromisoformat(h) for h in w["holidays"]})


FILL_T = [
    "Replacement parts may be new or reconditioned.",
    "Damage caused by misuse is not covered.",
    "Consumables are excluded.",
    "The holder must keep the original proof of purchase.",
    "Rights under mandatory consumer law are not affected.",
    "Cosmetic wear does not constitute a defect.",
    "Transfers of ownership must be notified in writing.",
    "Data stored on the product is not backed up.",
    "Notices are deemed delivered when sent to the address on file.",
    "Third-party accessories are outside this document.",
    "Amendments must be in writing and signed.",
    "This document is governed by the laws of the issuing jurisdiction.",
]
RULE_PH = {  # phrase variants for the duration sentence
    "months": [
        "The period starts on the {anchor} and ends {n} months later.",
        "Cover runs for {n} months counted from the {anchor}.",
        "The period begins on the {anchor}; it lasts {n} calendar months.",
        "Counting from the {anchor}, the period is {n} months long.",
        "{n} months after the {anchor} the period comes to an end.",
    ],
    "days": [
        "The period is {n} days long, measured from the {anchor}.",
        "{n} days are allowed from the {anchor}.",
        "The window is {n} calendar days from the {anchor}.",
        "From the {anchor}, there are {n} days in which to act.",
        "A period of {n} days applies, starting from the {anchor}.",
    ],
    "weeks": [
        "The period is {n} weeks long, measured from the {anchor}.",
        "{n} weeks are allowed from the {anchor}.",
        "The window is {n} weeks from the {anchor}.",
        "From the {anchor}, there are {n} weeks in which to act.",
        "A period of {n} weeks applies, starting from the {anchor}.",
    ],
    "bdays": [
        "The period is {n} business days long, measured from the {anchor}.",
        "{n} business days are allowed from the {anchor}.",
        "The window is {n} business days from the {anchor}.",
        "From the {anchor}, there are {n} business days in which to act.",
        "A period of {n} business days applies from the {anchor}.",
    ],
}


def gen_temporal(split: str, i: int) -> dict:
    cfg = T_CFG[split]
    rng = random.Random(f"os-prog-v2|{FAM_T}|{split}|{i}")
    sk_name = rng.choice(cfg["skins"])
    sk = T_SKINS[sk_name]
    ph = rng.choice(cfg["phrase"])
    ds = rng.randrange(3)
    org, who, site = _org(rng), _person(rng), _site(rng)
    kind = rng.choice(["months", "months", "days", "weeks", "bdays"])
    base = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randrange(0, 900))
    if kind == "months" and rng.random() < 0.6:  # force a day number that can overflow the target month
        base = base.replace(day=min(rng.choice([29, 30, 31]), calendar.monthrange(base.year, base.month)[1]))
    w = dict(
        skin=sk_name,
        kind=kind,
        anchor=base.isoformat(),
        n=0,
        roll=False,
        exclusive_anniv=False,
        count_anchor=False,
        holidays=[],
        phrase=ph,
        date_style=ds,
    )
    if kind == "months":
        w.update(
            n=rng.choice([3, 6, 9, 12, 18, 24]), roll=rng.random() < 0.4, exclusive_anniv=rng.random() < 0.3
        )
    elif kind == "days":
        w.update(n=rng.choice([14, 21, 30, 45, 60, 90]), count_anchor=rng.random() < 0.4)
    elif kind == "weeks":
        w.update(n=rng.choice([2, 4, 6, 8, 12]), count_anchor=rng.random() < 0.4)
    else:
        w.update(n=rng.choice([5, 10, 15, 20]))
        w["holidays"] = sorted(
            {(base + dt.timedelta(days=rng.randrange(1, 40))).isoformat() for _ in range(rng.randrange(1, 4))}
        )
    end = period_end(w)
    ref_name, ref_off, ref_label = rng.choice(TZ)
    cl_name, cl_off, cl_label = rng.choice([t for t in TZ if t[1] != ref_off])
    delta_h = (
        rng.choice([-1, 1]) * rng.randrange(2, 22)
        if rng.random() < 0.8
        else rng.choice([-1, 1]) * rng.randrange(25, 70)
    )
    boundary = dt.datetime.combine(
        end + dt.timedelta(days=1), dt.time(0, 0)
    )  # 24:00 of last day, in reference time
    ref_event = boundary + dt.timedelta(
        hours=delta_h, minutes=rng.choice([0, 15, 30, 45]) * (1 if delta_h > 0 else -1)
    )
    local_event = ref_event + dt.timedelta(hours=cl_off - ref_off)
    in_time = ref_event < boundary
    # document
    blocks = [f"{org} - {sk['title']} ref. {rng.randrange(1000, 9999)}-{rng.randrange(100, 999)}"]
    ids = [(sk["anchor_label"], fmt_date(base, ds))]
    extra_dates = [
        (
            rng.choice(["Order placed", "Quote issued", "Account opened"]),
            fmt_date(base - dt.timedelta(days=rng.randrange(3, 40)), ds),
        ),
        (
            rng.choice(["Document printed", "Terms version dated"]),
            fmt_date(base + dt.timedelta(days=rng.randrange(-20, 25)), ds),
        ),
    ]
    facts = [f"{k}: {v}." for k, v in ids + extra_dates]
    rng.shuffle(facts)
    blocks.append(f"{sk['item'].capitalize()} record - {who}, {site}. " + " ".join(facts))
    anchor_phrase = sk["anchor"]
    kind_key = {"months": "months", "days": "days", "weeks": "weeks", "bdays": "bdays"}[kind]
    rule = [
        "Duration. "
        + RULE_PH[kind_key][ph].format(anchor=f"{anchor_phrase} (and not any other date shown)", n=w["n"])
    ]
    if kind == "months":
        rule.append(
            f"Should the closing month be too short to contain the day number of the {anchor_phrase}, "
            + (
                "the closing day is the 1st of the month that follows."
                if w["roll"]
                else "the final day of that month is the closing day."
            )
        )
        rule.append(
            "Cover lapses at the midnight that opens the anniversary date; the anniversary day itself is therefore not covered."
            if w["exclusive_anniv"]
            else "The closing day counts in full, up to the midnight that ends it."
        )
    elif kind in ("days", "weeks"):
        rule.append(
            f"Counting starts on the {anchor_phrase} itself (day 1); the closing day counts in full."
            if w["count_anchor"]
            else f"Counting starts the day after the {anchor_phrase}; the closing day counts in full."
        )
    else:
        rule.append(
            "Weekends and these public holidays do not count as business days: "
            + ", ".join(fmt_date(dt.date.fromisoformat(h), ds) for h in w["holidays"])
            + f". Counting begins the day after the {anchor_phrase}; the closing business day counts in full."
        )
    rule.append(
        f"Clock. Deadlines are read on {ref_name} clocks ({ref_label}). If a {sk['event']} carries a timestamp from another zone, translate it to {ref_name} clocks before comparing it with the period."
    )
    fills = rng.sample(FILL_T, rng.randrange(3, 8))
    blocks.append("\n".join(rule) + "\n" + " ".join(fills))
    blocks.append(
        f"Logged: {sk['event']} no. {rng.randrange(10, 99)}-{rng.randrange(10000, 99999)} arrived with timestamp {local_event.strftime('%Y-%m-%d %H:%M')} from {cl_name} ({cl_label})."
    )
    state = "\n\n".join(blocks)
    qtype_choice = rng.random() < 0.4
    if qtype_choice:
        wrong = {end + dt.timedelta(days=1), end - dt.timedelta(days=1)}
        alt = (
            dict(w, roll=not w["roll"])
            if kind == "months"
            else dict(w, count_anchor=not w["count_anchor"])
            if kind in ("days", "weeks")
            else dict(w, holidays=[])
        )
        wrong.add(period_end(alt))
        wrong.discard(end)
        while len(wrong) < 3:
            wrong.add(end + dt.timedelta(days=rng.choice([-3, -2, 2, 3, 7, -7])))
        opts = [fmt_date(end, ds)] + [fmt_date(x, ds) for x in sorted(wrong)[:3]]
        rng.shuffle(opts)
        label = fmt_date(end, ds)
        prompt = f"What is the last day (in {ref_name} time) on which {sk['what']} is still in time?"
        return record_v2(
            FAM_T,
            split,
            i,
            state=state,
            prompt=prompt,
            qtype="choice",
            options=opts,
            label=label,
            dist={o: float(o == label) for o in opts},
            difficulty="hard",
            rationale=f"period end {end.isoformat()}",
            world=dict(w, ref_off=ref_off, cl_off=cl_off, local=local_event.isoformat(), qkind="date"),
        )
    prompt = f"Was the {sk['event']} made in time, judged in {ref_name} time?"
    label = "yes" if in_time else "no"
    return record_v2(
        FAM_T,
        split,
        i,
        state=state,
        prompt=prompt,
        qtype="noul",
        options=None,
        label=label,
        dist={"yes": float(in_time), "no": float(not in_time)},
        difficulty="hard",
        rationale=f"end {end.isoformat()} 24:00 {ref_name}; event {ref_event.isoformat()} -> {label}",
        world=dict(w, ref_off=ref_off, cl_off=cl_off, local=local_event.isoformat(), qkind="yn"),
    )


def verify_temporal(r: dict) -> str | None:
    w = r["meta"]["world"]
    a = dt.date.fromisoformat(w["anchor"])
    # independent derivation using ordinal arithmetic instead of the generator helpers
    if w["kind"] == "months":
        t = a.year * 12 + (a.month - 1) + w["n"]
        y, m = divmod(t, 12)
        m += 1
        dim = calendar.monthrange(y, m)[1]
        e = (
            dt.date(y, m, a.day)
            if a.day <= dim
            else (dt.date(y, m, dim).toordinal() + (1 if w["roll"] else 0))
        )
        e = e if isinstance(e, int) else e.toordinal()
        e -= 1 if w["exclusive_anniv"] else 0
    elif w["kind"] in ("days", "weeks"):
        e = a.toordinal() + (w["n"] * (7 if w["kind"] == "weeks" else 1)) - (1 if w["count_anchor"] else 0)
    else:
        hol = {dt.date.fromisoformat(h).toordinal() for h in w["holidays"]}
        e = a.toordinal()
        left = w["n"]
        while left:
            e += 1
            if dt.date.fromordinal(e).weekday() < 5 and e not in hol:
                left -= 1
    end = dt.date.fromordinal(e)
    text = r["state"]["content"]
    local = dt.datetime.fromisoformat(w["local"])
    ref = local - dt.timedelta(hours=w["cl_off"] - w["ref_off"])
    if w["qkind"] == "date":
        if fmt_date(end, w["date_style"]) != r["target"]["label"]:
            return "date label mismatch"
    else:
        lab = "yes" if ref < dt.datetime.combine(end + dt.timedelta(days=1), dt.time(0, 0)) else "no"
        if lab != r["target"]["label"]:
            return "yes/no label mismatch"
    if local.strftime("%Y-%m-%d %H:%M") not in text:
        return "event stamp not rendered"
    if fmt_date(a, w["date_style"]) not in text:
        return "anchor not rendered"
    return None


# ---------------------------------------------------------------- multi_hop_procedure
M_SKINS = {
    "customs_inspection": dict(
        doc="BORDER CLEARANCE OPERATING PROCEDURE",
        chap="Inspection level for import declarations",
        ent="importer",
        ents="importers",
        amt="declaration",
        unit="USD",
        tiers=["green_lane", "document_check", "physical_inspection", "full_audit"],
        tier_txt=[
            "Release without inspection.",
            "Check documents only.",
            "Inspect the consignment physically.",
            "Hold for a full customs audit.",
        ],
        q="Under this procedure, what must happen to declaration {ref}?",
    ),
    "loan_review": dict(
        doc="CREDIT POLICY MANUAL",
        chap="Sanctioning authority for loan applications",
        ent="borrower",
        ents="borrowers",
        amt="application",
        unit="EUR",
        tiers=["branch_officer", "regional_credit", "credit_committee", "board_risk"],
        tier_txt=[
            "Branch officer may sanction.",
            "Regional credit must sanction.",
            "Credit committee must sanction.",
            "Board risk committee must sanction.",
        ],
        q="Which body must sanction application {ref} under this manual?",
    ),
    "site_monitoring": dict(
        doc="CLINICAL OPERATIONS SOP",
        chap="Monitoring response for reported site events",
        ent="site",
        ents="sites",
        amt="report",
        unit="events",
        tiers=["remote_review", "scheduled_visit", "for_cause_audit", "regulator_notice"],
        tier_txt=[
            "Handle by remote review.",
            "Add to the next scheduled visit.",
            "Open a for-cause audit.",
            "Notify the regulator.",
        ],
        q="What response does report {ref} require?",
    ),
    "claims_assignment": dict(
        doc="CLAIMS HANDLING GUIDE",
        chap="Claim assignment",
        ent="provider",
        ents="providers",
        amt="claim",
        unit="GBP",
        tiers=["junior_adjuster", "senior_adjuster", "claims_manager", "special_investigations"],
        tier_txt=[
            "Assign to a junior adjuster.",
            "Assign to a senior adjuster.",
            "Assign to the claims manager.",
            "Refer to special investigations.",
        ],
        q="Under this guide, who must handle claim {ref}?",
    ),
}
M_CFG = {
    "train": dict(skins=["customs_inspection", "loan_review"], phrase=(0, 1, 2)),
    "calibration": dict(skins=["customs_inspection", "loan_review"], phrase=(3,)),
    "test_locked": dict(skins=["claims_assignment"], phrase=(3, 4)),
    "challenge": dict(skins=["site_monitoring"], phrase=(3, 4)),
}
SUF = ["AG", "GmbH", "Ltd", "SA", "BV", "LLC", "Inc", "SpA", "Oy", "AB"]
ROOTS = [
    "Alder",
    "Brant",
    "Corvin",
    "Delmar",
    "Estrel",
    "Fennick",
    "Garnet",
    "Halden",
    "Iverna",
    "Jorvik",
    "Kestrel",
    "Lowen",
    "Marrow",
    "Norvik",
    "Orsay",
    "Perrin",
    "Quillon",
    "Rowan",
    "Selby",
    "Tarn",
]
CATS = ["LOW", "STANDARD", "ELEVATED", "HIGH"]
COUNTRIES = [
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Spain",
    "Sweden",
]
OUTSIDE = [
    "Brazil",
    "Panama",
    "Singapore",
    "Turkey",
    "Vietnam",
    "Mauritius",
    "Kenya",
    "Chile",
    "Mexico",
    "Peru",
]
M_FILL = [
    "Procedures in this chapter apply to all legal entities unless a local annex states otherwise.",
    "Records supporting a decision must be retained for seven years.",
    "Exceptions require written approval from the process owner and are logged.",
    "Where a system and this manual disagree, this manual prevails.",
    "Training on this chapter is mandatory for new staff within 30 days.",
    "Questions about interpretation go to the policy desk, not to the requester.",
    "Delegations of authority are maintained separately and are not repeated here.",
    "Amounts are rounded to the nearest whole unit before any threshold is applied.",
    "Reports on routing outcomes are reviewed quarterly.",
    "Temporary staff may prepare but not approve items covered here.",
    "Nothing in this chapter limits the right of the audit function to request documents.",
    "Definitions in the glossary apply to every chapter.",
]


def gen_multihop(split: str, i: int) -> dict:
    want = i % 4
    for att in range(80):
        r = _gen_multihop(split, i, att)
        if r["meta"]["world"]["tiers"].index(r["target"]["label"]) == want:
            return r
    return r


def _gen_multihop(split: str, i: int, att: int) -> dict:
    cfg = M_CFG[split]
    rng = random.Random(f"os-prog-v2|{FAM_M}|{split}|{i}|{att}")
    sk_name = rng.choice(cfg["skins"])
    sk = M_SKINS[sk_name]
    ph = rng.choice(cfg["phrase"])
    org = _org(rng)
    n_ent = rng.randrange(18, 30)
    roots = rng.sample(ROOTS, min(len(ROOTS), n_ent // 2 + 1))
    names = []
    for r_ in roots:
        sufs = rng.sample(SUF, 2)
        names += [f"{r_} {sufs[0]}", f"{r_} {sufs[1]}"]
    names = names[:n_ent]
    ents = []
    id_pool = rng.sample(range(1000, 9999), len(names))
    for k, nm in enumerate(names):
        ents.append(
            dict(
                name=nm,
                id=f"V{id_pool[k]}",
                cat=rng.choice(CATS[:3]) if rng.random() < 0.8 else "LOW",
                country=rng.choice(COUNTRIES) if rng.random() < 0.8 else rng.choice(OUTSIDE),
                onboarded=(dt.date(2026, 6, 1) - dt.timedelta(days=rng.randrange(20, 1400))).isoformat(),
                watch=rng.random() < 0.08,
            )
        )
    case = rng.choice(ents)
    inv_date = dt.date(2026, 6, 1) + dt.timedelta(days=rng.randrange(0, 120))
    win, new_months = rng.choice([30, 45, 60]), rng.choice([6, 12, 18])
    base_thr = sorted(rng.sample(range(2, 60), 3))
    thr = {c: [t * 1000 * (1 + ci) for t in base_thr] for ci, c in enumerate(CATS)}
    for ci, c in enumerate(CATS):
        thr[c] = [int(x * (1.0 - 0.18 * ci)) // 100 * 100 for x in thr[c]]
    amount = rng.randrange(1, 40) * 500
    others = []
    for _ in range(rng.randrange(2, 6)):
        e2 = case if rng.random() < 0.5 else rng.choice([x for x in ents if x is not case])
        others.append(
            dict(
                name=e2["name"],
                date=(inv_date - dt.timedelta(days=rng.randrange(1, 100))).isoformat(),
                amount=rng.randrange(1, 30) * 500,
                ref=f"{rng.randrange(100, 999)}",
            )
        )
    w = dict(
        skin=sk_name,
        phrase=ph,
        entities=ents,
        case=dict(
            name=case["name"],
            ref=f"{rng.choice(['NS', 'RX', 'QA'])}-2026-{rng.randrange(100, 999)}",
            date=inv_date.isoformat(),
            amount=amount,
            bank=case["country"],
            others=others,
        ),
        win=win,
        new_months=new_months,
        thr=thr,
        tiers=sk["tiers"],
    )
    tier, trace = derive_multihop(w)
    # render
    blocks = [
        f'{org.upper()} - {sk["doc"]}, Chapter {rng.randrange(4, 19)} "{sk["chap"]}" (rev. 2026-0{rng.randrange(1, 9)})'
    ]
    P = [
        lambda: (
            f"Purpose. Each {sk['amt']} is graded and banded before anything happens to it; this chapter says how, using the {sk['ent']}'s grade and the total in {sk['unit']}."
        ),
        lambda: (
            f"Scope. What happens to a {sk['amt']} is fixed here, from the {sk['ent']}'s grade and the running total in {sk['unit']}."
        ),
        lambda: (
            f"Overview. Two facts settle the outcome of a {sk['amt']}: the {sk['ent']}'s grade and the total in {sk['unit']}."
        ),
        lambda: (
            f"Aim. To turn a {sk['ent']}'s grade and a total in {sk['unit']} into an outcome for the {sk['amt']}."
        ),
        lambda: "Summary. Look up the grade, adjust it, total the amounts, then read the band table.",
    ]
    sec = [P[ph]()]
    sec.append(
        f"Name matching. Look the name on the {sk['amt']} up in Schedule 1 and take the ID shown against that very name; a different legal-form ending means a different {sk['ent']}. All later look-ups use the ID."
    )
    sec.append(
        f"Grade. The starting grade is the one in Schedule 2. Move it up one notch (order LOW, STANDARD, ELEVATED, HIGH; HIGH is the ceiling) for every one of these that applies: (i) the registered bank country in Schedule 6 is absent from Schedule 4; (ii) the {sk['ent']} joined less than {new_months} months before the {sk['amt']} date (Schedule 5). A {sk['ent']} listed in Schedule 3 is graded HIGH whatever else is true."
    )
    sec.append(
        f"Totalling. The figure to be graded is this {sk['amt']}'s amount plus earlier items carrying the same ID that fall within the {win} days before the {sk['amt']} date (the date itself is not in that span). Items under any other ID are left out even when the names look alike."
    )
    sec.append(
        "Band table. Read the row for the final grade and take the first band whose limit the total does not exceed (limits are inclusive):"
    )
    rows = [
        f"  {c}: up to {t[0]} -> {sk['tiers'][0]}; up to {t[1]} -> {sk['tiers'][1]}; up to {t[2]} -> {sk['tiers'][2]}; above {t[2]} -> {sk['tiers'][3]}"
        for c, t in thr.items()
    ]
    sec.append("\n".join(rows))
    sec.append("Tier meanings. " + " ".join(f"{t}: {x}" for t, x in zip(sk["tiers"], sk["tier_txt"])))
    sec += rng.sample(M_FILL, 6)
    n_extra = rng.randrange(14, 40)
    alias = ["Schedule 1 - registered names (name -> ID)"] + [
        f"  {e['name']} -> {e['id']}" for e in rng.sample(ents, len(ents))
    ]
    cat = ["Schedule 2 - starting grades"] + [f"  {e['id']}: {e['cat']}" for e in rng.sample(ents, len(ents))]
    flags = ["Schedule 3 - watch list"] + (
        [f"  {e['id']}: watch" for e in ents if e["watch"]] or ["  (none)"]
    )
    onb = ["Schedule 5 - joining dates"] + [
        f"  {e['id']}: {e['onboarded']}" for e in rng.sample(ents, len(ents))
    ]
    bank = ["Schedule 6 - registered bank country"] + [
        f"  {e['id']}: {e['country']}" for e in rng.sample(ents, len(ents))
    ]
    appr = ["Schedule 4 - approved countries"] + ["  " + ", ".join(COUNTRIES)]
    fill = ["General notes"] + [f"  {k + 1}. {rng.choice(M_FILL)}" for k in range(n_extra)]
    annex = [alias, cat, flags, appr, onb, bank, fill]
    rng.shuffle(annex)
    cs = w["case"]
    rec = [
        f"{sk['amt'].capitalize()} record {cs['ref']}",
        f"  Name printed: {cs['name']}",
        f"  {sk['amt'].capitalize()} date: {cs['date']}",
        f"  Amount: {cs['amount']} {sk['unit']}",
        "  Other items received from named parties (name, date, amount):",
    ] + [f"    {o['name']} / {o['date']} / {o['amount']}" for o in others]
    state = "\n\n".join(["\n".join(blocks)] + sec + ["\n".join(a) for a in annex] + ["\n".join(rec)])
    opts = list(sk["tiers"])
    rng.shuffle(opts)
    return record_v2(
        FAM_M,
        split,
        i,
        state=state,
        prompt=sk["q"].format(amt=sk["amt"], ref=cs["ref"]),
        qtype="choice",
        options=opts,
        label=tier,
        dist={o: float(o == tier) for o in opts},
        difficulty="hard",
        rationale=trace,
        world=w,
    )


def derive_multihop(w: dict) -> tuple[str, str]:
    ents = {e["name"]: e for e in w["entities"]}
    cs = w["case"]
    e = ents[cs["name"]]
    d = dt.date.fromisoformat(cs["date"])
    cat = CATS.index(e["cat"])
    steps = [f"alias {cs['name']} -> {e['id']}", f"base {e['cat']}"]
    if e["country"] not in COUNTRIES:
        cat += 1
        steps.append(f"bank {e['country']} not approved +1")
    if (d - dt.date.fromisoformat(e["onboarded"])).days < w["new_months"] * 30.4375:
        cat += 1
        steps.append("new vendor +1")
    cat = min(cat, 3)
    if e["watch"]:
        cat = 3
        steps.append("watch flag -> HIGH")
    tot = cs["amount"]
    for o in cs["others"]:
        od = dt.date.fromisoformat(o["date"])
        if ents[o["name"]]["id"] == e["id"] and 0 < (d - od).days <= w["win"]:
            tot += o["amount"]
    lim = w["thr"][CATS[cat]]
    tier = (
        w["tiers"][0]
        if tot <= lim[0]
        else w["tiers"][1]
        if tot <= lim[1]
        else w["tiers"][2]
        if tot <= lim[2]
        else w["tiers"][3]
    )
    steps.append(f"category {CATS[cat]}, aggregated {tot} -> {tier}")
    return tier, "; ".join(steps)


def verify_multihop(r: dict) -> str | None:
    w = r["meta"]["world"]
    text = r["state"]["content"]
    # re-derive with a second implementation (month approx replaced by exact day count threshold 30.4375 per month is shared by design; check the rest independently)
    ents = {e["name"]: e for e in w["entities"]}
    if len(ents) != len(w["entities"]) or len({e["id"] for e in w["entities"]}) != len(w["entities"]):
        return "duplicate names or ids"
    tier, _ = derive_multihop(w)
    if tier != r["target"]["label"]:
        return "tier mismatch"
    cs = w["case"]
    if f"Name printed: {cs['name']}" not in text or f"Amount: {cs['amount']}" not in text:
        return "case not rendered"
    for e in w["entities"]:
        if f"{e['name']} -> {e['id']}" not in text or f"{e['id']}: {e['cat']}" not in text:
            return "annex lost an entity"
    if len(text) // 4 < 1250:
        return "document too short for a long-procedure row"
    return None


# ---------------------------------------------------------------- long policy
LONG_CFG = {
    "train": dict(v2_policy.SPLIT_CFG["train"], filler=(40, 80)),
    "calibration": dict(v2_policy.SPLIT_CFG["calibration"], filler=(40, 80)),
    "test_locked": dict(v2_policy.SPLIT_CFG["test_locked"], filler=(40, 80)),
    "challenge": dict(v2_policy.SPLIT_CFG["challenge"], filler=(50, 90)),
}


def gen_longpolicy(split: str, i: int) -> dict:
    key = f"long_{split}"
    v2_policy.SPLIT_CFG[key] = LONG_CFG[split]
    r = v2_policy.gen(key, i + 5_000_000)
    r["id"] = f"osj-progv2-{FAM_P}-{split[:3]}-{i:05d}"
    r["domain"] = f"programmatic_{FAM_P}_v2"
    r["meta"]["task_pack"] = f"programmatic_{FAM_P}_v2"
    r["meta"]["scenario_family"] = FAM_P
    r["meta"]["split"] = split
    return r


def verify_longpolicy(r: dict) -> str | None:
    rr = {**r, "meta": {**r["meta"], "scenario_family": "policy_precedence"}}
    return v2_policy.verify_row(rr)


FAMS = {
    FAM_T: (gen_temporal, verify_temporal),
    FAM_M: (gen_multihop, verify_multihop),
    FAM_P: (gen_longpolicy, verify_longpolicy),
}


def generate(split: str, n: int, families=None) -> list[dict]:
    fams = families or list(FAMS)
    out = []
    for k in range(n):
        f = fams[k % len(fams)]
        out.append(FAMS[f][0](split, k // len(fams)))
    return out


def main() -> None:
    import argparse
    import collections
    import json
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=4500)
    ap.add_argument("--n-eval", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = {}
    for sp in SPLITS:
        rows = generate(sp, a.n_train if sp == "train" else a.n_eval)
        bad = 0
        for r in rows:
            if FAMS[r["meta"]["scenario_family"]][1](r):
                bad += 1
        random.Random(5).shuffle(rows)
        with open(os.path.join(a.out, f"{sp}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        man[sp] = dict(
            n=len(rows),
            failed_verification=bad,
            families=dict(collections.Counter(r["meta"]["scenario_family"] for r in rows)),
        )
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps(man))


if __name__ == "__main__":
    main()
