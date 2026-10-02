"""Programmatic families, v2: effective-dating with real structural and surface diversity.

Why v2 exists (measured, scripts/analysis/data_audit.py, 2026-09-30): prog_v1's families are ONE world
structure each with 4-5 phrasing options and small entity pools; training draws only variants 0-1, so ~95% of a
row's 5-word phrases recur across its family, and the "held-out" eval splits still share 81-92% of their phrasing
with train. temporal_numeric (0.13 on JevBench hard) also let the model shortcut: the rule always said the DATE OF
LOSS governs, so the answer never depended on reading the rule.

What changes here
  * 7 domain skins (vocabulary + value kind + the three dates' names), not one insurance schedule.
  * 3 governing-date rules (event date / filing-or-processing date / period-start date); the rule is stated in the
    document in one of 5 phrasings, so the answer genuinely depends on reading it.
  * 6 layouts (lines, markdown table, prose, date ranges, JSON, email) and 3 date styles.
  * Generated organisation names (28x28x18) instead of a 12-name pool.
  * Splits hold out more than phrasing: eval rule phrasings (3,4) are never in train (0,1,2); calibration also holds
    out the JSON layout and a date style; test_locked holds out a whole SKIN (per_diem); challenge holds out a whole
    skin (warranty_terms) AND a layout (email). Those two are real transfer tests.
Gold is computed from a structured world stored in meta.world and re-derived independently by verify(); verify()
also checks that the rendered text actually contains every revision's date and value (rendering fidelity).

  python -m os_datagen.programmatic.v2 --selfcheck
  python -m os_datagen.programmatic.v2 --out ../data/synthetic/prog_v2 --n-train 6000 --n-eval 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import date, timedelta

FAMILY = "effective_dating"
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

TRAIN_SKINS = ["property_sublimit", "rent_escalation", "customs_duty", "sla_credit", "price_list"]
SPLIT_CFG = {
    "train": dict(
        skins=TRAIN_SKINS,
        formats=["lines", "md_table", "prose", "ranges"],
        date_styles=(0, 1),
        phrase=(0, 1, 2),
    ),
    "calibration": dict(skins=TRAIN_SKINS, formats=["json"], date_styles=(2,), phrase=(3,)),
    "test_locked": dict(
        skins=["per_diem"],
        formats=["lines", "md_table", "prose", "ranges"],
        date_styles=(0, 1),
        phrase=(3, 4),
    ),
    "challenge": dict(skins=["warranty_terms"], formats=["email"], date_styles=(2,), phrase=(3, 4)),
}

SKINS = {
    "property_sublimit": dict(
        vk="usd",
        noun="property sublimit",
        subject="claim",
        vcol="Sublimit (USD)",
        titles=["SCHEDULE OF SUBLIMITS", "PROPERTY SUBLIMIT SCHEDULE", "SUBLIMIT REVISION LOG"],
        ev="date of loss",
        tr="date the claim was filed",
        th="policy inception date",
        ev_row="Date of loss (incident occurred)",
        tr_row="Date claim was filed with us",
        th_row="Policy incepted",
        depts=["Claims", "Intake", "Field Service", "Underwriting", "Compliance", "Dispatch"],
    ),
    "rent_escalation": dict(
        vk="usd",
        noun="monthly rent",
        subject="payment",
        vcol="Monthly rent (USD)",
        titles=["RENT SCHEDULE", "LEASE RENT ESCALATION TABLE", "SCHEDULE OF RENTAL RATES"],
        ev="payment due date",
        tr="date the invoice was issued",
        th="lease commencement date",
        ev_row="Payment due date",
        tr_row="Invoice issued on",
        th_row="Lease commenced",
        depts=["Leasing", "Property Accounts", "Tenant Services", "Facilities", "Collections"],
    ),
    "customs_duty": dict(
        vk="pct",
        noun="duty rate",
        subject="shipment",
        vcol="Duty rate",
        titles=["TARIFF SCHEDULE", "DUTY RATE NOTICE", "SCHEDULE OF CUSTOMS RATES"],
        ev="date of export",
        tr="date of customs entry",
        th="date the contract was signed",
        ev_row="Goods exported on",
        tr_row="Customs entry lodged on",
        th_row="Contract signed on",
        depts=["Trade Compliance", "Brokerage", "Import Desk", "Logistics", "Audit"],
    ),
    "sla_credit": dict(
        vk="pct",
        noun="service credit",
        subject="outage",
        vcol="Credit (% of monthly fee)",
        titles=["SERVICE CREDIT SCHEDULE", "SLA CREDIT TIERS", "UPTIME CREDIT TABLE"],
        ev="date of the outage",
        tr="date the ticket was opened",
        th="service start date",
        ev_row="Outage began",
        tr_row="Support ticket opened",
        th_row="Service start",
        depts=["Support", "Network Operations", "Account Management", "Billing", "Reliability"],
    ),
    "price_list": dict(
        vk="unit",
        noun="unit price",
        subject="order",
        vcol="Unit price (USD)",
        titles=["PRICE LIST", "UNIT PRICE SCHEDULE", "PRICING REVISIONS"],
        ev="order date",
        tr="delivery date",
        th="quote date",
        ev_row="Order placed",
        tr_row="Delivered",
        th_row="Quote issued",
        depts=["Sales Operations", "Procurement", "Order Desk", "Finance", "Fulfilment"],
    ),
    "per_diem": dict(
        vk="unit",
        noun="daily allowance",
        subject="trip",
        vcol="Allowance (USD/day)",
        titles=["PER DIEM SCHEDULE", "TRAVEL ALLOWANCE RATES", "DAILY ALLOWANCE REVISIONS"],
        ev="trip start date",
        tr="date the expense report was submitted",
        th="policy renewal date",
        ev_row="Trip began",
        tr_row="Expense report submitted",
        th_row="Policy renewed",
        depts=["Travel Desk", "Expenses", "People Operations", "Finance", "Audit"],
    ),
    "warranty_terms": dict(
        vk="months",
        noun="warranty period",
        subject="purchase",
        vcol="Coverage (months)",
        titles=["WARRANTY TERMS", "COVERAGE PERIOD SCHEDULE", "WARRANTY REVISION HISTORY"],
        ev="purchase date",
        tr="date the product was registered",
        th="manufacture date",
        ev_row="Purchased on",
        tr_row="Registered on",
        th_row="Manufactured on",
        depts=["Customer Care", "Warranty Claims", "Service Centre", "Returns", "Quality"],
    ),
}

_SYL = [
    "bar",
    "cor",
    "del",
    "fen",
    "gal",
    "hol",
    "ist",
    "jun",
    "kel",
    "lor",
    "mer",
    "nor",
    "ost",
    "pal",
    "qui",
    "ras",
    "sel",
    "tor",
    "ur",
    "van",
    "wes",
    "yor",
    "zan",
    "bri",
    "cin",
    "dra",
    "eld",
    "fro",
]
_TAIL = [
    "Logistics",
    "Mutual",
    "Instruments",
    "Freight Co",
    "Health",
    "Utilities",
    "Dairy",
    "Works",
    "Optics",
    "Rail",
    "Foundry",
    "Textiles",
    "Marine",
    "Analytics",
    "Holdings",
    "Supply",
    "Assurance",
    "Labs",
]
_SUR = [
    "Okonjo",
    "Valtonen",
    "Prakash",
    "Ferreiro",
    "Nakamura",
    "Abubakar",
    "Lindqvist",
    "Mwangi",
    "Delacroix",
    "Sigurdsson",
    "Haddad",
    "Kowalczyk",
    "Ibarra",
    "Thorne",
    "Yildiz",
    "Marchetti",
    "Osei",
    "Brandvold",
    "Castellan",
    "Rahimi",
]
_SITE = ["Depot", "Annex", "Bay", "Pier", "Unit", "Lot", "Shed", "Dock", "Wing", "Yard"]
_GENERIC = [
    "This schedule supersedes all schedules previously issued to {org}.",
    "Figures exclude applicable taxes unless stated otherwise.",
    "Queries should be directed to {who} in {dept}.",
    "The schedule is reviewed annually and may be revised on written notice.",
    "This document is confidential to {org}.",
    "Rounding is to the nearest unit shown.",
    "Prepared for {org} at {site}.",
    "No verbal amendment to this schedule is effective.",
    "A copy is held on file at {site}.",
    "Reference number: {ref}.",
]

PROMPTS = [
    "Which {noun} applies to this {subject}?",
    "Determine the {noun} that governs this {subject}.",
    "What {noun} should be applied to this {subject}?",
    "Read the document and select the {noun} in force for this {subject}.",
]
RULES = [
    "The {noun} is determined by the schedule in force on the {X}.",
    "Rule: apply the {noun} that was effective on the {X}; changes made after that date are ignored.",
    "For every {subject}, the {noun} is read off the revision in effect on the {X}, whatever the other dates are.",
    "Governing date: the {X}. Look up the {noun} that applied on that date.",
    "Which revision applies is decided solely by the {X}; other dates on file do not matter.",
]


def _org(rng):
    return rng.choice(_SYL).capitalize() + rng.choice(_SYL) + " " + rng.choice(_TAIL)


def _person(rng):
    return f"{rng.choice('ABCDEFGHJKLMNPRST')}. {rng.choice(_SUR)}"


def _site(rng):
    return f"{rng.choice(_SITE)} {rng.randrange(1, 60)}"


def _fmt_date(d: date, style: int) -> str:
    return (
        d.isoformat(),
        f"{d.day} {MONTHS[d.month - 1]} {d.year}",
        f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
    )[style]


def _fmt_val(vk: str, v) -> str:
    return {
        "usd": lambda: f"${v:,}",
        "pct": lambda: f"{v:.2f}%",
        "unit": lambda: f"${v:.2f}",
        "months": lambda: f"{v} months",
    }[vk]()


def _values(rng, vk, n):
    if vk == "usd":
        return rng.sample([x * 250 for x in range(20, 401)], n)
    if vk == "pct":
        return rng.sample([round(x * 0.25, 2) for x in range(4, 101)], n)
    if vk == "unit":
        return [c / 100 for c in rng.sample(range(500, 9999), n)]
    return rng.sample([6, 9, 12, 15, 18, 24, 30, 36, 48, 60], n)


def _window(revs, m):
    lo = revs[m][0]
    hi = revs[m + 1][0] - timedelta(days=1) if m + 1 < len(revs) else lo + timedelta(days=400)
    return lo, hi


def _day_in(rng, revs, m):
    lo, hi = _window(revs, m)
    span = (hi - lo).days
    return lo + timedelta(days=rng.randrange(1, span - 1) if span > 3 else 0)


def _render_schedule(fmt, sk, revs, ds, vk, rng, order):
    idx = list(range(len(revs)))
    if fmt in ("lines", "md_table", "json") and order == "desc":
        idx.reverse()
    if fmt in ("lines", "md_table", "json") and order == "shuffle":
        rng.shuffle(idx)

    def D(d):
        return _fmt_date(d, ds)

    def V(v):
        return _fmt_val(vk, v)

    if fmt == "lines":
        return [f"  Rev {k + 1}  effective {D(revs[k][0])}  {sk['noun']} {V(revs[k][1])}" for k in idx]
    if fmt == "md_table":
        return [f"| Revision | Effective from | {sk['vcol']} |", "|---|---|---|"] + [
            f"| {k + 1} | {D(revs[k][0])} | {V(revs[k][1])} |" for k in idx
        ]
    if fmt == "json":
        body = ", ".join(f'{{"effective": "{D(revs[k][0])}", "value": "{V(revs[k][1])}"}}' for k in idx)
        return [f'{{"schedule": [{body}]}}']
    if fmt == "ranges":
        out = []
        ks = list(range(len(revs))) if order != "desc" else list(reversed(range(len(revs))))
        for k in ks:
            end = f"{D(revs[k + 1][0] - timedelta(days=1))}" if k + 1 < len(revs) else "onward"
            out.append(
                f"- {D(revs[k][0])} to {end}: {V(revs[k][1])}"
                if end != "onward"
                else f"- {D(revs[k][0])} onward: {V(revs[k][1])}"
            )
        return out
    if fmt == "prose":
        parts = [f"From {D(revs[0][0])}, the {sk['noun']} was {V(revs[0][1])}."]
        for k in range(1, len(revs)):
            parts.append(
                rng.choice(
                    [
                        f"On {D(revs[k][0])} it changed to {V(revs[k][1])}.",
                        f"With effect from {D(revs[k][0])} it became {V(revs[k][1])}.",
                        f"From {D(revs[k][0])} onward it stood at {V(revs[k][1])}.",
                    ]
                )
            )
        return [" ".join(parts)]
    # email
    parts = [
        f"Quick recap of how the {sk['noun']} has moved: it started at {V(revs[0][1])} ({D(revs[0][0])})"
    ]
    for k in range(1, len(revs)):
        parts.append(f"then {V(revs[k][1])} from {D(revs[k][0])}")
    return [", ".join(parts) + "."]


def gen(split: str, i: int) -> dict:
    cfg = SPLIT_CFG[split]
    rng = random.Random(f"os-prog-v2|{split}|{i}")
    skin, fmt, ds = rng.choice(cfg["skins"]), rng.choice(cfg["formats"]), rng.choice(cfg["date_styles"])
    sk = SKINS[skin]
    vk = sk["vk"]
    org, who, site = _org(rng), _person(rng), _site(rng)
    dept = rng.choice(sk["depts"])
    n_rev = rng.randrange(3, 7)
    vals = _values(rng, vk, n_rev)
    d = date(2023, 1, 1) + timedelta(days=rng.randrange(0, 500))
    revs = []
    for v in vals:
        revs.append((d, v))
        d += timedelta(days=rng.randrange(30, 420))
    for _ in range(20):  # windows for (period-start, event, filing): i <= j <= k
        j = rng.randrange(n_rev)
        wi = rng.randrange(0, j + 1)
        wk = rng.randrange(j, n_rev)
        if not (wi == j == wk) or rng.random() < 0.1:
            break
    ev = _day_in(rng, revs, j)
    th = _day_in(rng, revs, wi)
    tr = _day_in(rng, revs, wk)
    if wi == j:
        th = min(th, ev)
    if wk == j:
        tr = max(tr, ev)
    dates = {"th": th, "ev": ev, "tr": tr}
    kind = rng.choice(["ev", "tr", "th"])
    gi = {"th": wi, "ev": j, "tr": wk}[kind]
    gold_v = revs[gi][1]
    gold = _fmt_val(vk, gold_v)
    pool = {gold_v, revs[wi][1], revs[j][1], revs[wk][1]}
    others = [r[1] for r in revs if r[1] not in pool]
    rng.shuffle(others)
    while len(pool) < 3 and others:
        pool.add(others.pop())
    if others and rng.random() < 0.5:
        pool.add(others.pop())
    options = [_fmt_val(vk, v) for v in pool]
    rng.shuffle(options)

    order = rng.choice(["asc", "asc", "desc", "shuffle"])
    title = f"{org} — {rng.choice(sk['titles'])} ({site})"
    rule = RULES[rng.choice(cfg["phrase"])].format(noun=sk["noun"], subject=sk["subject"], X=sk[kind])
    facts = [
        f"  {sk['ev_row']}: {_fmt_date(ev, ds)}",
        f"  {sk['tr_row']}: {_fmt_date(tr, ds)}",
        f"  {sk['th_row']}: {_fmt_date(th, ds)}",
    ]
    rng.shuffle(facts)
    gen_notes = [
        n.format(
            org=org, who=who, dept=dept, site=site, ref=f"{rng.choice('ABCDEFGH')}{rng.randrange(1000, 9999)}"
        )
        for n in rng.sample(_GENERIC, rng.randrange(1, 4))
    ]
    sched = _render_schedule(fmt, sk, revs, ds, vk, rng, order)
    blocks = [[title], sched, [f"Prepared by {who}, {dept}."] + facts, gen_notes]
    rule_pos = rng.choice(["top", "mid", "bottom"])
    blocks.insert({"top": 1, "mid": 2, "bottom": 4}[rule_pos], [rule])
    state = "\n\n".join("\n".join(b) for b in blocks)
    prompt = rng.choice(PROMPTS).format(noun=sk["noun"], subject=sk["subject"])
    distinct = len({revs[wi][1], revs[j][1], revs[wk][1]})
    world = {
        "skin": skin,
        "format": fmt,
        "date_style": ds,
        "rule": kind,
        "order": order,
        "n_rev": n_rev,
        "revs": [[r[0].isoformat(), r[1]] for r in revs],
        "dates": {k: v.isoformat() for k, v in dates.items()},
    }
    return {
        "id": f"osj-progv2-{FAMILY}-{split[:3]}-{i:05d}",
        "domain": f"programmatic_{FAMILY}_v2",
        "source": "os-datagen:programmatic",
        "state": {"content": state},
        "question": {"type": "choice", "prompt": prompt, "options": options},
        "target": {"label": gold, "dist": {o: (1.0 if o == gold else 0.0) for o in options}},
        "meta": {
            "split": split,
            "task_pack": f"programmatic_{FAMILY}_v2",
            "namespace": "programmatic",
            "scenario_family": FAMILY,
            "difficulty": "hard" if distinct > 1 else "medium",
            "label_quality": "deterministic",
            "label_source": "code_oracle",
            "acceptable": [gold],
            "soft_target": False,
            "world": world,
            "rationale": f"rule={kind}: governing date {dates[kind].isoformat()} falls in revision {gi + 1}",
        },
    }


def verify_row(r: dict) -> str | None:
    """Return None if OK, else a failure reason. Re-derives gold from the structured world by an independent route."""
    w, text = r["meta"]["world"], r["state"]["content"]
    sk = SKINS[w["skin"]]
    vk = sk["vk"]
    revs = [(date.fromisoformat(a), b) for a, b in w["revs"]]
    gov = date.fromisoformat(w["dates"][w["rule"]])
    cur = None
    for d, v in sorted(revs):  # step through revisions in time order (generator used window indices)
        if d <= gov:
            cur = v
    if cur is None or _fmt_val(vk, cur) != r["target"]["label"]:
        return "gold mismatch"
    if r["target"]["label"] not in r["question"]["options"] or len(set(r["question"]["options"])) != len(
        r["question"]["options"]
    ):
        return "bad options"
    ds = w["date_style"]
    for d, v in revs:
        if _fmt_date(d, ds) not in text or _fmt_val(vk, v) not in text:
            return f"render lost a revision fact ({_fmt_date(d, ds)} / {_fmt_val(vk, v)})"
    for k in ("th", "ev", "tr"):
        if _fmt_date(date.fromisoformat(w["dates"][k]), ds) not in text:
            return f"render lost the {k} date"
    if sk[w["rule"]] not in text:
        return "render lost the rule's governing-date concept"
    return None


def generate(split: str, n: int) -> list[dict]:
    return [gen(split, i) for i in range(n)]


def verify(rows: list[dict]) -> tuple[int, list[str]]:
    bad = []
    for r in rows:
        try:
            why = verify_row(r)
        except Exception as e:
            why = f"{type(e).__name__}: {e}"
        if why:
            bad.append(f"{r['id']}: {why}")
    return len(rows) - len(bad), bad


def record_v2(
    family: str,
    split: str,
    i: int,
    *,
    state: str,
    prompt: str,
    qtype: str,
    options: list[str] | None,
    label: str,
    dist: dict[str, float],
    difficulty: str,
    rationale: str,
    world: dict,
    extra: dict | None = None,
) -> dict:
    """Shared record builder for every v2 family (same shape as prog_v1's _record, v2 ids/domains, structured world kept in meta)."""
    q: dict = {"type": qtype, "prompt": prompt}
    labels = options if qtype == "choice" else ["yes", "no"]
    if qtype == "choice":
        q["options"] = options
    tot = sum(dist.get(ln, 0.0) for ln in labels) or 1.0
    norm = {ln: round(dist.get(ln, 0.0) / tot, 6) for ln in labels}
    return {
        "id": f"osj-progv2-{family}-{split[:3]}-{i:05d}",
        "domain": f"programmatic_{family}_v2",
        "source": "os-datagen:programmatic",
        "state": {"content": state},
        "question": q,
        "target": {"label": label, "dist": norm},
        "meta": {
            "split": split,
            "task_pack": f"programmatic_{family}_v2",
            "namespace": "programmatic",
            "scenario_family": family,
            "difficulty": difficulty,
            "label_quality": "deterministic",
            "label_source": "code_oracle",
            "acceptable": [label],
            "soft_target": max(norm.values()) < 0.999,
            "world": world,
            "rationale": rationale,
            **(extra or {}),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        rows = [r for s in SPLITS for r in generate(s, 400)]
        ok, bad = verify(rows)
        print(f"verified {ok}/{len(rows)} rows")
        if bad:
            print("FAILURES:", bad[:10])
            raise SystemExit(1)
        return
    if not a.out:
        raise SystemExit("--out required unless --selfcheck")
    os.makedirs(a.out, exist_ok=True)
    for s in SPLITS:
        rows = generate(s, a.n_train if s == "train" else a.n_eval)
        ok, bad = verify(rows)
        if bad:
            raise SystemExit(f"{s}: verification failed for {len(bad)} rows: {bad[:5]}")
        with open(os.path.join(a.out, f"{s}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{a.out}/{s}.jsonl: {len(rows)} rows, {ok} verified")


if __name__ == "__main__":
    main()
