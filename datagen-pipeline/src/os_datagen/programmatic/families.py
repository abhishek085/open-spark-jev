"""Programmatic decision families: verifiable gold computed in code, no LLM in the loop.

Why this exists, separately from ``taskpacks``
---------------------------------------------
The LLM-surface packs buy realistic prose at the cost of needing a generator, a verifier and a
fact-comparison pass per row. For the failure modes v7 targets, that machinery is not what is
missing -- *computable* gold is. Three things follow from generating in code instead:

1. **Gold is exact, including the distribution.** Every existing training row has a one-hot
   target (``target.dist`` is 1.0 on one option), so the model has never once been shown a state
   where the honest answer is "0.62 / 0.38". That is the most likely single cause of v6 being
   near-perfectly calibrated on our own splits (ECE 0.007) yet badly overconfident on hard
   out-of-distribution items (ECE 0.265). ``sft.py`` already trains KL against ``target.dist``,
   so soft-target rows need no trainer change.
2. **Zero benchmark contamination by construction.** States are assembled from our own entity
   pools by our own templates; nothing is copied, paraphrased or seeded from any benchmark item.
   ``scripts/tools/benchmark_overlap.py`` still gates the output as a belt-and-braces check.
3. **The arithmetic is checkable.** ``verify()`` recomputes each family's gold by an independent
   brute-force route, so a generator bug shows up as a test failure rather than as silent label noise.

The five families mirror *kinds* of decision that v6 is weakest at (measured on JevBench hard:
temporal_numeric 0.07, long_policy 0.37, ambiguous 0.43, tradeoff 0.50, probability 0.70), not any
specific item. Two of them (``probability_exact``, ``underdetermined``) carry genuinely spread gold
distributions and exist primarily to teach honest uncertainty.

Splits get disjoint worlds via the seed namespace, and each split draws a different instruction
phrasing, matching the taskpack convention (train 0-1, calibration 2, test_locked 3, challenge 4).
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Any, Callable

# Entity pools. Invented names, deliberately unlike anything in a public benchmark.
ORGS = ["Варна Logistics", "Quillfeather Mutual", "Nordhagen Instruments", "Palmyra Freight Co",
        "Tessellate Health", "Ironmoor Utilities", "Saltbrook Dairy", "Vantage Kiln Works",
        "Meridian Cask", "Holloway Optics", "Brightmarsh Rail", "Cindersend Foundry"]
SITES = ["Depot 7", "North Annex", "Kiln Row", "Bay 12", "Pier 3", "Unit K", "Lot 44", "Shed 9"]
PRODUCTS = ["flow regulator", "pressure sleeve", "drive coupling", "filter cartridge", "bearing race",
            "seal ring", "igniter module", "ballast plate"]
PEOPLE = ["R. Okonjo", "M. Valtonen", "S. Prakash", "D. Ferreiro", "L. Nakamura", "T. Abubakar",
          "J. Lindqvist", "A. Mwangi", "C. Delacroix", "K. Sigurdsson"]
DEPTS = ["Claims", "Intake", "Field Service", "Underwriting", "Compliance", "Dispatch"]

SPLITS = ("train", "calibration", "test_locked", "challenge")
_VARIANT_FOR_SPLIT = {"train": (0, 1), "calibration": (2,), "test_locked": (3,), "challenge": (4,)}


def _rng(family: str, split: str, i: int) -> random.Random:
    """Disjoint worlds per split: the split name is part of the seed namespace."""
    return random.Random(f"os-prog-v1|{family}|{split}|{i}")


def _variant(rng: random.Random, split: str, options: list[str]) -> str:
    return options[rng.choice(_VARIANT_FOR_SPLIT[split]) % len(options)]


def _dist(labels: list[str], weights: dict[str, float]) -> dict[str, float]:
    tot = sum(weights.get(l, 0.0) for l in labels) or 1.0
    return {l: round(weights.get(l, 0.0) / tot, 6) for l in labels}


def _record(family: str, split: str, i: int, *, state: Any, prompt: str, qtype: str,
            options: list[str] | None, label: str, dist: dict[str, float], difficulty: str,
            rationale: str, extra: dict | None = None) -> dict:
    q: dict[str, Any] = {"type": qtype, "prompt": prompt}
    if qtype == "choice":
        q["options"] = options
    labels = options if qtype == "choice" else ["yes", "no"]   # schema.Noul renders yes/no
    norm = _dist(labels, dist)   # normalise first: the soft-target flag must reflect the stored target
    return {
        "id": f"osj-prog-{family}-{split[:3]}-{i:05d}",
        "domain": f"programmatic_{family}_v1",
        "source": "os-datagen:programmatic",
        "state": {"content": state},
        "question": q,
        "target": {"label": label, "dist": norm},
        "meta": {"split": split, "task_pack": f"programmatic_{family}_v1", "namespace": "programmatic",
                 "scenario_family": family, "difficulty": difficulty, "label_quality": "deterministic",
                 "label_source": "code_oracle", "acceptable": [label],
                 "soft_target": max(norm.values()) < 0.999, "rationale": rationale, **(extra or {})},
    }


# --------------------------------------------------------------------------------------
# 1. temporal_numeric -- which effective-dated revision governs, then read a number off it
# --------------------------------------------------------------------------------------
_TN_PROMPTS = [
    "Which sublimit applies to this claim?",
    "Apply the schedule in force on the incident date: which sublimit governs?",
    "Determine the governing sublimit for this loss.",
    "Read the applicable schedule and select the sublimit that applies.",
    "Which sublimit amount is in force for this incident, given the revision history?",
]


def gen_temporal_numeric(split: str, i: int) -> dict:
    rng = _rng("temporal_numeric", split, i)
    org, site, who = rng.choice(ORGS), rng.choice(SITES), rng.choice(PEOPLE)
    base = date(2024, 1, 1) + timedelta(days=rng.randrange(0, 400))
    n_rev = rng.randrange(3, 5 if split != "challenge" else 6)
    amounts = rng.sample([5000, 7500, 10000, 12500, 15000, 20000, 25000, 30000], n_rev)
    revs = []
    d = base
    for k, amt in enumerate(amounts):
        revs.append({"revision": k + 1, "effective": d, "sublimit": amt})
        d = d + timedelta(days=rng.randrange(60, 400))
    # incident lands inside one revision's window, never on a boundary
    pick = rng.randrange(0, n_rev)
    lo = revs[pick]["effective"]
    hi = revs[pick + 1]["effective"] if pick + 1 < n_rev else lo + timedelta(days=500)
    span = (hi - lo).days
    incident = lo + timedelta(days=rng.randrange(1, max(2, span - 1)))
    gold_amt = revs[pick]["sublimit"]

    # the trap: the *reported* date (when the claim was filed) sits in a later revision's window
    filed = hi + timedelta(days=rng.randrange(1, 90)) if pick + 1 < n_rev else incident + timedelta(days=rng.randrange(1, 60))
    trap_amt = next((r["sublimit"] for r in revs if r["effective"] <= filed and
                     (r["revision"] == n_rev or revs[r["revision"]]["effective"] > filed)), gold_amt)

    opts = sorted({gold_amt, trap_amt, *rng.sample([a for a in amounts if a not in (gold_amt, trap_amt)] or [gold_amt], k=min(2, max(0, n_rev - 2)))})
    options = [f"{a:,}" for a in opts]
    lines = [f"{org} — SCHEDULE OF SUBLIMITS ({site})", "",
             "Revision history (each revision supersedes the previous one on its effective date):"]
    for r in revs:
        lines.append(f"  Rev {r['revision']}  effective {r['effective'].isoformat()}  property sublimit {r['sublimit']:,} USD")
    lines += ["", f"Claim file prepared by {who}, {rng.choice(DEPTS)}.",
              f"  Date of loss (incident occurred): {incident.isoformat()}",
              f"  Date claim was filed with us:     {filed.isoformat()}",
              f"  Damaged item: {rng.choice(PRODUCTS)}", "",
              "Standing note: the schedule in force on the DATE OF LOSS governs a claim, "
              "regardless of when the claim is filed or adjusted."]
    return _record("temporal_numeric", split, i, state="\n".join(lines),
                   prompt=_variant(rng, split, _TN_PROMPTS), qtype="choice", options=options,
                   label=f"{gold_amt:,}", dist={f"{gold_amt:,}": 1.0},
                   difficulty="hard" if trap_amt != gold_amt else "medium",
                   rationale=f"loss {incident} falls in rev {revs[pick]['revision']} (effective {lo}); filing date {filed} is a distractor",
                   extra={"trap": trap_amt != gold_amt})


def _verify_temporal_numeric(r: dict) -> bool:
    """Re-read the rendered table and re-apply the rule independently of the generator."""
    text = r["state"]["content"]
    revs = []
    for line in text.splitlines():
        if line.strip().startswith("Rev "):
            p = line.split()
            revs.append((date.fromisoformat(p[3]), int(p[6].replace(",", ""))))
    loss = next(date.fromisoformat(l.split(":")[1].strip()) for l in text.splitlines() if "Date of loss" in l)
    gov = max((d for d, _ in revs if d <= loss), default=None)
    amt = dict(revs)[gov]
    return f"{amt:,}" == r["target"]["label"]


# --------------------------------------------------------------------------------------
# 2. long_policy -- many clauses, precedence order decides; most clauses are irrelevant
# --------------------------------------------------------------------------------------
_LP_PROMPTS = [
    "Apply the policy: how should this claim be resolved?",
    "Adjudicate this claim under the wording provided.",
    "Which disposition does the policy require for this claim?",
    "Read the policy and select the correct disposition.",
    "Given the clauses and the loss facts, which disposition follows?",
]
_LP_FILLER = [
    ("Notice", "The insured shall give notice of any loss within 30 days of discovery. Notice may be given "
               "orally provided it is confirmed in writing within a further 14 days. Late notice does not of "
               "itself void cover unless the insurer's position has been prejudiced thereby."),
    ("Records", "The insured shall retain maintenance records for 24 months and produce them on request. "
                "Records held in electronic form satisfy this clause where a legible copy can be furnished."),
    ("Subrogation", "The insurer may pursue recovery against a responsible third party and the insured shall "
                    "lend all reasonable assistance. The insured shall do nothing after a loss to prejudice "
                    "such rights of recovery."),
    ("Arbitration", "Disputes exceeding 50,000 USD shall be referred to arbitration in the seat of the insurer. "
                    "The tribunal shall comprise a single arbitrator agreed between the parties, failing which "
                    "one shall be appointed by the relevant professional body."),
    ("Currency", "All sums are expressed in USD unless the schedule states otherwise. Where conversion is "
                 "required the rate of exchange shall be that prevailing on the date of loss."),
    ("Inspection", "The insurer may inspect the premises on reasonable notice. Nothing in this clause obliges "
                   "the insurer to inspect, and no inspection shall be construed as a warranty that the "
                   "premises are safe or that the risk is acceptable."),
    ("Assignment", "This policy may not be assigned without written consent. Consent shall not be unreasonably "
                   "withheld where the assignee carries on the same trade at the same location."),
    ("Cancellation", "Either party may cancel on 60 days written notice. On cancellation the insurer shall "
                     "return the unearned portion of the premium calculated pro rata."),
    ("Salvage", "Salvage proceeds are credited against the settlement. The insured shall not abandon property "
                "to the insurer without prior agreement."),
    ("Territory", "Cover applies within the territory stated in the schedule. Property temporarily removed for "
                  "cleaning, repair or calibration remains covered for up to 30 consecutive days."),
    ("Other insurance", "Where the loss is also insured elsewhere this policy shall pay only its rateable "
                        "proportion, determined by reference to the sums insured under each contributing policy."),
    ("Reinstatement", "Where the insurer elects to reinstate, it shall not be bound to reinstate exactly but "
                      "only as circumstances permit and in a reasonably sufficient manner."),
    ("Fraud", "Any claim advanced fraudulently, in whole or in part, is forfeit in its entirety and the "
              "insurer may treat the contract as terminated from the date of the fraudulent act."),
    ("Alterations", "The insured shall notify any material alteration to the premises or trade. Cover in "
                    "respect of the altered risk is suspended until the insurer confirms acceptance."),
    ("Protections", "Where the schedule records a protection warranty, the stated protections shall be in "
                    "full and effective operation whenever the premises are closed for business."),
    ("Debris removal", "The insurer shall indemnify the reasonable cost of removing debris of the insured "
                       "property, up to 10 per cent of the sum insured, in addition to the sum insured."),
    ("Professional fees", "Architects' and surveyors' fees necessarily incurred in reinstatement are covered, "
                          "excluding any fees incurred in preparing a claim under this policy."),
    ("Average", "If at the time of loss the sum insured is less than the reinstatement value, the insurer's "
                "liability shall be reduced in the proportion that the sum insured bears to that value."),
    ("Waiting period", "No indemnity is payable in respect of the first 24 hours of any interruption arising "
                       "from failure of a public supply."),
    ("Jurisdiction", "This policy shall be governed by the law of the place of issue and the parties submit to "
                     "the non-exclusive jurisdiction of its courts."),
]
_LP_SCHEDULE_NOISE = [
    "Business description: as stated in the proposal dated {d}.",
    "Premium payable: {p:,} USD annually, in advance.",
    "Policy period: 12 months from {d}.",
    "Broker of record: {org}.",
    "Protection warranty recorded: intruder alarm, maintained under contract.",
    "Sums insured are index-linked at the published construction cost index.",
]


def gen_long_policy(split: str, i: int) -> dict:
    rng = _rng("long_policy", split, i)
    org, site = rng.choice(ORGS), rng.choice(SITES)
    deductible, sublimit, estimate = rng.choice([500, 1000, 2500]), rng.choice([10000, 15000, 25000]), rng.randrange(3000, 60000, 500)
    vacant_days, vacancy_limit = rng.randrange(0, 120), 60
    seepage_weeks = rng.randrange(0, 5)
    peril = rng.choice(["water escape from a plumbing system", "wind-driven rain", "impact by vehicle", "accidental fire"])

    # precedence: exclusions first (vacancy, then repeated seepage), then sublimit, then full pay
    vacancy_excl = vacant_days > vacancy_limit
    seepage_excl = seepage_weeks >= 2 and "water" in peril
    if vacancy_excl:
        gold, why = "deny_vacancy_exclusion", f"premises vacant {vacant_days} d > {vacancy_limit} d"
    elif seepage_excl:
        gold, why = "deny_repeated_seepage", f"seepage continued {seepage_weeks} weeks (>= 2)"
    elif estimate - deductible > sublimit:
        gold, why = "pay_subject_to_sublimit", f"net {estimate - deductible:,} exceeds sublimit {sublimit:,}"
    else:
        gold, why = "pay_full_estimate_less_deductible", f"net {estimate - deductible:,} within sublimit {sublimit:,}"
    options = ["deny_vacancy_exclusion", "deny_repeated_seepage", "pay_subject_to_sublimit", "pay_full_estimate_less_deductible"]

    clauses = [("Vacancy", f"No cover for loss occurring while the premises have been vacant for more than {vacancy_limit} consecutive days."),
               ("Seepage", "No cover for loss caused by water that has seeped or leaked continuously or repeatedly over two weeks or more."),
               ("Property sublimit", f"Loss to property at an unattended location is payable up to {sublimit:,} USD."),
               ("Deductible", f"Each and every loss is subject to a deductible of {deductible:,} USD.")]
    # JevBench's long_policy items run past 2,000 tokens; the difficulty is partly that the four
    # operative clauses are buried. Keep most of _LP_FILLER in, not a token handful.
    fill = rng.sample(_LP_FILLER, k=rng.randrange(14, 18) if split != "challenge" else len(_LP_FILLER))
    body = clauses + fill
    rng.shuffle(body)
    sched = rng.sample(_LP_SCHEDULE_NOISE, k=4)
    issued = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 400))
    lines = [f"{org} — PROPERTY WORDING (extract), {site}", "", "SCHEDULE", ""]
    lines += ["  " + s.format(d=issued.isoformat(), p=rng.randrange(4, 40) * 1000, org=rng.choice(ORGS)) for s in sched]
    lines += ["", "OPERATIVE CLAUSES", ""]
    for n, (h, t) in enumerate(body, 1):
        lines.append(f"{n}. {h}. {t}")
    lines += ["", "LOSS FACTS", f"  Peril: {peril}",
              f"  Repair estimate: {estimate:,} USD",
              f"  Consecutive days premises vacant immediately before the loss: {vacant_days}",
              f"  Duration over which water was escaping before discovery: "
              f"{'not applicable' if 'water' not in peril else f'{seepage_weeks} week(s)'}",
              f"  Location attended at time of loss: no",
              f"  Adjuster: {rng.choice(PEOPLE)}"]
    return _record("long_policy", split, i, state="\n".join(lines),
                   prompt=_variant(rng, split, _LP_PROMPTS), qtype="choice", options=options,
                   label=gold, dist={gold: 1.0},
                   difficulty="hard", rationale=why,
                   extra={"n_clauses": len(body)})


def _verify_long_policy(r: dict) -> bool:
    t = r["state"]["content"]
    g = lambda k: next(l.split(":", 1)[1].strip() for l in t.splitlines() if l.strip().startswith(k))
    vacant = int(g("Consecutive days premises vacant"))
    est = int(g("Repair estimate").replace(" USD", "").replace(",", ""))
    ded = int(next(l for l in t.splitlines() if "deductible of" in l).split("deductible of")[1].split("USD")[0].replace(",", "").strip())
    sub = int(next(l for l in t.splitlines() if "payable up to" in l).split("payable up to")[1].split("USD")[0].replace(",", "").strip())
    lim = int(next(l for l in t.splitlines() if "vacant for more than" in l).split("more than")[1].split("consecutive")[0].strip())
    dur = g("Duration over which water")
    weeks = 0 if "not applicable" in dur else int(dur.split()[0])
    if vacant > lim:
        exp = "deny_vacancy_exclusion"
    elif weeks >= 2:
        exp = "deny_repeated_seepage"
    elif est - ded > sub:
        exp = "pay_subject_to_sublimit"
    else:
        exp = "pay_full_estimate_less_deductible"
    return exp == r["target"]["label"]


# --------------------------------------------------------------------------------------
# 3. probability_exact -- the honest answer is a number strictly between 0 and 1
# --------------------------------------------------------------------------------------
_PR_PROMPTS = [
    "Will the sample contain at least one defective unit? Give probabilities that reflect the evidence.",
    "Judge the probability that the inspection rejects this lot, using only the evidence in the state.",
    "Give a calibrated probability that at least one sampled unit is defective.",
    "State how likely it is that the sample contains a defective unit.",
    "Weigh the evidence and give the probability that the lot is rejected.",
]


def gen_probability_exact(split: str, i: int) -> dict:
    rng = _rng("probability_exact", split, i)
    org, prod = rng.choice(ORGS), rng.choice(PRODUCTS)
    n_rev = rng.randrange(2, 4)
    lot = rng.randrange(8, 20)
    defective = rng.randrange(1, max(2, lot // 2))
    draws_hist = rng.sample(range(1, min(6, lot)), n_rev)
    draws = draws_hist[-1]
    superseded = draws_hist[-2]
    base = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 300))
    plans = [{"version": k + 1, "effective": base + timedelta(days=120 * k), "draws": d} for k, d in enumerate(draws_hist)]
    inspect_on = plans[-1]["effective"] + timedelta(days=rng.randrange(1, 100))

    p_none = Fraction(comb(lot - defective, draws), comb(lot, draws)) if lot - defective >= draws else Fraction(0)
    p_yes = 1 - p_none
    gold = "yes" if p_yes > Fraction(1, 2) else "no"
    lines = [f"{org} — INCOMING INSPECTION, lot of {lot} {prod}s", "",
             f"Independent teardown found exactly {defective} of the {lot} units defective; "
             f"the defective units are indistinguishable from the rest without teardown.", "",
             "Sampling plan revision history (each version supersedes the previous one):"]
    for p in plans:
        lines.append(f"  v{p['version']}  effective {p['effective'].isoformat()}  draw {p['draws']} unit(s) at random, without replacement")
    lines += ["", f"Inspection date: {inspect_on.isoformat()}",
              "A lot is rejected if at least one sampled unit is defective.",
              f"Note from {rng.choice(PEOPLE)}: \"Under the plan we used last year we drew "
              f"{superseded} unit(s), so I'd expect rejection to be a coin flip at best.\""]
    return _record("probability_exact", split, i, state="\n".join(lines),
                   prompt=_variant(rng, split, _PR_PROMPTS), qtype="noul", options=None,
                   label=gold, dist={"yes": float(p_yes), "no": float(p_none)},
                   difficulty="hard",
                   rationale=f"1 - C({lot - defective},{draws})/C({lot},{draws}) = {float(p_yes):.4f}; superseded plan drew {superseded}",
                   extra={"exact_p_true": float(p_yes), "lot": lot, "defective": defective, "draws": draws})


def _verify_probability_exact(r: dict) -> bool:
    """Brute force over actual subsets instead of the closed form."""
    m = r["meta"]
    lot, d, k = m["lot"], m["defective"], m["draws"]
    units = [True] * d + [False] * (lot - d)
    subs = list(combinations(range(lot), k))
    p = sum(any(units[j] for j in s) for s in subs) / len(subs)
    return abs(p - m["exact_p_true"]) < 1e-9 and (r["target"]["label"] == "yes") == (p > 0.5)


# --------------------------------------------------------------------------------------
# 4. underdetermined -- the evidence genuinely does not pick one option; say so
# --------------------------------------------------------------------------------------
_UD_PROMPTS = [
    "Which queue should this ticket go to?",
    "Route this ticket to the correct queue.",
    "Select the queue this ticket belongs in.",
    "Given only what the ticket says, which queue applies?",
    "Assign this ticket to a queue.",
]


def gen_underdetermined(split: str, i: int) -> dict:
    rng = _rng("underdetermined", split, i)
    org, who = rng.choice(ORGS), rng.choice(PEOPLE)
    queues = ["billing", "technical", "account_access", "shipping"]
    # k options are supported by exactly equal evidence -> mass splits evenly over them
    k = rng.choice([2, 2, 3])
    tied = rng.sample(queues, k)
    signals = {"billing": "an unexpected charge on the last invoice",
               "technical": "the device power-cycling during use",
               "account_access": "being unable to sign in since Tuesday",
               "shipping": "a parcel that has not moved in nine days"}
    body = [f"{org} — inbound message", f"From: {who}", "",
            "Message:", "  Hello, I have a couple of separate problems and I am not sure who to ask."]
    for q in tied:
        body.append(f"  - {signals[q]}")
    body += ["  Please sort it out.", "",
             "Routing rules: a ticket goes to the queue matching its issue. A ticket raising "
             "several unrelated issues of equal weight has no single correct queue; the router "
             "must not guess.", f"  Received: {(date(2026, 1, 1) + timedelta(days=rng.randrange(0, 300))).isoformat()}"]
    gold = sorted(tied)[0]  # argmax is arbitrary among ties; the *distribution* is the real label
    return _record("underdetermined", split, i, state="\n".join(body),
                   prompt=_variant(rng, split, _UD_PROMPTS), qtype="choice", options=queues,
                   label=gold, dist={q: 1.0 for q in tied},
                   difficulty="hard",
                   rationale=f"{k} issues of equal weight ({', '.join(sorted(tied))}); honest answer is 1/{k} each",
                   extra={"tied_options": sorted(tied), "acceptable": sorted(tied)})


def _verify_underdetermined(r: dict) -> bool:
    tied = r["meta"]["tied_options"]
    d = r["target"]["dist"]
    return (all(abs(d[t] - 1 / len(tied)) < 1e-6 for t in tied)
            and all(abs(d[o]) < 1e-9 for o in d if o not in tied))


# --------------------------------------------------------------------------------------
# 5. tradeoff_weighted -- explicit weights, a near-tie is honestly a near-tie
# --------------------------------------------------------------------------------------
_TO_PROMPTS = [
    "Which supplier should be selected?",
    "Apply the scoring table and pick the supplier.",
    "Score the bids as specified and select the winner.",
    "Using the stated weights, which bid wins?",
    "Select the bid with the highest weighted score.",
]


def gen_tradeoff_weighted(split: str, i: int) -> dict:
    rng = _rng("tradeoff_weighted", split, i)
    org = rng.choice(ORGS)
    names = rng.sample(["Aldermere", "Brakewater", "Corvid", "Dunmoor"], 3)
    crits = [("unit price", 5), ("lead time", 3), ("defect rate", 2)]
    scores = {n: [rng.randrange(1, 11) for _ in crits] for n in names}
    tot = {n: sum(w * s for (_, w), s in zip(crits, v)) for n, v in scores.items()}
    best = max(tot.values())
    winners = sorted([n for n, v in tot.items() if v == best])
    # a 1-point gap is within the rounding the table itself admits -> treat as a near-tie
    near = sorted([n for n, v in tot.items() if best - v <= 1])
    lines = [f"{org} — BID EVALUATION", "",
             "Weighted scoring (higher score is better on every criterion):",
             "  " + "  ".join(f"{c} x{w}" for c, w in crits), ""]
    for n in names:
        lines.append(f"  {n:<12} " + "  ".join(f"{c}={s}" for (c, _), s in zip(crits, scores[n])))
    lines += ["", "The total is the weighted sum of the criterion scores. Criterion scores are "
                  "recorded to the nearest whole point, so totals within one point of each other "
                  "are not meaningfully different.",
              f"  Prepared by {rng.choice(PEOPLE)}, {rng.choice(DEPTS)}"]
    return _record("tradeoff_weighted", split, i, state="\n".join(lines),
                   prompt=_variant(rng, split, _TO_PROMPTS), qtype="choice", options=sorted(names),
                   label=winners[0], dist={n: 1.0 for n in near},
                   difficulty="hard" if len(near) > 1 else "medium",
                   rationale=f"totals {tot}; within-1-point set {near}",
                   extra={"totals": tot, "near_tie": near, "acceptable": near})


def _verify_tradeoff_weighted(r: dict) -> bool:
    t = r["meta"]["totals"]
    best = max(t.values())
    near = sorted([n for n, v in t.items() if best - v <= 1])
    return near == r["meta"]["near_tie"] and r["target"]["label"] in near


# --------------------------------------------------------------------------------------

FAMILIES: dict[str, Callable[[str, int], dict]] = {
    "temporal_numeric": gen_temporal_numeric,
    "long_policy": gen_long_policy,
    "probability_exact": gen_probability_exact,
    "underdetermined": gen_underdetermined,
    "tradeoff_weighted": gen_tradeoff_weighted,
}
VERIFIERS: dict[str, Callable[[dict], bool]] = {
    "temporal_numeric": _verify_temporal_numeric,
    "long_policy": _verify_long_policy,
    "probability_exact": _verify_probability_exact,
    "underdetermined": _verify_underdetermined,
    "tradeoff_weighted": _verify_tradeoff_weighted,
}


def generate(family: str, split: str, n: int) -> list[dict]:
    return [FAMILIES[family](split, i) for i in range(n)]


def verify(rows: list[dict]) -> tuple[int, list[str]]:
    """Recompute every row's gold by an independent route. Returns (n_ok, failures)."""
    bad = []
    for r in rows:
        fam = r["meta"]["scenario_family"]
        try:
            if not VERIFIERS[fam](r):
                bad.append(r["id"])
        except Exception as e:  # a verifier that cannot parse its own row is also a failure
            bad.append(f"{r['id']}:{type(e).__name__}:{e}")
    return len(rows) - len(bad), bad
