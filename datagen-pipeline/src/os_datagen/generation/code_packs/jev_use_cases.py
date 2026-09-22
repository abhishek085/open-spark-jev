"""Code-only packs for Jev's structured-state use cases: entity resolution, fraud risk, financial triage, real-time control, lead scoring,
data-row validity and semantic grep. Each row's label is computed by a rule engine from declared features; text is rendered from the same
features. Vocabulary pools (names, tickers, hosts...) and paraphrases are partitioned across splits so held-out splits see unseen surface forms."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any, Callable

ORACLE = "jev_use_cases_rules_v1"
SPLIT_POOLS = {"train": list(range(0, 7)), "calibration": [7], "locked_test": [8], "challenge": [9]}
SPLIT_FRAC = {"train": 0.70, "calibration": 0.10, "locked_test": 0.10, "challenge": 0.10}
INSTR_IDX = {"train": (0, 1), "calibration": (2,), "locked_test": (3,), "challenge": (4,)}

FIRST = ("Robert Elena Marcus Priya Jonas Hana Tomas Amara Lucas Sofia Daniel Mira Victor Nadia Oscar Leila Felix Ingrid Mateo Yuki Carlos Freya Andre Zainab Peter Lina "
         "Simon Aiko Hugo Rania Ethan Clara Ivan Noor Louis Petra Rafael Selma Anton Dina Kevin Marta Idris Greta Pablo Sana Jules Vera Omar Alma Theo Rosa Nikhil Elise "
         "Bruno Tara Samir Olga Damian Jana Kwame Livia Emil Asha Gareth Malin").split()
LAST = ("Martinez Kowalski Okafor Lindqvist Tanaka Haddad Bergmann Novak Sullivan Ferreira Chen Petrov Abadi Moreau Jensen Ito Silva Rahman Costa Weber "
        "Nakamura Fischer Duarte Hansen Ortiz Kaplan Mensah Varga Rossi Larsen Qureshi Bianchi Popescu Alvarez Dubois Nilsson Adeyemi Horvat Santos Klein "
        "Mizrahi Reyes Sokolov Barros Lund Osei Ricci Stein Vidal Yilmaz Berg Cardoso Ahmed Ivanova Lopez Meyer Gomez Sato Ekström Baptiste Zhou Marek").split()
NICK = {"Robert": "Bob", "Elena": "Lena", "Marcus": "Marc", "Daniel": "Dan", "Sofia": "Sophie", "Victor": "Vic", "Peter": "Pete", "Simon": "Si", "Ethan": "Ethan",
        "Kevin": "Kev", "Andre": "Andy", "Louis": "Lou", "Samir": "Sam", "Damian": "Dami", "Theo": "Theo", "Felix": "Fee", "Oscar": "Ozzy", "Carlos": "Carlo"}
COMPANY_A = "Apex Bright Cobalt Delta Ember Forge Granite Harbor Ionic Juniper Keystone Lumen Meridian Nimbus Orchid Pioneer Quartz Ridge Summit Talon Umber Vertex Willow Zenith Aurora Beacon Cedar Dynamo Everest Fjord".split()
COMPANY_B = "Systems Labs Logistics Foods Analytics Capital Robotics Media Health Energy Works Networks Studio".split()
STREETS = "Oak Maple Cedar Pine Elm Birch Willow Harbor Lake Hill River Park Mill Sunset Ridge Bay Forest Garden Union Market Station".split()
CITIES = "Portland Denver Austin Leeds Lyon Porto Malmö Gdańsk Utrecht Bergen Turin Basel Ghent Aarhus Cork Split Graz Kraków Bilbao Tampere".split()
TICKERS = "ACME BRLX CNDR DYNA EMBR FRGE GRNT HRBR IONC JNPR KYST LUMN MRDN NMBS ORCH PION QRTZ RDGE SMMT TLON UMBR VRTX WLOW ZNTH AURA BCON CEDR DYMO EVRS FJRD".split()
HOSTS = "api-1 api-2 db-primary db-replica cache-01 gateway-a gateway-b worker-7 worker-9 auth-svc billing-svc search-svc queue-02 edge-eu edge-us".split()
USERS = "alice bob carol dave erin frank grace heidi ivan judy karl lena mona nils omar pia quinn rosa sven tess uma vic wes xena yuri zoe".split()


def part(seq: list[Any], pool: int, k: int = 10) -> list[Any]:
    out = seq[pool::k]
    return out or seq


def digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def fmt_phone(d: str, style: int) -> str:
    return [f"({d[:3]}) {d[3:6]}-{d[6:]}", f"{d[:3]}.{d[3:6]}.{d[6:]}", f"+1 {d[:3]} {d[3:6]} {d[6:]}", f"{d[:3]}-{d[3:6]}-{d[6:]}"][style % 4]


def pick_split_pool(split: str, rng: random.Random) -> int:
    return rng.choice(SPLIT_POOLS[split])


def instr(variants: list[str], split: str, rng: random.Random) -> str:
    return variants[rng.choice(INSTR_IDX[split]) % len(variants)]


# ---------------------------------------------------------------- entity resolution
ER_OPTS = [("match", "The two records describe the same real-world entity."), ("no_match", "The two records describe different entities."),
           ("needs_review", "The evidence is insufficient or conflicting; a person should review.")]
ER_POLICY = ("Adjudication policy: identical or clearly variant names plus a shared email or phone number is a match. An exact name plus the same address, with no conflicting "
             "email or phone, is a match. A different name with no shared email or phone is not a match, and so are records whose email and phone both differ under a non-identical name. "
             "Anything else (for example the same name with only a differing or missing identifier) needs review.")
ER_INSTR = ["Do these two records refer to the same entity?", "Adjudicate the pair: match, no match, or needs review?",
            "Under the policy, what is the right resolution for this record pair?", "Decide whether record A and record B are the same person.", "Apply the adjudication policy to the two records."]


def _er_label(fl: dict[str, str]) -> str:
    n, e, p, a = fl["name"], fl["email"], fl["phone"], fl["addr"]
    hard_same = e == "same" or p == "same"
    both_conflict = e == "different" and p == "different"
    if n in ("exact", "variant") and hard_same and not (e == "same" and p == "different") and not (p == "same" and e == "different"):
        return "match"
    if n == "exact" and a == "same" and not both_conflict and e != "different" and p != "different":
        return "match"
    if n == "different" and not hard_same:
        return "no_match"
    if both_conflict and n != "exact":
        return "no_match"
    return "needs_review"


def gen_entity_resolution(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    fn, ln = rng.choice(part(FIRST, pool)), rng.choice(part(LAST, pool))
    co = f"{rng.choice(part(COMPANY_A, pool, 10))} {rng.choice(COMPANY_B)}"
    dom = co.split()[0].lower() + ".com"
    city, street, num = rng.choice(CITIES), rng.choice(STREETS), rng.randint(2, 980)
    ph = digits(rng, 10)
    fl = {"name": rng.choice(["exact", "exact", "variant", "variant", "different", "different"]), "email": rng.choice(["same", "same", "different", "missing"]),
          "phone": rng.choice(["same", "same", "different", "missing"]), "addr": rng.choice(["same", "variant", "different", "missing"])}
    A = {"name": f"{fn} {ln}", "email": f"{fn.lower()}.{ln.lower()}@{dom}", "phone": fmt_phone(ph, rng.randint(0, 3)), "address": f"{num} {street} Street, {city}", "company": co}
    fn2, ln2 = rng.choice(part(FIRST, pool)), rng.choice(part(LAST, pool))
    if fl["name"] == "different":
        nm = f"{fn2} {ln2}" if (fn2, ln2) != (fn, ln) else f"{fn2} {ln2}son"
    elif fl["name"] == "exact":
        nm = A["name"]
    else:
        v = rng.choice(["nick", "initial", "typo", "reorder"])
        nm = {"nick": f"{NICK.get(fn, fn[:3])} {ln}", "initial": f"{fn[0]}. {ln}", "typo": f"{fn} {ln[:-2]}{ln[-1]}{ln[-2]}", "reorder": f"{ln}, {fn}"}[v]
        if nm == A["name"]:
            nm = f"{fn[0]}. {ln}"
    other = f"{fn2.lower()}.{ln2.lower()}@{co.split()[0].lower()}.com"
    B = {"name": nm, "company": co if rng.random() < 0.7 else f"{co.split()[0]} Group"}
    B["email"] = {"same": A["email"], "different": other if other != A["email"] else "info@" + dom, "missing": None}[fl["email"]]
    B["phone"] = {"same": fmt_phone(ph, rng.randint(0, 3)), "different": fmt_phone(digits(rng, 10), rng.randint(0, 3)), "missing": None}[fl["phone"]]
    B["address"] = {"same": A["address"], "variant": f"{num} {street} St, {city}", "different": f"{rng.randint(2, 980)} {rng.choice(STREETS)} Avenue, {rng.choice(CITIES)}", "missing": None}[fl["addr"]]
    B = {k: v for k, v in B.items() if v is not None}
    state = {"policy": ER_POLICY, "record_a": A, "record_b": B}
    return {"state": state, "type": "choice", "options": ER_OPTS, "label": _er_label(fl), "instr": ER_INSTR, "features": fl}


# ---------------------------------------------------------------- fraud risk
FR_OPTS = [("approve", "Low risk: let the transaction through."), ("review", "Elevated risk: hold for a person to review."), ("block", "High risk: decline the transaction.")]
FR_POLICY = ("Risk points: amount over 5x the 30-day average +3 (over 2x +1); new device +1; transaction country differs from home country +2; 5 or more transactions in the last hour +3 "
             "(3 or 4 +1); account younger than 7 days +2 (younger than 30 days +1); high-risk merchant type (gift cards, crypto, electronics) +2; shipping address differs from billing +1; "
             "card present -1. Total 7 or more: block. A gift-card purchase on a new device from a foreign country: block. Total 4 to 6: review. Otherwise approve.")
FR_INSTR = ["What should the payments system do with this transaction?", "Score the transaction under the policy and choose approve, review or block.",
            "Apply the risk policy to the transaction: which action?", "Decide the handling of this payment given the risk points policy.", "Route this transaction according to the fraud policy."]
MERCH = {"low": ["groceries", "utilities", "pharmacy", "public transport", "restaurant"], "high": ["gift cards", "crypto exchange", "electronics"]}


def _fraud_label(t: dict[str, Any]) -> tuple[str, int]:
    pts = 0
    r = t["amount"] / max(t["avg_amount_30d"], 1)
    pts += 3 if r > 5 else (1 if r > 2 else 0)
    pts += 1 if t["new_device"] else 0
    pts += 2 if t["country"] != t["home_country"] else 0
    pts += 3 if t["txns_last_hour"] >= 5 else (1 if t["txns_last_hour"] >= 3 else 0)
    pts += 2 if t["account_age_days"] < 7 else (1 if t["account_age_days"] < 30 else 0)
    pts += 2 if t["merchant_type"] in MERCH["high"] else 0
    pts += 1 if t["shipping_differs"] else 0
    pts -= 1 if t["card_present"] else 0
    if pts >= 7 or (t["merchant_type"] == "gift cards" and t["new_device"] and t["country"] != t["home_country"]):
        return "block", pts
    return ("review" if pts >= 4 else "approve"), pts


def gen_fraud(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    home = rng.choice(["US", "GB", "DE", "FR", "SE", "PT"])
    avg = round(rng.uniform(20, 400) * (1 + 0.1 * pool), 2)
    t = {"amount": round(avg * rng.choice([0.3, 0.8, 1.0, 1.5, 2.5, 3, 6, 8, 12]), 2), "avg_amount_30d": avg, "home_country": home,
         "country": home if rng.random() < 0.6 else rng.choice(["NG", "BR", "RU", "VN", "US", "GB", "DE"]), "new_device": rng.random() < 0.35,
         "txns_last_hour": rng.choice([1, 1, 1, 2, 3, 4, 5, 7]), "account_age_days": rng.choice([2, 5, 12, 25, 60, 200, 900, 2400]),
         "merchant_type": rng.choice(MERCH["low"] + MERCH["high"] + ["clothing", "hotel", "airline"]), "shipping_differs": rng.random() < 0.25, "card_present": rng.random() < 0.3}
    lab, pts = _fraud_label(t)
    return {"state": {"policy": FR_POLICY, "transaction": t}, "type": "choice", "options": FR_OPTS, "label": lab, "instr": FR_INSTR, "features": {"points": pts}}


# ---------------------------------------------------------------- financial triage
FT_OPTS = [("BUY", "Open or add to a position."), ("HOLD", "Keep the current position; do nothing new."), ("SKIP", "Do not trade this name now.")]
FT_POLICY = ("Screening rules for this exercise: SKIP if earnings are 2 or fewer days away, annualised volatility is over 60%, or news sentiment is below -0.5. Otherwise BUY if price is above its "
             "50-day average, RSI is between 50 and 70, the 5-day return is positive and sentiment is at least 0.2. Otherwise HOLD.")
FT_INSTR = ["Triage this market snapshot: BUY, HOLD or SKIP?", "Apply the screening rules to the snapshot.", "What is the triage decision for this ticker under the rules?",
            "Choose the action the screening rules produce for this snapshot.", "Run the snapshot through the rule set and give the resulting action."]


def gen_fin(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    ma = round(rng.uniform(10, 400), 2)
    s = {"ticker": rng.choice(part(TICKERS, pool)), "price": round(ma * rng.uniform(0.9, 1.12), 2), "ma50": ma, "rsi14": rng.randint(22, 82), "return_5d_pct": round(rng.uniform(-6, 7), 1),
         "sentiment": round(rng.uniform(-0.9, 0.9), 2), "days_to_earnings": rng.choice([1, 2, 3, 8, 20, 45, 70]), "volatility_pct": rng.choice([18, 25, 34, 48, 58, 62, 75])}
    if rng.random() < 0.4:  # near-BUY snapshots: each rule is close to its boundary so the decisive condition varies
        s.update(price=round(ma * rng.uniform(1.0, 1.1), 2), rsi14=rng.randint(46, 74), return_5d_pct=round(rng.uniform(-1.5, 5), 1), sentiment=round(rng.uniform(0.05, 0.8), 2),
                 days_to_earnings=rng.choice([3, 8, 20, 45]), volatility_pct=rng.choice([25, 34, 48, 58]))
    if s["days_to_earnings"] <= 2 or s["volatility_pct"] > 60 or s["sentiment"] < -0.5:
        lab = "SKIP"
    elif s["price"] > s["ma50"] and 50 <= s["rsi14"] <= 70 and s["return_5d_pct"] > 0 and s["sentiment"] >= 0.2:
        lab = "BUY"
    else:
        lab = "HOLD"
    return {"state": {"rules": FT_POLICY, "snapshot": s}, "type": "choice", "options": FT_OPTS, "label": lab, "instr": FT_INSTR, "features": {}}


# ---------------------------------------------------------------- real-time control
GC_OPTS = [("heal", "Use a medkit now."), ("take_cover", "Move to cover."), ("reload", "Reload the weapon."), ("retreat", "Fall back out of the fight."),
           ("attack", "Fire at the nearest enemy."), ("advance", "Move forward toward the objective.")]
GC_POLICY = ("Priority rules, first match wins: health under 25 with a medkit: heal; health under 25 without one: retreat; ammo 0: reload if cover is available, else retreat; "
             "3 or more enemies visible and cover available: take_cover; nearest enemy within weapon range: attack; otherwise advance.")
RB_OPTS = [("continue", "Keep going at the current speed."), ("slow", "Reduce speed."), ("stop", "Stop in place."), ("return_to_dock", "Head back to the charging dock.")]
RB_POLICY = ("Priority rules, first match wins: battery under 15%: return_to_dock; obstacle within 0.5 m: stop; obstacle within 2 m, or carrying a fragile payload with an obstacle within 4 m: slow; otherwise continue.")
CT_INSTR = ["Choose the next action for the controller.", "Apply the priority rules to the current state: which action?", "What should the agent do this tick under the rules?",
            "Select the action the rule list yields for this state.", "Decide the next control action from the state and the priority rules."]


def gen_control(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    if rng.random() < 0.55:
        g = {"health": rng.choice([8, 15, 22, 24, 25, 40, 70, 100]), "medkits": rng.choice([0, 0, 1, 2]), "ammo": rng.choice([0, 0, 3, 12, 30]), "cover_available": rng.random() < 0.5,
             "enemies_visible": rng.choice([0, 1, 1, 2, 3, 4]), "nearest_enemy_m": rng.choice([5, 12, 25, 40, 80]), "weapon_range_m": rng.choice([20, 30, 40])}
        if g["health"] < 25:
            lab = "heal" if g["medkits"] > 0 else "retreat"
        elif g["ammo"] == 0:
            lab = "reload" if g["cover_available"] else "retreat"
        elif g["enemies_visible"] >= 3 and g["cover_available"]:
            lab = "take_cover"
        elif g["enemies_visible"] >= 1 and g["nearest_enemy_m"] <= g["weapon_range_m"]:
            lab = "attack"
        else:
            lab = "advance"
        return {"state": {"rules": GC_POLICY, "game_state": g}, "type": "choice", "options": GC_OPTS, "label": lab, "instr": CT_INSTR, "features": {"env": "game"}}
    r = {"battery_pct": rng.choice([9, 14, 15, 16, 30, 55, 90]), "obstacle_m": rng.choice([0.3, 0.5, 0.9, 1.8, 2.0, 3.5, 6.0, 12.0]), "speed_mps": rng.choice([0.4, 0.8, 1.2]),
         "fragile_payload": rng.random() < 0.4}
    if r["battery_pct"] < 15:
        lab = "return_to_dock"
    elif r["obstacle_m"] <= 0.5:
        lab = "stop"
    elif r["obstacle_m"] <= 2 or (r["fragile_payload"] and r["obstacle_m"] <= 4):
        lab = "slow"
    else:
        lab = "continue"
    return {"state": {"rules": RB_POLICY, "robot_state": r}, "type": "choice", "options": RB_OPTS, "label": lab, "instr": CT_INSTR, "features": {"env": "robot"}}


# ---------------------------------------------------------------- lead scoring (Score 0-3)
LS_LEVELS = [(0, "Poor fit or no buying signal."), (1, "Weak lead."), (2, "Promising lead."), (3, "Hot lead: prioritise now.")]
LS_POLICY = ("Lead points: director/VP/C-level +2 (manager +1); company of 50-5000 employees +1; budget confirmed +2; buying timeline within 3 months +2 (within 6 months +1); "
             "demo requested +2; industry matches our focus +1. Score 0 for up to 2 points, 1 for 3-4, 2 for 5-6, 3 for 7 or more.")
LS_INSTR = ["How strong is this lead (0-3)?", "Score the prospect under the lead policy.", "Under the points policy, what score does this lead get?",
            "Rate the lead from 0 to 3 using the stated points.", "Assign the qualification level for this prospect."]


def gen_lead(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    p = {"title": rng.choice(["Intern", "Engineer", "Analyst", "Team Manager", "Director of Operations", "VP Engineering", "CTO", "Office Coordinator"]),
         "company": f"{rng.choice(part(COMPANY_A, pool))} {rng.choice(COMPANY_B)}", "employees": rng.choice([8, 30, 75, 240, 900, 4800, 12000]), "budget_confirmed": rng.random() < 0.35,
         "timeline_months": rng.choice([None, 1, 2, 4, 6, 9, 18]), "demo_requested": rng.random() < 0.35, "industry_match": rng.random() < 0.5}
    t = p["title"]
    pts = 2 if any(k in t for k in ("Director", "VP", "CTO")) else (1 if "Manager" in t else 0)
    pts += 1 if 50 <= p["employees"] <= 5000 else 0
    pts += 2 if p["budget_confirmed"] else 0
    tm = p["timeline_months"]
    pts += 2 if tm is not None and tm <= 3 else (1 if tm is not None and tm <= 6 else 0)
    pts += 2 if p["demo_requested"] else 0
    pts += 1 if p["industry_match"] else 0
    lab = 0 if pts <= 2 else 1 if pts <= 4 else 2 if pts <= 6 else 3
    return {"state": {"policy": LS_POLICY, "prospect": p}, "type": "score", "levels": LS_LEVELS, "label": lab, "instr": LS_INSTR, "features": {"points": pts}}


# ---------------------------------------------------------------- data row validity (Boolean)
DV_INSTR = ["Is this row valid under the schema?", "Does the record satisfy every constraint in the schema?", "Would this row pass validation against the schema?",
            "Check the row against the schema: valid or not?", "Decide whether the record conforms to the schema."]
FIELD_LIB = {"id": {"type": "integer", "required": True, "min": 1}, "email": {"type": "string", "format": "email", "required": True}, "age": {"type": "integer", "min": 0, "max": 120},
             "status": {"type": "string", "enum": ["active", "paused", "closed"], "required": True}, "amount": {"type": "number", "min": 0}, "start_date": {"type": "string", "format": "date"},
             "end_date": {"type": "string", "format": "date"}, "country": {"type": "string", "enum": ["US", "GB", "DE", "FR", "SE", "PT"]}}


def _good(rng: random.Random, name: str, pool: int) -> Any:
    return {"id": rng.randint(1, 99999), "email": f"{rng.choice(part(USERS, pool))}@{rng.choice(['example.com', 'mail.org', 'corp.io'])}", "age": rng.randint(1, 110),
            "status": rng.choice(["active", "paused", "closed"]), "amount": round(rng.uniform(0, 5000), 2), "start_date": f"20{rng.randint(15, 24)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "end_date": None, "country": rng.choice(["US", "GB", "DE", "FR", "SE", "PT"])}[name]


def gen_rowcheck(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    names = rng.sample([n for n in FIELD_LIB if n != "end_date"], rng.randint(3, 5))
    if "start_date" in names and rng.random() < 0.6:
        names.append("end_date")
    schema = {n: FIELD_LIB[n] for n in names}
    row = {n: _good(rng, n, pool) for n in names if n != "end_date"}
    if "end_date" in names:
        y, m, d = (int(x) for x in row["start_date"].split("-"))
        row["end_date"] = f"{y + 1}-{m:02d}-{d:02d}"
    valid = rng.random() < 0.5
    if not valid:
        for n in rng.sample(names, rng.randint(1, min(2, len(names)))):
            kind = rng.choice(["missing", "range", "type", "enum", "format"])
            spec = schema[n]
            if kind == "missing" and spec.get("required"):
                row.pop(n, None)
            elif kind == "range" and n in ("age", "amount", "id"):
                row[n] = {"age": rng.choice([-3, 140, 200]), "amount": -round(rng.uniform(1, 99), 2), "id": 0}[n]
            elif kind == "type" and spec["type"] in ("integer", "number"):
                row[n] = "n/a"
            elif kind == "enum" and "enum" in spec:
                row[n] = "unknown"
            elif kind == "format" and spec.get("format") == "email":
                row[n] = row[n].replace("@", "_at_")
            elif kind == "format" and spec.get("format") == "date":
                row[n] = "31/02/2021"
            elif n == "end_date" and "start_date" in row and __import__("re").fullmatch(r"\d{4}-\d{2}-\d{2}", str(row["start_date"])):
                y, m, d = (int(x) for x in row["start_date"].split("-"))
                row["end_date"] = f"{y - 1}-{m:02d}-{d:02d}"
            else:
                continue
    ok = _row_valid(schema, row)
    return {"state": {"schema": schema, "row": row}, "type": "boolean", "label": ok, "instr": DV_INSTR, "features": {}}


def _row_valid(schema: dict[str, Any], row: dict[str, Any]) -> bool:
    import re
    for n, s in schema.items():
        if n not in row or row[n] is None:
            if s.get("required"):
                return False
            continue
        v = row[n]
        if s["type"] == "integer" and (not isinstance(v, int) or isinstance(v, bool)):
            return False
        if s["type"] == "number" and (not isinstance(v, (int, float)) or isinstance(v, bool)):
            return False
        if s["type"] == "string" and not isinstance(v, str):
            return False
        if "min" in s and v < s["min"] or "max" in s and v > s["max"]:
            return False
        if "enum" in s and v not in s["enum"]:
            return False
        if s.get("format") == "email" and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]+", v):
            return False
        if s.get("format") == "date" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            return False
    if row.get("start_date") and row.get("end_date") and row["end_date"] < row["start_date"]:
        return False
    return True


# ---------------------------------------------------------------- semantic grep (Boolean)
SG_CLASSES = {  # class -> paraphrase templates (index 4 is reserved for the challenge split, 3 for locked test, 2 for calibration)
    "auth_failure": ["Failed password for {u} from {ip}", "authentication failed for user {u} (attempt {n})", "login rejected: invalid credentials for {u}", "{u}: bad token, access denied from {ip}", "Sign-in refused for account {u} after {n} wrong attempts"],
    "auth_success": ["Accepted password for {u} from {ip}", "user {u} authenticated successfully", "login ok for {u}", "{u}: session started via SSO from {ip}", "Account {u} signed in without issues"],
    "timeout": ["upstream {h} timed out after {n}00 ms", "request to {h} exceeded deadline", "read timeout while calling {h}", "{h}: no response within {n}s", "Call to {h} was abandoned because it took too long"],
    "connection_refused": ["connect to {h}:5432 failed: connection refused", "{h} refused the connection", "ECONNREFUSED {h}", "could not open a socket to {h}: refused", "Nothing is listening on {h}; the connection attempt was rejected"],
    "disk_full": ["No space left on device /var/data on {h}", "write failed: disk {h} is 100% full", "{h}: filesystem full, cannot append", "volume on {h} has no free blocks", "Storage on {h} is exhausted; new writes are failing"],
    "oom_kill": ["Out of memory: killed process {n}12 on {h}", "{h}: OOM killer terminated the worker", "container on {h} exceeded its memory limit and was killed", "memory cgroup out of memory on {h}", "The kernel reclaimed memory on {h} by killing a process"],
    "deploy_started": ["deploy {n} started on {h}", "rolling out release 1.{n} to {h}", "{h}: deployment begun", "starting release pipeline for {h}", "A new version is now being rolled out to {h}"],
    "deploy_failed": ["deploy {n} failed on {h}: health checks did not pass", "rollout to {h} aborted, rolling back", "{h}: release rejected after failed smoke test", "deployment error on {h}: image pull failed", "The new version could not be installed on {h} and was reverted"],
    "healthcheck_ok": ["health check passed for {h}", "{h} is healthy (latency {n}ms)", "GET /healthz 200 from {h}", "probe ok: {h}", "All checks green for {h}"],
    "slow_query": ["slow query ({n}s) on {h}: SELECT ...", "{h}: query exceeded 2s threshold", "long-running statement detected on {h}", "db {h} reports a query taking {n} seconds", "A database call on {h} took unusually long to finish"],
}
SG_CRITERIA = [("The line reports a failed login or authentication attempt.", {"auth_failure"}), ("The line says a service or host ran out of memory or disk space.", {"disk_full", "oom_kill"}),
               ("The line describes a failure to reach another service over the network.", {"timeout", "connection_refused"}), ("The line is about a problem with a deployment or release.", {"deploy_failed"}),
               ("The line shows something going wrong.", {"auth_failure", "timeout", "connection_refused", "disk_full", "oom_kill", "deploy_failed", "slow_query"}),
               ("The line is routine, healthy or informational activity.", {"auth_success", "deploy_started", "healthcheck_ok"}),
               ("The line is about database performance.", {"slow_query"}), ("The line is about user access to the system, successful or not.", {"auth_failure", "auth_success"})]
SG_INSTR = ["Does this log line meet the criterion?", "Is the criterion satisfied by the log line?", "Would a semantic filter with this criterion keep the line?",
            "Judge the log line against the criterion: does it qualify?", "Decide whether the line matches the described criterion."]


def gen_grep(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    crit, ok = rng.choice(SG_CRITERIA)
    positive = rng.random() < 0.5
    pool_classes = list(ok) if positive else [c for c in SG_CLASSES if c not in ok]
    cls = rng.choice(pool_classes)
    idx = {"train": rng.randint(0, 1), "calibration": 2, "locked_test": 3, "challenge": 4}[split]
    line = SG_CLASSES[cls][idx].format(u=rng.choice(part(USERS, pool)), ip=f"10.{rng.randint(0, 250)}.{rng.randint(0, 250)}.{rng.randint(1, 250)}", h=rng.choice(part(HOSTS, pool)), n=rng.randint(2, 9))
    ts = f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}T{rng.randint(10, 23)}:{rng.randint(10, 59)}:{rng.randint(10, 59)}Z"
    return {"state": {"criterion": crit, "line": f"{ts} {line}"}, "type": "boolean", "label": cls in ok, "instr": SG_INSTR, "features": {"cls": cls}}


# ---------------------------------------------------------------- batch 2: TypeSafe automation use cases (records, risk, SOC, routing, people, code)
def _noise(rng: random.Random, pool: int) -> dict[str, Any]:
    """Fields the decision does not depend on (ids, host, time): the model must learn to ignore them, and they keep small state spaces from collapsing under dedupe."""
    return {"case_id": f"{rng.choice(['INC', 'CASE', 'TKT'])}-{rng.randint(1000, 99999)}", "host": rng.choice(part(HOSTS, pool)), "opened_at": f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}T{rng.randint(10, 23)}:{rng.randint(10, 59)}Z"}


def _choice(state: dict[str, Any], opts: list[tuple[str, str]], label: Any, instr: list[str], features: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"state": state, "type": "choice", "options": opts, "label": label, "instr": instr, "features": features or {}}


def _instr(topic: str) -> list[str]:
    return [f"Which {topic} decision does the policy give for this case?", f"Apply the policy: what is the {topic} decision?", f"Under the stated policy, choose the {topic} outcome.",
            f"Decide the {topic} outcome for this case from the policy.", f"Work through the policy and give the {topic} decision."]


INV_OPTS = [("pay", "Approve the invoice for payment."), ("hold", "Hold the invoice pending information or approval."), ("dispute", "Dispute the invoice with the vendor."),
            ("fraud_review", "Send the invoice to the fraud team.")]
INV_POLICY = ("Invoice rules, first match wins: bank details changed since the last payment, or an unknown vendor above 5000: fraud_review. No matching purchase order, or the invoiced amount "
              "more than 5% over the purchase order: dispute. Possible duplicate invoice, or no authorised approval: hold. Otherwise pay.")


def gen_invoice(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    po = round(rng.uniform(200, 20000) * (1 + 0.05 * pool), 2)
    i = {"vendor": f"{rng.choice(part(COMPANY_A, pool))} {rng.choice(COMPANY_B)}", "vendor_known": rng.random() < 0.75, "amount": round(po * rng.choice([0.9, 1.0, 1.0, 1.03, 1.05, 1.08, 1.4]), 2),
         "po_amount": po if rng.random() < 0.85 else None, "duplicate_suspected": rng.random() < 0.12, "bank_details_changed": rng.random() < 0.1, "approved_by_authorised": rng.random() < 0.8}
    if not i["vendor_known"] and rng.random() < 0.5:
        i["amount"] = round(rng.uniform(3000, 9000), 2)
    if i["bank_details_changed"] or (not i["vendor_known"] and i["amount"] > 5000):
        lab = "fraud_review"
    elif i["po_amount"] is None or i["amount"] > i["po_amount"] * 1.05:
        lab = "dispute"
    elif i["duplicate_suspected"] or not i["approved_by_authorised"]:
        lab = "hold"
    else:
        lab = "pay"
    return _choice({"policy": INV_POLICY, "invoice": i}, INV_OPTS, lab, _instr("invoice"))


CLM_OPTS = [("STP", "Straight-through payment without manual review."), ("SIU", "Refer to the special investigations unit."), ("specialist", "Assign to a specialist adjuster.")]
CLM_POLICY = ("Claims routing: fraud points: 3 or more claims in the last 12 months +2; policy started under 30 days ago +2; documents incomplete +1; loss over 25000 +1. 4 or more points: SIU. "
              "Otherwise, an injury, a loss over 50000 or an inactive policy: specialist. Otherwise STP.")


def gen_claim(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    c = {"claim_type": rng.choice(["auto collision", "water damage", "theft", "fire", "hail damage", "liability"]), "loss_amount": rng.choice([800, 3200, 9000, 24000, 26000, 48000, 60000, 120000]),
         "claims_last_12m": rng.choice([0, 0, 1, 2, 3, 4]), "policy_age_days": rng.choice([5, 20, 45, 180, 700, 2000]), "documents_complete": rng.random() < 0.75, "injury_involved": rng.random() < 0.2,
         "policy_active": rng.random() < 0.93}
    pts = (2 if c["claims_last_12m"] >= 3 else 0) + (2 if c["policy_age_days"] < 30 else 0) + (0 if c["documents_complete"] else 1) + (1 if c["loss_amount"] > 25000 else 0)
    lab = "SIU" if pts >= 4 else "specialist" if (c["injury_involved"] or c["loss_amount"] > 50000 or not c["policy_active"]) else "STP"
    return _choice({"policy": CLM_POLICY, "claim": c}, CLM_OPTS, lab, _instr("claim routing"), {"points": pts})


SOC_TRIAGE_OPTS = [("auto_close", "Close the alert; no action needed."), ("notify_user", "Notify the affected user and keep monitoring."), ("queue_tier2", "Queue for a tier-2 analyst."),
                   ("contain_now", "Start containment immediately.")]
SOC_POLICY = ("SOC triage rules, first match wins: lateral movement observed, or a confirmed threat on a critical asset: contain_now. Known false-positive pattern, or severity 1-2 with no confirmed "
              "threat and no user impact: auto_close. User impacted, severity 3 or lower and no confirmed threat: notify_user. Otherwise queue_tier2.")


def gen_soc_triage(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    a = {"alert_type": rng.choice(["impossible travel login", "malware detected", "unusual outbound traffic", "privilege change", "phishing report", "port scan", "endpoint tamper"]),
         "severity": rng.randint(1, 5), "asset_critical": rng.random() < 0.3, "threat_confirmed": rng.random() < 0.3, "user_impacted": rng.random() < 0.4,
         "lateral_movement": rng.random() < 0.12, "known_false_positive_pattern": rng.random() < 0.2}
    if a["lateral_movement"] or (a["threat_confirmed"] and a["asset_critical"]):
        lab = "contain_now"
    elif a["known_false_positive_pattern"] or (a["severity"] <= 2 and not a["threat_confirmed"] and not a["user_impacted"]):
        lab = "auto_close"
    elif a["user_impacted"] and a["severity"] <= 3 and not a["threat_confirmed"]:
        lab = "notify_user"
    else:
        lab = "queue_tier2"
    return _choice({"policy": SOC_POLICY, "alert": a}, SOC_TRIAGE_OPTS, lab, _instr("triage"))


ESC_OPTS = [("page_oncall", "Page the on-call engineer."), ("handoff_ir", "Hand off to the incident-response team."), ("notice_users", "Send a notice to affected users."), ("stay_queue", "Leave in the queue.")]
ESC_POLICY = ("Escalation rules, first match wins: triage contain_now on a critical asset: page_oncall. Triage contain_now otherwise, or severity 4-5 on a high-criticality asset: handoff_ir. "
              "Triage notify_user: notice_users. Otherwise stay_queue.")


def gen_soc_escalation(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    e = {"triage": rng.choice(["auto_close", "notify_user", "queue_tier2", "contain_now"]), "severity": rng.randint(1, 5), "asset_criticality": rng.choice(["low", "medium", "high", "critical"])}
    if e["triage"] == "contain_now" and e["asset_criticality"] == "critical":
        lab = "page_oncall"
    elif e["triage"] == "contain_now" or (e["severity"] >= 4 and e["asset_criticality"] == "high"):
        lab = "handoff_ir"
    elif e["triage"] == "notify_user":
        lab = "notice_users"
    else:
        lab = "stay_queue"
    e.update(_noise(rng, pool))
    return _choice({"policy": ESC_POLICY, "case": e}, ESC_OPTS, lab, _instr("escalation"))


REC_OPTS = [("remain_isolated", "Keep the host isolated."), ("limited_restore", "Restore with restrictions."), ("full_restore", "Restore fully.")]
REC_POLICY = ("Recovery rules: monitoring not clean, or contained fewer than 4 hours: remain_isolated. Contained at least 24 hours with the root cause fixed and monitoring clean: full_restore. "
              "Otherwise limited_restore.")


def gen_soc_recovery(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    r = {"hours_contained": rng.choice([1, 3, 4, 8, 20, 24, 30, 72]), "monitoring_clean": rng.random() < 0.7, "root_cause_fixed": rng.random() < 0.55}
    if not r["monitoring_clean"] or r["hours_contained"] < 4:
        lab = "remain_isolated"
    elif r["hours_contained"] >= 24 and r["root_cause_fixed"]:
        lab = "full_restore"
    else:
        lab = "limited_restore"
    r.update(_noise(rng, pool))
    return _choice({"policy": REC_POLICY, "case": r}, REC_OPTS, lab, _instr("recovery"))


LOGS_OPTS = [("auth_logs", "Collect authentication logs."), ("process_tree", "Collect the process tree."), ("network_logs", "Collect network logs."), ("identity_logs", "Collect identity-provider logs.")]
LOGS_MAP = {"brute force login": "auth_logs", "credential stuffing": "auth_logs", "password spraying": "auth_logs", "ransomware execution": "process_tree", "malicious script": "process_tree",
            "fileless malware": "process_tree", "data exfiltration": "network_logs", "command-and-control beaconing": "network_logs", "port scanning": "network_logs", "dns tunnelling": "network_logs",
            "privilege escalation in the directory": "identity_logs", "MFA fatigue attack": "identity_logs", "stolen session token": "identity_logs", "rogue OAuth app": "identity_logs"}
LOGS_POLICY = "Investigation: collect the log source that best shows the attack type: authentication attempts, process activity, network traffic or identity-provider events."


def gen_soc_logs(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    atk = rng.choice(list(LOGS_MAP))
    c = {"attack_type": atk, "asset_category": rng.choice(["laptop", "server", "cloud workload", "identity provider", "mail gateway"]), "scope": rng.choice(["one host", "several hosts", "one account", "many accounts"])}
    c.update(_noise(rng, pool))
    return _choice({"policy": LOGS_POLICY, "case": c}, LOGS_OPTS, LOGS_MAP[atk], _instr("log-collection"))


FC_LEVELS = [(0, "Low priority: likely benign."), (1, "Routine review."), (2, "High priority."), (3, "Urgent: escalate now.")]
FC_POLICY = ("Alert priority points: structuring pattern (several deposits just under the reporting limit) +3; high-risk jurisdiction +2; politically exposed person +2; prior suspicious-activity report "
             "on file +2; the rule's historical false-positive rate over 80% -2. Score 0 for 1 point or less, 1 for 2-3, 2 for 4-5, 3 for 6 or more.")


def gen_fincrime(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    a = {"total_amount": rng.choice([9500, 27000, 48000, 130000]), "structuring_pattern": rng.random() < 0.3, "high_risk_jurisdiction": rng.random() < 0.3, "politically_exposed": rng.random() < 0.15,
         "prior_sar": rng.random() < 0.2, "rule_false_positive_rate_pct": rng.choice([30, 55, 75, 85, 92]), "customer_since_years": rng.choice([0, 1, 4, 12])}
    pts = 3 * a["structuring_pattern"] + 2 * a["high_risk_jurisdiction"] + 2 * a["politically_exposed"] + 2 * a["prior_sar"] - (2 if a["rule_false_positive_rate_pct"] > 80 else 0)
    lab = 0 if pts <= 1 else 1 if pts <= 3 else 2 if pts <= 5 else 3
    return {"state": {"policy": FC_POLICY, "alert": a}, "type": "score", "levels": FC_LEVELS, "label": lab, "instr": _instr("alert-priority"), "features": {"points": pts}}


SKILLS = "Python SQL Go Rust Kubernetes Terraform React TypeScript Spark Airflow Docker Postgres Kafka Redis GraphQL Java Swift Django Linux Git Tableau".split()
RC_LEVELS = [(0, "Reject: a required skill is missing."), (1, "Basic fit."), (2, "Good fit."), (3, "Strong fit: fast-track.")]
RC_POLICY = ("Screening: a missing required skill scores 0. Otherwise start at 1; add 1 if at least half of the nice-to-have skills are present; add 1 if experience is at least 2 years above the minimum.")


def gen_recruit(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    sk = part(SKILLS, pool, 5) if pool % 2 else SKILLS
    req = rng.sample(sk, min(2, len(sk)))
    nice = rng.sample([x for x in SKILLS if x not in req], 4)
    have = [x for x in req if rng.random() < 0.85] + [x for x in nice if rng.random() < 0.5] + rng.sample(SKILLS, 2)
    j = {"required_skills": req, "nice_to_have": nice, "min_years": rng.choice([2, 3, 5])}
    cand = {"skills": sorted(set(have)), "years_experience": rng.choice([1, 3, 5, 7, 9, 12])}
    if not all(x in cand["skills"] for x in req):
        lab = 0
    else:
        lab = min(3, 1 + (len([x for x in nice if x in cand["skills"]]) * 2 >= len(nice)) + (cand["years_experience"] >= j["min_years"] + 2))
    return {"state": {"policy": RC_POLICY, "job": j, "candidate": cand}, "type": "score", "levels": RC_LEVELS, "label": lab, "instr": _instr("candidate-fit"), "features": {}}


MR_OPTS = [("small", "Route to the small, cheap, fast model."), ("medium", "Route to the mid-size model."), ("large", "Route to the large, most capable model.")]
MR_POLICY = ("Routing rules, first match wins: multi-step reasoning, or code generation where accuracy is critical: large. Latency budget under 500 ms: small. Extraction or classification with under 4000 "
             "context tokens and accuracy not critical: small. Otherwise medium.")


def gen_modelroute(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    t = {"task": rng.choice(["extraction", "classification", "summarisation", "code generation", "multi-step reasoning", "creative writing", "translation"]), "context_tokens": rng.choice([300, 1500, 3900, 4100, 12000, 60000]),
         "accuracy_critical": rng.random() < 0.4, "latency_budget_ms": rng.choice([300, 450, 800, 2000, 10000]), "needs_tools": rng.random() < 0.4}
    if t["task"] == "multi-step reasoning" or (t["task"] == "code generation" and t["accuracy_critical"]):
        lab = "large"
    elif t["latency_budget_ms"] < 500:
        lab = "small"
    elif t["task"] in ("extraction", "classification") and t["context_tokens"] < 4000 and not t["accuracy_critical"]:
        lab = "small"
    else:
        lab = "medium"
    return _choice({"policy": MR_POLICY, "request": t}, MR_OPTS, lab, _instr("model-routing"))


RF_INSTR = ["Is the customer eligible for a refund?", "Under the refund policy, does this request qualify?", "Should the refund be approved under the stated policy?",
            "Check the request against the refund policy: eligible?", "Decide refund eligibility from the policy and the order facts."]
RF_POLICY = ("Refund policy: final-sale items are never refundable. Otherwise an unopened or opened item is refundable within the return window; a damaged item is refundable within twice the return window.")


def gen_refund(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    o = {"item": rng.choice(["headphones", "blender", "desk lamp", "backpack", "keyboard", "kettle"]), "days_since_delivery": rng.choice([3, 10, 14, 20, 29, 31, 45, 60, 90]),
         "return_window_days": rng.choice([14, 30]), "condition": rng.choice(["unopened", "opened", "damaged"]), "final_sale": rng.random() < 0.15}
    ok = (not o["final_sale"]) and ((o["condition"] in ("unopened", "opened") and o["days_since_delivery"] <= o["return_window_days"]) or (o["condition"] == "damaged" and o["days_since_delivery"] <= 2 * o["return_window_days"]))
    o.update({"order_id": f"ORD-{rng.randint(10000, 99999)}"})
    return {"state": {"policy": RF_POLICY, "order": o}, "type": "boolean", "label": ok, "instr": RF_INSTR, "features": {}}


CH_LEVELS = [(0, "Low churn risk."), (1, "Watch."), (2, "At risk."), (3, "Very likely to churn.")]
CH_POLICY = ("Churn points: no login for 14+ days +3 (7-13 days +2, 3-6 days +1); sessions in the last 7 days under half the prior week +2; no purchase in 60 days +1; fewer than 2 active friends +1. "
             "Score 0 for 0-1 points, 1 for 2-3, 2 for 4-5, 3 for 6 or more.")


def gen_churn(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    p = {"days_since_login": rng.choice([0, 1, 4, 6, 8, 12, 16, 30]), "sessions_last_7d": rng.choice([0, 1, 3, 6, 10]), "sessions_prior_7d": rng.choice([2, 5, 8, 12]), "days_since_purchase": rng.choice([5, 30, 70, 200]),
         "active_friends": rng.choice([0, 1, 3, 8])}
    d = p["days_since_login"]
    pts = (3 if d >= 14 else 2 if d >= 7 else 1 if d >= 3 else 0) + (2 if p["sessions_last_7d"] < p["sessions_prior_7d"] / 2 else 0) + (1 if p["days_since_purchase"] >= 60 else 0) + (1 if p["active_friends"] < 2 else 0)
    lab = 0 if pts <= 1 else 1 if pts <= 3 else 2 if pts <= 5 else 3
    return {"state": {"policy": CH_POLICY, "player": p}, "type": "score", "levels": CH_LEVELS, "label": lab, "instr": _instr("churn-risk"), "features": {"points": pts}}


LINT_RULES = {
    "Do not use a bare except clause.": (["try:\n    {v} = load({a})\nexcept:\n    {v} = None", "try:\n    {f}({a})\nexcept:\n    pass"], ["try:\n    {v} = load({a})\nexcept ValueError:\n    {v} = None", "try:\n    {f}({a})\nexcept Exception as err:\n    log(err)"]),
    "Functions must not use mutable default arguments.": (["def {f}({a}, {v}=[]):\n    {v}.append({a})\n    return {v}", "def {f}({a}, {v}={{}}):\n    {v}[{a}] = 1\n    return {v}"],
                                                        ["def {f}({a}, {v}=None):\n    {v} = {v} or []\n    {v}.append({a})\n    return {v}", "def {f}({a}, {v}=()):\n    return list({v}) + [{a}]"]),
    "SQL statements must be parameterised, never built by string concatenation.": (["cur.execute(\"SELECT * FROM users WHERE name = '\" + {a} + \"'\")", "q = f\"DELETE FROM orders WHERE id = {{{a}}}\"\ncur.execute(q)"],
                                                                                    ["cur.execute(\"SELECT * FROM users WHERE name = %s\", ({a},))", "cur.execute(\"DELETE FROM orders WHERE id = ?\", ({a},))"]),
    "Library code must not print to stdout; use the logger.": (["def {f}({a}):\n    print('processing', {a})\n    return {a}", "def {f}({a}):\n    result = {a} * 2\n    print(result)\n    return result"],
                                                             ["def {f}({a}):\n    logger.info('processing %s', {a})\n    return {a}", "def {f}({a}):\n    result = {a} * 2\n    logger.debug(result)\n    return result"]),
}
LINT_INSTR = ["Does the code violate the convention?", "Is the stated convention broken by this snippet?", "Would a semantic linter flag this code under the convention?",
              "Check the snippet against the convention: violation or not?", "Decide whether the code breaks the coding rule."]


def gen_lint(rng: random.Random, pool: int, split: str) -> dict[str, Any]:
    rule = rng.choice(list(LINT_RULES))
    bad, good = LINT_RULES[rule]
    viol = rng.random() < 0.5
    tpl = rng.choice(bad if viol else good)
    names = {"f": rng.choice(part(["handle", "process", "sync", "fetch", "store", "update", "render", "merge", "load_item", "send"], pool, 5)), "a": rng.choice(part(["item", "user_id", "payload", "name", "order_id", "key", "path"], pool, 3)),
             "v": rng.choice(part(["result", "cache", "rows", "acc", "data", "items"], pool, 3))}
    return {"state": {"convention": rule, "code": tpl.format(**names)}, "type": "boolean", "label": viol, "instr": LINT_INSTR, "features": {}}


PACKS: dict[str, tuple[Callable[..., dict[str, Any]], str, str]] = {  # pack -> (generator, composition family, task description)
    "harness_entity_resolution_v1": (gen_entity_resolution, "data_code_workflows", "entity resolution"),
    "harness_fraud_risk_v1": (gen_fraud, "security_policy", "fraud risk"),
    "harness_financial_triage_v1": (gen_fin, "data_code_workflows", "financial triage"),
    "harness_realtime_control_v1": (gen_control, "agent_harness", "real-time control"),
    "harness_lead_scoring_v1": (gen_lead, "communication_productivity", "lead scoring"),
    "harness_row_validity_v1": (gen_rowcheck, "data_code_workflows", "data-row validity"),
    "harness_semantic_grep_v1": (gen_grep, "data_code_workflows", "semantic grep"),
    "harness_invoice_processing_v1": (gen_invoice, "security_policy", "invoice processing"),
    "harness_insurance_claims_v1": (gen_claim, "security_policy", "insurance claims"),
    "harness_soc_triage_v1": (gen_soc_triage, "security_policy", "SOC triage"),
    "harness_soc_escalation_v1": (gen_soc_escalation, "security_policy", "SOC escalation"),
    "harness_soc_recovery_v1": (gen_soc_recovery, "security_policy", "SOC recovery"),
    "harness_soc_log_collection_v1": (gen_soc_logs, "security_policy", "SOC investigation"),
    "harness_financial_crime_alert_v1": (gen_fincrime, "security_policy", "financial crime"),
    "harness_recruiting_fit_v1": (gen_recruit, "communication_productivity", "recruiting"),
    "harness_model_routing_v1": (gen_modelroute, "agent_harness", "model routing"),
    "harness_refund_eligibility_v1": (gen_refund, "communication_productivity", "refund eligibility"),
    "harness_churn_risk_v1": (gen_churn, "communication_productivity", "churn risk"),
    "harness_semantic_lint_v1": (gen_lint, "data_code_workflows", "semantic linting"),
}


def make_rows(pack: str, count: int, seed: int) -> list[dict[str, Any]]:
    gen = PACKS[pack][0]
    rng = random.Random(f"{pack}-{seed}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split, frac in SPLIT_FRAC.items():
        target = int(count * frac)
        quota: dict[str, int] = defaultdict(int)
        got: list[dict[str, Any]] = []
        tries = 0
        probe = gen(random.Random(0), 0, split)
        n_labels = len(probe["options"]) if probe["type"] == "choice" else len(probe["levels"]) if probe["type"] == "score" else 2
        while len(got) < target and tries < target * 150:
            tries += 1
            pool = pick_split_pool(split, rng)
            r = gen(rng, pool, split)
            key = hashlib.sha1(json.dumps(r["state"], sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            lab = str(r["label"])
            if quota[lab] >= target / n_labels * 1.3 + 1:
                continue
            seen.add(key)
            quota[lab] += 1
            r.update(split=split, pool=pool, pack=pack, rid=f"{pack[8:12]}-{split[:2]}-{len(got):05d}", opt_order=None)
            got.append(r)
        rows += got
    return rows
