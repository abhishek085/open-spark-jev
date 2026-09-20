"""Controlled vocabularies. Each split draws from its own disjoint entity pool and style family so
held-out splits vary source/entity names and text style (split isolation)."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Pool:
    pool_id: str
    objects: tuple[str, ...]
    adjectives: tuple[str, ...]
    first_names: tuple[str, ...]
    last_names: tuple[str, ...]
    orgs: tuple[str, ...]
    cities: tuple[str, ...]
    topics: tuple[str, ...]
    files: tuple[str, ...]

    def person(self, rng: random.Random) -> str:
        return f"{rng.choice(self.first_names)} {rng.choice(self.last_names)}"

    def people(self, rng: random.Random, n: int) -> list[str]:
        out: list[str] = []
        while len(out) < n:
            p = self.person(rng)
            if p not in out:
                out.append(p)
        return out


POOLS: dict[str, Pool] = {
    "pool_a": Pool(
        "pool_a",
        ("vase", "lantern", "ladder", "kettle", "compass", "satchel", "mirror", "anvil"),
        ("red", "fragile", "shiny", "heavy", "waterproof", "antique", "flammable"),
        ("Mara", "Teodor", "Ilse", "Bram", "Nadia", "Corin"),
        ("Voss", "Halloran", "Pemberton", "Okafor", "Lindqvist", "Marsh"),
        ("Marrowgate Logistics", "Ellery Print Works", "Tidewell Foods", "Quillon Labs"),
        ("Harrowfield", "Brindlecombe", "Oxley Cross", "Wenmouth"),
        ("warehouse audit", "budget review", "vendor onboarding", "office move"),
        ("sales.csv", "orders.csv", "inventory.csv", "shipments.csv"),
    ),
    "pool_b": Pool(
        "pool_b",
        ("teapot", "blanket", "trumpet", "bucket", "clock", "drum", "pillow", "helmet"),
        ("green", "brittle", "noisy", "foldable", "sturdy", "polished", "reusable"),
        ("Yara", "Emeric", "Sunniva", "Lucan", "Odalys", "Perrin"),
        ("Castellan", "Ripley", "Abara", "Fenwick", "Dunmore", "Solberg"),
        ("Kestrel Freight", "Lumen Stationers", "Bracken Dairy", "Orrin Analytics"),
        ("Stillwater", "Peverell", "Ashgrove", "Norwick"),
        ("payroll calendar", "safety training", "tenant survey", "product launch"),
        ("revenue.csv", "tickets.csv", "expenses.csv", "returns.csv"),
    ),
    "pool_c": Pool(
        "pool_c",
        ("crate", "violin", "hammock", "telescope", "ribbon", "wagon", "umbrella", "barrel"),
        ("blue", "delicate", "portable", "insulated", "ancient", "rusty", "tunable"),
        ("Zeno", "Halvard", "Imara", "Tobias", "Wren", "Anselm"),
        ("Quennell", "Draycott", "Nakamura-Lowe", "Petrov", "Greaves", "Alder"),
        ("Fenmoor Textiles", "Salter & Reed", "Pinecrest Clinics", "Vantage Harbor"),
        ("Lowmarsh", "Carrowdale", "Eastmere", "Thistlebridge"),
        ("grant renewal", "fleet maintenance", "curriculum review", "data migration"),
        ("metrics.csv", "bookings.csv", "usage.csv", "claims.csv"),
    ),
    "pool_d": Pool(
        "pool_d",
        ("sled", "easel", "canoe", "bell", "tent", "saddle", "flute", "globe"),
        ("orange", "flimsy", "compact", "weathered", "hollow", "glossy", "adjustable"),
        ("Kaia", "Rowan", "Sabine", "Dmitri", "Elowen", "Joaquin"),
        ("Brandt", "Ishikawa", "Moreau", "Tan", "Whitlock", "Duarte"),
        ("Cinder Mills", "Aurel Systems", "Northfold Bank", "Gossamer Health"),
        ("Rookwood", "Pellham", "Dunmarrow", "Silverlake"),
        ("site inspection", "hiring plan", "loyalty program", "server upgrade"),
        ("events.csv", "leads.csv", "visits.csv", "logs.csv"),
    ),
}

# split -> [(split_family, pool_id, tone, instruction-variant indexes)]. Disjoint across splits.
SPLIT_STYLES: dict[str, list[tuple[str, str, str, tuple[int, ...]]]] = {
    "train": [("train_family_a", "pool_a", "plain", (0, 1)), ("train_family_b", "pool_a", "terse", (0, 1))],
    "calibration": [("calibration_family_a", "pool_b", "formal", (2,))],
    "locked_test": [("locked_test_family_a", "pool_c", "conversational", (3,))],
    "challenge": [("challenge_family_a", "pool_d", "noisy", (4,))],
}


def style_for(rng: random.Random, split: str) -> tuple[str, Pool, str, tuple[int, ...]]:
    fam, pool_id, tone, variants = rng.choice(SPLIT_STYLES[split])
    return fam, POOLS[pool_id], tone, variants


def pool_ids_by_split() -> dict[str, set[str]]:
    return {s: {p for _, p, _, _ in v} for s, v in SPLIT_STYLES.items()}
