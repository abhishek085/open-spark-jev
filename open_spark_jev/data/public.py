"""Adapters that turn public Hugging Face datasets into decision records.

Each adapter maps a dataset onto one primitive and keeps the option count <= MAX_OPTIONS.
They are intentionally tiny and declarative so adding a dataset is a 10-line change.
``datasets`` is imported lazily; the module is importable without it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from .corpus import Record

Adapter = Callable[[int, int], Iterator[Record]]  # (limit, seed) -> records
REGISTRY: dict[str, Adapter] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn

    return deco


def _load(path: str, split: str, config: str | None = None, limit: int = 0, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
    if limit:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    return ds


@register("ag_news")
def ag_news(limit: int = 2000, seed: int = 0) -> Iterator[Record]:
    names = ["world", "sports", "business", "sci_tech"]
    for i, row in enumerate(_load("ag_news", "train", limit=limit, seed=seed)):
        yield Record(
            id=f"ag_news-{i}", domain="classification", source="public:ag_news",
            state={"content": row["text"], "schema_hint": "news headline and lead", "domain": "classification"},
            question={"type": "choice", "prompt": "Which section does this news item belong to?", "options": names},
            target={"label": names[row["label"]]},
        )


@register("emotion")
def emotion(limit: int = 2000, seed: int = 0) -> Iterator[Record]:
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for i, row in enumerate(_load("dair-ai/emotion", "train", "split", limit=limit, seed=seed)):
        yield Record(
            id=f"emotion-{i}", domain="classification", source="public:dair-ai/emotion",
            state={"content": row["text"], "schema_hint": "short social media post", "domain": "classification"},
            question={"type": "choice", "prompt": "What is the dominant emotion expressed?", "options": names},
            target={"label": names[row["label"]]},
        )


@register("banking77_top20")
def banking77(limit: int = 3000, seed: int = 0) -> Iterator[Record]:
    """Intent routing. 77 intents exceed the 26-slot menu, so keep the 20 most frequent."""
    from collections import Counter

    ds = _load("PolyAI/banking77", "train", limit=0, seed=seed)
    names = ds.features["label"].names
    top = [c for c, _ in Counter(ds["label"]).most_common(20)]
    keep = [names[c] for c in top]
    n = 0
    for i, row in enumerate(ds.shuffle(seed=seed)):
        if row["label"] not in top:
            continue
        yield Record(
            id=f"banking77-{i}", domain="routing", source="public:PolyAI/banking77",
            state={"content": row["text"], "schema_hint": "customer message to a bank", "domain": "routing"},
            question={"type": "choice", "prompt": "Which intent best matches this customer message?", "options": keep},
            target={"label": names[row["label"]]},
        )
        n += 1
        if limit and n >= limit:
            break


@register("toxic_chat")
def toxic_chat(limit: int = 2000, seed: int = 0) -> Iterator[Record]:
    for i, row in enumerate(_load("lmsys/toxic-chat", "train", "toxicchat0124", limit=limit, seed=seed)):
        yield Record(
            id=f"toxicchat-{i}", domain="moderation", source="public:lmsys/toxic-chat",
            state={"content": row["user_input"], "schema_hint": "user message to a chatbot", "domain": "moderation"},
            question={"type": "noul", "prompt": "This message is toxic or violates a reasonable content policy."},
            target={"label": "yes" if row["toxicity"] == 1 else "no"},
        )


@register("boolq")
def boolq(limit: int = 2000, seed: int = 0) -> Iterator[Record]:
    for i, row in enumerate(_load("google/boolq", "train", limit=limit, seed=seed)):
        yield Record(
            id=f"boolq-{i}", domain="reading", source="public:google/boolq",
            state={"content": row["passage"], "schema_hint": "reference passage", "domain": "reading"},
            question={"type": "noul", "prompt": row["question"].rstrip("?") + "?"},
            target={"label": "yes" if row["answer"] else "no"},
        )


@register("yelp_score")
def yelp(limit: int = 2000, seed: int = 0) -> Iterator[Record]:
    levels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    for i, row in enumerate(_load("Yelp/yelp_review_full", "train", limit=limit, seed=seed)):
        yield Record(
            id=f"yelp-{i}", domain="scoring", source="public:Yelp/yelp_review_full",
            state={"content": row["text"], "schema_hint": "customer review", "domain": "scoring"},
            question={"type": "score", "prompt": "How many stars did the reviewer give?", "levels": levels},
            target={"label": levels[row["label"]]},
        )


def build(names: list[str], limit: int, seed: int = 0) -> list[Record]:
    out: list[Record] = []
    for n in names:
        out.extend(REGISTRY[n](limit, seed))
    return out
