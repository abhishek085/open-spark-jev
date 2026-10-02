#!/usr/bin/env python
"""Metadata-only survey of public Hugging Face datasets as candidates for the guarded public-data slice.

Queries the Hub API (no dataset files downloaded), keeps permissively licensed datasets that are NOT one of our eval
sources, and writes a table for human approval. Nothing here trains on anything; approval + scripts/tools/benchmark_overlap.py
come before any download.

  python scripts/analysis/public_survey.py --out runs/v7/public_candidates.json
"""

import argparse, json
from huggingface_hub import HfApi

QUERIES = [
    "prompt injection",
    "tool call safety",
    "function calling",
    "intent classification",
    "content moderation",
    "jailbreak",
    "retrieval relevance judgement",
    "agent action risk",
    "policy compliance classification",
    "ticket routing triage",
    "fraud detection text",
    "llm as judge preference",
    "entity resolution",
    "pii detection",
]
PERMISSIVE = {
    "apache-2.0",
    "mit",
    "cc-by-4.0",
    "cc0-1.0",
    "cc-by-sa-4.0",
    "bsd-3-clause",
    "odc-by",
    "cc-by-3.0",
    "openrail",
    "cc-by-2.0",
}
# our eval sources (training on any of these would contaminate external scores) + the general Nimble suite
EVAL_SOURCES = [
    "banking77",
    "boolq",
    "civil_comments",
    "paws",
    "pubmedqa",
    "squad",
    "summeval",
    "multinli",
    "vitaminc",
    "aegis",
    "helpsteer",
    "massive",
    "ag_news",
    "yelp",
    "emotion",
    "toxic-chat",
    "toxic_chat",
    "jevbench",
    "jev-",
    "kev",
    "typesafe",
    "deepset",
    "cybernative",
    "code_vulnerability",
    "jev-sec-bench",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-query", type=int, default=30)
    a = ap.parse_args()
    api, seen, rows = HfApi(), set(), []
    for q in QUERIES:
        try:
            for d in api.list_datasets(search=q, limit=a.per_query, sort="downloads"):
                if d.id in seen:
                    continue
                seen.add(d.id)
                lic = next((t.split(":", 1)[1] for t in (d.tags or []) if t.startswith("license:")), None)
                size = next(
                    (t.split(":", 1)[1] for t in (d.tags or []) if t.startswith("size_categories:")), None
                )
                tasks = [t.split(":", 1)[1] for t in (d.tags or []) if t.startswith("task_categories:")]
                low = d.id.lower()
                rows.append(
                    {
                        "id": d.id,
                        "query": q,
                        "license": lic,
                        "size": size,
                        "tasks": tasks[:3],
                        "downloads": d.downloads,
                        "likes": d.likes,
                        "permissive": lic in PERMISSIVE,
                        "is_eval_source": any(e in low for e in EVAL_SOURCES),
                    }
                )
        except Exception as e:
            print("query failed:", q, type(e).__name__, str(e)[:100])
    keep = [r for r in rows if r["permissive"] and not r["is_eval_source"]]
    keep.sort(key=lambda r: -(r["downloads"] or 0))
    json.dump(
        {
            "n_seen": len(rows),
            "n_kept": len(keep),
            "kept": keep,
            "dropped_eval_source": [r["id"] for r in rows if r["is_eval_source"]],
            "dropped_license": [
                (r["id"], r["license"]) for r in rows if not r["permissive"] and not r["is_eval_source"]
            ][:60],
        },
        open(a.out, "w"),
        indent=1,
    )
    print(f"seen {len(rows)} datasets; kept {len(keep)} permissive non-eval-source; wrote {a.out}")
    for r in keep[:45]:
        print(
            f"  {r['id'][:52]:52s} {str(r['license']):12s} {str(r['size'])[:18]:18s} dl={r['downloads']:>8} {','.join(r['tasks'])[:40]}  [{r['query']}]"
        )


if __name__ == "__main__":
    main()
