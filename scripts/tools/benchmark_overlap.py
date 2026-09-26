#!/usr/bin/env python
"""Strict benchmark-contamination guard.

Fingerprints every public JevBench item (state, question text, option labels/criteria, gold rationale)
and checks any training/generated JSONL against it. Only hashes are stored, never benchmark text, so
the fingerprint file is safe to commit.

Three independent signals; ANY of them fails the row:
  exact   normalised full-text match of the state
  shingle >= --containment of a row's word 8-gram shingles found in one benchmark item (or vice versa)
  entity  *rare* entities (multi-word proper names, id-like tokens, CamelCase, number-with-unit; single capitalised words are ignored as sentence-initial noise); >= --min-entities of them that appears in a benchmark item
          and is not a common word (fictional company names, ticket ids, etc.)

Usage
  build:  benchmark_overlap.py build --bench ../external/jevbench/datasets/public --out data/benchmarks/jevbench_fingerprints.json
  check:  benchmark_overlap.py check --fp data/benchmarks/jevbench_fingerprints.json data/synthetic/x.jsonl [--fail]
Exit code 1 (with --fail) if any row is contaminated.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter

SH = 8
WORD = re.compile(r"[a-z0-9][a-z0-9\-_/\.]*[a-z0-9]|[a-z0-9]")
ENT = re.compile(r"\b(?:[A-Z][A-Za-z]{2,}(?:[ -][A-Z][A-Za-z]{2,})+|[A-Z]{2,}[-_ ]?\d[\w\-]*|[A-Za-z]{2,}-\d[\w\-]*|[A-Z][a-z]+[A-Z]\w+|\d[\d,\.]*\s?(?:USD|EUR|GBP|kg|km|mg|ms|%))\b")
COMMON = set("""The This That These Those There Their They Then When Where Which While With Without Will Would Should Could
Customer Support Policy Order Account Please Question Option Answer State Tier Team Manager Section Rule Rules
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August
September October November December Yes False True None Null Note Notes Version Standard Premium Basic Enterprise""".split())


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def norm(s: str) -> str:
    return " ".join(WORD.findall(s.lower()))


def flat(o) -> str:
    if isinstance(o, dict):
        return " ".join(flat(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return " ".join(flat(v) for v in o)
    return "" if o is None else str(o)


def shingles(text: str) -> set[str]:
    w = norm(text).split()
    return {h(" ".join(w[i:i + SH])) for i in range(max(0, len(w) - SH + 1))}


def entities(text: str) -> set[str]:
    out = set()
    for m in ENT.findall(text):
        m = m.strip()
        if m.split()[0] in COMMON:
            continue
        out.add(h(m.lower()))
    return out


def item_text(it: dict) -> tuple[str, str]:
    """(state text, everything else)."""
    q = it.get("question", {})
    state = flat(it.get("state"))
    rest = " ".join([flat(q), flat(it.get("labels")), flat(it.get("provenance", {}).get("rationale"))])
    return state, rest


def build(args):
    items = []
    for f in sorted(glob.glob(f"{args.bench}/*.jsonl")):
        items += [json.loads(l) for l in open(f) if l.strip()]
    fps = []
    ent_df = Counter()
    for it in items:
        st, rest = item_text(it)
        e = entities(st + " " + rest)
        ent_df.update(e)
        fps.append({"id": h(it["id"]), "state_hash": h(norm(st)), "shingles": sorted(shingles(st + " " + rest)), "entities": sorted(e)})
    json.dump({"n_items": len(fps), "shingle_size": SH, "items": fps}, open(args.out, "w"))
    print(f"fingerprinted {len(fps)} benchmark items -> {args.out}")


def check(args):
    fp = json.load(open(args.fp))
    exact = {i["state_hash"]: i["id"] for i in fp["items"]}
    sh_idx: dict[str, list[int]] = {}
    for k, i in enumerate(fp["items"]):
        for s in i["shingles"]:
            sh_idx.setdefault(s, []).append(k)
    ent_idx: dict[str, set[int]] = {}
    for k, i in enumerate(fp["items"]):
        for e in i["entities"]:
            ent_idx.setdefault(e, set()).add(k)
    rows = []
    for path in args.files:
        for ln, line in enumerate(open(path)):
            if line.strip():
                rows.append((path, ln, json.loads(line)))
    # entities frequent in the checked corpus are generic vocabulary, not benchmark-specific names
    corpus_df = Counter()
    for _, _, r in rows:
        corpus_df.update(entities(flat(r.get("state")) + " " + flat(r.get("question"))))
    rare_cap = max(3, int(args.corpus_rare * len(rows)))
    bad = Counter()
    ex = []
    n = len(rows)
    for path, ln, r in rows:
        if True:
            st = flat(r.get("state"))
            full = st + " " + flat(r.get("question"))
            reasons = []
            if h(norm(st)) in exact:
                reasons.append("exact")
            rs = shingles(full)
            if rs:
                hits = Counter(k for s in rs if s in sh_idx for k in sh_idx[s])
                if hits:
                    k, c = hits.most_common(1)[0]
                    tot = len(fp["items"][k]["shingles"]) or 1
                    if c / len(rs) >= args.containment or c / tot >= args.containment:
                        reasons.append(f"shingle({c}/{len(rs)})")
            re_ = entities(full)
            shared = [e for e in re_ if e in ent_idx and len(ent_idx[e]) <= args.max_entity_items and corpus_df[e] <= rare_cap]
            if len(shared) >= args.min_entities:
                reasons.append(f"entity({len(shared)})")
            if reasons:
                for x in reasons:
                    bad[x.split("(")[0]] += 1
                if len(ex) < 10:
                    ex.append((path, ln, r.get("id"), reasons))
    print(f"checked {n} rows against {fp['n_items']} benchmark items: contaminated={sum(bad.values())} by_signal={dict(bad)}")
    for e in ex:
        print("  ", e)
    if args.fail and bad:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--bench", required=True)
    b.add_argument("--out", required=True)
    c = sp.add_parser("check")
    c.add_argument("--fp", required=True)
    c.add_argument("files", nargs="+")
    c.add_argument("--containment", type=float, default=0.10)
    c.add_argument("--min-entities", type=int, default=3)
    c.add_argument("--max-entity-items", type=int, default=3)
    c.add_argument("--corpus-rare", type=float, default=0.002)
    c.add_argument("--fail", action="store_true")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else check(a)


if __name__ == "__main__":
    main()
