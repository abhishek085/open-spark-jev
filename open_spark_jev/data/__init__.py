"""Decision corpora: a single JSONL record format shared by real, synthetic and simulated data.

Record schema (one per line)::

    {
      "id": "sim-routing-000123",
      "domain": "routing",                # slicing key
      "source": "simulator|teacher|public:<name>",
      "state": {"content": ..., "schema_hint": ..., "domain": ...},
      "question": {"type": "choice", "prompt": "...", "options": [...], ...},
      "target": {
          "label": "billing",              # hard label (index resolved against question.labels)
          "dist": {"billing": 0.8, ...}    # optional soft target: known posterior or teacher dist
      },
      "meta": {...}                        # anything else (difficulty, injected=true, ...)
    }
"""

from .corpus import Record, iter_records, read_jsonl, write_jsonl

__all__ = ["Record", "iter_records", "read_jsonl", "write_jsonl"]
