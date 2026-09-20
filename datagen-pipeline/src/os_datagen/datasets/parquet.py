"""Optional materialized analysis format (needs `pip install os-datagen[parquet]`). JSONL stays the primary format."""
from __future__ import annotations

import json
from pathlib import Path

from ..utils.jsonl import read_jsonl


def jsonl_to_parquet(src: Path, dst: Path) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pyarrow is not installed; pip install 'os-datagen[parquet]'") from e
    rows = []
    for r in read_jsonl(src):
        flat = {"record_id": r["record_id"], "task_pack": r["task_pack"], "split": r["split"], "decision_type": r["decision"]["type"],
                "difficulty": r["quality"]["difficulty"], "label_quality": r["truth"]["label_quality"],
                "preferred": r["truth"].get("preferred_option") or str(r["truth"].get("preferred_level") if r["truth"].get("preferred_level") is not None else r["truth"].get("truth")),
                "namespace": r["meta"].get("namespace"), "scenario_family": r["meta"].get("scenario_family"),
                "state_json": json.dumps(r["decision"]["state"], ensure_ascii=False), "question_json": json.dumps(r["decision"]["question"], ensure_ascii=False),
                "truth_json": json.dumps(r["truth"]), "provenance_json": json.dumps(r["provenance"])}
        rows.append(flat)
    pq.write_table(pa.Table.from_pylist(rows), dst)
    return len(rows)
