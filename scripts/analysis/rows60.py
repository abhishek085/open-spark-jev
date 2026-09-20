"""Per-row predictions of several checkpoints on the 60-row tool-call set, plus Jev's recorded answers."""
import json, sys
from open_spark_jev.data.corpus import read_jsonl
from open_spark_jev.eval.benchmark import run
from open_spark_jev.eval.external import _jev_probs
from open_spark_jev.experimental.variant_scorer import VariantScorer, is_variant
from open_spark_jev.model import MenuScorer

MODELS = {
    "sft-v2": "checkpoints/sft-qwen3-1.7b", "sft-m17": "checkpoints/sft-m17-qwen3-1.7b",
    "a0": "checkpoints/variants/a0", "a0-public": "checkpoints/variants/a0-public",
    "a0-toolcall": "checkpoints/variants4/a0", "a2": "checkpoints/variants/a2",
}
recs = read_jsonl("data/benchmarks/external/ext-toolcall-risk.jsonl")
out = {"rows": []}
for r in recs:
    q = r.question_obj()
    jp = _jev_probs(r, q.labels)
    out["rows"].append({"id": r.id, "slice": r.domain.split("/")[-1], "call": str(r.state["content"]), "gold": q.labels[r.label_index()],
                        "labels": q.labels, "jev": q.labels[max(range(len(jp)), key=jp.__getitem__)] if jp else None,
                        "jev_p": jp, "pred": {}})
for name, path in MODELS.items():
    try:
        sc = VariantScorer(path) if is_variant(path) else MenuScorer(path, use_state_cache=False)
    except Exception as e:  # noqa
        print("skip", name, e, file=sys.stderr); continue
    answers, _ = run(sc, recs, batch=8)
    for row, a in zip(out["rows"], answers):
        row["pred"][name] = {"sel": a.selected, "conf": round(max(a.probs), 3), "probs": [round(p, 3) for p in a.probs]}
    del sc
    import torch; torch.cuda.empty_cache()
    print("done", name, file=sys.stderr)
json.dump(out, open("runs/rows60.json", "w"), indent=1)
