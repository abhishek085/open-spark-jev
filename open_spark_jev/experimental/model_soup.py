"""A13 (docs/NOVELTY.md): weight-space averaging ("model soup") of sibling checkpoints.

Averaging the merged weights of two (or more) checkpoints trained from the same base on the same
data, differing only in a training hyperparameter, usually improves calibration for free -- no
extra training, no extra inference cost, unlike output ensembling (which pays N forward passes).

v7.2-4b and v7.3-4b are true siblings: identical base, identical data (25,568 rows), identical
LoRA rank/schedule -- the only difference is lambda_brier (0.5 vs 3.0). That makes them the
cleanest pair in the ladder to soup, exactly as docs/NOVELTY.md's A13 entry proposes.

This is a lossy edit like truncate.py, so it always writes a NEW checkpoint and never touches
the sources. Calibration temperatures are NOT inherited (averaging weights invalidates any
source checkpoint's fitted T) -- refit with eval/osdg.py before shipping.

  python -m open_spark_jev.experimental.model_soup \
      --models checkpoints/v7.2-4b checkpoints/v7.3-4b --out checkpoints/v7.6-4b
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="+", required=True, help="two or more sibling checkpoint dirs to average"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if len(a.models) < 2:
        raise SystemExit("--models needs at least two checkpoints to average")

    with open(os.path.join(a.models[0], "config.json")) as f:
        cfg0 = json.load(f)
    for m in a.models[1:]:
        with open(os.path.join(m, "config.json")) as f:
            if json.load(f) != cfg0:
                raise SystemExit(
                    f"{m} has a different config.json than {a.models[0]} -- not true siblings, refusing to soup"
                )

    state_dicts = [load_file(os.path.join(m, "model.safetensors")) for m in a.models]
    keys0 = set(state_dicts[0])
    for m, sd in zip(a.models[1:], state_dicts[1:]):
        if set(sd) != keys0:
            raise SystemExit(f"{m} has different parameter keys than {a.models[0]} -- refusing to soup")

    averaged = {}
    for k in keys0:
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        averaged[k] = stacked.mean(dim=0).to(state_dicts[0][k].dtype)

    os.makedirs(a.out, exist_ok=True)
    save_file(averaged, os.path.join(a.out, "model.safetensors"))
    for f in (
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        src = os.path.join(a.models[0], f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, f))

    meta = {
        "sources": a.models,
        "method": "uniform weight average of merged checkpoints",
        "note": "Calibration temperatures are NOT inherited -- averaging weights invalidates any "
        "source checkpoint's fitted T. Refit with eval/osdg.py before relying on confidences.",
    }
    json.dump(meta, open(os.path.join(a.out, "soup.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
