"""A10 / v7.5: score options by their own text instead of by an assigned letter.

The baseline readout assigns option *i* the letter ``A``+i and reads the logit of that letter token.
Three costs come with that binding, and all three are structural rather than a matter of training
longer:

  * **Option order changes the answer.** The letter carries no meaning, so the mapping from position
    to letter is information the model has to use, and it does: option-order sensitivity is a
    standing known limitation, and ``eval/osdg.py`` exists partly to measure it. Permutation
    averaging (``scripts/analysis/calib_v7_study.py``) reduces it at the cost of N forward passes.
  * **It stops at 26 options.** Jev advertises up to 255. There is no 27th bare letter token.
  * **The letter must be bound to the option's meaning inside one forward pass**, from a menu the
    model has only just read.

Scoring the option's own text removes all three. For each option we score the *continuation*
``P(option text | state, question)`` under the same model, with no letters in the prompt and no new
parameters: ``score_i = sum log P(token | ...) / len(tokens)**alpha``. Because every option is scored
against an identical prefix, the result is **permutation-invariant by construction** -- not
approximately, exactly -- and the option count is unbounded.

This is deliberately *not* the A3 mistake. A3 bolted on a freshly initialised cross-attention head and
lost to the baseline because one LoRA epoch cannot train a new head from scratch. Here there is nothing
new to train: the scoring function is the pretrained language model itself, so a zero-shot comparison
against the letter readout is meaningful on day one. That also makes it cheap to refute.

The cost is compute: k options means k short continuations instead of one letter position, batched
after a shared prefix -- a multiplier on the suffix, not on the state.

  python -m open_spark_jev.experimental.option_readout --model checkpoints/v6-4b \
      --items data/synthetic/prog_v1/test_locked.jsonl --limit 400 --out runs/v7/option_readout_v6.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..calibration import ece as ece_fn
from ..data.corpus import read_jsonl
from ..model import MenuScorer
from ..prompting import ASSISTANT_HEADER, render_prefix
from ..schema import Choice, Noul, Question, Score


def option_display(q: Question, label: str) -> str:
    """The words we score, matching how the menu would have described the option."""
    if label == "abstain":
        return "Abstain - the state does not contain enough information to decide"
    if isinstance(q, Noul):
        return {"yes": "Yes, the claim is true", "no": "No, the claim is false"}[label]
    return label


def letterless_question_block(q: Question) -> str:
    """The question, options listed but NOT lettered, and no 'answer with a letter' instruction."""
    lines: list[str] = []
    if isinstance(q, Choice):
        lines += ["### Question (choice)", q.prompt.strip(), "Options:"]
    elif isinstance(q, Score):
        lines += ["### Question (score)", q.prompt.strip()]
        if q.rubric:
            lines += ["Rubric:", q.rubric.strip()]
        lines.append("Levels (ordered from lowest to highest):")
    elif isinstance(q, Noul):
        lines += ["### Question (noul)", "Claim: " + q.prompt.strip(),
                  "Is the claim true of the STATE?", "Options:"]
    else:  # pragma: no cover
        raise TypeError(type(q))
    for label in q.labels:
        lines.append(f"- {option_display(q, label)}")
    lines.append("Answer with the best option.")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def option_text_logprobs(scorer: MenuScorer, state, q: Question, alpha: float = 1.0) -> np.ndarray:
    """Length-normalised log P(option text | state, question) per option, in label order."""
    prefix = scorer._encode(render_prefix(state, max_chars=scorer.max_state_chars))
    head = scorer._encode(f"{letterless_question_block(q)}<|im_end|>\n{ASSISTANT_HEADER}")
    conts = [scorer._encode(option_display(q, l)) for l in q.labels]
    pad = scorer.tokenizer.pad_token_id or 0
    seqs = [prefix + head + c for c in conts]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    ids, mask = ids.to(scorer.device), mask.to(scorer.device)
    logp = torch.log_softmax(scorer.lm(input_ids=ids, attention_mask=mask).logits.float(), -1)
    start = len(prefix) + len(head)
    out = []
    for i, c in enumerate(conts):
        # token c[j] is predicted by the distribution at position start + j - 1
        tot = sum(float(logp[i, start + j - 1, t]) for j, t in enumerate(c))
        out.append(tot / (len(c) ** alpha if alpha else 1.0))
    return np.array(out)


def softmax(z):
    z = np.asarray(z, float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=1.0, help="length-normalisation exponent")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = read_jsonl(a.items)[: a.limit or None]
    scorer = MenuScorer(a.model, use_state_cache=False)
    rows = []
    t0 = time.time()
    for n, r in enumerate(recs):
        q, state = r.question_obj(), r.state_obj()
        gi = q.labels.index(r.target["label"])
        letter = scorer.decide(state, [q], temperature=1.0, return_logits=True)[0]
        lz = letter.raw_logits if letter.raw_logits is not None else list(np.log(np.clip(letter.probs, 1e-12, 1)))
        oz = option_text_logprobs(scorer, state, q, a.alpha)
        rows.append({"id": r.id, "family": r.meta.get("scenario_family", "na"), "gold_idx": gi,
                     "n_labels": len(q.labels), "qtype": q.type,
                     "letter_logits": [round(float(v), 4) for v in lz],
                     "option_text_logprobs": [round(float(v), 4) for v in oz]})
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)

    def report(key: str, scale: float) -> dict:
        probs = [softmax(np.array(r[key]) * scale) for r in rows]
        y = [r["gold_idx"] for r in rows]
        hit = np.array([float(int(np.argmax(p)) == g) for p, g in zip(probs, y)])
        conf = np.array([float(p.max()) for p in probs])
        K = max(len(p) for p in probs)
        P = [list(p) + [0.0] * (K - len(p)) for p in probs]
        return {"accuracy": round(float(hit.mean()), 4), "mean_conf": round(float(conf.mean()), 4),
                "ece": round(float(ece_fn(P, y)), 4)}

    per_family: dict[str, dict] = {}
    for f in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == f]

        def acc(key: str, sub=sub) -> float:
            return round(float(np.mean([int(np.argmax(r[key]) == r["gold_idx"]) for r in sub])), 4)

        per_family[f] = {"n": len(sub), "letter": acc("letter_logits"), "option_text": acc("option_text_logprobs")}

    summary = {"model": a.model, "items": a.items, "n": len(rows), "alpha": a.alpha,
               "letter_readout": report("letter_logits", 1.0),
               "option_text_readout": {str(s): report("option_text_logprobs", s) for s in (1.0, 2.0, 4.0, 8.0)},
               "per_family_accuracy": per_family,
               "note": "option-text scores are length-normalised log-probs, so their natural scale is not a "
                       "logit scale; the sweep over multipliers stands in for a fitted temperature.",
               "seconds": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
