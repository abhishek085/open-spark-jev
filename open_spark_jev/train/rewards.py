"""Reward functions for the RLCD phase.

All rewards operate on a *decision distribution* ``p`` over a question's labels, an
optional ground-truth label / posterior, and the question + record metadata. They return a
scalar in roughly [-1, 1]. They are composable: ``CompositeReward`` sums weighted terms.

Proper scoring rules (calibration-driven)
-----------------------------------------
* ``brier_reward``     : 1 - sum_k (p_k - y_k)^2 in [−1, 1]. Against the *posterior* when a
                         simulator provides one, else against the one-hot label.
* ``log_reward``       : log p[y] clipped at -5, rescaled to [-1, 1]. Sharper than Brier;
                         use with care on noisy labels.

Principle-driven
----------------
* ``abstain_reward``   : rewards abstaining when the posterior is flat (max < tau) and
                         penalises abstaining when it is peaked; zero when no abstain option.
* ``conservative_reward``: for questions tagged with a ``safe_label`` in meta (e.g. "escalate",
                         "page_oncall", "block"), penalises confident *unsafe* answers when the
                         true label is the safe one, more than the reverse.
* ``injection_reward`` : for records flagged ``meta.injected``, rewards agreement between the
                         decision on the injected state and on the clean state (needs
                         ``clean_probs``). This is the direct anti-prompt-injection signal.
* ``rm_reward``        : a learned reward model score in [0, 1] (see rlcd.py), rescaled.

The composite reward used in configs/train/rlcd_contrastive.yaml (mechanism 1) and
configs/train/rlcd_direct.yaml (mechanism 2, rm weight forced to 0) is:
    R = 1.0*brier + 0.3*abstain + 0.3*conservative + 0.5*injection + 0.5*rm
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

RewardFn = Callable[..., float]


def _onehot(n: int, i: int) -> list[float]:
    v = [0.0] * n
    v[i] = 1.0
    return v


def brier_reward(p: Sequence[float], labels: Sequence[str], target_label: str | None = None, target_dist: Sequence[float] | None = None, **_) -> float:
    if target_dist is None:
        if target_label is None:
            return 0.0
        target_dist = _onehot(len(labels), list(labels).index(target_label))
    return 1.0 - sum((a - b) ** 2 for a, b in zip(p, target_dist))


def log_reward(p: Sequence[float], labels: Sequence[str], target_label: str | None = None, **_) -> float:
    if target_label is None:
        return 0.0
    lp = math.log(max(p[list(labels).index(target_label)], 1e-9))
    lp = max(lp, -5.0)
    return 1.0 + 2.0 * lp / 5.0  # log p = 0 -> 1, log p = -5 -> -1


def abstain_reward(p: Sequence[float], labels: Sequence[str], target_dist: Sequence[float] | None = None, tau: float = 0.5, **_) -> float:
    if "abstain" not in labels:
        return 0.0
    ai = list(labels).index("abstain")
    if target_dist is None:
        return 0.0
    non_abstain = [d for i, d in enumerate(target_dist) if i != ai]
    ambiguous = max(non_abstain) < tau * sum(non_abstain) if sum(non_abstain) > 0 else True
    pa = p[ai]
    return (pa - (1 - pa)) if ambiguous else ((1 - pa) - pa)


def conservative_reward(p: Sequence[float], labels: Sequence[str], target_label: str | None = None, safe_label: str | None = None, **_) -> float:
    if safe_label is None or safe_label not in labels or target_label is None:
        return 0.0
    si = list(labels).index(safe_label)
    if target_label == safe_label:
        # Missing a case that needed the safe action: penalise proportional to unsafe mass.
        return -(1.0 - p[si]) * 1.5
    # Safe action taken when not needed: mild penalty (cost of over-escalation).
    return -p[si] * 0.5


def injection_reward(p: Sequence[float], labels: Sequence[str], clean_probs: Sequence[float] | None = None, injected: bool = False, **_) -> float:
    if not injected or clean_probs is None:
        return 0.0
    tv = 0.5 * sum(abs(a - b) for a, b in zip(p, clean_probs))  # total variation in [0, 1]
    return 1.0 - 2.0 * tv


def rm_reward(p: Sequence[float], labels: Sequence[str], rm_score: float | None = None, **_) -> float:
    if rm_score is None:
        return 0.0
    return 2.0 * rm_score - 1.0


@dataclass
class CompositeReward:
    terms: dict[str, tuple[RewardFn, float]] = field(default_factory=dict)

    @classmethod
    def from_weights(cls, weights: dict[str, float]) -> CompositeReward:
        fns = {
            "brier": brier_reward, "log": log_reward, "abstain": abstain_reward,
            "conservative": conservative_reward, "injection": injection_reward, "rm": rm_reward,
        }
        return cls({k: (fns[k], w) for k, w in weights.items() if w})

    def __call__(self, **kw) -> tuple[float, dict[str, float]]:
        parts = {name: fn(**kw) for name, (fn, _) in self.terms.items()}
        total = sum(self.terms[name][1] * v for name, v in parts.items())
        return total, parts


SAFE_LABELS = {"moderation": "escalate", "incident": "page_oncall", "risk": "critical", "api_trace": "no", "security": "yes"}
