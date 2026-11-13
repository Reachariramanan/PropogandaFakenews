"""Myopic value-of-information stopping for evidence acquisition (pure numpy).

Belief: log-odds L of "claim true". The next evidence unit has stance s in {support, refute, neutral}
with predictive probs `stance_probs` and likelihood ratios `lr` (LR>1 for support, <1 for refute,
1 for neutral). Posterior after s: L + log lr_s.  Bayes-risk of acting now with 0-1 loss is
min(p, 1-p); EVOI = risk_now - E_s[risk_after] >= 0. Continue while EVOI > cost.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def bayes_risk(p_true: float) -> float:
    return float(min(p_true, 1.0 - p_true))


def entropy_bits(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def predictive_stance_probs(p_true: float, sens: Sequence[float]) -> np.ndarray:
    """P(s) = p*P(s|true) + (1-p)*P(s|false); `sens` = (P(s|true), P(s|false)) rows stacked as [S,2]."""
    sens = np.asarray(sens, float)
    return p_true * sens[:, 0] + (1 - p_true) * sens[:, 1]


def evoi(log_odds: float, stance_probs: Sequence[float], lr: Sequence[float], metric: str = "risk") -> float:
    """Expected reduction in Bayes risk (or entropy bits) from observing one more evidence unit."""
    ps = np.asarray(stance_probs, float)
    ps = ps / ps.sum()
    post = sigmoid(log_odds + np.log(np.asarray(lr, float)))
    f = bayes_risk if metric == "risk" else entropy_bits
    return max(0.0, f(float(sigmoid(log_odds))) - float(sum(w * f(float(p)) for w, p in zip(ps, post))))


def should_continue(log_odds: float, stance_probs, lr, cost: float, metric: str = "risk") -> bool:
    return evoi(log_odds, stance_probs, lr, metric) > cost
