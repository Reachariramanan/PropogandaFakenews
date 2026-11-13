"""Information gain in bits (frameworks note Method 4): KL(posterior || prior)."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from fnp.math.woe import woe_posterior


def kl_categorical_bits(post, prior) -> float:
    p = np.clip(np.asarray(post, float), 1e-12, 1.0)
    q = np.clip(np.asarray(prior, float), 1e-12, 1.0)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * (np.log2(p) - np.log2(q))))


def kl_bernoulli_bits(p_post: float, p_prior: float) -> float:
    return kl_categorical_bits([p_post, 1 - p_post], [p_prior, 1 - p_prior])


def leave_one_out_bits(evidence: Sequence[dict],
                       posterior_fn: Optional[Callable[[Sequence[dict]], float]] = None) -> np.ndarray:
    """Bits each evidence unit adds given all the others: KL(P_all || P_without_i)."""
    fn = posterior_fn or (lambda evs: woe_posterior(evs)["p_true"])
    p_all = fn(evidence)
    return np.array([kl_bernoulli_bits(p_all, fn(list(evidence[:i]) + list(evidence[i + 1:])))
                     for i in range(len(evidence))])
