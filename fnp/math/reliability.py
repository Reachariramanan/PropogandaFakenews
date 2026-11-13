"""Source reliability as a Beta posterior (methodologoes Part V step 3).

Prior:      r ~ Beta(a0, b0), a0 = n0*m, b0 = n0*(1-m), m = TYPE_RELIABILITY[type] clipped.
Data:       k correct verdicts out of n checked claims from this source (track record).
Posterior:  Beta(a0 + k, b0 + n - k); mean (a0+k)/(n0+n); central credible interval.
Bias hook:  r_eff = r * (1 - lambda*|bias|), bias in [-1, 1] (0 = neutral outlet).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import beta as _beta

from fnp.math.constants import M
from fnp.schema import TYPE_RELIABILITY


def type_prior_mean(evidence_type: Optional[str]) -> float:
    m = TYPE_RELIABILITY.get(evidence_type or "UNKNOWN", TYPE_RELIABILITY["UNKNOWN"])
    c = M("rel_clip")
    return float(np.clip(m, c, 1 - c))


def beta_prior(evidence_type: Optional[str], n0: Optional[float] = None) -> tuple[float, float]:
    n0 = M("rel_prior_n0") if n0 is None else n0
    m = type_prior_mean(evidence_type)
    return n0 * m, n0 * (1 - m)


def reliability_posterior(evidence_type: Optional[str], k: float = 0, n: float = 0,
                          n0: Optional[float] = None, level: Optional[float] = None) -> dict:
    """Posterior mean and equal-tailed credible interval of a source's reliability."""
    if k < 0 or n < k:
        raise ValueError("need 0 <= k <= n")
    a0, b0 = beta_prior(evidence_type, n0)
    a, b = a0 + k, b0 + (n - k)
    lev = M("rel_cred_level") if level is None else level
    lo, hi = _beta.ppf([(1 - lev) / 2, (1 + lev) / 2], a, b)
    return {"mean": a / (a + b), "lo": float(lo), "hi": float(hi), "a": a, "b": b,
            "n_obs": n}


def bias_discount(reliability: float, bias: Optional[float], lam: Optional[float] = None) -> float:
    if bias is None or not np.isfinite(bias):
        return float(reliability)
    lam = M("bias_discount") if lam is None else lam
    return float(reliability * (1.0 - lam * min(abs(bias), 1.0)))


def source_reliability(evidence_type: Optional[str], track: Optional[tuple[float, float]] = None,
                       bias: Optional[float] = None) -> float:
    k, n = track if track else (0, 0)
    return bias_discount(reliability_posterior(evidence_type, k, n)["mean"], bias)


def assign_reliability(evidence: Sequence[dict], track_records: Optional[dict] = None,
                       bias: Optional[dict] = None, overwrite: bool = False) -> list[dict]:
    """Copies of evidence dicts with `reliability` set from type prior + source track record.

    track_records: {source: (k_correct, n_checked)}; bias: {source: bias in [-1,1]}.
    An existing `reliability` is kept unless overwrite=True.
    """
    track_records = track_records or {}
    bias = bias or {}
    out = []
    for ev in evidence:
        d = dict(ev)
        if overwrite or d.get("reliability") is None or not np.isfinite(d.get("reliability", np.nan)):
            src = d.get("source")
            d["reliability"] = source_reliability(d.get("evidence_type"), track_records.get(src),
                                                  bias.get(src))
        out.append(d)
    return out
