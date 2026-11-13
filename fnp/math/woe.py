"""Weight-of-evidence posterior (methodologoes Part V final formula; frameworks note Method 3).

  s_i  = reliability_i * independence_i * relevance_i * stance_conf_i     in [0,1]
  dir  = +1 support, -1 refute, 0 otherwise
  L    = logit(prior) + kappa * sum_i dir_i * s_i
  P    = sigmoid(L)
context_mismatch strengths are summed into a separate MISLEADING channel and do not move L.
`fit_kappa` fits (kappa, prior) by maximum likelihood on gold labels.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from fnp.config import C
from fnp.math._util import fnum, logit, sigmoid
from fnp.math.reliability import type_prior_mean

DIRECTION = {"support": 1.0, "refute": -1.0, "context_mismatch": 0.0, "neutral": 0.0}


def evidence_strength(ev: dict) -> float:
    """s_i. Missing reliability -> type prior mean; missing independence/relevance/
    stance_conf -> 1.0 (so the caller MUST run assign_independence first)."""
    rel = fnum(ev.get("reliability"), type_prior_mean(ev.get("evidence_type")))
    ind = fnum(ev.get("independence"), 1.0)
    rlv = fnum(ev.get("relevance"), 1.0)
    sc = fnum(ev.get("stance_conf"), 1.0)
    return float(np.clip(rel * ind * rlv * sc, 0.0, 1.0))


def strengths(evidence: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(s, dir, is_mismatch) arrays."""
    s = np.array([evidence_strength(e) for e in evidence], dtype=float)
    d = np.array([DIRECTION.get(e.get("stance"), 0.0) for e in evidence], dtype=float)
    cm = np.array([e.get("stance") == "context_mismatch" for e in evidence], dtype=bool)
    return s, d, cm


def channel_sums(evidence: Sequence[dict]) -> dict:
    s, d, cm = strengths(evidence)
    return {"support": float(s[d > 0].sum()), "contradiction": float(s[d < 0].sum()),
            "context_mismatch": float(s[cm].sum())}


def woe_posterior(evidence: Sequence[dict], prior: Optional[float] = None,
                  kappa: Optional[float] = None) -> dict:
    prior = C("prior_true") if prior is None else prior
    kappa = C("woe_kappa") if kappa is None else kappa
    ch = channel_sums(evidence)
    net = ch["support"] - ch["contradiction"]
    L = float(logit(prior) + kappa * net)
    return {**ch, "net": net, "logit": L, "p_true": float(sigmoid(L)), "prior": prior,
            "kappa": kappa}


def _net_of(rec) -> float:
    if isinstance(rec, dict):
        if "net" in rec:
            return float(rec["net"])
        return float(rec.get("support", 0.0)) - float(rec.get("contradiction", 0.0))
    if isinstance(rec, (list, tuple)) and rec and isinstance(rec[0], dict):
        return woe_posterior(rec)["net"]
    return float(rec)


def fit_kappa(records: Sequence, y: Sequence[int], fit_prior: bool = True,
              l2: float = 1e-3) -> dict:
    """Max-likelihood (kappa >= 0, prior) for P(y=1) = sigmoid(logit(prior) + kappa*net).

    records: per-claim net evidence as floats, dicts with 'net' or 'support'/'contradiction',
    or raw evidence lists. y: 1 = claim true, 0 = false. A tiny L2 penalty keeps the fit
    finite on separable data.
    """
    x = np.array([_net_of(r) for r in records], dtype=float)
    yy = np.asarray(y, dtype=float)
    if x.size == 0:
        raise ValueError("no records")
    p0 = float(np.clip(yy.mean(), 0.02, 0.98))

    def nll(theta):
        b, k = (theta[0], theta[1]) if fit_prior else (logit(C("prior_true")), theta[0])
        z = b + k * x
        ll = yy * z - np.logaddexp(0.0, z)
        return -ll.sum() + l2 * k * k

    if fit_prior:
        res = minimize(nll, x0=[logit(p0), C("woe_kappa")], method="L-BFGS-B",
                       bounds=[(-10, 10), (0.0, 50.0)])
        b, k = res.x
    else:
        res = minimize(nll, x0=[C("woe_kappa")], method="L-BFGS-B", bounds=[(0.0, 50.0)])
        b, k = logit(C("prior_true")), res.x[0]
    return {"kappa": float(k), "prior": float(sigmoid(b)), "nll": float(res.fun),
            "success": bool(res.success), "n": int(x.size)}
