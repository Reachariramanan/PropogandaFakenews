"""Meta-analytic pooling of evidence (frameworks note Method 10).

Effect per informative evidence unit (support/refute only):
  q_i = reliability*relevance*stance_conf (strength WITHOUT independence), capped at ds cap
  y_i = dir_i * logit(0.5 + 0.5*q_i)          (log-odds that the claim is true, per unit)
  w_i = reliability*independence*relevance     (methodology note's precision proxy)
  v_i = 1 / (eps + scale*w_i)
Independence enters only the precision, so k copies of one origin (w/k each) give the same
pooled estimate and SE as one copy. Fixed-effect and DerSimonian-Laird random effects;
CI bounds are mapped through the sigmoid to p_true_lo / p_true_hi.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from fnp.config import C
from fnp.math._util import fnum, logit, sigmoid
from fnp.math.constants import M
from fnp.math.reliability import type_prior_mean
from fnp.math.woe import DIRECTION


def fixed_and_random(y: Sequence[float], v: Sequence[float], z: float | None = None) -> dict:
    """Inverse-variance fixed effect + DerSimonian-Laird random effects."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    z = M("meta_z") if z is None else z
    k = y.size
    if k == 0:
        nan = float("nan")
        return {"k": 0, "fe": nan, "fe_se": nan, "re": nan, "re_se": nan, "re_lo": nan,
                "re_hi": nan, "fe_lo": nan, "fe_hi": nan, "Q": nan, "tau2": nan, "I2": nan}
    w = 1.0 / v
    fe = float(np.sum(w * y) / np.sum(w))
    fe_se = float(np.sqrt(1.0 / np.sum(w)))
    Q = float(np.sum(w * (y - fe) ** 2))
    df = k - 1
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - df) / c) if df > 0 and c > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q) if df > 0 and Q > 0 else 0.0
    ws = 1.0 / (v + tau2)
    re = float(np.sum(ws * y) / np.sum(ws))
    re_se = float(np.sqrt(1.0 / np.sum(ws)))
    return {"k": k, "fe": fe, "fe_se": fe_se, "fe_lo": fe - z * fe_se, "fe_hi": fe + z * fe_se,
            "re": re, "re_se": re_se, "re_lo": re - z * re_se, "re_hi": re + z * re_se,
            "Q": Q, "tau2": tau2, "I2": I2}


def evidence_effects(evidence: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    cap, eps, sc = C("ds_strength_cap"), M("meta_eps"), M("meta_precision_scale")
    ys, vs = [], []
    for e in evidence:
        d = DIRECTION.get(e.get("stance"), 0.0)
        if d == 0:
            continue
        rel = fnum(e.get("reliability"), type_prior_mean(e.get("evidence_type")))
        ind, rlv = fnum(e.get("independence"), 1.0), fnum(e.get("relevance"), 1.0)
        q = min(rel * rlv * fnum(e.get("stance_conf"), 1.0), cap)
        ys.append(d * float(logit(0.5 + 0.5 * q)))
        vs.append(1.0 / (eps + sc * rel * ind * rlv))
    return np.array(ys), np.array(vs)


def meta_pool(evidence: Sequence[dict]) -> dict:
    """Pooled log-odds + CI; p_true_lo/hi = sigmoid(RE CI). No evidence -> [0, 1]."""
    y, v = evidence_effects(evidence)
    r = fixed_and_random(y, v)
    if r["k"] == 0:
        r.update(p_true_lo=0.0, p_true_hi=1.0, p_pooled=float("nan"))
    else:
        r.update(p_true_lo=float(sigmoid(r["re_lo"])), p_true_hi=float(sigmoid(r["re_hi"])),
                 p_pooled=float(sigmoid(r["re"])))
    return r
