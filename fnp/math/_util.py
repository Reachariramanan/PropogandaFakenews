"""Small numeric helpers shared by fnp.math modules."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import expit, logit as _logit

EPS = 1e-12


def sigmoid(x):
    return expit(x)


def logit(p, eps: float = 1e-9):
    return _logit(np.clip(p, eps, 1 - eps))


def fnum(x: Any, default: float = float("nan")) -> float:
    """Float or `default` for None / non-numeric / NaN input."""
    if x is None:
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


def isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def soft_ge(x: float, thr: float, tau: float) -> float:
    """Logistic version of the indicator [x >= thr]; 0.5 exactly at the threshold."""
    return float(expit((x - thr) / tau))


def soft_le(x: float, thr: float, tau: float) -> float:
    return float(expit((thr - x) / tau))


def soft_or(*ps: float) -> float:
    q = 1.0
    for p in ps:
        q *= 1.0 - p
    return 1.0 - q
