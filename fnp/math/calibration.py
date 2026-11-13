"""Calibration utilities: temperature scaling, ECE, Brier, reliability diagram, Wilson CI,
risk-coverage curve."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from scipy.stats import norm


def _logp(probs):
    return np.log(np.clip(np.asarray(probs, float), 1e-12, 1.0))


def apply_temperature(probs, T: float) -> np.ndarray:
    """softmax(log p / T): T > 1 softens, T < 1 sharpens."""
    return softmax(_logp(probs) / T, axis=1)


def fit_temperature(probs, y) -> float:
    lp, y = _logp(probs), np.asarray(y, int)

    def nll(T):
        return -log_softmax(lp / T, axis=1)[np.arange(len(y)), y].mean()

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def reliability_diagram(probs, y, n_bins: int = 15) -> dict:
    P, y = np.asarray(probs, float), np.asarray(y, int)
    conf, pred = P.max(axis=1), P.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    sc = np.bincount(idx, weights=conf, minlength=n_bins)
    sa = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"edges": edges, "count": cnt, "confidence": sc / cnt, "accuracy": sa / cnt}


def ece(probs, y, n_bins: int = 15) -> float:
    r = reliability_diagram(probs, y, n_bins)
    m = r["count"] > 0
    return float(np.sum(r["count"][m] / r["count"].sum() * np.abs(r["accuracy"][m] - r["confidence"][m])))


def brier(probs, y) -> float:
    P, y = np.asarray(probs, float), np.asarray(y, int)
    oh = np.eye(P.shape[1])[y]
    return float(np.mean(np.sum((P - oh) ** 2, axis=1)))


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def risk_coverage(probs, y) -> dict:
    """Selective-prediction curve: sort by confidence, risk = error rate of the kept prefix."""
    P, y = np.asarray(probs, float), np.asarray(y, int)
    order = np.argsort(-P.max(axis=1), kind="stable")
    err = (P.argmax(axis=1) != y)[order].astype(float)
    n = np.arange(1, len(y) + 1)
    risk = np.cumsum(err) / n
    return {"coverage": n / len(y), "risk": risk, "aurc": float(risk.mean())}
