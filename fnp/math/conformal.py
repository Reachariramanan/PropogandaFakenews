"""Split-conformal prediction sets (LAC and APS) over schema.DECIDABLE, and the label
decision that turns a set into a verdict label.

LAC score: 1 - p_y.  APS score: sum of probs of classes ranked at or above y (optionally
randomised with a seeded U). qhat = the ceil((n+1)(1-alpha))/n empirical quantile
(method="higher"), giving marginal coverage >= 1-alpha under exchangeability.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from fnp.config import C
from fnp.schema import DECIDABLE


def _scores(P: np.ndarray, y: np.ndarray, method: str, u: Optional[np.ndarray]) -> np.ndarray:
    if method == "lac":
        return 1.0 - P[np.arange(len(y)), y]
    order = np.argsort(-P, axis=1, kind="stable")
    sortedP = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(sortedP, axis=1)
    rank = np.argmax(order == y[:, None], axis=1)
    s = cum[np.arange(len(y)), rank]
    if u is not None:
        s = s - u * sortedP[np.arange(len(y)), rank]
    return s


class SplitConformal:
    def __init__(self, method: str = "lac", alpha: Optional[float] = None,
                 randomized: bool = False, seed: int = 0, labels: Sequence[str] = DECIDABLE):
        assert method in ("lac", "aps")
        self.method, self.randomized, self.seed = method, randomized, seed
        self.alpha = C("conformal_alpha") if alpha is None else alpha
        self.labels = tuple(labels)
        self.qhat: Optional[float] = None

    def _u(self, n: int, offset: int) -> Optional[np.ndarray]:
        if self.method == "aps" and self.randomized:
            return np.random.default_rng(self.seed + offset).random(n)
        return None

    def fit(self, cal_probs, cal_y) -> "SplitConformal":
        P = np.asarray(cal_probs, float)
        y = np.array([self.labels.index(v) if isinstance(v, str) else int(v) for v in cal_y])
        s = _scores(P, y, self.method, self._u(len(y), 0))
        n = len(s)
        lev = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = float(np.quantile(s, lev, method="higher"))
        return self

    def predict_mask(self, probs) -> np.ndarray:
        if self.qhat is None:
            raise RuntimeError("fit first")
        P = np.atleast_2d(np.asarray(probs, float))
        K = P.shape[1]
        if self.method == "lac":
            return (1.0 - P) <= self.qhat
        u = self._u(len(P), 1)
        mask = np.zeros_like(P, dtype=bool)
        for k in range(K):
            yk = np.full(len(P), k)
            mask[:, k] = _scores(P, yk, "aps", u) <= self.qhat
        return mask

    def predict_set(self, probs) -> list[list[str]]:
        return [[self.labels[k] for k in np.nonzero(row)[0]] for row in self.predict_mask(probs)]


def decide_label(probs: dict[str, float], pred_set: Optional[Sequence[str]]) -> dict:
    """Primary label, secondary tags and plausibly_fake from probs over DECIDABLE (+ set).

    * no set given          -> argmax
    * |set| == 1            -> that label
    * set == {FAKE, PROPAGANDA} -> FAKE primary, PROPAGANDA secondary (labels co-occur)
    * |set| > 1 otherwise, or empty set -> INCONCLUSIVE
    plausibly_fake: label != FAKE and P(FAKE)+P(MISLEADING) >= C('plausible_low'); the tier
    is "likely" at >= C('plausible_high'), else "plausible".
    secondary: other decidable labels in the set or with P >= M('secondary_min').
    """
    from fnp.math.constants import M
    dp = {k: float(probs.get(k, 0.0)) for k in DECIDABLE}
    if pred_set is None:
        label = max(dp, key=dp.get)
    else:
        ps = set(pred_set)
        if len(ps) == 1:
            label = next(iter(ps))
        elif ps == {"FAKE", "PROPAGANDA"}:
            label = "FAKE"
        else:
            label = "INCONCLUSIVE"
    sec = sorted(k for k in DECIDABLE if k != label and
                 (dp[k] >= M("secondary_min") or (pred_set is not None and k in pred_set and label != "INCONCLUSIVE")))
    pfake = dp["FAKE"] + dp["MISLEADING"]
    plaus = label != "FAKE" and pfake >= C("plausible_low")
    tier = "" if not plaus else ("likely" if pfake >= C("plausible_high") else "plausible")
    return {"label": label, "secondary": sec, "plausibly_fake": plaus, "fake_tier": tier,
            "p_fake_or_misleading": pfake}
