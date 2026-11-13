"""Contextual linear bandit for routing a claim to {laya, gemma, human}.

Per-arm Bayesian linear regression  r = w_a . x + noise  (prior N(0, I/lam), noise var s2).
`choose` = Thompson sampling (w ~ posterior) or LinUCB (mean + alpha*sqrt(x'A^-1 x)); `explore=False`
uses the posterior mean. Offline replay is full-information: every arm's outcome is logged.

Context x = [laya max-prob, laya entropy, top1-top2 margin, ds_uncertainty, n_eff, p_prop,
kb_match_score, 1]. Reward = log-score of the arm's decision on gold (human = gold, log-loss 0)
minus cost. COSTS are ASSUMED (laya 0.001, gemma 0.05, human 1.0; normalised tokens/latency/labour).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ARMS = ("laya", "gemma", "human")
COSTS = {"laya": 0.001, "gemma": 0.05, "human": 1.0}     # assumed, see 02_ASSUMPTIONS_REGISTER
CONTEXT_NAMES = ("laya_max_prob", "laya_entropy", "laya_margin", "ds_uncertainty", "n_eff", "p_prop",
                 "kb_match_score", "bias")


def context_vector(laya_probs: Sequence[float], ds_uncertainty: float = 0.0, n_eff: float = 0.0,
                   p_prop: float = 0.0, kb_match_score: float = 0.0) -> np.ndarray:
    p = np.asarray(laya_probs, float)
    p = p / p.sum()
    s = np.sort(p)[::-1]
    ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / math.log(len(p)))
    return np.array([s[0], ent, s[0] - s[1], ds_uncertainty, min(n_eff, 10) / 10, p_prop, kb_match_score, 1.0])


def arm_reward(arm: str, probs_on_gold: float) -> float:
    """log-score of the arm's decision on gold minus cost. Human decides = gold -> log-loss 0."""
    ll = 0.0 if arm == "human" else math.log(max(probs_on_gold, 1e-4))
    return ll - COSTS[arm]


class RoutingBandit:
    def __init__(self, dim: int = len(CONTEXT_NAMES), arms: Sequence[str] = ARMS, lam: float = 1.0,
                 noise_var: float = 0.25, alpha: float = 0.5, seed: int = 0, mode: str = "thompson"):
        self.arms, self.dim, self.mode, self.alpha, self.s2, self.lam = tuple(arms), dim, mode, alpha, noise_var, lam
        self.A = {a: lam * np.eye(dim) for a in self.arms}      # precision (unscaled by noise)
        self.b = {a: np.zeros(dim) for a in self.arms}
        self.rng = np.random.default_rng(seed)

    def _post(self, a):
        cov = np.linalg.inv(self.A[a])
        return cov @ self.b[a], cov

    def update(self, x: np.ndarray, arm: str, reward: float) -> None:
        x = np.asarray(x, float)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def scores(self, x: np.ndarray, explore: bool = False) -> dict:
        x = np.asarray(x, float)
        out = {}
        for a in self.arms:
            mu, cov = self._post(a)
            if not explore:
                out[a] = float(mu @ x)
            elif self.mode == "linucb":
                out[a] = float(mu @ x + self.alpha * math.sqrt(max(x @ cov @ x, 0)))
            else:
                w = self.rng.multivariate_normal(mu, self.s2 * cov)
                out[a] = float(w @ x)
        return out

    def choose(self, context: np.ndarray, explore: bool = False) -> str:
        s = self.scores(context, explore)
        return max(self.arms, key=lambda a: s[a])

    def fit_replay(self, contexts: np.ndarray, rewards: np.ndarray, epochs: int = 1, shuffle: bool = True) -> "RoutingBandit":
        """contexts [N,d]; rewards [N, n_arms] (full information, arm order = self.arms): every arm is updated
        on every logged row (offline replay)."""
        n = len(contexts)
        for _ in range(epochs):
            for i in (self.rng.permutation(n) if shuffle else range(n)):
                for j, a in enumerate(self.arms):
                    self.update(contexts[i], a, rewards[i, j])
        return self

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"arms": self.arms, "dim": self.dim, "mode": self.mode, "alpha": self.alpha,
                                          "noise_var": self.s2, "lam": self.lam,
                                          "A": {a: self.A[a].tolist() for a in self.arms},
                                          "b": {a: self.b[a].tolist() for a in self.arms}}))

    @classmethod
    def load(cls, path, seed: int = 0) -> "RoutingBandit":
        d = json.loads(Path(path).read_text())
        m = cls(d["dim"], d["arms"], d["lam"], d["noise_var"], d["alpha"], seed, d["mode"])
        m.A = {a: np.array(v) for a, v in d["A"].items()}
        m.b = {a: np.array(v) for a, v in d["b"].items()}
        return m


def fallback_policy(laya_max_prob: float, conformal_set_size: int = 1, tau: float = 0.7) -> str:
    """Deterministic policy: escalate to gemma if the conformal set is ambiguous or laya is unsure."""
    return "gemma" if conformal_set_size > 1 or laya_max_prob < tau else "laya"
