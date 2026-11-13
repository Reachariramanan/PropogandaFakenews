"""Coordination detection by non-negative CP on source x claim x time x evidence-type counts
(frameworks note Method 9; Kolda & Bader 2009).

For each component r with factors (a_r sources, b_r claims, c_r time, d_r types):
  conc_r  = 1 - H(a_r)/log(S)              concentration on few sources
  peak_r  = 1 - H(c_r)/log(T)              temporal peakiness
  multi_r = min(1, exp(H(a_r)) - 1)        gate: a single-source component is not coordination
  score   = max_r conc_r * peak_r * multi_r
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac_hals

from fnp.math.constants import M


def build_tensor(records: Sequence[tuple], shape: tuple[int, int, int, int]) -> np.ndarray:
    """records: (source_idx, claim_idx, time_idx, type_idx[, count])."""
    X = np.zeros(shape)
    for r in records:
        X[r[0], r[1], r[2], r[3]] += r[4] if len(r) > 4 else 1.0
    return X


def _norm_entropy(v: np.ndarray) -> tuple[float, float]:
    p = np.clip(v, 0, None)
    tot = p.sum()
    if tot <= 0 or v.size < 2:
        return 1.0, 1.0
    p = p / tot
    H = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return H / np.log(v.size), float(np.exp(H))


def coordination_score(X: np.ndarray, rank: int | None = None, n_iter: int | None = None) -> dict:
    rank = int(M("cp_rank") if rank is None else rank)
    n_iter = int(M("cp_n_iter") if n_iter is None else n_iter)
    if X.sum() <= 0:
        return {"score": 0.0, "component": -1, "components": [], "top_sources": []}
    # Deterministic restarts (SVD + seeded random); keep the lowest reconstruction error so
    # the result does not depend on thread scheduling inside the SVD init.
    T = tl.tensor(X.astype(float))
    best_cp, best_err = None, np.inf
    for init, seed in (("svd", None), ("random", 0), ("random", 1)):
        kw = {"random_state": seed} if seed is not None else {}
        cp = non_negative_parafac_hals(T, rank=rank, n_iter_max=n_iter, init=init, **kw)
        err = float(np.linalg.norm(X - np.asarray(tl.cp_to_tensor(cp))))
        if err < best_err:
            best_cp, best_err = cp, err
    A, B, Ct, D = [np.asarray(f) for f in best_cp.factors]
    comps = []
    for r in range(rank):
        hs, n_src = _norm_entropy(A[:, r])
        ht, _ = _norm_entropy(Ct[:, r])
        s = (1 - hs) * (1 - ht) * min(1.0, n_src - 1.0) if A[:, r].sum() > 0 else 0.0
        comps.append(max(0.0, s))
    best = int(np.argmax(comps))
    a = A[:, best]
    top = np.where(a >= 0.5 * a.max())[0].tolist() if a.max() > 0 else []
    return {"score": float(comps[best]), "component": best, "components": comps,
            "top_sources": top, "top_times": np.argsort(-Ct[:, best])[:3].tolist()}
