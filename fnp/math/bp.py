"""Loopy sum-product BP on a pairwise binary MRF over claims (frameworks note Method 1).

Node potential phi_i(x) = [1-p_i, p_i]; edge (i, j, sign, strength) has Ising potential
psi(x_i, x_j) = exp(J * s_i * s_j), s = 2x-1, J = scale*sign*strength. sign=+1 entails
(agreement favoured), -1 contradicts. Exact on trees; approximate with loops.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from fnp.math.constants import M


def _psi(J: float) -> np.ndarray:
    return np.exp(J * np.array([[1.0, -1.0], [-1.0, 1.0]]))


def loopy_bp(p: Sequence[float], edges: Sequence[tuple], damping: float | None = None,
             max_iter: int | None = None, tol: float | None = None) -> dict:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    n = p.size
    phi = np.stack([1 - p, p], axis=1)
    d = M("bp_damping") if damping is None else damping
    max_iter = int(M("bp_max_iter") if max_iter is None else max_iter)
    tol = M("bp_tol") if tol is None else tol
    sc = M("bp_coupling_scale")
    # directed messages: k -> (src, dst, psi[src_state, dst_state])
    src, dst, psis = [], [], []
    for (i, j, sign, w) in edges:
        P = _psi(sc * sign * w)
        src += [i, j]; dst += [j, i]; psis += [P, P.T]
    src, dst, psis = np.array(src, int), np.array(dst, int), np.array(psis).reshape(-1, 2, 2)
    m = np.full((len(src), 2), 0.5)
    rev = np.arange(len(src)) ^ 1               # paired reverse message index
    converged, it = len(src) == 0, 0
    for it in range(1, max_iter + 1):
        if not len(src):
            break
        logb = np.log(phi).copy()
        np.add.at(logb, dst, np.log(m))
        # cavity: belief at src minus message dst->src
        cav = np.exp(logb[src] - np.log(m[rev]))
        new = np.einsum("ka,kab->kb", cav, psis)
        new /= new.sum(axis=1, keepdims=True)
        new = d * m + (1 - d) * new
        delta = np.abs(new - m).max()
        m = new
        if delta < tol:
            converged = True
            break
    logb = np.log(phi).copy()
    if len(src):
        np.add.at(logb, dst, np.log(m))
    b = np.exp(logb - logb.max(axis=1, keepdims=True))
    b /= b.sum(axis=1, keepdims=True)
    return {"marginals": b[:, 1], "converged": bool(converged), "n_iter": it}


def exact_marginals(p: Sequence[float], edges: Sequence[tuple]) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    n = p.size
    if n > 12:
        raise ValueError("exact enumeration limited to <= 12 nodes")
    sc = M("bp_coupling_scale")
    X = np.array(list(itertools.product([0, 1], repeat=n)))
    logw = np.where(X == 1, np.log(p), np.log(1 - p)).sum(axis=1)
    S = 2 * X - 1
    for (i, j, sign, w) in edges:
        logw += sc * sign * w * S[:, i] * S[:, j]
    wts = np.exp(logw - logw.max())
    return (wts[:, None] * X).sum(axis=0) / wts.sum()
