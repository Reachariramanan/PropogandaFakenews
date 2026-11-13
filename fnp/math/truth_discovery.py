"""Truth discovery over a sources x claims matrix of categorical reports.

* Dawid & Skene (1979) EM: latent true class z_i in {0..K-1}, each source s has a confusion
  matrix pi_s[k, j] = P(source reports j | truth k). Laplace-smoothed M-step, majority-vote init.
* TruthFinder (Yin, Han & Yu 2008): iterative trust <-> fact confidence with dampening gamma
  and inter-fact implication rho.

Observation matrix convention: `obs` int array (n_sources, n_items), -1 = no report.
`stance_matrix` turns per-claim evidence lists into this matrix (support=0, refute=1,
context_mismatch=2, neutral -> -1 because it says nothing about truth).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.special import logsumexp

from fnp.math.constants import M

STANCE_CODE = {"support": 0, "refute": 1, "context_mismatch": 2}


def stance_matrix(evidence_by_claim: dict[str, Sequence[dict]]) -> tuple[list[str], list[str], np.ndarray]:
    """(sources, claim_ids, obs). If a source gives several stances on one claim, the one
    with the highest stance_conf wins."""
    claims = sorted(evidence_by_claim)
    sources = sorted({ev.get("source") or "?" for evs in evidence_by_claim.values() for ev in evs})
    si = {s: i for i, s in enumerate(sources)}
    obs = -np.ones((len(sources), len(claims)), dtype=int)
    best = np.full(obs.shape, -1.0)
    for j, c in enumerate(claims):
        for ev in evidence_by_claim[c]:
            code = STANCE_CODE.get(ev.get("stance"))
            if code is None:
                continue
            i = si[ev.get("source") or "?"]
            conf = float(ev.get("stance_conf", 1.0) or 0.0)
            if conf > best[i, j]:
                best[i, j], obs[i, j] = conf, code
    return sources, claims, obs


def _onehot(obs: np.ndarray, J: int) -> np.ndarray:
    S, I = obs.shape
    O = np.zeros((S, I, J))
    s_idx, i_idx = np.nonzero(obs >= 0)
    O[s_idx, i_idx, obs[s_idx, i_idx]] = 1.0
    return O


def dawid_skene(obs: np.ndarray, n_classes: Optional[int] = None, n_labels: Optional[int] = None,
                label_to_class: Optional[Sequence[int]] = None, max_iter: int = 200,
                tol: float = 1e-7, smoothing: Optional[float] = None,
                init_posterior: Optional[np.ndarray] = None) -> dict:
    """Dawid-Skene EM.

    obs: (S, I) ints in 0..J-1, -1 missing. K true classes (default J).
    label_to_class: length-J map from reported label to the class it votes for in the
      majority-vote init (-1 = abstain). Identity when K == J.
    Returns posterior T (I,K), class_prior (K,), confusion (S,K,J), reliability (S,) =
    P(report maps to the true class), loglik trace, converged flag.
    """
    obs = np.asarray(obs, dtype=int)
    S, I = obs.shape
    J = int(n_labels if n_labels is not None else max(obs.max() + 1, 1))
    K = int(n_classes if n_classes is not None else J)
    lam = M("ds_laplace") if smoothing is None else smoothing
    if label_to_class is None:
        if K != J:
            raise ValueError("label_to_class is required when n_classes != n_labels")
        label_to_class = list(range(J))
    l2c = np.asarray(label_to_class, dtype=int)
    O = _onehot(obs, J)                                     # (S,I,J)

    if init_posterior is not None:
        T = np.asarray(init_posterior, dtype=float)
    else:                                                   # majority vote (soft ties)
        votes = np.zeros((I, K))
        cnt = O.sum(axis=0)                                 # (I,J)
        for j in range(J):
            if l2c[j] >= 0:
                votes[:, l2c[j]] += cnt[:, j]
        votes += 1e-3                                       # items with no votes -> uniform
        T = votes / votes.sum(axis=1, keepdims=True)

    ll_trace: list[float] = []
    converged = False
    for _ in range(max_iter):
        # M-step
        prior = (T.sum(axis=0) + lam) / (I + K * lam)
        counts = np.einsum("ik,sij->skj", T, O) + lam      # (S,K,J)
        conf = counts / counts.sum(axis=2, keepdims=True)
        # E-step
        log_like = np.einsum("sij,skj->ik", O, np.log(conf)) + np.log(prior)  # (I,K)
        norm = logsumexp(log_like, axis=1, keepdims=True)
        T = np.exp(log_like - norm)
        ll = float(norm.sum())
        if ll_trace and abs(ll - ll_trace[-1]) < tol * max(1.0, abs(ll)):
            ll_trace.append(ll)
            converged = True
            break
        ll_trace.append(ll)

    # reliability: P(report j maps to the true class), averaged over the class prior
    hit = np.zeros((K, J))
    for j in range(J):
        if l2c[j] >= 0:
            hit[l2c[j], j] = 1.0
    reported = conf * (l2c >= 0)[None, None, :]
    rel = np.einsum("k,skj,kj->s", prior, conf, hit) / np.maximum(
        np.einsum("k,skj->s", prior, reported), 1e-12)
    return {"posterior": T, "class_prior": prior, "confusion": conf, "reliability": rel,
            "loglik": ll_trace, "converged": converged, "n_iter": len(ll_trace)}


def truthfinder(obs: np.ndarray, n_values: Optional[int] = None,
                implication: Optional[np.ndarray] = None, gamma: Optional[float] = None,
                rho: Optional[float] = None, init_trust: Optional[float] = None,
                max_iter: int = 100, tol: float = 1e-6) -> dict:
    """TruthFinder (Yin et al. 2008).

    Facts are (item, value) pairs. With trust t(w) and tau(w) = -ln(1 - t(w)):
      sigma(f)  = sum_{w provides f} tau(w)
      sigma*(f) = sigma(f) + rho * sum_{f' on same item} sigma(f') * imp(f' -> f)
      s(f)      = 1 / (1 + exp(-gamma * sigma*(f)))          (dampened confidence)
      t(w)      = mean_{f provided by w} s(f)
    Stops when 1 - cos(t_new, t_old) < tol. imp defaults to (0 - base_sim) between
    different values of the same item.
    Returns confidence (I,V) of each value (raw TruthFinder s(f), NaN for absent facts),
    posterior (I,V) = confidences renormalised over present values (our convention, the
    paper does not define one), trust (S,), converged, n_iter.
    """
    obs = np.asarray(obs, dtype=int)
    S, I = obs.shape
    V = int(n_values if n_values is not None else obs.max() + 1)
    g = M("tf_gamma") if gamma is None else gamma
    r = M("tf_rho") if rho is None else rho
    t = np.full(S, M("tf_init_trust") if init_trust is None else init_trust)
    if implication is None:
        implication = np.full((V, V), -M("tf_base_sim"))
        np.fill_diagonal(implication, 0.0)
    O = _onehot(obs, V)                                     # (S,I,V)
    present = O.sum(axis=0) > 0                             # (I,V)
    n_prov = O.sum(axis=(1, 2))                             # facts per source
    converged, it = False, 0
    s = np.zeros((I, V))
    for it in range(1, max_iter + 1):
        tau = -np.log(np.clip(1.0 - t, 1e-10, None))
        sigma = np.einsum("s,siv->iv", tau, O)              # (I,V)
        sigma_star = sigma + r * (sigma * present) @ implication   # row: sum_f' sigma(f') imp(f'->f)
        s = 1.0 / (1.0 + np.exp(-g * sigma_star))
        s = np.where(present, s, 0.0)
        t_new = np.einsum("siv,iv->s", O, s) / np.maximum(n_prov, 1)
        t_new = np.where(n_prov > 0, t_new, t)
        cos = t_new @ t / max(np.linalg.norm(t_new) * np.linalg.norm(t), 1e-12)
        t = t_new
        if 1 - cos < tol:
            converged = True
            break
    conf = np.where(present, s, np.nan)
    tot = np.where(present, s, 0.0).sum(axis=1, keepdims=True)
    post = np.where(present, s, 0.0) / np.where(tot > 0, tot, 1.0)
    post[tot[:, 0] == 0] = 1.0 / V
    return {"confidence": conf, "posterior": post, "trust": t, "converged": converged,
            "n_iter": it}
