"""Evidence independence (methodologoes Part V step 4; frameworks note, 'counting is the enemy').

Evidence units that share an origin are grouped by union-find over four links:
  * equal `origin_key` (cited filing, KB id, wire story id),
  * equal normalised `url`,
  * a shared entry in an optional `source_links` list (outbound links of the evidence page),
  * text-embedding cosine >= C('dup_cosine').
Each unit then gets ind_i = 1/|cluster|^alpha, so with alpha = 1 a cluster contributes the
weight of exactly one (average) member to any additive fusion.

`copy_probability` is a simplified AccuCopy (Dong, Berti-Equille & Srivastava 2009) Bayes
factor for "source A copies source B" from the values they share.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

import numpy as np

from fnp.config import C
from fnp.math.constants import M


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:          # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def labels(self) -> np.ndarray:
        """Cluster labels 0..k-1 in order of first appearance (deterministic)."""
        roots = [self.find(i) for i in range(len(self.parent))]
        remap: dict[int, int] = {}
        return np.array([remap.setdefault(r, len(remap)) for r in roots], dtype=int)


_URL_RE = re.compile(r"^https?://(www\.)?", re.I)


def normalise_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = _URL_RE.sub("", url.strip()).split("#")[0].rstrip("/").lower()
    u = re.sub(r"[?&](utm_[^=&]+|fbclid|gclid)=[^&]*", "", u)
    return u or None


def cluster_evidence(evidence: Sequence[dict], embeddings: Optional[np.ndarray] = None,
                     cos_threshold: Optional[float] = None) -> np.ndarray:
    """Origin-cluster labels (len(evidence),) via union-find over shared keys and cosine."""
    n = len(evidence)
    uf = UnionFind(n)
    first_seen: dict[tuple[str, str], int] = {}
    for i, ev in enumerate(evidence):
        keys: list[tuple[str, str]] = []
        if ev.get("origin_key"):
            keys.append(("origin", str(ev["origin_key"])))
        nu = normalise_url(ev.get("url"))
        if nu:
            keys.append(("url", nu))
        for link in ev.get("source_links") or ():
            nl = normalise_url(link)
            if nl:
                keys.append(("url", nl))     # a link to page X == being page X's copy
        for k in keys:
            if k in first_seen:
                uf.union(i, first_seen[k])
            else:
                first_seen[k] = i
    if embeddings is not None and n > 1:
        thr = C("dup_cosine") if cos_threshold is None else cos_threshold
        E = np.asarray(embeddings, dtype=float)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        S = E @ E.T
        ii, jj = np.where(np.triu(S >= thr, k=1))
        for a, b in zip(ii.tolist(), jj.tolist()):
            uf.union(a, b)
    return uf.labels()


def independence_weights(labels: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
    """ind_i = 1 / |cluster(i)|^alpha."""
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return np.zeros(0)
    a = M("independence_alpha") if alpha is None else alpha
    sizes = np.bincount(labels)[labels].astype(float)
    return 1.0 / sizes ** a


def assign_independence(evidence: Sequence[dict], embeddings: Optional[np.ndarray] = None,
                        alpha: Optional[float] = None) -> list[dict]:
    """Copies of the evidence dicts with `independence` and `origin_cluster` set."""
    labels = cluster_evidence(evidence, embeddings)
    w = independence_weights(labels, alpha)
    out = []
    for ev, lab, wi in zip(evidence, labels, w):
        d = dict(ev)
        d["independence"] = float(wi)
        d["origin_cluster"] = int(lab)
        out.append(d)
    return out


def kish_n_eff(weights: Iterable[float]) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2; 0 for no / all-zero weights."""
    w = np.asarray(list(weights), dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    return float(w.sum() ** 2 / np.sum(w ** 2))


def cluster_n_eff(labels: np.ndarray, unit_weights: np.ndarray) -> float:
    """Kish n_eff over origin clusters: weights are summed within a cluster first, so k
    copies of one origin count as one unit (plain Kish over copies would say k)."""
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return 0.0
    W = np.bincount(labels, weights=np.asarray(unit_weights, dtype=float))
    return kish_n_eff(W)


# ---------------------------------------------------------------------------
# AccuCopy-style dependence between two sources (Dong et al. 2009, simplified)
# ---------------------------------------------------------------------------
def copy_probability(values_a: dict, values_b: dict, truth: dict,
                     acc_a: float, acc_b: float, n_false: Optional[float] = None,
                     copy_rate: Optional[float] = None, prior: Optional[float] = None) -> dict:
    """Posterior P(A and B are dependent) from the objects both report on.

    For each shared object: k_t = same value and it equals `truth`, k_f = same false value,
    k_d = different values. Under independence (A accuracy a, B accuracy b, n uniform false
    values):  Pt = a*b,  Pf = (1-a)(1-b)/n,  Pd = 1 - Pt - Pf.
    Under copying with rate c (one copies the other's value with prob c; the direction is
    not resolved -- we use the mean accuracy m=(a+b)/2 for the copied value):
      Pt' = m*c + Pt*(1-c),  Pf' = (1-m)*c + Pf*(1-c),  Pd' = Pd*(1-c).
    Bayes: P(dep | obs) = 1 / (1 + (1-prior)/prior * prod(P_indep)/prod(P_dep)).
    Sharing a *false* (rare) value is the strong copy signal because Pf is tiny.
    """
    n = M("copy_n_false") if n_false is None else n_false
    c = M("copy_rate") if copy_rate is None else copy_rate
    alpha = M("copy_prior") if prior is None else prior
    a = float(np.clip(acc_a, 1e-6, 1 - 1e-6))
    b = float(np.clip(acc_b, 1e-6, 1 - 1e-6))
    m = 0.5 * (a + b)
    shared = [o for o in values_a if o in values_b and o in truth]
    kt = sum(1 for o in shared if values_a[o] == values_b[o] == truth[o])
    kf = sum(1 for o in shared if values_a[o] == values_b[o] != truth[o])
    kd = len(shared) - kt - kf
    pt, pf = a * b, (1 - a) * (1 - b) / n
    pd = max(1 - pt - pf, 1e-12)
    pt2, pf2, pd2 = m * c + pt * (1 - c), (1 - m) * c + pf * (1 - c), pd * (1 - c)
    log_bf = (kt * (np.log(pt2) - np.log(pt)) + kf * (np.log(pf2) - np.log(pf))
              + kd * (np.log(pd2) - np.log(pd)))       # log P(obs|dep) - log P(obs|indep)
    log_post_odds = np.log(alpha) - np.log(1 - alpha) + log_bf
    p = float(1.0 / (1.0 + np.exp(-log_post_odds)))
    return {"p_copy": p, "k_true": kt, "k_false": kf, "k_diff": kd, "log_bayes_factor": float(log_bf)}


def pairwise_copy_matrix(source_values: dict[str, dict], truth: dict,
                         accuracy: dict[str, float], **kw) -> tuple[list[str], np.ndarray]:
    """Symmetric matrix of copy_probability over all source pairs (diagonal = 1)."""
    names = sorted(source_values)
    P = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = copy_probability(source_values[names[i]], source_values[names[j]], truth,
                                 accuracy.get(names[i], 0.8), accuracy.get(names[j], 0.8), **kw)
            P[i, j] = P[j, i] = r["p_copy"]
    return names, P


def copy_adjusted_independence(names: list[str], P: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Merge sources whose pairwise copy probability >= threshold and return 1/|group|."""
    uf = UnionFind(len(names))
    ii, jj = np.where(np.triu(P >= threshold, k=1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        uf.union(a, b)
    w = independence_weights(uf.labels(), 1.0)
    return {nm: float(x) for nm, x in zip(names, w)}
