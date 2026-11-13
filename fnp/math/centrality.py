"""Evidence centrality via PageRank on the independence-collapsed support graph (Method 5).

Nodes: one per origin cluster (all copies of one origin collapse to one node), one per claim
c and one per its negation ~c. Edge origin -> c with weight = sum of support strengths,
origin -> ~c for refutes, plus back-edges of the same weight (HITS-like mutual
reinforcement). Teleport mass is placed on origin nodes in proportion to reliability.
centrality(c) = r(c) / (r(c) + r(~c)), 0.5 if neither receives mass.
"""
from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np

from fnp.math._util import fnum
from fnp.math.constants import M
from fnp.math.independence import cluster_evidence
from fnp.math.woe import strengths


def claim_centrality(evidence_by_claim: dict[str, Sequence[dict]]) -> dict[str, float]:
    G = nx.DiGraph()
    pers: dict = {}
    for cid, evs in evidence_by_claim.items():
        G.add_node(("c", cid)); G.add_node(("n", cid))
        if not evs:
            continue
        labels = [e.get("origin_cluster") for e in evs]
        if any(l is None for l in labels):
            labels = cluster_evidence(evs).tolist()
        s, d, _ = strengths(evs)
        for e, lab, si, di in zip(evs, labels, s, d):
            if di == 0 or si <= 0:
                continue
            o = ("o", e.get("origin_key") or f"{cid}:{lab}")
            tgt = ("c" if di > 0 else "n", cid)
            for u, v in ((o, tgt), (tgt, o)):
                w = G.get_edge_data(u, v, {"weight": 0.0})["weight"]
                G.add_edge(u, v, weight=w + float(si))
            pers[o] = max(pers.get(o, 0.0), fnum(e.get("reliability"), 0.5))
    out = {cid: 0.5 for cid in evidence_by_claim}
    if not pers:
        return out
    r = nx.pagerank(G, alpha=M("pagerank_alpha"), personalization=pers, weight="weight")
    for cid in evidence_by_claim:
        a, b = r.get(("c", cid), 0.0), r.get(("n", cid), 0.0)
        out[cid] = float(a / (a + b)) if a + b > 1e-12 else 0.5
    return out
