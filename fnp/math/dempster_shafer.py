"""Dempster-Shafer fusion (frameworks note Method 2).

Masses are dicts {frozenset: mass}. The n-ary conjunctive combination (unnormalised, keeps
m(empty) = K) is associative; the final step either renormalises by 1-K (Dempster) or moves
K onto the whole frame Theta (Yager 1987). Yager is the default because Dempster's rule
behaves badly under high conflict (Zadeh's paradox).
Claim frame {T, F}: support -> m({T}) = min(s, cap), refute -> m({F}) = min(s, cap), the
rest to Theta. context_mismatch and neutral evidence are vacuous here.
"""
from __future__ import annotations

from typing import Sequence

from fnp.config import C
from fnp.math.woe import strengths

T, F = "T", "F"
THETA = frozenset({T, F})
EMPTY = frozenset()


def conjunctive(m1: dict, m2: dict) -> dict:
    out: dict = {}
    for a, x in m1.items():
        for b, y in m2.items():
            k = a & b
            out[k] = out.get(k, 0.0) + x * y
    return out


def combine(masses: Sequence[dict], frame: frozenset, rule: str = "yager") -> dict:
    """Combine mass functions; returns masses incl. EMPTY=0 and key 'K' stored separately."""
    acc = {frame: 1.0}
    for m in masses:
        acc = conjunctive(acc, m)
    K = acc.pop(EMPTY, 0.0)
    if rule == "dempster":
        if K >= 1.0 - 1e-15:
            raise ValueError("total conflict: Dempster's rule undefined")
        acc = {a: v / (1.0 - K) for a, v in acc.items()}
    elif rule == "yager":
        acc[frame] = acc.get(frame, 0.0) + K
    else:
        raise ValueError(rule)
    return {"m": acc, "K": K}


def bel(m: dict, A: frozenset) -> float:
    return sum(v for B, v in m.items() if B and B <= A)


def pl(m: dict, A: frozenset) -> float:
    return sum(v for B, v in m.items() if B & A)


def claim_masses(evidence: Sequence[dict], cap: float | None = None) -> list[dict]:
    cap = C("ds_strength_cap") if cap is None else cap
    s, d, _ = strengths(evidence)
    out = []
    for si, di in zip(s, d):
        if di == 0 or si <= 0:
            continue
        x = min(float(si), cap)
        out.append({frozenset({T if di > 0 else F}): x, THETA: 1.0 - x})
    return out


def ds_fuse(evidence: Sequence[dict], rule: str = "yager", cap: float | None = None) -> dict:
    r = combine(claim_masses(evidence, cap), THETA, rule)
    m = r["m"]
    bT, bF = bel(m, frozenset({T})), bel(m, frozenset({F}))
    pT, pF = pl(m, frozenset({T})), pl(m, frozenset({F}))
    return {"bel_T": bT, "bel_F": bF, "pl_T": pT, "pl_F": pF,
            "uncertainty": pT - bT, "conflict": r["K"]}
