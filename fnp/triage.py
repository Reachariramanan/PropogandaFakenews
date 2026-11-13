"""Shared fake/misleading call for stored verdicts (Propaganda Lab stage 33, triage export stage 35).

The stacker's PROPAGANDA probability is a length artefact (docs/review/05 #1/#7), so per claim the
PROPAGANDA mass is removed and the rest renormalised over FAKE/MISLEADING/LEGIT/INCONCLUSIVE, but only
when >= MIN_NONPROP_MASS remains (otherwise that decider made no factual call). Stacker first, the
Gemma adjudicator as fallback, else INCONCLUSIVE. The claim with the highest P(FAKE)+P(MISLEADING)
decides the document.
"""
from __future__ import annotations

from typing import Optional

FM = ("FAKE", "MISLEADING", "LEGIT", "INCONCLUSIVE")
MIN_NONPROP_MASS = 0.2


def fm_probs(p: Optional[dict]) -> Optional[dict]:
    p = p or {}
    z = sum(p.get(k, 0.0) for k in FM)
    if z < MIN_NONPROP_MASS:
        return None
    q = {k: p.get(k, 0.0) / z for k in FM}
    return {"label": max(FM, key=q.get), "probs": {k: round(v, 4) for k, v in q.items()},
            "fake_score": round(q["FAKE"] + q["MISLEADING"], 4)}


def fake_call(claims: list[dict]) -> dict:
    """claims: [{"stacker": probs, "gemma": probs|None}] -> {label, probs, fake_score, basis}."""
    best = {"label": "INCONCLUSIVE", "probs": {k: 0.0 for k in FM}, "fake_score": 0.0, "basis": "none"}
    for c in claims:
        for basis in ("stacker", "gemma"):
            r = fm_probs(c.get(basis))
            if r is not None:
                if r["fake_score"] >= best["fake_score"]:
                    best = {**r, "basis": basis}
                break
    return best
