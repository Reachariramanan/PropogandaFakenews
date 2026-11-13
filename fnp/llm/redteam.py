"""Hostile reviewer over the adjudicator's structured dossier and proposal."""
from __future__ import annotations

import json
from typing import Optional

from . import prompts as P
from . import validators as V
from .adjudicate import sanitize_state
from .cache import cached_call

MAX_TOKENS = 2500


def review(state: dict, proposed: dict, *, retry_failed: bool = True) -> dict:
    """-> dict(issues[{issue, severity}], adjusted_probs, confidence_adjustment, ok, error, meta).
    On failure ok=False, no issues, adjusted_probs = the proposal's probs, adjustment 0."""
    prop = sanitize_state({k: proposed[k] for k in ("probs", "label", "rationale", "key_evidence_ids")
                           if k in proposed})
    user = P.render(P.REDTEAM_USER, STATE=json.dumps(sanitize_state(state), ensure_ascii=False, indent=1),
                    PROPOSED=json.dumps(prop, ensure_ascii=False, indent=1))
    res = cached_call("redteam", P.REDTEAM_VERSION, P.REDTEAM_SYSTEM, user, V.validate_redteam,
                      MAX_TOKENS, retry_failed=retry_failed)
    meta = {k: res.get(k) for k in ("tokens", "latency_s", "cached", "attempts")}
    if not res["ok"]:
        pp = V.normalise_probs(proposed["probs"]) if proposed.get("probs") else None
        return {"ok": False, "error": res.get("error"), "issues": [], "adjusted_probs": pp,
                "confidence_adjustment": 0.0, "meta": meta}
    return {**V.parse_redteam(res["obj"]), "ok": True, "error": None, "meta": meta}
