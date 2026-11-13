"""Chronology check over structured claims (no raw text)."""
from __future__ import annotations

import json

from ..schema import Claim
from . import prompts as P
from . import validators as V
from .cache import cached_call

MAX_TOKENS = 2000


def check(claims: list[Claim]) -> list[dict]:
    """-> list of dict(flag, prob, claim_ids, reason); [] when nothing is flagged, there are no
    claims, or the call failed (see `check_ex` for the diagnostics)."""
    return check_ex(claims)["flags"]


def check_ex(claims: list[Claim]) -> dict:
    if not claims:
        return {"flags": [], "ok": True, "error": None, "meta": {}}
    slim = [{"claim_id": c["claim_id"], "claim": c.get("claim", ""), "date": c.get("date"),
             "claim_type": c.get("claim_type"), "entities": c.get("entities", [])} for c in claims]
    user = P.render(P.TIMELINE_USER, CLAIMS=json.dumps(slim, ensure_ascii=False, indent=1))
    res = cached_call("timeline", P.TIMELINE_VERSION, P.TIMELINE_SYSTEM, user,
                      V.validate_timeline, MAX_TOKENS)
    meta = {k: res.get(k) for k in ("tokens", "latency_s", "cached", "attempts")}
    if not res["ok"]:
        return {"flags": [], "ok": False, "error": res.get("error"), "meta": meta}
    flags = V.parse_timeline(res["obj"], {c["claim_id"] for c in slim})
    return {"flags": flags, "ok": True, "error": None, "meta": meta}
