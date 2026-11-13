"""L7 Gemma adjudicator: STRUCTURED state only, never raw article text."""
from __future__ import annotations

import json
from typing import Any, Optional

from ..schema import LABELS
from . import prompts as P
from . import validators as V
from .cache import cached_call

MAX_TOKENS = 2000
MAX_STR = 500
FORBIDDEN_KEYS = {"text", "body", "article", "raw_text", "content", "full_text", "html"}


def sanitize_state(state: Any, _depth: int = 0) -> Any:
    """Guard for the 'adjudicator never sees raw text' rule: refuse raw-text keys and cap
    every string (evidence summaries/claims are short; an article body is not)."""
    if isinstance(state, dict):
        bad = FORBIDDEN_KEYS & {str(k).lower() for k in state}
        if bad:
            raise ValueError(f"adjudicator state must be structured; found raw-text key(s) {sorted(bad)}")
        return {k: sanitize_state(v, _depth + 1) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return [sanitize_state(v, _depth + 1) for v in state]
    if isinstance(state, float):
        return round(state, 4)
    if isinstance(state, str) and len(state) > MAX_STR:
        return state[:MAX_STR] + "..."
    return state


def evidence_ids(state: dict) -> set[str]:
    ids: set[str] = set()
    for key in ("evidence", "top_evidence"):
        for e in state.get(key) or []:
            if isinstance(e, dict) and e.get("evidence_id"):
                ids.add(str(e["evidence_id"]))
    return ids


def _fallback(error: Optional[str], meta: dict) -> dict:
    probs = {l: 0.0 for l in LABELS}
    probs["INCONCLUSIVE"] = 1.0
    return {"ok": False, "error": error, "probs": probs, "label": "INCONCLUSIVE",
            "rationale": "", "key_evidence_ids": [], "meta": meta}


def decide(state: dict, *, retry_failed: bool = True) -> dict:
    """-> dict(probs over LABELS summing to 1, label, rationale, key_evidence_ids, ok, error, meta).
    On failure ok=False with all mass on INCONCLUSIVE: callers must check `ok`."""
    clean = sanitize_state(state)
    user = P.render(P.ADJUDICATE_USER, STATE=json.dumps(clean, ensure_ascii=False, indent=1))
    res = cached_call("adjudicate", P.ADJUDICATE_VERSION, P.ADJUDICATE_SYSTEM, user,
                      V.validate_verdict, MAX_TOKENS, retry_failed=retry_failed)
    meta = {k: res.get(k) for k in ("tokens", "latency_s", "cached", "attempts")}
    if not res["ok"]:
        return _fallback(res.get("error"), meta)
    out = V.parse_verdict(res["obj"], evidence_ids(clean) or None)
    return {**out, "ok": True, "error": None, "meta": meta}
