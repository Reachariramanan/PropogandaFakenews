"""L1 claim extractor: raw text -> list[Claim] (the only reader of raw text, with L2/L3)."""
from __future__ import annotations

from typing import Optional

from ..schema import Claim
from . import prompts as P
from . import validators as V
from .cache import cached_call

MAX_TOKENS = 2500


class ClaimList(list):
    """list[Claim] that also carries the call diagnostics."""
    ok: bool = True
    error: Optional[str] = None
    dropped: dict
    meta: dict
    truncated: bool = False


def extract_claims(doc_id: str, text: str, lang: Optional[str] = None, *,
                   retry_failed: bool = True) -> ClaimList:
    out = ClaimList()
    out.dropped, out.meta = {}, {}
    body, trunc = P.truncate_text(text or "")
    out.truncated = trunc
    if not body.strip():
        return out
    note = (f"\nNOTE: the text was truncated to its first {P.MAX_TEXT_CHARS} characters."
            if trunc else "")
    user = P.render(P.CLAIMS_USER, NOTE=note, TEXT=body)
    res = cached_call("claims", P.CLAIMS_VERSION, P.CLAIMS_SYSTEM, user, V.validate_claims,
                      MAX_TOKENS, retry_failed=retry_failed)
    out.meta = {k: res.get(k) for k in ("tokens", "latency_s", "cached", "attempts")}
    if not res["ok"]:
        out.ok, out.error = False, res.get("error")
        return out
    # Quotes are checked against the text the model actually saw.
    claims, out.dropped = V.parse_claims(res["obj"], doc_id, body, lang)
    out.extend(claims)
    return out
