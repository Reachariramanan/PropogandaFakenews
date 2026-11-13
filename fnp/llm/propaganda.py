"""L2 propaganda tagger: raw text -> list[Span] (+ doc-level one_sided / framing)."""
from __future__ import annotations

from typing import Optional

from ..schema import Span
from . import prompts as P
from . import validators as V
from .cache import cached_call

MAX_TOKENS = 3000
MAX_CHARS = 8000


class SpanList(list):
    """list[Span] that also carries doc-level fields and diagnostics.
    Offsets index into the original (untruncated) text because truncation keeps the prefix."""
    ok: bool = True
    error: Optional[str] = None
    one_sided: float = 0.0
    framing: str = ""
    dropped: dict
    meta: dict
    truncated: bool = False


def tag_spans(doc_id: str, text: str, *, retry_failed: bool = True) -> SpanList:
    out = SpanList()
    out.dropped, out.meta = {}, {}
    body, trunc = P.truncate_text(text or "", MAX_CHARS)
    out.truncated = trunc
    if not body.strip():
        return out
    note = f"\nNOTE: the text was truncated to its first {MAX_CHARS} characters." if trunc else ""
    user = P.render(P.PROPAGANDA_USER, NOTE=note, TEXT=body)
    res = cached_call("propaganda", P.PROPAGANDA_VERSION, P.PROPAGANDA_SYSTEM, user,
                      V.validate_propaganda, MAX_TOKENS, retry_failed=retry_failed)
    out.meta = {k: res.get(k) for k in ("tokens", "latency_s", "cached", "attempts")}
    if not res["ok"]:
        out.ok, out.error = False, res.get("error")
        return out
    spans, out.dropped, out.one_sided, out.framing = V.parse_spans(res["obj"], doc_id, body)
    out.extend(spans)
    return out
