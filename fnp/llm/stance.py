"""L3 stance judge: (claim, evidence passage) -> stance / evidence type / relevance."""
from __future__ import annotations

from typing import Iterable, Optional

from ..gemma import run_batch
from ..schema import Claim
from . import prompts as P
from . import validators as V
from .cache import cached_call

MAX_TOKENS = 1500
MAX_EVIDENCE_CHARS = 3500


def _failed(error: Optional[str]) -> dict:
    return {"ok": False, "error": error, "stance": "neutral", "stance_conf": 0.0,
            "evidence_type": "UNKNOWN", "relevance": 0.0, "context_mismatch_reason": None,
            "rationale": "", "meta": {}}


def judge(claim: Claim, evidence_text: str, evidence_meta: Optional[dict] = None, *,
          retry_failed: bool = True) -> dict:
    """-> dict(stance, stance_conf, evidence_type, relevance, context_mismatch_reason,
    rationale, ok, error, meta). On failure ok=False and the neutral/zero-confidence defaults."""
    m = evidence_meta or {}
    ev, trunc = P.truncate_text(evidence_text or "", MAX_EVIDENCE_CHARS)
    if not ev.strip():
        return _failed("empty_evidence")
    user = P.render(
        P.STANCE_USER, CLAIM=claim.get("claim", ""), DATE=str(claim.get("date") or "unknown"),
        SPEAKER=str(claim.get("speaker") or "unknown"),
        SOURCE=str(m.get("source") or "unknown"), PUBLISHED=str(m.get("published") or "unknown"),
        URL=str(m.get("url") or "unknown"), EVIDENCE=ev + ("\n[...truncated]" if trunc else ""))
    res = cached_call("stance", P.STANCE_VERSION, P.STANCE_SYSTEM, user, V.validate_stance,
                      MAX_TOKENS, retry_failed=retry_failed)
    meta = {k: res.get(k) for k in ("tokens", "latency_s", "cached", "attempts")}
    if not res["ok"]:
        d = _failed(res.get("error"))
        d["meta"] = meta
        return d
    return {**V.parse_stance(res["obj"]), "ok": True, "error": None, "meta": meta}


def judge_many(pairs: Iterable[tuple], concurrency: Optional[int] = None) -> list[dict]:
    """pairs of (claim, evidence_text[, evidence_meta]); results keep input order."""
    return run_batch(list(pairs), lambda p: judge(p[0], p[1], p[2] if len(p) > 2 else None),
                     concurrency)
