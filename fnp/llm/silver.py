"""Direct (raw-text) Gemma labelling: silver/distillation labels for Laya and the
'Gemma-only' ablation baseline. NOT part of the evidence pipeline."""
from __future__ import annotations

from typing import Union

from ..schema import LABELS
from . import prompts as P
from . import validators as V
from .cache import cached_call

MAX_TOKENS = 1800
MAX_CHARS = 4000


def label(text_or_claim: Union[str, dict], *, retry_failed: bool = True) -> dict:
    """-> dict(probs over LABELS, label, rationale, ok, error, meta). Accepts a string or a Claim."""
    text = text_or_claim.get("claim", "") if isinstance(text_or_claim, dict) else str(text_or_claim or "")
    body, trunc = P.truncate_text(text, MAX_CHARS)
    note = f"\nNOTE: truncated to the first {MAX_CHARS} characters." if trunc else ""
    user = P.render(P.SILVER_USER, NOTE=note, TEXT=body)
    res = cached_call("silver", P.SILVER_VERSION, P.SILVER_SYSTEM, user, V.validate_silver,
                      MAX_TOKENS, retry_failed=retry_failed)
    meta = {k: res.get(k) for k in ("tokens", "latency_s", "cached", "attempts")}
    if not res["ok"]:
        probs = {l: (1.0 if l == "INCONCLUSIVE" else 0.0) for l in LABELS}
        return {"ok": False, "error": res.get("error"), "probs": probs, "label": "INCONCLUSIVE",
                "rationale": "", "meta": meta}
    out = V.parse_verdict(res["obj"])
    out.pop("key_evidence_ids", None)
    return {**out, "ok": True, "error": None, "meta": meta}
