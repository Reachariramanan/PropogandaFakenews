"""Deterministic compact state text and the typed question set for the Laya System-1 decider.

Laya reads `[CLS] <type> instructions [SEP] [MASK] opt0 ... [SEP] state [SEP]` and the state is
cut from the END when it is too long, so the most decision-relevant parts come first:
claim, headline signals, propagandistic spans, propagation, top-5 evidence, remaining features.
The same input always yields the same string (no dict-order, float-format or time dependence).
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

from fnp.schema import FEATURE_NAMES, LABELS

MAX_STATE_CHARS = 1800
TOP_EVIDENCE = 5

# Features shown in the headline SIGNALS line (the rest go to MORE, which truncation drops first).
KEY_FEATURES = ("p_true", "p_true_lo", "p_true_hi", "support", "contradiction", "context_mismatch",
                "ds_uncertainty", "ds_conflict", "n_evidence", "n_eff", "kb_match_score", "p_prop",
                "one_sidedness", "coordination", "check_worthy")

_WS = re.compile(r"\s+")

# ----------------------------------------------------------------------------- questions
# Definitions are the project's (docs/ARCHITECTURE.md); wording is kept short because the option
# descriptions are part of the head and compete with the state for the 512-token budget.
LABEL_DEFS = {
    "FAKE": "the core claim is false or fabricated",
    "PROPAGANDA": "persuasion techniques, one-sided framing or coordinated amplification aimed at "
                  "influence; claims may be true or unverifiable",
    "LEGIT": "supported by independent reliable evidence, no significant manipulation",
    "MISLEADING": "real content with false context, partly true or cherry-picked",
    "INCONCLUSIVE": "insufficient or conflicting evidence",
}
assert tuple(LABEL_DEFS) == LABELS

QUESTIONS: dict[str, dict] = {
    "verdict": {
        "type": "choice",
        "instructions": "Give the verdict for the claim from the evidence and signals.",
        "criteria": dict(LABEL_DEFS),
    },
    "is_propaganda": {
        "type": "noul",
        "instructions": "Does the text use propaganda techniques or one-sided persuasion aimed at influence?",
        "criteria": {"false": "no manipulation", "true": "propaganda techniques present"},
    },
    "context_mismatch": {
        "type": "noul",
        "instructions": "Is real content presented in a false or mismatched context (date, place, event)?",
        "criteria": {"false": "context matches", "true": "context is false or mismatched"},
    },
    "check_worthy": {
        "type": "noul",
        "instructions": "Is this a factual claim that is worth fact-checking?",
        "criteria": {"false": "opinion or trivial", "true": "check-worthy factual claim"},
    },
}
NOUL_QUESTIONS = ("is_propaganda", "context_mismatch", "check_worthy")


# ----------------------------------------------------------------------------- formatting
def _num(v: Any) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, bool):
        return "1" if v else "0"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _clip(str(v), 24)
    if math.isnan(f) or math.isinf(f):
        return "unknown"
    s = f"{round(f, 2):.2f}"
    return s[:-3] if s.endswith(".00") else s


def _clip(text: Any, n: int) -> str:
    t = _WS.sub(" ", str(text or "")).strip()
    return t if len(t) <= n else t[: max(0, n - 1)].rstrip() + "…"


def _kv(d: dict, keys, ) -> str:
    return " ".join(f"{k}={_num(d[k])}" for k in keys if k in d)


def _score(ev: dict) -> float:
    def g(k, default):
        v = ev.get(k)
        try:
            f = float(v)
            return default if math.isnan(f) else f
        except (TypeError, ValueError):
            return default
    return g("reliability", 0.3) * g("independence", 1.0) * g("relevance", 0.5) * max(g("stance_conf", 0.5), 0.05)


def rank_evidence(evidence: list[dict]) -> list[dict]:
    """Strongest first; ties broken by input order, so the result is deterministic."""
    idx = sorted(range(len(evidence)), key=lambda i: (-_score(evidence[i]), i))
    return [evidence[i] for i in idx]


def _evidence_line(i: int, ev: dict, chars: int) -> str:
    head = "|".join([str(ev.get("stance") or "neutral"),
                     str(ev.get("evidence_type") or "UNKNOWN"),
                     "rel=" + _num(ev.get("reliability"))])
    src = _clip(ev.get("source") or "", 24)
    body = _clip(ev.get("text") or "", chars)
    return f"{i}. {head}" + (f" [{src}]" if src else "") + f": {body}"


def _spans_line(spans: Optional[dict]) -> str:
    if not spans:
        return ""
    parts = []
    for k in sorted(spans):
        v = spans[k]
        if isinstance(v, dict):
            v = ",".join(f"{a}:{_num(b)}" for a, b in sorted(v.items(), key=lambda kv: (-float(kv[1] or 0), kv[0]))[:4])
        elif isinstance(v, (list, tuple)):
            v = ",".join(_clip(x, 24) for x in list(v)[:4])
        else:
            v = _num(v)
        parts.append(f"{k}={v}")
    return "SPANS: " + " ".join(parts)


def _prop_line(propagation: Optional[dict]) -> str:
    if not propagation:
        return ""
    return "PROPAGATION: " + " ".join(f"{k}={_num(propagation[k])}" for k in sorted(propagation)[:8])


def build_state(claim: dict, features: dict, evidence: list[dict],
                spans_summary: Optional[dict] = None, propagation: Optional[dict] = None) -> str:
    """Compact deterministic state text, at most `MAX_STATE_CHARS` characters."""
    claim = claim or {}
    features = features or {}
    ranked = rank_evidence(list(evidence or []))
    top = ranked[:TOP_EVIDENCE]

    key = [k for k in KEY_FEATURES if k in features]
    rest = [k for k in FEATURE_NAMES if k in features and k not in key]
    rest += sorted(k for k in features if k not in FEATURE_NAMES and k not in key)

    meta = " ".join(f"{n}={_clip(claim[k], 32)}" for n, k in
                    (("type", "claim_type"), ("lang", "lang"), ("speaker", "speaker"),
                     ("date", "date"), ("mode", "quoted_or_reported")) if claim.get(k))

    def render(claim_chars: int, ev_chars: int, with_more: bool) -> str:
        lines = [f"CLAIM: {_clip(claim.get('claim', ''), claim_chars)}"]
        if meta:
            lines.append("META: " + meta)
        if key:
            lines.append("SIGNALS: " + _kv(features, key))
        for ln in (_spans_line(spans_summary), _prop_line(propagation)):
            if ln:
                lines.append(ln)
        if top:
            lines.append(f"EVIDENCE (top {len(top)} of {len(ranked)}):")
            lines += [_evidence_line(i + 1, e, ev_chars) for i, e in enumerate(top)]
        else:
            lines.append("EVIDENCE: none")
        if with_more and rest:
            lines.append("MORE: " + _kv(features, rest))
        return "\n".join(lines)

    for claim_chars, ev_chars, more in ((500, 150, True), (400, 110, True), (320, 80, True),
                                        (260, 60, False), (200, 40, False)):
        s = render(claim_chars, ev_chars, more)
        if len(s) <= MAX_STATE_CHARS:
            return s
    return s[:MAX_STATE_CHARS - 1].rstrip() + "…"
