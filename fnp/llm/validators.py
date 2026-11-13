"""Structural validators (for `chat_json(validate=...)`) and parse -> schema functions.

Validators return None if the object is acceptable, else a short reason; they never mutate.
Per-item problems that should not discard a whole answer (a non-verbatim quote, an unknown
technique) are handled in the parse functions, which drop the item and count the drop.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..schema import (CLAIM_TYPES, EVIDENCE_TYPES, LABELS, PROPAGANDA_TECHNIQUES, STANCES,
                      Claim, Span)
from ..text_utils import evidence_in_text, find_span, sha16
from .prompts import MAX_CLAIMS

PROB_FLOOR = 0.002          # keeps log-probs finite downstream; applied before renormalising
MAX_SPANS = 25
MAX_FRAGMENT_CHARS = 600
SEVERITIES = ("low", "medium", "high")

_QUOTED = ("asserted", "quoted", "reported")
_TECH_KEY = {re.sub(r"[^a-z0-9]", "", t.lower()): t for t in PROPAGANDA_TECHNIQUES}
_ET_KEY = {e.lower(): e for e in EVIDENCE_TYPES}


def to_float(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if x == x else None
    if isinstance(x, str):
        try:
            v = float(x.strip())
            return v if v == v else None
        except ValueError:
            return None
    return None


def unit(x: Any, default: Optional[float] = None) -> Optional[float]:
    """Number clipped to [0,1] (None -> default)."""
    v = to_float(x)
    return default if v is None else min(1.0, max(0.0, v))


def _num01(x: Any) -> bool:
    v = to_float(x)
    return v is not None and -0.001 <= v <= 1.001


def norm_label(x: Any) -> Optional[str]:
    s = str(x or "").strip().upper()
    return s if s in LABELS else None


def norm_technique(x: Any) -> Optional[str]:
    return _TECH_KEY.get(re.sub(r"[^a-z0-9]", "", str(x or "").lower()))


def norm_enum(x: Any, allowed: tuple[str, ...]) -> Optional[str]:
    s = str(x or "").strip().lower()
    for a in allowed:
        if a.lower() == s:
            return a
    return None


def norm_evidence_type(x: Any) -> str:
    return _ET_KEY.get(str(x or "").strip().lower().replace(" ", "_"), "UNKNOWN")


def _is_list_of_dicts(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, dict) for i in x)


# ------------------------------------------------------------------ probabilities
def _probs_problem(p: Any) -> Optional[str]:
    if not isinstance(p, dict):
        return "probs_not_object"
    vals = {}
    for k, v in p.items():
        lab = norm_label(k)
        f = to_float(v)
        if lab is None:
            return f"unknown_label:{k}"
        if f is None or f < 0:
            return f"bad_prob:{k}"
        vals[lab] = f
    if not vals:
        return "probs_empty"
    if sum(vals.values()) <= 0:
        return "probs_sum_zero"
    return None


def normalise_probs(p: dict) -> dict[str, float]:
    """Over LABELS in fixed order: missing -> 0, floor, renormalise to sum exactly 1."""
    raw = {l: 0.0 for l in LABELS}
    for k, v in p.items():
        lab = norm_label(k)
        f = to_float(v)
        if lab and f is not None and f > 0:
            raw[lab] += f
    s = sum(raw.values())
    if s <= 0:
        return {l: 1.0 / len(LABELS) for l in LABELS}
    raw = {l: max(v / s, PROB_FLOOR) for l, v in raw.items()}
    s2 = sum(raw.values())
    return {l: v / s2 for l, v in raw.items()}


def argmax_label(probs: dict[str, float]) -> str:
    return max(LABELS, key=lambda l: probs[l])   # ties -> earlier label in LABELS order


# ------------------------------------------------------------------ claims
def validate_claims(obj: dict) -> Optional[str]:
    cl = obj.get("claims")
    if not _is_list_of_dicts(cl):
        return "claims_not_list_of_objects"
    for i, c in enumerate(cl):
        if not isinstance(c.get("claim"), str) or not c["claim"].strip():
            return f"claim[{i}]_missing_text"
        if not isinstance(c.get("evidence_quote"), str):
            return f"claim[{i}]_missing_quote"
        if "check_worthy" in c and not _num01(c["check_worthy"]):
            return f"claim[{i}]_check_worthy_range"
    return None


def parse_claims(obj: dict, doc_id: str, text: str, lang: Optional[str] = None
                 ) -> tuple[list[Claim], dict]:
    """-> (claims, drops). Claims whose evidence_quote is not verbatim in `text` are dropped."""
    drops = {"non_verbatim": 0, "empty": 0, "duplicate": 0, "over_cap": 0}
    lang = lang or (str(obj["language"])[:8] if isinstance(obj.get("language"), str) else None)
    out: list[Claim] = []
    seen: set[str] = set()
    for c in obj.get("claims", []):
        claim = re.sub(r"\s+", " ", c["claim"]).strip()
        quote = c["evidence_quote"].strip()
        if not claim:
            drops["empty"] += 1
            continue
        if not evidence_in_text(quote, text):
            drops["non_verbatim"] += 1
            continue
        cid = sha16(doc_id, claim)
        if cid in seen:
            drops["duplicate"] += 1
            continue
        seen.add(cid)
        ctype = norm_enum(c.get("claim_type"), CLAIM_TYPES) or "other"
        qr = norm_enum(c.get("quoted_or_reported"), _QUOTED) or "asserted"
        speaker = c.get("speaker")
        date = c.get("date")
        ents = c.get("entities")
        out.append({
            "claim_id": cid, "doc_id": doc_id, "claim": claim,
            "speaker": str(speaker).strip() or None if speaker not in (None, "") else None,
            "claim_type": ctype, "quoted_or_reported": qr,
            "check_worthy": unit(c.get("check_worthy"), 0.5),
            "date": str(date).strip() or None if date not in (None, "") else None,
            "entities": [str(e).strip() for e in ents if str(e).strip()] if isinstance(ents, list) else [],
            "evidence_quote": quote, "lang": lang,
        })
    if len(out) > MAX_CLAIMS:
        out.sort(key=lambda c: -c["check_worthy"])
        drops["over_cap"] = len(out) - MAX_CLAIMS
        out = out[:MAX_CLAIMS]
    return out, drops


# ------------------------------------------------------------------ propaganda
def validate_propaganda(obj: dict) -> Optional[str]:
    sp = obj.get("spans")
    if not _is_list_of_dicts(sp):
        return "spans_not_list_of_objects"
    for i, s in enumerate(sp):
        if not isinstance(s.get("fragment"), str):
            return f"span[{i}]_missing_fragment"
        if "confidence" in s and not _num01(s["confidence"]):
            return f"span[{i}]_confidence_range"
    if not _num01(obj.get("one_sided")):
        return "one_sided_missing_or_out_of_range"
    if "framing" in obj and obj["framing"] is not None and not isinstance(obj["framing"], str):
        return "framing_not_string"
    return None


def locate_fragment(fragment: str, text: str) -> Optional[tuple[int, int]]:
    """Offsets of `fragment` in `text`: exact, case-insensitive, then whitespace-flexible."""
    fragment = fragment.strip()
    if not fragment:
        return None
    pos = find_span(fragment, text)
    if pos and text[pos[0]:pos[1]].lower() == fragment.lower():
        return pos
    parts = fragment.split()
    if parts:
        m = re.search(r"\s+".join(re.escape(p) for p in parts), text, re.I)
        if m:
            return m.start(), m.end()
    return None


def parse_spans(obj: dict, doc_id: str, text: str) -> tuple[list[Span], dict, float, str]:
    """-> (spans, drops, one_sided, framing). Offsets index into `text` (the full original)."""
    drops = {"non_verbatim": 0, "bad_technique": 0, "too_long": 0, "duplicate": 0, "over_cap": 0}
    out: list[Span] = []
    seen: set[tuple[int, int, str]] = set()
    for s in obj.get("spans", []):
        tech = norm_technique(s.get("technique"))
        if tech is None:
            drops["bad_technique"] += 1
            continue
        frag = s["fragment"].strip()
        if len(frag) > MAX_FRAGMENT_CHARS:
            drops["too_long"] += 1
            continue
        pos = locate_fragment(frag, text)
        if pos is None:
            drops["non_verbatim"] += 1
            continue
        key = (pos[0], pos[1], tech)
        if key in seen:
            drops["duplicate"] += 1
            continue
        seen.add(key)
        out.append({"doc_id": doc_id, "start": pos[0], "end": pos[1], "text": text[pos[0]:pos[1]],
                    "technique": tech, "confidence": unit(s.get("confidence"), 0.5)})
    if len(out) > MAX_SPANS:
        out.sort(key=lambda s: -s["confidence"])
        drops["over_cap"] = len(out) - MAX_SPANS
        out = out[:MAX_SPANS]
    out.sort(key=lambda s: (s["start"], s["end"]))
    framing = obj.get("framing")
    return out, drops, unit(obj.get("one_sided"), 0.0), (framing.strip() if isinstance(framing, str) else "")


# ------------------------------------------------------------------ stance
def validate_stance(obj: dict) -> Optional[str]:
    st = norm_enum(obj.get("stance"), STANCES)
    if st is None:
        return f"bad_stance:{obj.get('stance')}"
    for k in ("stance_conf", "relevance"):
        if not _num01(obj.get(k)):
            return f"{k}_missing_or_out_of_range"
    r = obj.get("context_mismatch_reason")
    if st == "context_mismatch" and not (isinstance(r, str) and r.strip()):
        return "context_mismatch_needs_reason"
    if r is not None and not isinstance(r, str):
        return "reason_not_string"
    return None


def parse_stance(obj: dict) -> dict:
    st = norm_enum(obj.get("stance"), STANCES)
    r = obj.get("context_mismatch_reason")
    r = r.strip() if isinstance(r, str) and r.strip() else None
    return {"stance": st, "stance_conf": unit(obj.get("stance_conf"), 0.0),
            "evidence_type": norm_evidence_type(obj.get("evidence_type")),
            "relevance": unit(obj.get("relevance"), 0.0),
            "context_mismatch_reason": r if st == "context_mismatch" else None,
            "rationale": str(obj.get("rationale") or "").strip()}


# ------------------------------------------------------------------ timeline
def validate_timeline(obj: dict) -> Optional[str]:
    fl = obj.get("flags")
    if not _is_list_of_dicts(fl):
        return "flags_not_list_of_objects"
    for i, f in enumerate(fl):
        if not isinstance(f.get("flag"), str) or not f["flag"].strip():
            return f"flag[{i}]_missing_flag"
        if not _num01(f.get("prob")):
            return f"flag[{i}]_prob_missing_or_out_of_range"
        if not isinstance(f.get("claim_ids"), list):
            return f"flag[{i}]_claim_ids_not_list"
    return None


def parse_timeline(obj: dict, valid_ids: set[str]) -> list[dict]:
    out = []
    for f in obj.get("flags", []):
        ids = [str(i) for i in f["claim_ids"] if str(i) in valid_ids]
        if not ids:                       # a flag about no known claim is unusable
            continue
        out.append({"flag": f["flag"].strip().lower().replace(" ", "_"), "prob": unit(f["prob"], 0.5),
                    "claim_ids": ids, "reason": str(f.get("reason") or "").strip()})
    return out


# ------------------------------------------------------------------ adjudicate / silver / redteam
def validate_verdict(obj: dict) -> Optional[str]:
    bad = _probs_problem(obj.get("probs"))
    if bad:
        return bad
    if "label" in obj and obj["label"] is not None and norm_label(obj["label"]) is None:
        return f"bad_label:{obj['label']}"
    if "key_evidence_ids" in obj and not isinstance(obj["key_evidence_ids"], list):
        return "key_evidence_ids_not_list"
    return None


def parse_verdict(obj: dict, valid_evidence_ids: Optional[set[str]] = None) -> dict:
    """label is always the argmax of the renormalised probs (a stated label that disagrees
    with its own probabilities is overridden; `label_stated` keeps what the model said)."""
    probs = normalise_probs(obj["probs"])
    ids = [str(i) for i in obj.get("key_evidence_ids") or []]
    if valid_evidence_ids is not None:
        ids = [i for i in ids if i in valid_evidence_ids]
    return {"probs": probs, "label": argmax_label(probs), "label_stated": norm_label(obj.get("label")),
            "rationale": str(obj.get("rationale") or "").strip(), "key_evidence_ids": ids}


def validate_silver(obj: dict) -> Optional[str]:
    return validate_verdict({k: v for k, v in obj.items() if k != "key_evidence_ids"})


def validate_redteam(obj: dict) -> Optional[str]:
    iss = obj.get("issues")
    if not _is_list_of_dicts(iss):
        return "issues_not_list_of_objects"
    for i, it in enumerate(iss):
        if not isinstance(it.get("issue"), str):
            return f"issue[{i}]_missing_text"
        if norm_enum(it.get("severity"), SEVERITIES) is None:
            return f"issue[{i}]_bad_severity"
    bad = _probs_problem(obj.get("adjusted_probs"))
    if bad:
        return "adjusted_" + bad
    ca = to_float(obj.get("confidence_adjustment"))
    if ca is None or not -1.001 <= ca <= 1.001:
        return "confidence_adjustment_missing_or_out_of_range"
    return None


def parse_redteam(obj: dict) -> dict:
    return {"issues": [{"issue": it["issue"].strip(), "severity": norm_enum(it["severity"], SEVERITIES)}
                       for it in obj["issues"]],
            "adjusted_probs": normalise_probs(obj["adjusted_probs"]),
            "confidence_adjustment": max(-1.0, min(1.0, to_float(obj["confidence_adjustment"])))}
