"""Text normalisation and verbatim-evidence checks (same rules as EDA entity_routing)."""
from __future__ import annotations

import hashlib
import re
import unicodedata


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "")).strip().casefold()


def evidence_in_text(evidence: str, text: str) -> bool:
    """Verbatim check (whitespace/case/Unicode-normalised). Empty evidence is not evidence."""
    ev = norm(evidence)
    return bool(ev) and ev in norm(text)


def find_span(fragment: str, text: str) -> tuple[int, int] | None:
    """Char offsets of `fragment` in `text`: exact first, then case-insensitive."""
    if not fragment:
        return None
    i = text.find(fragment)
    if i < 0:
        i = text.lower().find(fragment.lower())
    return (i, i + len(fragment)) if i >= 0 else None


def sha16(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(norm(p) for p in parts).encode("utf-8")).hexdigest()[:16]
