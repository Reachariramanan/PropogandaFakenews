"""Text hygiene and simhash for the realtime store (pure stdlib, no I/O).

clean_text: strip HTML and control characters, collapse whitespace, truncate on a word boundary.
simhash64:  64-bit word-shingle simhash of title + text, plus hamming helpers.
"""
import hashlib
import html
import re

EXCERPT_CHARS = 1500
TITLE_CHARS = 400

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.I | re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
# a real tag starts with a letter or '/': so "a < b and c > d" and "I <3 you" survive
_TAG = re.compile(r"</?[A-Za-z][^<>]{0,300}>")
_BREAKS = re.compile(r"<\s*(br|/p|/div|/li|/h[1-6])\b[^<>]{0,100}>", re.I)
# C0/C1 controls (tab/newline become spaces below), zero-width and bidi marks, BOM, line/para separators
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏‪-‮⁠-⁤﻿]")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.U)


def clean_text(value, limit=EXCERPT_CHARS):
    """Return plain single-spaced text of at most `limit` characters, cut on a word boundary ('' for None)."""
    if value is None:
        return ""
    s = value if isinstance(value, str) else str(value)
    if "<" in s:
        s = _COMMENT.sub(" ", s)
        s = _SCRIPT_STYLE.sub(" ", s)
        s = _BREAKS.sub(" ", s)
        s = _TAG.sub(" ", s)
    if "&" in s:
        s = html.unescape(s)
    s = _CONTROL.sub("", s)
    s = _WS.sub(" ", s.replace(" ", " ").replace(" ", " ")).strip()
    return truncate_words(s, limit)


def truncate_words(s, limit):
    """Cut `s` to <= limit chars on a word boundary, adding an ellipsis when something was removed."""
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    cut = s[: max(limit - 1, 0)]
    sp = cut.rfind(" ")
    if sp >= int(limit * 0.5):  # no tiny stub: only back up to a space when it keeps at least half
        cut = cut[:sp]
    return cut.rstrip(" ,.;:-") + "…"


def normalize_lang(value):
    """Language code from the database column: lowercase, legacy codes mapped, 'und' when missing/unknown."""
    if value is None:
        return "und"
    v = str(value).strip().lower().replace("_", "-")
    if v in ("", "unknown", "und", "none", "null", "n/a", "xx", "zxx"):
        return "und"
    v = {"in": "id", "iw": "he", "ji": "yi"}.get(v, v)
    return v[:12]


# ---- simhash ---------------------------------------------------------------------------------------------------

_MASK = (1 << 64) - 1
MIN_TOKENS = 4  # fewer words than this give no usable fingerprint (would collide on short reposts)


def _h64(token):
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def tokens(text):
    return [w.lower() for w in _WORD.findall(text or "")]


def simhash64(title, text=None, shingle=3):
    """Unsigned 64-bit simhash of title + text over word shingles; None when there are too few words."""
    words = tokens("%s %s" % (title or "", text or ""))
    if len(words) < MIN_TOKENS:
        return None
    n = min(shingle, len(words))
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    weights = [0] * 64
    for g in grams:
        h = _h64(g)
        for b in range(64):
            weights[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(64):
        if weights[b] > 0:
            out |= 1 << b
    return out & _MASK


def to_signed(u):
    """Unsigned 64-bit -> signed (fits SQLite INTEGER)."""
    return None if u is None else (u - (1 << 64) if u >= (1 << 63) else u)


def to_unsigned(s):
    return None if s is None else int(s) & _MASK


def hamming(a, b):
    return bin((a ^ b) & _MASK).count("1")
