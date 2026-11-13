"""Fail-soft web retrieval: DuckDuckGo HTML search + trafilatura article fetch, disk-cached."""
from __future__ import annotations

import hashlib
import html as _html
import json
import re
import urllib.parse
import urllib.request
from typing import Optional

from ..config import CACHE_DIR

WEB_CACHE = CACHE_DIR / "web"
UA = "Mozilla/5.0 (compatible; FNP-research-bot/0.1; polite, low-volume)"
TIMEOUT = 15

_RESULT = re.compile(r'<a[^>]+class="result__a"[^>]+href="(?P<u>[^"]+)"[^>]*>(?P<t>.*?)</a>', re.S)
_SNIP = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(?P<s>.*?)</a>', re.S)
_TAG = re.compile(r"<[^>]+>")


def _key(kind: str, s: str) -> str:
    return hashlib.sha1(f"{kind}|{s}".encode()).hexdigest()


def _cache_get(kind, s):
    p = WEB_CACHE / f"{_key(kind, s)}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def _cache_put(kind, s, val):
    try:
        WEB_CACHE.mkdir(parents=True, exist_ok=True)
        (WEB_CACHE / f"{_key(kind, s)}.json").write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(_TAG.sub("", s))).strip()


def _real_url(u: str) -> str:
    u = _html.unescape(u)
    if u.startswith("//"):
        u = "https:" + u
    q = urllib.parse.urlparse(u)
    if "duckduckgo.com" in q.netloc and q.path.startswith("/l/"):
        t = urllib.parse.parse_qs(q.query).get("uddg")
        if t:
            return t[0]
    return u


def parse_ddg_html(page: str, k: int = 5) -> list[dict]:
    out = []
    parts = re.split(r'(?=<div[^>]+class="[^"]*\bresult\b)', page)
    for blk in parts:
        m = _RESULT.search(blk)
        if not m:
            continue
        if "result--ad" in blk[:200]:
            continue
        s = _SNIP.search(blk)
        out.append(dict(title=_clean(m["t"]), url=_real_url(m["u"]), snippet=_clean(s["s"]) if s else ""))
        if len(out) >= k:
            break
    return out


def web_search(query: str, k: int = 5) -> list[dict]:
    try:
        c = _cache_get("search", query)
        if c is not None:
            return c[:k]
        req = urllib.request.Request("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query),
                                     headers={"User-Agent": UA, "Accept-Language": "en"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            page = r.read().decode("utf-8", "replace")
        res = parse_ddg_html(page, max(k, 10))
        if res:
            _cache_put("search", query, res)
        return res[:k]
    except Exception:
        return []


def fetch_article(url: str) -> Optional[dict]:
    try:
        c = _cache_get("article", url)
        if c is not None:
            return c or None
        import trafilatura
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read(5_000_000).decode("utf-8", "replace")
        text = trafilatura.extract(raw, url=url, include_comments=False)
        if not text:
            return None
        md = trafilatura.extract_metadata(raw)
        art = dict(url=url, title=(md.title if md and md.title else ""), text=text,
                   published=(md.date if md and md.date else None))
        _cache_put("article", url, art)
        return art
    except Exception:
        return None
