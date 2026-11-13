"""Country attribution for the Propaganda Lab ("about" = countries a text mentions).

Gazetteer (data/geo/gazetteer_v1.json), built once by `build_gazetteer()`:
  * pycountry names (name / common / official) and geonamescache capitals + cities >= 1M people
    (a city name shared by two countries with >= 1/3 the population of the larger is dropped);
  * Gemma-drafted multilingual aliases + demonyms (en, hi, bn, ur, ta, ru, uk, zh, ar, fr, es, de),
    one cached call per country; an alias claimed by two countries is dropped from both.
Matching (`CountryMatcher.match`): Aho-Corasick over casefolded text, word-boundary check for
alphanumeric scripts (CJK has none), AMBIGUOUS terms only count when the same country is also hit
by another method in the text. A text mentioning k countries gives each weight 1/k.
ASSUMPTIONS: a mention means the text is "about" that country; demonyms count like names.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

import ahocorasick

from fnp.config import DATA_DIR

GAZ_PATH = DATA_DIR / "geo" / "gazetteer_v1.json"
ALIAS_LANGS = "en, hi, bn, ur, ta, ru, uk, zh, ar, fr, es, de"
CITY_MIN_POP = 1_000_000
# Words that are a country name but usually mean something else; count only with corroboration.
AMBIGUOUS = {"georgia", "jordan", "turkey", "chad", "niger", "guinea", "congo", "indian", "indians",
             "china", "mali", "peru", "cuba", "oman", "iran", "iraq", "india", "us", "usa", "uk",
             "dominica", "panama", "jersey", "man", "american", "americans", "korea", "sudan",
             "samoa", "monaco", "lebanon", "victoria", "salvador", "colombo", "kingston", "georgetown",
             "santiago", "valencia", "hyderabad", "aurangabad", "barcelona", "sydney", "melbourne",
             "perth", "cambridge", "birmingham", "manchester", "paris", "london", "delhi"}
# ...but these are unambiguous enough on their own in this corpus; listed so the set above stays
# conservative only for genuinely polysemous words.
NOT_AMBIGUOUS = {"china", "india", "iran", "iraq", "usa", "uk", "korea", "sudan", "lebanon",
                 "london", "paris", "delhi", "oman", "cuba", "peru", "panama"}
AMBIGUOUS -= NOT_AMBIGUOUS
STOP_TERMS = {"us", "man", "chad", "jordan", "victoria", "georgia", "turkey", "guinea", "niger",
              "indian", "indians", "american", "americans"}   # dropped unless corroborated (see match)

# Forced after de-duplication (high-volume terms lost to collisions or not generated).
# Kashmir/Ladakh -> IND follows the India worldview used by the map.
OVERRIDES = {
    "USA": ["american", "americans", "u.s.", "u.s.a.", "u.s", "америка", "американский", "американцы",
            "أمريكا", "أمريكي", "امریکہ", "امریکی", "अमेरिका", "अमेरिकी", "আমেরিকা", "আমেরিকান",
            "அமெரிக்கா", "美国", "美國", "washington d.c.", "white house", "pentagon"],
    "GBR": ["uk", "u.k.", "british", "britain", "great britain", "england", "scotland", "wales",
            "ब्रिटेन", "ब्रिटिश", "британия", "великобритания", "英国"],
    "UKR": ["україна", "український", "українська", "kyiv", "kiev", "киев", "київ"],
    "CHN": ["चीन", "चीनी", "中国", "中國", "китай", "китайский", "চীন", "چین", "الصين"],
    "PSE": ["gaza", "west bank", "gaza strip", "गाज़ा", "गाजा", "газа", "غزة"],
    "IND": ["kashmir", "jammu and kashmir", "j&k", "ladakh", "कश्मीर", "हिंदुस्तान", "hindustan", "bharat"],
    "KOR": ["seoul"], "PRK": ["pyongyang", "dprk"], "COD": ["drc", "kinshasa"], "COG": ["brazzaville"],
    "TWN": ["taipei", "台湾", "臺灣"], "RUS": ["kremlin", "россия", "русский", "rossiya"],
    "ISR": ["idf", "tel aviv"], "IRN": ["tehran", "ईरान"], "PAK": ["पाकिस्तान", "पाक"],
}

SHORT_OK = {"usa", "uae", "ksa", "prc", "drc", "dprk", "rok", "pak"}

ALIAS_SYSTEM = (
    "You produce gazetteer entries. Return ONLY JSON: {\"aliases\": [..], \"demonyms\": [..]}. "
    "aliases = names people actually use for the country in news and social media, in these "
    f"languages: {ALIAS_LANGS} (native script), plus common English short forms and abbreviations "
    "that are unambiguous. demonyms = adjective/people forms in the same languages "
    "(e.g. Indian, भारतीय, ভারতীয়). No city names, no leaders, no historical empires, at most 40 "
    "items per list, each 2-40 characters.")


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold().strip()


def _base_entries() -> dict:
    import geonamescache
    import pycountry
    gc = geonamescache.GeonamesCache()
    gcc = gc.get_countries()
    by3 = {c["iso3"]: c for c in gcc.values()}
    out = {}
    for c in pycountry.countries:
        terms = {}
        for k in ("name", "common_name", "official_name"):
            v = getattr(c, k, None)
            if v:
                terms[norm(v)] = "name"
                if "," in v:                        # "Korea, Republic of" -> "Korea"
                    terms[norm(v.split(",")[0])] = "name"
        g = by3.get(c.alpha_3)
        if g and g.get("capital"):
            terms[norm(g["capital"])] = "capital"
        out[c.alpha_3] = {"name": getattr(c, "common_name", None) or c.name, "iso2": c.alpha_2,
                          "tld": (g or {}).get("tld", ""), "terms": terms}
    # cities >= CITY_MIN_POP, unique enough
    cities = {}
    for ct in gc.get_cities().values():
        if ct["population"] < CITY_MIN_POP:
            continue
        cc = gcc.get(ct["countrycode"])
        if not cc:
            continue
        for nm in {ct["name"], *[a for a in ct.get("alternatenames", []) if a and len(a) > 3][:0]}:
            cities.setdefault(norm(nm), []).append((ct["population"], cc["iso3"]))
    for nm, lst in cities.items():
        lst.sort(reverse=True)
        if len({i for _, i in lst}) > 1 and lst[1][0] * 3 >= lst[0][0]:
            continue
        iso = lst[0][1]
        if iso in out:
            out[iso]["terms"].setdefault(nm, "city")
    return out


def _gemma_aliases(iso3: str, name: str) -> dict:
    from fnp.llm.cache import cached_call

    def val(o):
        if not isinstance(o, dict) or not isinstance(o.get("aliases"), list) or not isinstance(o.get("demonyms"), list):
            return "need aliases and demonyms lists"
        return None
    r = cached_call("geo_alias", "geo_alias_v1", ALIAS_SYSTEM,
                    json.dumps({"country": name, "iso3": iso3}, ensure_ascii=False), val, 1200)
    return r.get("obj") if r.get("ok") else {"aliases": [], "demonyms": []}


def build_gazetteer(path: Path = GAZ_PATH, use_gemma: bool = True, concurrency: int = 8) -> dict:
    base = _base_entries()
    if use_gemma:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(concurrency) as ex:
            res = dict(zip(base, ex.map(lambda k: _gemma_aliases(k, base[k]["name"]), base)))
        for k, o in res.items():
            for kind, lst in (("alias", o.get("aliases", [])), ("demonym", o.get("demonyms", []))):
                for t in lst:
                    if not isinstance(t, str) or not 2 <= len(t.strip()) <= 40:
                        continue
                    nt = norm(t)
                    # short Latin abbreviations collide with acronyms (e.g. "rss" -> South Sudan)
                    if re.fullmatch(r"[a-z.&\-]{1,4}", nt) and nt not in SHORT_OK:
                        continue
                    base[k]["terms"].setdefault(nt, kind)
    owners = {}
    for k, e in base.items():
        for t in e["terms"]:
            owners.setdefault(t, set()).add(k)
    dup = {t for t, s in owners.items() if len(s) > 1}
    for e in base.values():
        e["terms"] = {t: m for t, m in e["terms"].items()
                      if t not in dup and (len(t) >= 3 or _is_cjk(t))}
    for iso, terms in OVERRIDES.items():
        for t in terms:
            for e in base.values():
                e["terms"].pop(norm(t), None)
            base[iso]["terms"][norm(t)] = "override"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": "gazetteer_v1", "dropped_shared_terms": sorted(dup),
                                "countries": base}, ensure_ascii=False, indent=0))
    return base


def _is_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" or "가" <= ch <= "힯" for ch in s)


def _word_char(ch: str) -> bool:
    return ch.isalnum() or unicodedata.category(ch).startswith("M")   # letters + combining marks


# Phrases that contain a country term but are not about that country; hits inside them are dropped.
BLOCKERS = ["latin america", "south america", "central america", "north america", "latin american",
            "south american", "american samoa", "new mexico", "new jersey", "guinea pig", "turkey day",
            "indian ocean", "native american", "native americans", "american indian", "west indies",
            "east india company", "jordan peterson", "michael jordan", "chad smith", "french fries",
            "china town", "chinatown", "dutch oven", "english language", "british columbia"]
_FLAG_RE = re.compile("[\U0001F1E6-\U0001F1FF]{2}")


class CountryMatcher:
    def __init__(self, gaz: Optional[dict] = None):
        gaz = gaz or json.loads(GAZ_PATH.read_text())["countries"]
        self.names = {k: v["name"] for k, v in gaz.items()}
        self.iso2 = {v["iso2"]: k for k, v in gaz.items() if v.get("iso2")}
        self.A = ahocorasick.Automaton()
        for iso, e in gaz.items():
            for t, m in e["terms"].items():
                self.A.add_word(t, (t, iso, m))
        for b in BLOCKERS:
            self.A.add_word(b, (b, None, "blocker"))
        self.A.make_automaton()

    def flags(self, text: str) -> list[tuple[str, str, str]]:
        out = []
        for f in _FLAG_RE.findall(text or ""):
            cc = "".join(chr(ord(ch) - 0x1F1E6 + 65) for ch in f)
            if cc in self.iso2:
                out.append((f, self.iso2[cc], "flag"))
        return out

    def hits(self, text: str) -> list[tuple[str, str, str]]:
        s = norm(text)
        raw, blocks = [], []
        for end, (t, iso, m) in self.A.iter(s):
            st = end - len(t) + 1
            if not _is_cjk(t):
                if st > 0 and _word_char(s[st - 1]):
                    continue
                if end + 1 < len(s) and _word_char(s[end + 1]):
                    continue
            if iso is None:
                blocks.append((st, end))
            else:
                raw.append((st, end, t, iso, m))
        out = [(t, iso, m) for st, end, t, iso, m in raw
               if not any(bs <= st and end <= be for bs, be in blocks)]
        return out + self.flags(text)

    def match(self, text: str) -> list[dict]:
        """-> [{iso3, weight, methods, terms}] (empty = unlocated)."""
        hs = self.hits(text or "")
        if not hs:
            return []
        strong = {iso for t, iso, m in hs if t not in AMBIGUOUS and t not in STOP_TERMS}
        keep = {}
        for t, iso, m in hs:
            if (t in AMBIGUOUS or t in STOP_TERMS) and iso not in strong:
                continue
            e = keep.setdefault(iso, {"iso3": iso, "methods": set(), "terms": set()})
            e["methods"].add(m)
            e["terms"].add(t)
        k = len(keep)
        return [{"iso3": e["iso3"], "weight": 1.0 / k, "methods": sorted(e["methods"]),
                 "terms": sorted(e["terms"])[:5]} for e in keep.values()]


@lru_cache(maxsize=1)
def matcher() -> CountryMatcher:
    return CountryMatcher()


_TLD_RE = re.compile(r"\.([a-z]{2})$")


@lru_cache(maxsize=1)
def _tld_map() -> dict:
    gaz = json.loads(GAZ_PATH.read_text())["countries"]
    m = {e["tld"].lstrip(".").lower(): k for k, e in gaz.items() if e.get("tld")}
    for generic in ("tv", "io", "ai", "co", "me", "fm", "ly", "to", "cc", "ws", "gg", "la"):
        m.pop(generic, None)                   # ccTLDs used as generic vanity domains
    return m


def domain_country(domain: str, outlets: Optional[dict] = None) -> Optional[tuple[str, str]]:
    """Publisher country for a news domain -> (iso3, method) or None. Outlet table wins over ccTLD."""
    d = (domain or "").lower().removeprefix("www.").strip(".")
    if not d:
        return None
    if outlets:
        parts = d.split(".")
        for i in range(len(parts) - 1):
            if ".".join(parts[i:]) in outlets:
                return outlets[".".join(parts[i:])], "outlet"
    m = _TLD_RE.search(d)
    if m and m.group(1) in _tld_map():
        return _tld_map()[m.group(1)], "cctld"
    return None
