"""Origin attribution ("where the publisher / channel is based") for the Propaganda Lab.

Social: a channel registry for broadcast-style platforms (telegram, youtube, instagram, facebook,
linkedin, pinterest, snapchat). The highest-volume accounts are drafted by Gemma from the account
name plus 3 sample posts; only answers with confidence >= MIN_CONF are used, and every row goes
to docs/review/07_OUTLET_COUNTRY.csv for hand review (a reviewed iso3 there overrides Gemma).
Reddit users and long-tail accounts stay `unlocated`; language is NEVER used as a proxy.
News: fnp.geo.countries.domain_country (outlet table, then ccTLD), when the news DB is reachable.
"""
from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from fnp.config import DATA_DIR, FNP_ROOT

REG_PATH = DATA_DIR / "geo" / "channels_v1.json"
REVIEW_CSV = FNP_ROOT / "docs" / "review" / "07_OUTLET_COUNTRY.csv"
BROADCAST = {"telegram": 60, "youtube_video": 200, "instagram": 200, "facebook": 60, "linkedin": 60,
             "pinterest": 30, "snapchat": 100}          # platform -> top-N accounts by volume
MIN_CONF = 0.7

SYSTEM = ("You identify where a media account is based. Return ONLY JSON "
          "{\"iso3\": \"<ISO 3166-1 alpha-3>\" or null, \"confidence\": 0..1, \"basis\": \"<short reason>\"}. "
          "Use the account name and the sample posts. The country is where the publisher/organisation "
          "operates from (e.g. RT -> RUS, Times of India -> IND, SCMP -> HKG, BBC Chinese -> GBR), not "
          "the topic of the posts. If it is an individual, aggregator or unclear, return null.")


def _ask(platform: str, name: str, samples: list[str]) -> dict:
    from fnp.llm.cache import cached_call

    def val(o):
        if not isinstance(o, dict) or "iso3" not in o or "confidence" not in o:
            return "need iso3 and confidence"
        if o["iso3"] is not None and (not isinstance(o["iso3"], str) or len(o["iso3"]) != 3):
            return "iso3 must be 3 letters or null"
        return None
    u = json.dumps({"platform": platform, "account": name, "sample_posts": [(s or "")[:300] for s in samples]},
                   ensure_ascii=False)
    r = cached_call("geo_origin", "geo_origin_v1", SYSTEM, u, val, 300)
    return r.get("obj") if r.get("ok") else {"iso3": None, "confidence": 0.0, "basis": "gemma failed"}


def build_channel_registry(posts: pd.DataFrame, concurrency: int = 8) -> dict:
    """posts: columns platform, author_name, text. Writes REG_PATH and the review CSV."""
    import pycountry
    valid = {c.alpha_3 for c in pycountry.countries}
    jobs = []
    for plat, top in BROADCAST.items():
        p = posts[(posts.platform == plat) & posts.author_name.notna()]
        for name, n in p.author_name.value_counts().head(top).items():
            jobs.append((plat, name, int(n), p[p.author_name == name].text.head(3).tolist()))
    with ThreadPoolExecutor(concurrency) as ex:
        ans = list(ex.map(lambda j: _ask(j[0], j[1], j[3]), jobs))
    reviewed = {}
    if REVIEW_CSV.exists():
        for r in csv.DictReader(REVIEW_CSV.open()):
            if r.get("reviewed_iso3", "").strip():
                reviewed[(r["platform"], r["account"])] = r["reviewed_iso3"].strip().upper()
    reg, rows = {}, []
    for (plat, name, n, _), a in zip(jobs, ans):
        iso = (a.get("iso3") or "").upper() or None
        conf = float(a.get("confidence") or 0)
        use = reviewed.get((plat, name)) or (iso if iso in valid and conf >= MIN_CONF else None)
        if use and use != "NONE":
            reg[f"{plat}|{name}"] = {"iso3": use, "method": "reviewed" if (plat, name) in reviewed else "gemma_registry",
                                     "confidence": conf}
        rows.append({"platform": plat, "account": name, "n_posts": n, "gemma_iso3": iso or "",
                     "confidence": round(conf, 2), "basis": (a.get("basis") or "")[:160],
                     "used_iso3": use or "", "reviewed_iso3": reviewed.get((plat, name), "")})
    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REG_PATH.write_text(json.dumps({"version": "channels_v1", "min_conf": MIN_CONF, "channels": reg},
                                   ensure_ascii=False, indent=0))
    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["n_posts"]))
    return reg


def load_registry(path: Path = REG_PATH) -> dict:
    return json.loads(path.read_text())["channels"] if path.exists() else {}


def origin_of(platform: str, author: str, registry: dict) -> tuple[str, str] | None:
    e = registry.get(f"{platform}|{author}")
    return (e["iso3"], e["method"]) if e else None
