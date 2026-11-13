"""Newest-first item sources for the live scan. Everything here is READ-ONLY.

social : EDA's incoming queue (posts newer than the last extract), then the EDA parquet.
news   : shaktidb_news.scraped_articles_trusted over Postgres (needs the network route;
         columns are discovered at runtime because the schema was not visible when written).
Each source yields dicts {id, source, platform, url, text, lang, posted_at}.
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

import duckdb
import pandas as pd

from fnp.config import FNP_ROOT
from fnp.eda_bridge import read_parquet

MIN_CHARS = 25
INCOMING = FNP_ROOT.parent / "EDA" / "data" / "incoming_posts.duckdb"
TEXT_COLS = ("content", "article_text", "body", "text", "full_text", "description")
TITLE_COLS = ("title", "headline")
TIME_COLS = ("published_at", "publish_date", "published", "date", "scraped_at", "created_at")


def _rows(df: pd.DataFrame, source: str, plat_default: str = "") -> Iterator[dict]:
    for r in df.itertuples(index=False):
        t = (r.text or "").strip()
        if len(t) < MIN_CHARS:
            continue
        yield {"id": f"{source}:{r.post_uid}", "source": source,
               "platform": getattr(r, "platform", plat_default) or plat_default,
               "url": getattr(r, "url", None), "text": t, "lang": getattr(r, "lang", None),
               "posted_at": str(r.posted_at)}


def social_local(limit: Optional[int] = None) -> Iterator[dict]:
    """Incoming queue first (newest), then the parquet, both ordered posted_at desc."""
    seen = set()
    n = 0
    if INCOMING.exists():
        try:
            con = duckdb.connect(str(INCOMING), read_only=True)
            q = con.execute("SELECT post_uid, platform, lang, text, posted_at FROM incoming_posts "
                            "ORDER BY posted_at DESC").df()
            con.close()
            for d in _rows(q, "social"):
                seen.add(d["id"]); n += 1
                yield d
                if limit and n >= limit:
                    return
        except Exception:
            pass
    t = read_parquet("post_facts")[["post_uid", "platform", "url", "text", "lang", "posted_at"]]
    t = t.sort_values("posted_at", ascending=False)
    for d in _rows(t, "social"):
        if d["id"] in seen:
            continue
        yield d
        n += 1
        if limit and n >= limit:
            return


def _pg(prefix: str):
    import psycopg2
    c = psycopg2.connect(host=os.environ[f"{prefix}_DB_HOST"], port=int(os.environ[f"{prefix}_DB_PORT"]),
                         dbname=os.environ[f"{prefix}_DB_NAME"], user=os.environ[f"{prefix}_DB_USER"],
                         password=os.environ[f"{prefix}_DB_PASSWORD"], connect_timeout=8)
    c.set_session(readonly=True, isolation_level="REPEATABLE READ")
    return c


def pg_reachable(prefix: str = "NEWS") -> tuple[bool, str]:
    try:
        _pg(prefix).close()
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def news_pg(limit: int = 2000, offset: int = 0) -> Iterator[dict]:
    """Newest articles from scraped_articles_trusted (read-only, one SELECT per call)."""
    c = _pg("NEWS")
    try:
        cur = c.cursor()
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='scraped_articles_trusted'")
        cols = [r[0] for r in cur.fetchall()]
        text_c = [x for x in TEXT_COLS if x in cols]
        title_c = next((x for x in TITLE_COLS if x in cols), None)
        time_c = next((x for x in TIME_COLS if x in cols), None)
        if not text_c and not title_c:
            raise RuntimeError(f"no text-like column in scraped_articles_trusted: {cols}")
        sel = ["id"] + ([title_c] if title_c else []) + text_c[:1] + ([time_c] if time_c else []) \
            + [x for x in ("url", "link", "lang", "language", "source", "domain") if x in cols]
        q = (f'SELECT {", ".join(dict.fromkeys(sel))} FROM scraped_articles_trusted '
             f'ORDER BY {time_c or "id"} DESC NULLS LAST LIMIT %s OFFSET %s')
        cur.execute(q, (limit, offset))
        names = [d[0] for d in cur.description]
        for row in cur.fetchall():
            r = dict(zip(names, row))
            body = (r.get(text_c[0]) or "") if text_c else ""
            text = ((r.get(title_c) or "") + ". " + body).strip(". ").strip() if title_c else body
            if len(text) < MIN_CHARS:
                continue
            yield {"id": f"news:{r['id']}", "source": "news",
                   "platform": r.get("source") or r.get("domain") or "news",
                   "url": r.get("url") or r.get("link"), "text": text[:6000],
                   "lang": r.get("lang") or r.get("language"),
                   "posted_at": str(r.get(time_c)) if time_c else ""}
        c.rollback()
    finally:
        c.close()
