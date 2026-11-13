"""Incremental ingest from the news/social DBs into annot.item (source DBs: read-only).

newest   : watermark on the source's ingestion time (scraped_at) with a keyset (ts, pk); each run reads
           only rows newer than the stored watermark (minus a lookback window, so late-arriving rows
           are still seen; duplicates are dropped by the unique keys). Candidates are taken newest-first
           and quota-balanced by language. If a run is truncated by `per_source`, the skipped gap is
           reachable through backfill.
backfill : random sample (TABLESAMPLE) across the whole table or a date window, time-balanced
           (round-robin over months) and language-quota balanced. Independent of the watermark.
Annotators are NOT served items in ingest order: the queue (service.next_item) is stratified-random.

  .venv/bin/python -m fnp.annotate.ingest --mode newest --per-source 60            # dry run
  .venv/bin/python -m fnp.annotate.ingest --mode newest --per-source 60 --apply
  .venv/bin/python -m fnp.annotate.ingest --mode backfill --per-source 60 --since 2026-01-01 --apply
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import timedelta
from typing import Iterable, Optional

from psycopg2 import sql

from ..config import C
from . import sources as S
from .db import connect_source, tx

OVERSAMPLE = 6  # candidate rows pulled per row wanted (filters + quotas + dedup discard many)


def ensure_sources(only: Optional[Iterable[str]] = None) -> dict[str, int]:
    """Register every source (idempotent) and return {platform: source_id}."""
    ids = {}
    with tx() as cur:
        for s in S.SOURCES:
            if only and s.platform not in only:
                continue
            cur.execute("INSERT INTO ingest_source (kind,platform,source_db,source_table) VALUES (%s,%s,%s,%s) "
                        "ON CONFLICT (kind,platform,source_table) DO UPDATE SET source_db=EXCLUDED.source_db "
                        "RETURNING source_id", (s.kind, s.platform, s.db, s.table))
            sid = cur.fetchone()["source_id"]
            cur.execute("INSERT INTO ingest_watermark (source_id, lookback) VALUES (%s, make_interval(hours => %s)) "
                        "ON CONFLICT DO NOTHING", (sid, int(C("ann_lookback_hours"))))
            ids[s.platform] = sid
    return ids


def _utc_naive(d):
    """Source tables mix timestamp and timestamptz; store everything as naive UTC."""
    if d is not None and d.tzinfo is not None:
        from datetime import timezone
        return d.astimezone(timezone.utc).replace(tzinfo=None)
    return d


def _fetch(src: S.Src, mode: str, n_cand: int, wm: Optional[dict], since, until, lookback: timedelta,
           seed: int) -> tuple[list[dict], int]:
    """-> (rows, n_raw). Read-only SELECTs on the source DB."""
    where = [S.length_where(src)]
    if mode == "newest" and wm and wm["newest_ts"] is not None:
        where.append(sql.SQL("{} > {}").format(sql.Identifier(src.ts), sql.Literal(wm["newest_ts"] - lookback)))
    if since:
        where.append(sql.SQL("{} >= {}").format(sql.Identifier(src.pub), sql.Literal(since)))
    if until:
        where.append(sql.SQL("{} < {}").format(sql.Identifier(src.pub), sql.Literal(until)))
    w = sql.SQL(" AND ").join(where)
    con = connect_source(src.kind)
    try:
        cur = con.cursor()
        if mode == "newest":
            q = S.select_sql(src, w, sql.SQL("ORDER BY {ts} DESC, {pk}::text DESC LIMIT {n}").format(
                ts=sql.Identifier(src.ts), pk=sql.Identifier(src.pk), n=sql.Literal(n_cand)))
        else:
            cur.execute("SELECT GREATEST(reltuples, 1)::bigint FROM pg_class WHERE oid = %s::regclass", (src.table,))
            est = cur.fetchone()[0]
            pct = min(100.0, max(0.01, 100.0 * n_cand * 3 / est))
            q = S.select_sql(src, w, sql.SQL("LIMIT {n}").format(n=sql.Literal(n_cand * 3)), sample_pct=pct)
        cur.execute(q)
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
        for r in rows:
            r["ts"], r["pub"] = _utc_naive(r["ts"]), _utc_naive(r["pub"])
        con.rollback()
    finally:
        con.close()
    return rows, len(rows)


def _prepare(src: S.Src, r: dict) -> Optional[dict]:
    lo, hi = S.length_bounds(src)
    body, cut = r["body"], None
    if src.truncate:
        body, cut = S.truncate_at_boundary(S.clean_text(body or ""), hi, C("ann_trunc_min_frac"))
    text, title, tl = S.compose(r["title"], body)
    if not (lo <= len(text) <= hi + (tl or 0)):
        return None
    letters = sum(ch.isalpha() for ch in text)
    if letters < 0.4 * min(len(text), 200):           # mostly emoji / urls / numbers
        return None
    if S.substance(text[tl:]) < C("ann_min_substance_chars"):   # link-only / mention-only post
        return None
    lang, grp = S.lang_group(r["lang"], text)
    if C(f"ann_share_{grp}") <= 0:                    # languages annotators were not recruited for are never ingested
        return None
    h = S.norm_hash(text)
    pub = r["pub"]
    return dict(source_pk=r["pk"], url=r["url"], author=(r["author"] or None), outlet=(r["outlet"] or None),
                published_at=pub, scraped_at=r["ts"], lang=lang, lang_group=grp,
                period=pub.strftime("%Y-%m") if pub else "unknown", title=title or None, title_len=tl, text=text,
                text_sha256=S.sha256(text), norm_hash=h, body_key=S.body_key(text, tl),
                dup_key=S.video_group_key(str(r["pk"])) if src.group_by_pk else S.dup_key(text),
                meta_extra={"text_column": src.text[0], **({"truncated": cut} if cut else {})},
                is_double=S.flag_fraction(h, "double") < C("ann_double_frac"),
                is_blind=S.flag_fraction(h, "blind") < C("ann_blind_frac"))


def _select(cands: list[dict], n: int, mode: str, rng: random.Random) -> list[dict]:
    """Language quotas, then fill from leftovers. newest: newest-first. backfill: round-robin over months."""
    shares = {g: C(f"ann_share_{g}") for g in ("en", "hi", "other") if C(f"ann_share_{g}") > 0}
    by = defaultdict(list)
    for c in cands:
        by[c["lang_group"]].append(c)

    def ordered(rows):
        if mode == "newest":
            return sorted(rows, key=lambda c: (c["scraped_at"], c["source_pk"]), reverse=True)
        per = defaultdict(list)
        for c in rows:
            per[c["period"]].append(c)
        for v in per.values():
            rng.shuffle(v)
        out, keys = [], sorted(per)
        while any(per.values()):
            rng.shuffle(keys)
            for k in keys:
                if per[k]:
                    out.append(per[k].pop())
        return out

    pools = {g: ordered(by[g]) for g in shares}
    take, rest = [], []
    for g, sh in shares.items():
        k = round(n * sh)
        take += pools[g][:k]
        rest += pools[g][k:]
    if len(take) < n:
        take += ordered(rest)[: n - len(take)]
    return take[:n]


INSERT = """INSERT INTO item (source_id, source_pk, ingest_mode, run_id, url, author, outlet, published_at, scraped_at,
    lang, lang_group, period, title, title_len, text, text_sha256, norm_hash, body_key, dup_key, is_double, is_blind, meta)
    VALUES (%(source_id)s, %(source_pk)s, %(mode)s, %(run_id)s, %(url)s, %(author)s, %(outlet)s, %(published_at)s,
    %(scraped_at)s, %(lang)s, %(lang_group)s, %(period)s, %(title)s, %(title_len)s, %(text)s, %(text_sha256)s,
    %(norm_hash)s, %(body_key)s, %(dup_key)s, %(is_double)s, %(is_blind)s, %(meta)s)
    ON CONFLICT DO NOTHING RETURNING item_id"""


def ingest_source(src: S.Src, source_id: int, mode: str, per_source: int, apply: bool, since=None, until=None,
                  seed: int = 11) -> dict:
    rng = random.Random(seed)
    with tx() as cur:
        cur.execute("SELECT newest_ts, newest_pk, lookback FROM ingest_watermark WHERE source_id=%s", (source_id,))
        wm = cur.fetchone() or {"newest_ts": None, "newest_pk": None,
                                "lookback": timedelta(hours=int(C("ann_lookback_hours")))}
        run_id = None
        if apply:
            cur.execute("INSERT INTO ingest_run (source_id, mode, params) VALUES (%s,%s,%s) RETURNING run_id",
                        (source_id, mode, json.dumps({"per_source": per_source, "since": str(since), "until": str(until)})))
            run_id = cur.fetchone()["run_id"]
    res = dict(platform=src.platform, mode=mode, fetched=0, filtered=0, dup=0, selected=0, inserted=0, error=None)
    try:
        rows, n_raw = _fetch(src, mode, per_source * OVERSAMPLE, wm, since, until, wm["lookback"], seed)
        res["fetched"] = n_raw
        prepared, seen, seen_bk = [], set(), set()
        for r in rows:
            p = _prepare(src, r)
            if p is None:
                res["filtered"] += 1
            elif p["norm_hash"] in seen or p["body_key"] in seen_bk:
                res["dup"] += 1
            else:
                seen.add(p["norm_hash"])
                seen_bk.add(p["body_key"])
                prepared.append(p)
        # drop what is already stored (so quota is not wasted on rows that would conflict)
        with tx() as cur:
            cur.execute("SELECT norm_hash FROM item WHERE norm_hash = ANY(%s)", (list(seen),))
            have = {r["norm_hash"] for r in cur.fetchall()}
            cur.execute("SELECT body_key FROM item WHERE body_key = ANY(%s)", (list(seen_bk),))   # a repost of a stored item
            have_bk = {r["body_key"].strip() for r in cur.fetchall()}
            cur.execute("SELECT source_pk FROM item WHERE source_id=%s AND source_pk = ANY(%s)",
                        (source_id, [p["source_pk"] for p in prepared]))
            have_pk = {r["source_pk"] for r in cur.fetchall()}
        fresh = [p for p in prepared if p["norm_hash"] not in have and p["source_pk"] not in have_pk and p["body_key"] not in have_bk]
        res["dup"] += len(prepared) - len(fresh)
        chosen = _select(fresh, per_source, mode, rng)
        res["selected"] = len(chosen)
        res["by_lang"] = {g: sum(c["lang_group"] == g for c in chosen) for g in ("en", "hi", "other")}
        if apply:
            n_ins = 0
            with tx() as cur:
                for p in chosen:
                    cur.execute(INSERT, {**p, "source_id": source_id, "mode": mode, "run_id": run_id,
                                         "meta": json.dumps({"normalisation": "nul-removed,crlf->lf,strip",
                                                             "source_table": src.table, **p["meta_extra"]})})
                    n_ins += cur.rowcount
                if mode == "newest" and rows:
                    top = max(rows, key=lambda r: (r["ts"], r["pk"]))
                    if wm["newest_ts"] is None or (top["ts"], top["pk"]) > (wm["newest_ts"], wm["newest_pk"] or ""):
                        cur.execute("UPDATE ingest_watermark SET newest_ts=%s, newest_pk=%s WHERE source_id=%s",
                                    (top["ts"], top["pk"], source_id))
                cur.execute("UPDATE ingest_watermark SET last_run_at=now(), rows_seen=rows_seen+%s WHERE source_id=%s",
                            (n_raw, source_id))
            res["inserted"] = n_ins          # only after the transaction committed
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    if apply:
        with tx() as cur:
            cur.execute("UPDATE ingest_run SET finished_at=now(), n_fetched=%s, n_filtered=%s, n_dup=%s, n_inserted=%s, "
                        "error=%s WHERE run_id=%s", (res["fetched"], res["filtered"], res["dup"], res["inserted"],
                                                     res["error"], run_id))
    return res


def run(mode: str, per_source: int, apply: bool, platforms=None, since=None, until=None, seed: int = 11) -> list[dict]:
    ids = ensure_sources(platforms) if apply else {}
    out = []
    for s in S.SOURCES:
        if platforms and s.platform not in platforms:
            continue
        out.append(ingest_source(s, ids.get(s.platform, 0), mode, per_source, apply, since, until, seed))
    return out


def servable_items() -> int:
    """Items in the annotation DB that can still be served or were finished (not excluded)."""
    with tx() as cur:
        cur.execute("SELECT count(*) n FROM item WHERE status <> 'excluded' AND lang_group = ANY(%s)",
                    ([g for g in ("en", "hi", "other") if C(f"ann_share_{g}") > 0],))
        return cur.fetchone()["n"]


def fill(total: int, per_source: int = 250, max_rounds: int = 60, platforms=None, progress=None) -> dict:
    """Grow the pool to `total` servable items: one newest-first round (catches up with the watermark), then random
    historical backfill rounds. Stops at `total`, when a round adds almost nothing, or after max_rounds."""
    start = servable_items()
    rounds, log = 0, []
    mode = "newest"
    while servable_items() < total and rounds < max_rounds:
        res = run(mode, per_source, True, platforms, None, None, seed=11 + rounds)
        added = sum(r["inserted"] for r in res)
        errs = [r["error"] for r in res if r["error"]]
        log.append({"round": rounds + 1, "mode": mode, "inserted": added, "errors": errs[:3]})
        if progress:
            progress(log[-1])
        rounds += 1
        mode = "backfill"
        if added < max(5, 0.02 * per_source * len(res)) and rounds > 2:
            break                                            # sources are (nearly) exhausted for the current filters
    return {"start": start, "end": servable_items(), "target": total, "rounds": log}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["newest", "backfill"], default="newest")
    ap.add_argument("--per-source", type=int, default=60)
    ap.add_argument("--platform", action="append", help="repeatable; default all")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--fill-to", type=int, help="grow the pool to this many servable items (newest round, then backfill rounds)")
    a = ap.parse_args()
    if a.fill_to:
        if not a.apply:
            print(f"dry run: would fill from {servable_items()} to {a.fill_to} items in rounds of {a.per_source}/source. Add --apply.")
            return
        print(json.dumps(fill(a.fill_to, a.per_source, platforms=set(a.platform or [])), indent=1))
        return
    for r in run(a.mode, a.per_source, a.apply, set(a.platform or []), a.since, a.until, a.seed):
        print(json.dumps(r))
    if not a.apply:
        print("dry run: nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
