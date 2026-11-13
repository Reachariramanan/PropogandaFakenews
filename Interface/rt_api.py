"""Read-only JSON API logic for the Evidence Base (/api/rt/*). server.py owns routing, auth and rate limiting.

Everything here reads the LOCAL SQLite cache (rt_store / realtime_sync) plus the FNP enrichment sidecar (rt_enrich);
it never opens a connection to the customer databases. Every connection is `query_only`, every statement is
parameterised, every value is validated against a whitelist, and a progress handler aborts statements that run
longer than the time budget. Fields outside the `inferred` block come from the database; `inferred` is FNP's.
"""
import json
import os
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import claimcheck  # noqa: E402
import realtime_sync  # noqa: E402
import rt_enrich  # noqa: E402
import rt_scheduler  # noqa: E402
import rt_store  # noqa: E402
from rt_text import hamming, to_unsigned  # noqa: E402

PAGE_SIZE_MAX = 50
OFFSET_MAX = 10000
TOP_N = 15
QUERY_SECONDS = 12.0
STATS_SECONDS = 40.0
COUNT_CAP = 10000        # full-text result counts stop here (total_capped=true): counting every match is the slow part
STATS_TTL = 30.0
HEALTH_TTL = 15.0
SNIPPET = 280
SORTS = ("recent", "p_fake", "p_prop", "relevance")
KINDS = ("news", "social")
LIKELY_FAKE = ("FAKE", "MISLEADING")
ROOT = os.path.dirname(HERE)
PL_META = os.path.join(os.environ.get("FNP_PROPLAB_DIR") or os.path.join(ROOT, "reports", "prop_lab", "v1"), "meta.json")
BATCH_CAVEAT_DEFAULT = ("p_prop and technique_probs come from an encoder trained on long news articles and not validated "
                        "on social media or short items; use them as a ranking signal, not a calibrated probability.")

_RE_PLATFORM = re.compile(r"^[a-z0-9_]{1,40}$")
_RE_LANG = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?$|^und$")
_RE_TOPIC = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
_RE_ITEM = re.compile(r"^[A-Za-z0-9_.:\-]{1,200}$")
_RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


def _g(q, key, default=""):
    v = (q.get(key) or [default])[0]
    return v.strip() if isinstance(v, str) else default


def _enum(q, key, allowed, default=""):
    v = _g(q, key, default).lower()
    if v and v not in allowed:
        raise ApiError(400, "%s must be one of: %s" % (key, ", ".join(allowed)))
    return v


def _int(q, key, default, lo, hi):
    raw = _g(q, key, "")
    if raw == "":
        return default
    if not re.fullmatch(r"\d{1,9}", raw):
        raise ApiError(400, "%s must be an integer between %d and %d" % (key, lo, hi))
    v = int(raw)
    if v < lo or v > hi:
        raise ApiError(400, "%s must be an integer between %d and %d" % (key, lo, hi))
    return v


def _date_bound(raw, key, upper):
    """YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ -> (operator, ISO string) for published_at. Bad input -> 400."""
    if not raw:
        return None
    try:
        if _RE_DATE.match(raw):
            d = datetime.strptime(raw, "%Y-%m-%d")
            return ("<", (d + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")) if upper else \
                (">=", d.strftime("%Y-%m-%dT00:00:00Z"))
        if _RE_ISO.match(raw):
            datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
            return ("<=", raw) if upper else (">=", raw)
    except ValueError:
        pass
    raise ApiError(400, "%s must be YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ" % key)


def _pct(n, d):
    return round(100.0 * n / d, 1) if d else None


class CountryNames:
    """ISO3 -> display name from the Propaganda Lab build meta (`names`), re-read when the file changes. {} when absent."""

    def __init__(self, path=None):
        self.path, self._sig, self._names = path or PL_META, None, {}

    def get(self, iso3):
        try:
            st = os.stat(self.path)
            sig = (st.st_mtime_ns, st.st_size)
            if sig != self._sig:
                with open(self.path, encoding="utf-8") as f:
                    n = json.load(f).get("names")
                self._names = {k: v for k, v in n.items() if isinstance(k, str) and isinstance(v, str)} \
                    if isinstance(n, dict) else {}
                self._sig = sig
        except (OSError, ValueError, AttributeError):
            self._names = {}
        return self._names.get(iso3)

    def __getitem__(self, iso3):
        return self.get(iso3)


def batch_block(meta, too_short, n_scored_rows):
    """The FNP batch coverage numbers from meta.json (plus the sidecar's own counts). Every number is FNP's; nothing is
    estimated here except `estimated_runs`, which is null unless meta gives a per-run limit."""
    m = meta or {}

    def num(k):
        v = m.get(k)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    scored = num("n_items_total")
    scored = int(scored) if scored is not None else n_scored_rows
    n_win, n_uns = num("n_in_window"), num("n_unscored_in_window")
    limit = None
    for k in ("max_items_per_run", "per_run_limit", "max_per_run"):
        if isinstance(m.get(k), int) and not isinstance(m.get(k), bool) and m[k] > 0:
            limit = m[k]
            break
    n_run = num("n_scored_this_run")
    est = rate = None
    if limit is not None and n_uns is not None:
        rate = max(1, min(int(n_run) if n_run is not None else limit, limit))
        est = -(-int(n_uns) // rate) if n_uns > 0 else 0
    return dict(
        present=bool(m), scored=scored, n_in_window=int(n_win) if n_win is not None else None,
        n_unscored_in_window=int(n_uns) if n_uns is not None else None,
        pct_of_window=round(100.0 * scored / n_win, 3) if n_win else None, built_at=m.get("built_at") if isinstance(m.get("built_at"), str) else None,
        next_run_at=m.get("next_run_at") if isinstance(m.get("next_run_at"), str) else None,
        interval_hours=num("interval_hours"), n_scored_this_run=int(n_run) if n_run is not None else None,
        n_failed_this_run=int(num("n_failed_this_run") or 0), run_seconds=num("run_seconds"),
        caveat=m.get("caveat") if isinstance(m.get("caveat"), str) else None,
        model_version=m.get("model_version") if isinstance(m.get("model_version"), str) else None,
        last_item_published_at=m.get("last_item_published_at") if isinstance(m.get("last_item_published_at"), dict) else None,
        per_run_limit=limit, runs_rate=rate, estimated_runs=est, too_short=too_short)


class RtApi:
    def __init__(self, db_path=None, enrich=None, fn_matches=None, clock=None, desk_url=None, country_names=None):
        self.db_path = db_path or rt_store.DEFAULT_DB
        self.enrich = enrich or rt_enrich.Enrich(store_path=self.db_path, background=True)
        self.fn_matches = fn_matches  # callable(text) -> dict, supplied by server.py (TF-IDF over the fact-checks)
        self.desk_url = desk_url      # callable(url) -> Desk item id or None, supplied by server.py (normalised URL index)
        self.country_names = country_names if country_names is not None else CountryNames()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._cache = {}
        self._cache_lock = threading.Lock()

    # ---- connections --------------------------------------------------------------------------------------------
    def _con(self, seconds=QUERY_SECONDS):
        if not os.path.exists(self.db_path):
            raise ApiError(503, "realtime cache not created yet; run: python3 Interface/realtime_sync.py --once")
        con = rt_store.open_store(self.db_path, create=False)
        rt_store.ensure_deep_view(con)  # temp view of the deep (Claim Check) results; must precede query_only
        con.execute("PRAGMA query_only=ON")
        self.enrich.attach(con)
        end = time.monotonic() + seconds
        con.set_progress_handler(lambda: 1 if time.monotonic() > end else 0, 20000)
        return con

    def _run(self, fn, seconds=QUERY_SECONDS):
        con = self._con(seconds)
        try:
            return fn(con)
        except sqlite3.OperationalError as e:
            if "interrupt" in str(e).lower():
                raise ApiError(503, "query too slow; narrow the filters")
            raise
        finally:
            con.close()

    def _cached(self, key, ttl, make):
        now = time.monotonic()
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < ttl:
                return hit[1]
        val = make()
        with self._cache_lock:
            self._cache[key] = (now, val)
            if len(self._cache) > 64:
                for k in sorted(self._cache, key=lambda k: self._cache[k][0])[:16]:
                    del self._cache[k]
        return val

    # ---- shared filter parsing ----------------------------------------------------------------------------------
    def _scope(self, q):
        """Validated scope filters used by stats (kind, platform, days)."""
        kind = _enum(q, "kind", KINDS)
        platform = _g(q, "platform").lower()
        if platform and not _RE_PLATFORM.match(platform):
            raise ApiError(400, "platform must be lowercase letters, digits or underscore")
        days = _int(q, "days", 0, 0, 365)
        return kind, platform, days

    def _scope_sql(self, kind, platform, days, extra=None):
        conds, args = [], []
        if kind:
            conds.append("i.kind = ?")
            args.append(kind)
        if platform:
            conds.append("i.platform = ?")
            args.append(platform)
        if days:
            conds.append("i.published_at >= ?")
            args.append(rt_store.utc_iso(self.clock() - timedelta(days=days)))
        for c, a in (extra or []):
            conds.append(c)
            args.extend(a)
        return (" WHERE " + " AND ".join(conds)) if conds else "", args

    # ---- /api/rt/status -----------------------------------------------------------------------------------------
    def status(self):
        st = self._cached("status", 20.0, lambda: realtime_sync.status(self.db_path, now=self.clock()))
        policy = realtime_sync.effective_policy()
        return dict(st, policy=policy, enrichment=self.enrich.status(),
                    source_freshness=source_freshness(st, policy, self.clock()),
                    generated_at=rt_store.utc_iso(self.clock()))

    # ---- /healthz block -----------------------------------------------------------------------------------------
    def health(self, env=None):
        env = os.environ if env is None else env
        enabled = env.get("FNP_RT_SYNC") == "1"
        out = dict(enabled=enabled, running=bool(realtime_sync.background_status().get("running")),
                   last_sync=None, lag_minutes=None, rows=0, errors=0)

        def make():
            if not os.path.exists(self.db_path):
                return {}
            con = rt_store.open_store(self.db_path, create=False)
            try:
                con.execute("PRAGMA query_only=ON")
                last = con.execute("SELECT max(ts) FROM sync_log").fetchone()[0]
                ok = con.execute("SELECT max(ts) FROM sync_log WHERE error IS NULL").fetchone()[0]
                errs = con.execute("SELECT count(*) FROM sync_log WHERE ts = ? AND error IS NOT NULL", (last,)).fetchone()[0] \
                    if last else 0
                rows = con.execute("SELECT count(*) FROM items").fetchone()[0]
                newest = con.execute("SELECT max(published_at) FROM items").fetchone()[0]
                return dict(last_sync=ok, rows=rows, errors=errs, newest=newest)
            finally:
                con.close()
        try:
            h = self._cached("health", HEALTH_TTL, make)
            out.update(last_sync=h.get("last_sync"), rows=h.get("rows", 0), errors=h.get("errors", 0))
            if h.get("newest"):
                out["lag_minutes"] = round((self.clock() - rt_store.parse_iso(h["newest"])).total_seconds() / 60.0, 1)
        except Exception:  # noqa: BLE001  healthz must never fail because of the realtime block
            out["errors"] = None
        try:
            sc = rt_scheduler.schedule_status(self.db_path, env=env)
            out["analysis"] = dict(enabled=sc["enabled"], last_run=sc["last_run"], next_run_at=sc["next_run_at"],
                                   analysed_deep=sc["totals"]["analysed_deep"], partial_items=sc["totals"]["partial_items"],
                                   rolling=sc["rolling"])
        except Exception:  # noqa: BLE001
            out["analysis"] = dict(enabled=False, last_run=None, next_run_at=None, analysed_deep=0, partial_items=0,
                                   rolling=None)
        return out

    # ---- /api/rt/schedule -----------------------------------------------------------------------------------------
    def schedule(self):
        """Scheduled deep-analysis status plus the last 10 cycles (counts only, never item text)."""
        return dict(rt_scheduler.schedule_status(self.db_path, log_rows=rt_scheduler.LOG_KEEP),
                    generated_at=rt_store.utc_iso(self.clock()))

    # ---- /api/rt/stats ------------------------------------------------------------------------------------------
    def stats(self, q):
        kind, platform, days = self._scope(q)
        enr = self.enrich.status()
        key = ("stats", kind, platform, days, enr.get("loaded_at"), enr.get("records"), self._deep_sig())
        # the sync_log tip changes whenever a sync finishes, so a cached page never outlives new data by > STATS_TTL
        return self._cached(key, STATS_TTL, lambda: self._run(lambda con: self._stats(con, kind, platform, days), STATS_SECONDS))

    def _deep_sig(self):
        """(rows, newest created_at) of `analyses`: part of the stats cache key so a new deep result shows at once."""
        try:
            con = rt_store.open_store(self.db_path, create=False)
            try:
                return tuple(con.execute("SELECT count(*), coalesce(max(created_at), '') FROM analyses").fetchone())
            finally:
                con.close()
        except Exception:  # noqa: BLE001
            return None

    def _stats(self, con, kind, platform, days):
        where, args = self._scope_sql(kind, platform, days)
        has_idx = con.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='items_stats'").fetchone()
        # the covering index makes the group-bys index-only scans (4-5x faster on 500k+ rows); older stores lack it
        FI = "items i INDEXED BY items_stats" if has_idx else "items i"
        one = lambda sql, a=args: con.execute(sql, a).fetchone()
        total = one("SELECT count(*) FROM items i" + where)[0]
        res = dict(scope=dict(kind=kind or None, platform=platform or None, days=days or None), total=total)
        res["by_kind"] = {r[0]: r[1] for r in con.execute("SELECT i.kind, count(*) FROM items i%s GROUP BY i.kind" % where, args)}
        res["by_platform"] = [dict(platform=r[0], count=r[1]) for r in con.execute(
            "SELECT i.platform, count(*) c FROM items i%s GROUP BY i.platform ORDER BY c DESC, i.platform" % where, args)]
        n_src = one("SELECT count(DISTINCT i.source) FROM " + FI + where)[0]
        res["by_source"] = dict(distinct=n_src, top=[dict(source=r[0] or "(unknown)", count=r[1]) for r in con.execute(
            "SELECT i.source, count(*) c FROM %s%s GROUP BY i.source ORDER BY c DESC, i.source LIMIT %d" % (FI, where, TOP_N), args)])
        # topics: the database classifies news only; everything else is the Unclassified bucket
        w2, a2 = self._scope_sql(kind, platform, days, [("i.db_topic_id IS NOT NULL", [])])
        topics = []
        for r in con.execute("SELECT i.db_topic_id, count(*) c FROM %s%s GROUP BY i.db_topic_id "
                             "ORDER BY c DESC, i.db_topic_id LIMIT %d" % (FI, w2, TOP_N), a2).fetchall():
            nm = con.execute("SELECT topic_name FROM items WHERE db_topic_id = ? AND topic_name IS NOT NULL LIMIT 1",
                             (r[0],)).fetchone()
            topics.append(dict(topic_id=r[0], name=(nm[0] if nm else None) or r[0], count=r[1]))
        n_classified = one("SELECT count(*) FROM " + FI + w2, a2)[0]
        res["by_topic"] = dict(
            top=topics, distinct=one("SELECT count(DISTINCT i.db_topic_id) FROM " + FI + w2, a2)[0],
            classified=n_classified, unclassified=dict(topic_id=None, name="Unclassified", count=total - n_classified,
                                                        unclassified=True),
            note="Topics are the database's own classification and exist for news only; social posts and not-yet "
                 "classified news fall in Unclassified.")
        n_lang = one("SELECT count(DISTINCT i.lang) FROM " + FI + where)[0]
        res["by_language"] = dict(distinct=n_lang, top=[dict(lang=r[0], count=r[1]) for r in con.execute(
            "SELECT i.lang, count(*) c FROM %s%s GROUP BY i.lang ORDER BY c DESC, i.lang LIMIT %d" % (FI, where, TOP_N), args)])
        dwhere, dargs = (where, args) if days else self._scope_sql(kind, platform, 30)
        day_rows = {}
        for r in con.execute("SELECT substr(i.published_at, 1, 10) d, i.kind, count(*) FROM %s%s GROUP BY d, i.kind "
                             "ORDER BY d" % (FI, dwhere), dargs):
            e = day_rows.setdefault(r[0], dict(day=r[0], news=0, social=0, total=0))
            e[r[1]] += r[2]
            e["total"] += r[2]
        res["by_day"] = list(day_rows.values())
        # ---- analysed coverage and verdict share: denominators always shown. Two sources, never double counted:
        # 'batch' = the FNP cheap-pass export (sidecar `enr`), 'deep' = a stored Claim Check result (view `deep`).
        # An item with both counts once in the verdict share, with the deep verdict (it is the better-informed one).
        per_kind = {}

        def kc(k):
            return per_kind.setdefault(k, dict(batch=0, deep=0, either=0, both=0))
        n_batch = n_deep = n_both = n_deep_only = with_prop = n_short = 0
        buckets = [0] * 10
        eff_verdicts = {}                      # effective verdict label -> items
        vsrc = {"batch": [0, 0], "deep": [0, 0]}   # source -> [items with a verdict, flagged fake/misleading]

        def verdict(src, label, flagged, n):
            eff_verdicts[label] = eff_verdicts.get(label, 0) + n
            vsrc[src][0] += n
            vsrc[src][1] += n if flagged else 0
        for k, v, bk, d_label, d_flag, short, n in con.execute(
                "SELECT i.kind, e.verdict, CASE WHEN e.p_prop IS NULL THEN -1 ELSE min(cast(e.p_prop * 10 AS INTEGER), 9) END, "
                "d.d_label, CASE WHEN d.id IS NULL THEN NULL WHEN d.d_pfm >= 0.5 THEN 1 ELSE 0 END, coalesce(e.too_short, 0), "
                "count(*) FROM enr.enr e CROSS JOIN items i ON i.id = e.id LEFT JOIN deep d ON d.id = e.id%s "
                "GROUP BY 1, 2, 3, 4, 5, 6" % where, args):
            n_batch += n
            n_short += n if short else 0
            kc(k)["batch"] += n
            kc(k)["either"] += n
            if bk >= 0:
                with_prop += n
                buckets[bk] += n
            if d_label is not None:
                n_both += n
                n_deep += n
                kc(k)["deep"] += n
                kc(k)["both"] += n
                verdict("deep", d_label, d_flag == 1, n)
            elif v is not None:
                verdict("batch", v, v in LIKELY_FAKE, n)
        wd = where + (" AND " if where else " WHERE ") + "e.id IS NULL"
        for k, d_label, d_flag, n in con.execute(
                "SELECT i.kind, d.d_label, d.d_pfm >= 0.5, count(*) FROM deep d CROSS JOIN items i ON i.id = d.id "
                "LEFT JOIN enr.enr e ON e.id = d.id%s GROUP BY 1, 2, 3" % wd, args):
            n_deep += n
            n_deep_only += n
            kc(k)["deep"] += n
            kc(k)["either"] += n
            verdict("deep", d_label, d_flag == 1, n)
        analysed = n_batch + n_deep_only
        with_verdict = vsrc["batch"][0] + vsrc["deep"][0]
        flagged = vsrc["batch"][1] + vsrc["deep"][1]
        last_deep = con.execute("SELECT max(deep_at) FROM deep").fetchone()[0]
        # partial deep results (propaganda signals only, the verdict timed out): counted apart, never in the verdict share
        n_partial = con.execute("SELECT count(*) FROM partial p CROSS JOIN items i ON i.id = p.id%s" % where, args).fetchone()[0]
        enr_st = self.enrich.status()
        fnp = batch_block(self.enrich.batch_meta(), n_short, n_batch)
        by_kind_cov = {k: dict(items=n, analysed=kc(k)["either"], pct=_pct(kc(k)["either"], n), batch=kc(k)["batch"],
                               deep=kc(k)["deep"], both=kc(k)["both"]) for k, n in res["by_kind"].items()}
        res["coverage"] = dict(
            items_in_scope=total, analysed=analysed, pct=_pct(analysed, total), by_kind=by_kind_cov,
            batch=dict(fnp, label="batch cheap pass (FNP export)", analysed=n_batch, items_in_scope=total,
                       pct=_pct(n_batch, total), built_at=enr_st.get("built_at") or fnp.get("built_at"),
                       too_short_in_scope=n_short, analysed_scorable=n_batch - n_short,
                       load=enr_st.get("load_report")),
            deep=dict(label="deep verdict (Claim Check)", analysed=n_deep, analysed_n=n_deep, partial_n=n_partial,
                      items_in_scope=total, pct=_pct(n_deep, total), last_at=last_deep),
            partial_n=n_partial,
            both=n_both, deep_only=n_deep_only,
            with_verdict=with_verdict, with_p_prop=with_prop, verdicts=eff_verdicts,
            note="Analysed = the item has a batch (FNP export) record or a deep (Claim Check) verdict; an item with both "
                 "is counted once. Most batch records carry no verdict. Partial = a deep run whose verdict timed out, so only propaganda "
                 "signals exist: counted apart in deep.partial_n and never in the verdict share. The batch export covers only part of the window "
                 "until the FNP timer has finished its runs: see batch.n_unscored_in_window.")
        res["likely_fake_misleading"] = dict(
            count=flagged, denominator=with_verdict, share=(flagged / with_verdict) if with_verdict else None,
            pct=_pct(flagged, with_verdict), fake=eff_verdicts.get("FAKE", 0), misleading=eff_verdicts.get("MISLEADING", 0),
            by_source=dict(batch=dict(count=vsrc["batch"][1], denominator=vsrc["batch"][0]),
                           deep=dict(count=vsrc["deep"][1], denominator=vsrc["deep"][0])),
            definition="Items counted: those with a verdict from either source (a deep verdict wins when an item has "
                       "both, so nothing is counted twice). Deep: fake_call P(FAKE) + P(MISLEADING) of at least 0.5; "
                       "batch: a FAKE or MISLEADING verdict. Items without a verdict are not counted either way.")
        res["p_prop_distribution"] = dict(
            n=with_prop, buckets=[dict(lo=i / 10, hi=(i + 1) / 10, count=c) for i, c in enumerate(buckets)],
            deep_only_without_p_prop=n_deep_only, too_short=n_short, basis="batch p_prop",
            note="Batch p_prop only. Items too short to score have no p_prop and are counted separately (too_short). "
                 "A deep result has no document-level p_prop, so deep-only items appear in the technique mix (from "
                 "their technique classes), not in this distribution.")
        tech = {}
        for t, n_items, m in con.execute(
                "SELECT t.technique, count(DISTINCT t.id) n, sum(t.cnt) m FROM enr.enr_tech t CROSS JOIN items i ON i.id = t.id%s "
                "GROUP BY t.technique" % where, args):
            tech[t] = [n_items, m or 0]
        for t, n_items, m in con.execute(
                "SELECT json_extract(j.value, '$.technique'), count(DISTINCT d.id), count(*) FROM deep d "
                "CROSS JOIN items i ON i.id = d.id LEFT JOIN enr.enr e ON e.id = d.id "
                "JOIN main.analyses a ON a.item_id = d.id AND a.engine_version = d.engine_version "
                "CROSS JOIN json_each(a.result_json, '$.step2.propaganda.techniques') j%s GROUP BY 1" % wd, args):
            if t:
                cur = tech.setdefault(t, [0, 0])
                cur[0] += n_items
                cur[1] += m
        res["technique_mix"] = dict(
            denominator_analysed=analysed, n_batch=n_batch, n_deep_only=n_deep_only, basis="batch techniques, else deep",
            note="Technique lists come from the batch record when an item has one (its strong techniques; items too short "
                 "to score are left out), otherwise from the deep result's technique classes; an item is never counted "
                 "from both.",
            top=[dict(technique=t, items=v[0], mentions=v[1]) for t, v in
                 sorted(tech.items(), key=lambda kv: (-kv[1][0], kv[0]))[:20]])
        newest = one("SELECT max(i.published_at) FROM items i" + where)[0]
        last_ok = con.execute("SELECT max(ts) FROM sync_log WHERE error IS NULL").fetchone()[0]
        now = self.clock()
        res["freshness"] = dict(
            newest_item_at=newest, lag_minutes=round((now - rt_store.parse_iso(newest)).total_seconds() / 60.0, 1) if newest else None,
            synced_at=last_ok, minutes_since_sync=round((now - rt_store.parse_iso(last_ok)).total_seconds() / 60.0, 1) if last_ok else None)
        res["enrichment"] = {k: v for k, v in self.enrich.status().items() if k != "coverage"}
        res["generated_at"] = rt_store.utc_iso(now)
        return res

    # ---- /api/rt/items ------------------------------------------------------------------------------------------
    def items(self, q):
        kind, platform, _days = self._scope(q)
        source = _g(q, "source")
        if len(source) > 160 or any(ord(c) < 32 for c in source):
            raise ApiError(400, "bad source")
        topic = _g(q, "topic")
        if topic and topic.lower() != "unclassified" and not _RE_TOPIC.match(topic):
            raise ApiError(400, "bad topic")
        lang = _g(q, "lang")
        if lang and not _RE_LANG.match(lang):
            raise ApiError(400, "bad lang")
        lo, hi = _date_bound(_g(q, "from"), "from", False), _date_bound(_g(q, "to"), "to", True)
        analysed = _enum(q, "analysed", ("yes", "no"))
        text_q = _g(q, "q")
        if len(text_q) > 300:
            raise ApiError(400, "q too long (max 300 characters)")
        match = None
        if text_q:
            match = rt_store.fts_query(text_q)
            if match is None:
                raise ApiError(400, "q has no searchable words")
        sort = _enum(q, "sort", SORTS, "relevance" if match else "recent")
        if sort == "relevance" and not match:
            raise ApiError(400, "sort=relevance needs q")
        size = _int(q, "size", 25, 1, PAGE_SIZE_MAX)
        page = _int(q, "page", 1, 1, 100000)
        offset = (page - 1) * size
        if offset > OFFSET_MAX:
            raise ApiError(400, "page too deep (limit %d results); narrow the filters" % OFFSET_MAX)

        conds, args = [], []
        if match:
            conds.append("items_fts MATCH ?")
            args.append(match)
        if kind:
            conds.append("i.kind = ?")
            args.append(kind)
        if platform:
            conds.append("i.platform = ?")
            args.append(platform)
        if source:
            conds.append("i.source = ?")
            args.append(source)
        if topic:
            if topic.lower() == "unclassified":
                conds.append("i.db_topic_id IS NULL")
            else:
                conds.append("i.db_topic_id = ?")
                args.append(topic)
        if lang:
            conds.append("i.lang = ?")
            args.append(lang)
        if lo:
            conds.append("i.published_at %s ?" % lo[0])
            args.append(lo[1])
        if hi:
            conds.append("i.published_at %s ?" % hi[0])
            args.append(hi[1])
        if analysed:
            conds.append("(e.id IS NOT NULL OR d.id IS NOT NULL OR pt.id IS NOT NULL)" if analysed == "yes" else
                         "(e.id IS NULL AND d.id IS NULL AND pt.id IS NULL)")
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        frm = "items_fts f CROSS JOIN items i ON i.rowid = f.rowid" if match else "items i"  # CROSS JOIN: FTS drives
        join = " LEFT JOIN enr.enr e ON e.id = i.id LEFT JOIN deep d ON d.id = i.id LEFT JOIN partial pt ON pt.id = i.id"
        order = {"recent": "i.published_at DESC, i.id DESC",
                 "relevance": "bm25(items_fts), i.published_at DESC, i.id",
                 "p_fake": "coalesce(d.d_pfm, e.p_fake) IS NULL, coalesce(d.d_pfm, e.p_fake) DESC, i.published_at DESC, i.id",
                 "p_prop": "e.p_prop IS NULL, e.p_prop DESC, i.published_at DESC, i.id"}[sort]
        need_join_for_count = bool(analysed)
        cols = ("i.id, i.kind, i.platform, i.source_table, i.source_id, i.url, i.title, i.text_excerpt, i.source, i.lang, "
                "i.published_at, i.db_topic_id, i.db_category_id, i.db_subtopic_id, i.topic_name, i.category_name, "
                "i.cluster_id, i.is_duplicate, i.engagement, i.synced_at, e.id AS e_id, e.p_prop, e.p_fake, "
                "e.p_misleading, e.verdict, e.model_version, e.built_at, e.src, e.too_short, e.n_spans, e.n_strong, d.id AS d_id, d.d_label, d.d_fake, d.d_mis, "
                "d.d_pfm, d.deep_at, d.engine_version AS d_engine, d.dup_of, pt.id AS pt_id, pt.partial_at, pt.retries AS pt_retries, "
                "pt.last_try_at AS pt_last, pt.reason AS pt_reason")

        def go(con):
            inner = "SELECT 1 FROM %s%s%s" % (frm, join if need_join_for_count else "", where)
            if match:  # a common word can match most of the cache: stop counting at COUNT_CAP
                total = con.execute("SELECT count(*) FROM (%s LIMIT %d)" % (inner, COUNT_CAP + 1), args).fetchone()[0]
            else:
                total = con.execute("SELECT count(*) FROM (%s)" % inner, args).fetchone()[0]
            rows = con.execute("SELECT %s FROM %s%s%s ORDER BY %s LIMIT ? OFFSET ?" % (cols, frm, join, where, order),
                               args + [size, offset]).fetchall()
            return total, rows
        total, rows = self._run(go)
        capped = total > COUNT_CAP and bool(match)
        if capped:
            total = COUNT_CAP
        techs = self.enrich.techniques_for([r["id"] for r in rows if r["e_id"]])
        items = []
        for r in rows:
            d = _item_dict(r, detail=False)
            batch = rt_enrich.Enrich.inferred_from_row(
                dict(p_prop=r["p_prop"], p_fake=r["p_fake"], p_misleading=r["p_misleading"], verdict=r["verdict"],
                     model_version=r["model_version"], built_at=r["built_at"], src=r["src"], too_short=r["too_short"],
                     n_spans=r["n_spans"], n_strong=r["n_strong"]),
                techs.get(r["id"], [])) if r["e_id"] else None
            deep = _deep_dict(r["d_label"], r["d_fake"], r["d_mis"], r["d_pfm"], r["deep_at"], r["d_engine"],
                              r["dup_of"]) if r["d_id"] else None
            part = _partial_dict(r["pt_id"] and r["partial_at"], r["pt_retries"], r["pt_last"], r["pt_reason"]) \
                if r["pt_id"] and not deep else None
            d["inferred"] = combine_inferred(batch, deep, partial=part)
            items.append(d)
        return dict(total=total, total_capped=capped, page=page, size=size, pages=max(1, -(-total // size)), sort=sort,
                    items=items,
                    filters=dict(q=text_q or None, kind=kind or None, platform=platform or None, source=source or None,
                                 topic=topic or None, lang=lang or None, **{"from": _g(q, "from") or None,
                                                                              "to": _g(q, "to") or None},
                                 analysed=analysed or None))

    # ---- /api/rt/sources (type-ahead for the source filter) ------------------------------------------------------
    def sources(self, q):
        """Distinct sources whose name contains `q`, with counts: [{source, count}], most items first."""
        kind, platform, _days = self._scope(q)
        text = _g(q, "q")
        if len(text) > 80 or any(ord(c) < 32 for c in text):
            raise ApiError(400, "q must be at most 80 characters")
        limit = _int(q, "limit", 20, 1, 50)
        conds, args = ["i.source IS NOT NULL", "i.source <> ''"], []
        if text:
            esc = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conds.append("i.source LIKE ? ESCAPE '\\'")
            args.append("%" + esc + "%")
        if kind:
            conds.append("i.kind = ?")
            args.append(kind)
        if platform:
            conds.append("i.platform = ?")
            args.append(platform)
        sql = ("SELECT i.source, count(*) c FROM items i WHERE " + " AND ".join(conds) +
               " GROUP BY i.source ORDER BY c DESC, i.source LIMIT ?")
        key = ("sources", text.lower(), kind, platform, limit, self._deep_sig())
        rows = self._cached(key, STATS_TTL, lambda: self._run(lambda con: con.execute(sql, args + [limit]).fetchall()))
        return dict(q=text or None, kind=kind or None, platform=platform or None, limit=limit,
                    sources=[dict(source=r[0], count=r[1]) for r in rows])

    # ---- /api/rt/item/<id> --------------------------------------------------------------------------------------
    def item(self, item_id):
        if not _RE_ITEM.match(item_id or ""):
            raise ApiError(404, "not found")

        def go(con):
            row = con.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                return None
            r = dict(row)
            r["simhash_u"] = to_unsigned(r["simhash"])
            ana = con.execute("SELECT engine_version, created_at, result_json FROM analyses WHERE item_id = ? "
                              "ORDER BY created_at DESC LIMIT 5", (item_id,)).fetchall()
            return r, self._near_dups(con, item_id, r["simhash_u"]), [dict(a) for a in ana]
        got = self._run(go)
        if got is None:
            raise ApiError(404, "not found")
        row, dups, ana = got
        d = _item_dict(row, detail=True)
        d["near_duplicates"] = dups
        d["nearest_factchecks"] = self._factchecks(row)
        d["analyses"] = []
        deep, deep_res, part, part_res = None, None, None, None
        for a in ana:
            try:
                res = json.loads(a["result_json"]) if len(a["result_json"]) < 400000 else None
            except ValueError:
                res = None
            d["analyses"].append(dict(engine_version=a["engine_version"], created_at=a["created_at"],
                                       result=claimcheck.with_blocks(res)))
            if deep is None:
                deep = deep_from_result(res, a["created_at"], a["engine_version"])
                deep_res = res if deep else None
            if part is None and not deep:
                part = partial_from_result(res, a["created_at"])
                part_res = res if part else None
        if deep:
            part, part_res = None, None
        d["inferred"] = combine_inferred(
            self.enrich.lookup(item_id, text=(row.get("title") or "", row.get("text_excerpt") or ""), desk=self.desk_url,
                               country_names=self.country_names), deep, deep_res or part_res, partial=part)
        return d

    @staticmethod
    def _near_dups(con, item_id, h, max_hamming=3):
        """Items whose simhash is within `max_hamming` bits. Same pigeonhole banding as rt_store.SimhashIndex, run as an
        indexless SQL scan so no per-process index (hundreds of MB at this scale) has to be held in memory."""
        if h is None:
            return dict(count=None, reason="no fingerprint (text too short)", max_hamming=max_hamming, sample=[])
        out = rt_store.near_dup_scan(con, item_id, h, max_hamming)
        return dict(count=len(out), max_hamming=max_hamming, sample=[
            dict(id=i, title=t, published_at=p, source=s, hamming=d) for d, p, i, t, s in out[:5]])

    def _factchecks(self, row):
        if self.fn_matches is None:
            return dict(available=False, matches=[])
        text = ("%s. %s" % (row.get("title") or "", row.get("text_excerpt") or ""))[:1500].strip()
        if len(text) < 3:
            return dict(available=True, state="no_text", matches=[])
        try:
            r = self.fn_matches(text)
        except Exception:  # noqa: BLE001
            return dict(available=False, matches=[])
        if not r:
            return dict(available=False, matches=[])
        return dict(available=True, state=r.get("state"), nearest_similarity=r.get("nearest_similarity"),
                    message=r.get("message"), matches=(r.get("matches") or [])[:5])


STALE_HOURS = 48          # newest item older than this while the sync is healthy = nothing new at the source
SYNC_FRESH_MIN = 30       # "healthy" needs a successful sync within this many minutes


def source_freshness(st, policy, now):
    """One row per source table from the store status: rows, newest item, share of the store, and two flags.
    stale  = the sync is healthy (last ok sync within 30 min, no error in the latest sync) but the newest item is older
             than 48 h, so the database itself has no newer rows (not a sync fault).
    capped = the stored row count reached the per-table cap, so older posts inside the window are not stored."""
    total = st.get("total_rows") or 0
    since = st.get("minutes_since_last_ok_sync")
    last = st.get("last_sync_at")
    n_err = sum(1 for e in (st.get("recent_errors") or []) if last and e.get("ts") == last)
    healthy = since is not None and since <= SYNC_FRESH_MIN and n_err == 0
    out = []
    for r in st.get("per_source") or []:
        newest = r.get("newest_published_at")
        age_h = None
        if newest:
            try:
                age_h = round((now - rt_store.parse_iso(newest)).total_seconds() / 3600.0, 1)
            except Exception:  # noqa: BLE001
                age_h = None
        cap = (policy or {}).get("news_limit_per_source" if r.get("kind") == "news" else "social_limit_per_table")
        out.append(dict(source_table=r.get("source_key"), kind=r.get("kind"), platform=r.get("platform"), rows=r.get("rows", 0),
                        newest_published_at=newest, age_hours=age_h,
                        share=round(r.get("rows", 0) / total, 4) if total else None,
                        stale=bool(healthy and age_h is not None and age_h > STALE_HOURS),
                        capped=bool(cap and r.get("rows", 0) >= cap), cap=cap if cap else None))
    return dict(rows=out, sync_healthy=healthy, stale_after_hours=STALE_HOURS)


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _deep_dict(label, p_fake, p_mis, p_fm, created_at, engine_version, dup_of=None):
    """The `inferred.deep` block: a stored Claim Check (check_v1) verdict. p_fake_misleading = P(FAKE) + P(MISLEADING)."""
    return dict(verdict=label, p_fake=_num(p_fake), p_misleading=_num(p_mis), p_fake_misleading=_num(p_fm),
                created_at=created_at, engine_version=engine_version, dup_of=dup_of if isinstance(dup_of, str) else None)


def deep_from_result(res, created_at, engine_version):
    """A stored result body -> the `inferred.deep` block, or None when it holds no engine verdict."""
    v = ((res.get("engine") or {}).get("verdict") if isinstance(res, dict) and isinstance(res.get("engine"), dict) else None)
    if not isinstance(v, dict) or not v.get("label"):
        return None
    pr = v.get("probs") if isinstance(v.get("probs"), dict) else {}
    f, m = _num(pr.get("FAKE")), _num(pr.get("MISLEADING"))
    return _deep_dict(v["label"], f, m, (f or 0.0) + (m or 0.0), created_at, engine_version, res.get("dup_of"))


def _partial_dict(created_at, retries, last_try_at, reason):
    """The `inferred.partial` block: a deep run whose verdict timed out; propaganda signals only, never a verdict."""
    return dict(state="partial", created_at=created_at, retries=int(retries or 0), last_try_at=last_try_at,
                reason=reason if isinstance(reason, str) else None, max_retries=rt_scheduler.RETRY_MAX,
                note="partial = propaganda signals only, the verdict timed out")


def partial_from_result(res, created_at):
    """A stored result body -> the `inferred.partial` block, or None when it is not a partial result."""
    if not isinstance(res, dict) or res.get("partial") is not True:
        return None
    info = res.get("partial_info") if isinstance(res.get("partial_info"), dict) else {}
    return _partial_dict(created_at, info.get("retries"), info.get("last_try_at"), info.get("reason"))


# detail fields of a real-format batch record that are repeated inside `inferred.batch` (and kept at the top level too)
BATCH_DETAIL_KEYS = ("technique_probs", "strong_techniques", "spans", "spans_dropped", "n_tokens", "n_spans", "n_strong",
                     "density", "countries", "nearest_factcheck", "techniques", "lang")


def combine_inferred(batch, deep, deep_result=None, partial=None):
    """One `inferred` block from the batch record (rt_enrich dict or None) and the deep block (or None); None when the
    item has neither. `source` is the one shown first ('deep' when it exists), `sources` lists every source present.
    Top-level verdict/p_fake/p_misleading are the deep ones when a deep result exists (counted once), else batch;
    p_prop and the technique list are the batch ones, falling back to the deep result's techniques."""
    if batch is None and deep is None and partial is None:
        return None
    out = dict(batch) if batch else dict(p_prop=None, p_fake=None, p_misleading=None, verdict=None, model_version=None,
                                         built_at=None, techniques=[])
    out["export_source"] = out.pop("source", None) if batch else None
    out["batch"] = dict(
        {k: batch[k] for k in BATCH_DETAIL_KEYS if k in batch}, p_prop=batch["p_prop"], p_fake=batch["p_fake"],
        p_misleading=batch["p_misleading"], verdict=batch["verdict"], model_version=batch["model_version"],
        built_at=batch["built_at"], too_short=bool(batch.get("too_short"))) if batch else None
    out["deep"] = deep
    out["partial"] = partial if not deep else None
    out["sources"] = (["batch"] if batch else []) + (["deep"] if deep else []) + (["partial"] if out["partial"] else [])
    out["source"] = "deep" if deep else "batch" if batch else "partial"
    if deep or out["partial"]:
        if deep:
            out["verdict"], out["p_fake"], out["p_misleading"] = deep["verdict"], deep["p_fake"], deep["p_misleading"]
        techs = (((deep_result or {}).get("step2") or {}).get("propaganda") or {}).get("techniques") \
            if isinstance(deep_result, dict) else None
        if not batch and isinstance(techs, list):
            agg = {}
            for t in techs:
                if isinstance(t, dict) and isinstance(t.get("technique"), str):
                    n, mx = agg.get(t["technique"], (0, None))
                    c = _num(t.get("confidence"))
                    agg[t["technique"]] = (n + 1, c if mx is None or (c is not None and c > mx) else mx)
            out["techniques"] = [dict(technique=k, count=n, max_confidence=m) for k, (n, m) in
                                 sorted(agg.items(), key=lambda kv: (-kv[1][0], kv[0]))]
            out["spans"] = [dict(text=t["text"][:200], technique=t["technique"], confidence=_num(t.get("confidence")),
                                 field=None) for t in techs[:12]
                            if isinstance(t, dict) and isinstance(t.get("text"), str) and isinstance(t.get("technique"), str)]
    return out


def _item_dict(r, detail):
    r = dict(r)
    text = r.get("text_excerpt") or ""
    d = dict(
        id=r["id"], kind=r["kind"], platform=r["platform"], source_table=r["source_table"], source_id=r["source_id"],
        url=r["url"], title=r["title"], source=r["source"], lang=r["lang"], published_at=r["published_at"],
        topic=dict(id=r["db_topic_id"], name=r["topic_name"]) if r["db_topic_id"] else None,
        category=dict(id=r["db_category_id"], name=r["category_name"]) if r["db_category_id"] else None,
        subtopic_id=r["db_subtopic_id"], cluster_id=r["cluster_id"],
        is_duplicate=None if r["is_duplicate"] is None else bool(r["is_duplicate"]),
        engagement=r["engagement"], synced_at=r["synced_at"])
    if detail:
        d["text_excerpt"] = text
    else:
        d["snippet"] = text if len(text) <= SNIPPET else text[:SNIPPET].rstrip() + "…"
    return d
