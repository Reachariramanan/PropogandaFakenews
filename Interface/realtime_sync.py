#!/usr/bin/env python3
"""Realtime database -> local SQLite cache (phase E1 of the Evidence Base).

READ-ONLY by construction. The two customer PostgreSQL accounts are superusers, so safety is enforced here:
  * every connection is opened with default_transaction_read_only=on, statement_timeout=30s and an idle timeout,
    set_session(readonly=True), and the session setting is verified before use;
  * every statement goes through ReadOnlyDB.query(), which refuses anything that is not a single SELECT/WITH;
  * no temp tables, one connection per database, strictly sequential queries.
Credentials come from environment variables, falling back to ~/.config/fnp/realtime.env (mode 600). They are never
logged, printed, stored in SQLite or put in an error message (see redact()).

Data model: see rt_store.py (items, items_fts, watermarks, sync_log, analyses). Item ids:
  social: '<source_table>:<source_id>'     e.g. 'reddit_posts:123'
  news:   'news:<source_table>:<article_id>' e.g. 'news:scraped_articles:2459300'

CLI:  python3 Interface/realtime_sync.py --once [--limit N] [--days D] [--sources a,b] [--status]
Phase E2 uses: sync_once, status, start_background, stop_background, background_status, near_duplicates, search_ids.
"""
import argparse
import json
import logging
import os
import re
import stat
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rt_store  # noqa: E402
from rt_text import EXCERPT_CHARS, TITLE_CHARS, clean_text, normalize_lang, simhash64  # noqa: E402

log = logging.getLogger("fnp.realtime")

DEFAULT_ENV_FILE = os.path.expanduser("~/.config/fnp/realtime.env")
READONLY_OPTIONS = ("-c default_transaction_read_only=on -c statement_timeout=30000 "
                    "-c idle_in_transaction_session_timeout=60000")
# Sync policy: news is large and fast-moving (short window, big cap); social is smaller (longer window).
# Env overrides: FNP_RT_NEWS_DAYS, FNP_RT_SOCIAL_DAYS, FNP_RT_NEWS_LIMIT, FNP_RT_SOCIAL_LIMIT (limits are per source
# table); FNP_RT_REFRESH_BUDGET bounds the per-cycle topic refresh. Explicit days/limit arguments (CLI) win for both.
DEFAULTS = dict(news_days=7, social_days=30, news_limit=400000, social_limit=100000, batch=2000, overlap_hours=2.0,
                refresh_budget=5000)
MAX_ENGAGEMENT = 2 ** 62


def _env_num(name, default, cast=float):
    try:
        return cast(os.environ[name])
    except (KeyError, ValueError):
        return default


# ---- credentials and redaction ---------------------------------------------------------------------------------

_SECRETS = set()
_SECRET_PATTERNS = (
    re.compile(r"(password\s*[=:]\s*)('[^']*'|\"[^\"]*\"|\S+)", re.I),
    re.compile(r"(passwd\s*[=:]\s*)('[^']*'|\"[^\"]*\"|\S+)", re.I),
)


def redact(text):
    """Remove every known password and any 'password=...' fragment from a string headed for a log or the store."""
    s = str(text)
    for sec in sorted(_SECRETS, key=len, reverse=True):
        if sec:
            s = s.replace(sec, "***")
    for pat in _SECRET_PATTERNS:
        s = pat.sub(lambda m: m.group(1) + "***", s)
    return s[:600]


class ConfigError(Exception):
    pass


class DbConfig:
    """Connection settings for one database. The password is never part of repr/str."""
    __slots__ = ("name", "host", "port", "dbname", "user", "_password")

    def __init__(self, name, host, port, dbname, user, password):
        self.name, self.host, self.port, self.dbname, self.user = name, host, int(port), dbname, user
        self._password = password
        if password:
            _SECRETS.add(password)

    @property
    def password(self):
        return self._password

    def __repr__(self):
        return "DbConfig(%s host=%s port=%s db=%s user=%s password=<hidden>)" % (
            self.name, self.host, self.port, self.dbname, self.user)

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("DbConfig is not picklable (holds a credential)")


def parse_env_file(path):
    """KEY=VALUE lines (# comments, optional quotes, optional 'export '). Returns {} if unreadable."""
    out = {}
    try:
        if os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            log.warning("credential file %s is accessible to group/others; run chmod 600", path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                k, v = line.split("=", 1)
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                out[k.strip()] = v
    except OSError:
        pass
    return out


def load_settings(env=None, env_file=None):
    """{'news': DbConfig, 'social': DbConfig}. Environment variables win over the credentials file."""
    env = os.environ if env is None else env
    file_vals = parse_env_file(env_file or env.get("FNP_RT_ENV_FILE") or DEFAULT_ENV_FILE)

    def get(k):
        v = env.get(k)
        return v if v not in (None, "") else file_vals.get(k)

    host = get("FNP_RT_HOST")
    out, missing = {}, []
    for name in ("NEWS", "SOCIAL"):
        vals = {f: get("FNP_RT_%s_%s" % (name, f)) for f in ("PORT", "DB", "USER", "PASSWORD")}
        absent = [("FNP_RT_%s_%s" % (name, f)) for f, v in vals.items() if not v]
        if not host:
            absent.append("FNP_RT_HOST")
        if absent:
            missing.append((name.lower(), absent))
            continue
        out[name.lower()] = DbConfig(name.lower(), host, vals["PORT"], vals["DB"], vals["USER"], vals["PASSWORD"])
    return out, dict(missing)


class DbError(Exception):
    """A database problem whose message has already been redacted."""


def default_connect(which, settings=None):
    """Open the single read-only connection for 'news' or 'social'. Raises ConfigError / DbError (redacted)."""
    if settings is None:
        settings, missing = load_settings()
    else:
        missing = {}
    cfg = settings.get(which)
    if cfg is None:
        raise ConfigError("no credentials for the %s database (missing: %s)" % (
            which, ", ".join(missing.get(which, ["unknown"]))))
    try:
        import psycopg2
    except ImportError:
        raise DbError("psycopg2 is not installed") from None
    conn = None
    try:
        conn = psycopg2.connect(host=cfg.host, port=cfg.port, dbname=cfg.dbname, user=cfg.user,
                                password=cfg.password, options=READONLY_OPTIONS, connect_timeout=10,
                                application_name="fnp-evidence-desk-sync")
        conn.set_session(readonly=True, autocommit=True)
        cur = conn.cursor()
        cur.execute("SELECT current_setting('default_transaction_read_only')")
        ro = cur.fetchone()[0]
        cur.close()
        if str(ro).lower() != "on":
            raise DbError("session is not read-only; refusing to use this connection")
        return conn
    except DbError:
        _close_quietly(conn)
        raise
    except Exception as e:  # psycopg2.Error and anything else: never leak the credential
        _close_quietly(conn)
        raise DbError("%s database: %s" % (which, redact(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__)) from None


def _close_quietly(conn):
    try:
        if conn is not None:
            conn.close()
    except Exception:
        pass


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|into|vacuum|lock|reindex|cluster|"
    r"pg_terminate_backend|pg_cancel_backend|set_config|nextval|setval|lo_import|lo_export|dblink)\b", re.I)
_SELECT_START = re.compile(r"^\s*(select|with)\b", re.I)


def check_select(sql):
    """Raise unless `sql` is a single SELECT/WITH statement with no write-capable keyword."""
    body = sql.strip().rstrip(";")
    if not _SELECT_START.match(body) or ";" in body or _FORBIDDEN.search(body):
        raise DbError("refused non-SELECT statement: %s" % body[:60].replace("\n", " "))


class ReadOnlyDB:
    """Wraps a DB-API connection; the only way this module talks to PostgreSQL."""

    def __init__(self, conn, label):
        self.conn, self.label = conn, label
        self.statements = 0

    def query(self, sql, params=None):
        check_select(sql)
        cur = self.conn.cursor()
        try:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        finally:
            try:
                cur.close()
            except Exception:
                pass
        self.statements += 1
        return [dict(zip(cols, r)) for r in rows]

    def close(self):
        _close_quietly(self.conn)


# ---- table map (inspected via information_schema on 2026-11-06) --------------------------------------------------

def _q(c):
    return '"%s"' % c


def _first(*cols):
    return "coalesce(%s)" % ", ".join("nullif(btrim(%s), '')" % _q(c) for c in cols)


def _eng(*cols):
    if not cols:
        return "NULL::bigint"
    allnull = " AND ".join("%s IS NULL" % _q(c) for c in cols)
    return "CASE WHEN %s THEN NULL ELSE (%s)::bigint END" % (allnull, " + ".join("coalesce(%s,0)" % _q(c) for c in cols))


def _lang(*cols):
    if not cols:
        return "NULL::text"
    parts = []
    for c in cols:
        if c == "language":
            parts.append("nullif(nullif(btrim(%s), ''), 'unknown')" % _q(c))
        else:
            parts.append("nullif(btrim(%s), '')" % _q(c))
    return "coalesce(%s)" % ", ".join(parts)


def _text(sql_expr):
    return "left(%s, 4000)" % sql_expr


def _tbl(platform, id_col, time_expr, body, author, url, lang, eng, title=None):
    return dict(platform=platform, id=_q(id_col), time=time_expr, title=_text(title) if title else "NULL::text",
                body=_text(body), author=author, url=url, lang=lang, eng=eng)


_L = _lang("language", "detected_language")
_NULL = "NULL::text"

# time column notes: naive timestamp columns are converted explicitly. facebook_profile_posts.posted_time and
# baidu_tieba_posts.posted_time are Asia/Kolkata wall-clock (verified against the epoch "timestamp" column and
# collected_time); discord_posts.timestamp is UTC (verified against raw_json); tiktok uses the epoch create_time.
SOCIAL_TABLES = {
    "instagram_posts": _tbl("instagram", "id", _q("posted_at"), _first("caption"), _first("user_name"),
                            _first("post_url"), _L, _eng("likes_count", "comment_count", "shares_count")),
    "twitter_posts": _tbl("twitter", "id", _q("posted_at"), _first("text"), _first("author_username"),
                          _first("post_url"), _L, _eng("like_count", "retweet_count", "reply_count", "quote_count")),
    "x_posts": _tbl("x", "tweet_id", _q("posted_at"), _first("text"), _first("author_username"),
                    _first("post_url"), _L, _eng("like_count", "retweet_count", "reply_count", "quote_count")),
    "telegram_posts": _tbl("telegram", "id", _q("posted_at"), _first("text"), _first("channel_name"),
                           _first("post_url"), _L, _eng("views")),
    "reddit_posts": _tbl("reddit", "id", _q("posted_at"), _first("body"), _first("author"), _first("url"), _L,
                         _eng(), title=_first("title")),
    "mastodon_posts": _tbl("mastodon", "id", _q("created_at"), _first("content_text", "content_html"),
                           _first("account_acct"), _first("url"), _L,
                           _eng("reblogs_count", "favourites_count", "replies_count")),
    "facebook_posts": _tbl("facebook", "id", _q("posted_time"), _first("post_text"), _first("author_name"),
                           _first("url"), _L, _eng("reactions_count", "comments_count", "shares_count")),
    "facebook_profile_posts": _tbl("facebook", "id", "(%s AT TIME ZONE 'Asia/Kolkata')" % _q("posted_time"),
                                   _first("message"), _first("author_name"), _first("url"), _L,
                                   _eng("reactions_count", "comments_count", "reshare_count")),
    "linkedin_posts": _tbl("linkedin", "id", _q("posted_at"), _first("main_text"), _first("author_page"),
                           _first("post_url"), _L, _eng("reaction_count", "comment_count"), title=_first("title")),
    "tiktok_posts": _tbl("tiktok", "post_id", "to_timestamp(%s)" % _q("create_time"), _first("desc"),
                         _first("author_username"), _first("share_url", "video_url"),
                         _lang("language", "desc_language", "detected_language"),
                         _eng("digg_count", "comment_count", "share_count")),
    "snapchat_posts": _tbl("snapchat", "id", _q("posted_at"), _first("caption"), _first("creator"),
                           _first("spotlight_url"), _L, _eng("likes", "comments", "shares"), title=_first("title")),
    "truthsocial_posts": _tbl("truthsocial", "post_id", _q("created_at"), _first("text_plain", "text_html"),
                              _first("author_handle"), _first("url"), _L,
                              _eng("reblogs_count", "favourites_count", "replies_count")),
    "discord_posts": _tbl("discord", "id", "(%s AT TIME ZONE 'UTC')" % _q("timestamp"), _first("content"),
                          _first("author"), _first("url"), _lang(), _eng()),
    "whatsapp_posts": _tbl("whatsapp", "id", _q("posted_at"), _first("text", "caption"), _first("channel_id"),
                           _NULL, _L, _eng()),
    "pinterest_posts": _tbl("pinterest", "id", _q("posted_at"), _first("description", "comment_text"),
                            _first("creator_username", "creator_name"), _first("pin_url"), _L, _eng("comment_count"),
                            title=_first("title")),
    "baidu_tieba_posts": _tbl("baidu_tieba", "id", "(%s AT TIME ZONE 'Asia/Kolkata')" % _q("posted_time"),
                              _first("content"), _first("author_name"), _first("post_url"), _L,
                              _eng("like_count", "reply_count"), title=_first("title")),
}

# tables deliberately not synced (reported in every sync report)
SKIPPED_SOCIAL = {
    "youtube_comments": "excluded for now (comments, 309k rows); enable on request",
    "reddit_ensembledata_seen": "helper/seen table, no post text",
}

# news_engine.article_fact.source_table values we can join to a body table
NEWS_SOURCES = {
    "scraped_articles": dict(content_table="raw.scraped_articles", lang=False),
    "scraped_articles_trusted": dict(content_table="raw.scraped_articles_trusted", lang=True),
}

NEWS_SQL = (
    "SELECT article_id AS sid, published_at AS ts, topic_id, category_id, subtopic_id, cluster_id, is_duplicate, "
    "source_domain, url, title FROM news_engine.article_fact "
    "WHERE source_table = %(st)s AND published_at >= %(lo)s AND published_at <= %(hi)s{keyset} "
    "ORDER BY published_at DESC, article_id DESC LIMIT %(n)s")
NEWS_KEYSET = " AND (published_at, article_id) < (%(kt)s, %(kid)s)"
NEWS_TAXONOMY_SQL = "SELECT code, name FROM news_engine.taxonomy_node"
NEWS_TABLES_SQL = ("SELECT source_table, count(*) AS n FROM news_engine.article_fact "
                   "WHERE published_at >= %(lo)s AND published_at <= %(hi)s GROUP BY source_table")
NEWS_CONTENT_SQL = ("SELECT id, left(article_content, 4000) AS content{lang} FROM {table} WHERE id = ANY(%(ids)s)")
NEWS_REFRESH_SQL = ("SELECT article_id AS sid, topic_id, category_id, subtopic_id, cluster_id, is_duplicate "
                    "FROM news_engine.article_fact WHERE source_table = %(st)s AND article_id = ANY(%(ids)s)")


def social_sql(cfg, keyset):
    sql = ("SELECT {id} AS sid, {time} AS ts, {title} AS title, {body} AS body, {author} AS author, {url} AS url, "
           "{lang} AS lang, {eng} AS eng FROM public.{{table}} WHERE {time} >= %(lo)s AND {time} <= %(hi)s").format(**cfg)
    if keyset:
        sql += " AND ({time}, {id}) < (%(kt)s, %(kid)s)".format(**cfg)
    return sql + " ORDER BY {time} DESC, {id} DESC LIMIT %(n)s".format(**cfg)


def _safe_url(u):
    u = (u or "").strip()
    return u[:2000] if u.lower().startswith(("http://", "https://")) else None


def _engagement(v):
    if v is None:
        return None
    try:
        return max(0, min(int(v), MAX_ENGAGEMENT))
    except (TypeError, ValueError):
        return None


def map_social(table, cfg, row, synced_at):
    """Database row -> item dict, or None when there is no usable text."""
    title = clean_text(row.get("title"), TITLE_CHARS)
    body = clean_text(row.get("body"), EXCERPT_CHARS)
    if not title and not body:
        return None
    sid = str(row["sid"])
    return dict(
        id="%s:%s" % (table, sid), kind="social", platform=cfg["platform"], source_table=table, source_id=sid,
        url=_safe_url(row.get("url")), title=title or None, text_excerpt=body, source=clean_text(row.get("author"), 160) or None,
        lang=normalize_lang(row.get("lang")), published_at=rt_store.utc_iso(row["ts"]),
        db_topic_id=None, db_category_id=None, db_subtopic_id=None, topic_name=None, category_name=None,
        cluster_id=None, is_duplicate=None, engagement=_engagement(row.get("eng")),
        simhash=simhash64(title, body), synced_at=synced_at)


def map_news(source_table, row, content, taxonomy, synced_at):
    """article_fact row (+ body text from the raw table) -> item dict, or None when no usable text."""
    title = clean_text(row.get("title"), TITLE_CHARS)
    body = clean_text(content.get("text"), EXCERPT_CHARS) if content.get("text") else content.get("excerpt", "")
    if not title and not body:
        return None
    sid = str(row["sid"])
    topic, cat = row.get("topic_id"), row.get("category_id")
    dup = row.get("is_duplicate")
    return dict(
        id="news:%s:%s" % (source_table, sid), kind="news", platform="news", source_table=source_table,
        source_id=sid, url=_safe_url(row.get("url")), title=title or None, text_excerpt=body,
        source=clean_text(row.get("source_domain"), 160) or None,
        lang=normalize_lang(content.get("lang")), published_at=rt_store.utc_iso(row["ts"]),
        db_topic_id=topic, db_category_id=cat, db_subtopic_id=row.get("subtopic_id"),
        topic_name=taxonomy.get(topic) if topic else None, category_name=taxonomy.get(cat) if cat else None,
        cluster_id=row.get("cluster_id"), is_duplicate=None if dup is None else int(bool(dup)),
        engagement=None, simhash=content.get("simhash") if "simhash" in content else simhash64(title, body),
        synced_at=synced_at)


# ---- the sync --------------------------------------------------------------------------------------------------

_sync_lock = threading.Lock()


def _normalize_sources(sources):
    """-> ordered list of source keys ('news:<table>' or a social table name); raises ValueError on unknown names."""
    all_keys = ["news:%s" % t for t in NEWS_SOURCES] + list(SOCIAL_TABLES)
    if not sources:
        return all_keys
    if isinstance(sources, str):
        sources = [s for s in sources.split(",")]
    out = []
    for s in (x.strip() for x in sources):
        if not s:
            continue
        if s in ("news", "all_news"):
            out += ["news:%s" % t for t in NEWS_SOURCES]
        elif s in ("social", "all_social"):
            out += list(SOCIAL_TABLES)
        elif s in NEWS_SOURCES:
            out.append("news:%s" % s)
        elif s in all_keys:
            out.append(s)
        else:
            raise ValueError("unknown source %r (known: news, social, %s)" % (s, ", ".join(all_keys)))
    return list(dict.fromkeys(out))


class RealtimeSync:
    def __init__(self, db_path=None, connect=None, clock=None, settings=None):
        self.db_path = db_path or rt_store.DEFAULT_DB
        self._connect = connect
        self._settings = (settings, {}) if isinstance(settings, dict) else settings
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    # -- connections (one per database per run) --
    def _open(self, which):
        if self._connect is not None:
            return ReadOnlyDB(self._connect(which), which)
        if self._settings is None:
            self._settings = load_settings()
        cfgs, missing = self._settings
        if which not in cfgs:
            _raise_missing(which, missing)
        return ReadOnlyDB(default_connect(which, cfgs), which)

    # -- main entry --
    def sync_once(self, limit=None, days=None, sources=None, per_source_limit=None, batch=None, overlap_hours=None,
                  stop_event=None, prune=False):
        keys = _normalize_sources(sources)
        pol = {}
        for kind in ("news", "social"):
            d = days if days is not None else _env_num("FNP_RT_%s_DAYS" % kind.upper(), DEFAULTS[kind + "_days"])
            c = _env_num("FNP_RT_%s_LIMIT" % kind.upper(), DEFAULTS[kind + "_limit"], int)
            if per_source_limit:
                c = int(per_source_limit)
            if limit:
                c = min(c, int(limit)) if per_source_limit else int(limit)
            pol[kind] = (float(d), int(c))
        batch = int(batch or _env_num("FNP_RT_BATCH", DEFAULTS["batch"], int))
        overlap = timedelta(hours=float(overlap_hours if overlap_hours is not None else
                                        _env_num("FNP_RT_OVERLAP_HOURS", DEFAULTS["overlap_hours"])))
        if not _sync_lock.acquire(blocking=False):
            return dict(skipped="another sync is already running", sources={}, rows_total=0, errors=[])
        t_start = time.time()
        now = self.clock()
        ctx = dict(now=now, synced_at=rt_store.utc_iso(now), cutoff=now - timedelta(days=pol["news"][0]),
                   hi=now + timedelta(days=1), cap=pol["news"][1], batch=batch, overlap=overlap, stop=stop_event,
                   taxonomy=None)
        report = dict(started_at=ctx["synced_at"], window_days={k: v[0] for k, v in pol.items()},
                      per_source_cap={k: v[1] for k, v in pol.items()}, batch=batch, sources={},
                      skipped_tables=[dict(table=t, reason=r) for t, r in SKIPPED_SOCIAL.items()],
                      errors=[], rows_total=0)
        store = None
        dbs = {}
        db_errors = {}
        try:
            store = rt_store.open_store(self.db_path)
            for key in keys:
                if stop_event is not None and stop_event.is_set():
                    report["stopped"] = True
                    break
                which = "news" if key.startswith("news:") else "social"
                ctx["cutoff"] = now - timedelta(days=pol[which][0])
                ctx["cap"] = pol[which][1]
                t0 = time.time()
                entry = dict(rows=0, fetched=0, skipped_empty=0, truncated=False, seconds=0.0, error=None)
                report["sources"][key] = entry
                try:
                    if which in db_errors:
                        raise DbError(db_errors[which])
                    if which not in dbs:
                        try:
                            dbs[which] = self._open(which)
                        except Exception as e:
                            db_errors[which] = self._safe_err(e)
                            raise
                    if which == "news":
                        self._sync_news(store, dbs["news"], key.split(":", 1)[1], ctx, entry, report)
                    else:
                        self._sync_social(store, dbs["social"], key, ctx, entry)
                except Exception as e:  # isolate: one source failing never aborts the run
                    entry["error"] = self._safe_err(e)
                    report["errors"].append(dict(source=key, error=entry["error"]))
                    log.warning("sync %s failed: %s", key, entry["error"])
                entry["seconds"] = round(time.time() - t0, 2)
                report["rows_total"] += entry["rows"]
                self._log(store, ctx["synced_at"], key, entry)
            if prune and store is not None:
                report["pruned"] = sum(
                    rt_store.prune(store, rt_store.utc_iso(now - timedelta(days=pol[k][0])), kind=k)
                    for k in ("news", "social"))
        except Exception as e:
            report["errors"].append(dict(source="*", error=self._safe_err(e)))
            log.warning("sync failed: %s", self._safe_err(e))
        finally:
            for db in dbs.values():
                db.close()
            if store is not None:
                store.close()
            _sync_lock.release()
        report["seconds"] = round(time.time() - t_start, 2)
        return report

    @staticmethod
    def _safe_err(e):
        return redact("%s: %s" % (type(e).__name__, e) if not isinstance(e, (DbError, ConfigError)) else e)

    @staticmethod
    def _log(store, ts, key, entry):
        try:
            store.execute("INSERT INTO sync_log(ts, source_key, rows, seconds, error, note) VALUES (?,?,?,?,?,?)",
                          (ts, key, entry["rows"], entry["seconds"], entry["error"],
                           "truncated at cap" if entry.get("truncated") else None))
        except Exception as e:  # logging must never break the sync
            log.warning("could not write sync_log: %s", redact(e))

    # -- watermarks --
    @staticmethod
    def _wm(store, key):
        r = store.execute("SELECT last_published_at, last_id, floor_published_at FROM watermarks WHERE source_key=?",
                          (key,)).fetchone()
        return (rt_store.parse_iso(r[0]), r[1], rt_store.parse_iso(r[2])) if r else (None, None, None)

    @staticmethod
    def _set_wm(store, key, newest, last_id, floor):
        store.execute(
            "INSERT INTO watermarks(source_key, last_published_at, last_id, floor_published_at) VALUES (?,?,?,?) "
            "ON CONFLICT(source_key) DO UPDATE SET last_published_at=excluded.last_published_at, "
            "last_id=excluded.last_id, floor_published_at=excluded.floor_published_at",
            (key, rt_store.utc_iso(newest), last_id, rt_store.utc_iso(floor)))

    # -- generic keyset pull of one range (newest first) --
    def _pull(self, store, fetch, build, lo, hi, cap, batch, entry, stop):
        """Pull rows in [lo, hi] newest-first by keyset, upserting each batch. Returns
        (newest_ts, newest_id, oldest_ts, truncated)."""
        kt = kid = None
        newest = newest_id = oldest = None
        got = 0
        truncated = False
        while got < cap:
            if stop is not None and stop.is_set():
                truncated = True
                break
            n = min(batch, cap - got)
            rows = fetch(lo, hi, kt, kid, n)
            if not rows:
                break
            items, empty = build(rows)
            rt_store.upsert_items(store, items)
            entry["rows"] += len(items)
            entry["skipped_empty"] += empty
            entry["fetched"] += len(rows)
            got += len(rows)
            if newest is None:
                newest, newest_id = rows[0]["ts"], str(rows[0]["sid"])
            oldest = rows[-1]["ts"]
            kt, kid = rows[-1]["ts"], rows[-1]["sid"]
            if len(rows) < n:
                break
            if got >= cap:
                truncated = True  # a full final batch: there may be more rows we did not read
        return newest, newest_id, oldest, truncated

    def _segments(self, store, key, ctx, entry, fetch, build, stored_where):
        """Top segment (rows newer than the watermark, with overlap) then, within the stored-row budget, a backfill
        segment below the contiguous floor. The watermark moves only after both segments finished without error
        (an exception skips _set_wm; rows already upserted stay, which is harmless because upserts are idempotent).
        Coverage is the interval [floor, watermark]; a truncated top segment that does not reach the old coverage
        resets it to the new interval and the gap is backfilled on later runs."""
        cutoff, hi, cap, batch, stop = ctx["cutoff"], ctx["hi"], ctx["cap"], ctx["batch"], ctx["stop"]
        wm, wm_id, floor = self._wm(store, key)
        lo = cutoff if wm is None else max(cutoff, wm - ctx["overlap"])
        n_new, n_id, n_old, trunc = self._pull(store, fetch, build, lo, hi, cap, batch, entry, stop)
        entry["truncated"] = bool(trunc)
        if n_new is None:
            return  # nothing in range (or nothing new): coverage unchanged
        if wm is None:
            newest, newest_id, new_floor = n_new, n_id, (n_old if trunc else cutoff)
        elif (not trunc) or n_old <= wm:  # the new segment reaches the old coverage: they join
            newest, newest_id = (n_new, n_id) if n_new > wm else (wm, wm_id)
            new_floor = floor if floor is not None else cutoff
        else:  # truncated and a gap remains: restart coverage from the new segment
            newest, newest_id, new_floor = n_new, n_id, n_old
        if new_floor > cutoff and not trunc and not (stop is not None and stop.is_set()):
            stored = store.execute("SELECT count(*) FROM items WHERE %s AND published_at >= ?" % stored_where,
                                   (rt_store.utc_iso(cutoff),)).fetchone()[0]
            budget = cap - stored
            if budget > 0:
                b_new, b_id, b_old, b_trunc = self._pull(store, fetch, build, cutoff, new_floor + timedelta(seconds=2),
                                                         budget, batch, entry, stop)
                new_floor = b_old if (b_trunc and b_old is not None) else cutoff
                entry["truncated"] = entry["truncated"] or bool(b_trunc)
        self._set_wm(store, key, newest, newest_id, new_floor)

    # -- social --
    def _sync_social(self, store, db, table, ctx, entry):
        cfg = SOCIAL_TABLES[table]
        sql_first = social_sql(cfg, False).replace("{table}", table)
        sql_next = social_sql(cfg, True).replace("{table}", table)

        def fetch(lo, hi, kt, kid, n):
            p = dict(lo=lo, hi=hi, n=n)
            if kt is not None:
                p.update(kt=kt, kid=kid)
            return db.query(sql_next if kt is not None else sql_first, p)

        def build(rows):
            items, empty = [], 0
            for r in rows:
                it = map_social(table, cfg, r, ctx["synced_at"])
                if it is None:
                    empty += 1
                else:
                    items.append(it)
            return items, empty

        self._segments(store, table, ctx, entry, fetch, build, "source_table = '%s'" % table)

    # -- news --
    def _taxonomy(self, db, ctx):
        if ctx["taxonomy"] is None:
            try:
                ctx["taxonomy"] = {r["code"]: r["name"] for r in db.query(NEWS_TAXONOMY_SQL)}
            except Exception as e:
                log.warning("taxonomy names unavailable: %s", self._safe_err(e))
                ctx["taxonomy"] = {}
        return ctx["taxonomy"]

    def _sync_news(self, store, db, source_table, ctx, entry, report):
        cfg = NEWS_SOURCES[source_table]
        taxonomy = self._taxonomy(db, ctx)
        if "news_tables_seen" not in report:  # informational: article_fact source tables we cannot join
            report["news_tables_seen"] = {}
            try:
                recent = ctx["now"] - timedelta(days=2)
                for r in db.query(NEWS_TABLES_SQL, dict(lo=recent, hi=ctx["hi"])):
                    report["news_tables_seen"][r["source_table"]] = r["n"]
                    if r["source_table"] not in NEWS_SOURCES:
                        report["skipped_tables"].append(dict(
                            table="news_engine.article_fact[source_table=%s]" % r["source_table"],
                            reason="no body table join configured (%s rows in the last 2 days)" % r["n"]))
            except Exception as e:
                log.warning("could not list article_fact source tables: %s", self._safe_err(e))
        sql_first = NEWS_SQL.format(keyset="")
        sql_next = NEWS_SQL.format(keyset=NEWS_KEYSET)
        content_sql = NEWS_CONTENT_SQL.format(table=cfg["content_table"], lang=", language AS lang" if cfg["lang"] else "")

        def fetch(lo, hi, kt, kid, n):
            p = dict(st=source_table, lo=lo, hi=hi, n=n)
            if kt is not None:
                p.update(kt=kt, kid=kid)
            return db.query(sql_next if kt is not None else sql_first, p)

        def build(rows):
            ids = [r["sid"] for r in rows]
            known = {}
            for chunk in _chunks(["news:%s:%s" % (source_table, i) for i in ids], 500):
                q = "SELECT id, text_excerpt, simhash, lang FROM items WHERE id IN (%s)" % ",".join("?" * len(chunk))
                for s in store.execute(q, chunk):
                    if s[1]:
                        known[s[0]] = dict(excerpt=s[1], simhash=rt_store.to_unsigned(s[2]), lang=s[3])
            need = [i for i in ids if "news:%s:%s" % (source_table, i) not in known]
            fresh = {}
            if need:
                for c in db.query(content_sql, dict(ids=need)):
                    fresh[c["id"]] = c
            items, empty = [], 0
            for r in rows:
                iid = "news:%s:%s" % (source_table, r["sid"])
                if iid in known:
                    k = known[iid]
                    content = dict(excerpt=k["excerpt"], lang=k["lang"])  # reuse stored text; title may be re-cleaned
                    it = map_news(source_table, r, content, taxonomy, ctx["synced_at"])
                else:
                    c = fresh.get(r["sid"], {})
                    it = map_news(source_table, r, dict(text=c.get("content"), lang=c.get("lang")), taxonomy,
                                  ctx["synced_at"])
                if it is None:
                    empty += 1
                else:
                    items.append(it)
            return items, empty

        self._segments(store, "news:%s" % source_table, ctx, entry, fetch, build,
                       "kind = 'news' AND source_table = '%s'" % source_table)
        self._refresh_classification(store, db, source_table, ctx, taxonomy)

    def _refresh_classification(self, store, db, source_table, ctx, taxonomy):
        """The database fills topic/category some time after ingest: re-read the classification of stored,
        still-unclassified articles (primary-key lookups, no body text)."""
        budget = int(_env_num("FNP_RT_REFRESH_BUDGET", DEFAULTS["refresh_budget"], int))
        rows = store.execute(
            "SELECT source_id FROM items WHERE kind='news' AND source_table=? AND db_topic_id IS NULL "
            "ORDER BY published_at DESC LIMIT ?", (source_table, budget)).fetchall()
        ids = [r[0] for r in rows]
        for chunk in _chunks(ids, ctx["batch"]):
            if ctx["stop"] is not None and ctx["stop"].is_set():
                return
            got = db.query(NEWS_REFRESH_SQL, dict(st=source_table, ids=[int(i) for i in chunk]))
            upd = []
            for g in got:
                if g.get("topic_id"):
                    dup = g.get("is_duplicate")
                    upd.append((g["topic_id"], g.get("category_id"), g.get("subtopic_id"),
                                taxonomy.get(g["topic_id"]), taxonomy.get(g.get("category_id")), g.get("cluster_id"),
                                None if dup is None else int(bool(dup)),
                                "news:%s:%s" % (source_table, g["sid"])))
            if upd:
                store.execute("BEGIN IMMEDIATE")
                try:
                    store.executemany(
                        "UPDATE items SET db_topic_id=?, db_category_id=?, db_subtopic_id=?, topic_name=?, "
                        "category_name=?, cluster_id=?, is_duplicate=? WHERE id=?", upd)
                    store.execute("COMMIT")
                except Exception:
                    store.execute("ROLLBACK")
                    raise


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _raise_missing(which, missing):
    raise ConfigError("no credentials for the %s database (missing: %s)" % (which, ", ".join(missing.get(which, []))))


# ---- module-level API (what server.py calls) --------------------------------------------------------------------

def sync_once(limit=None, days=None, sources=None, db_path=None, connect=None, clock=None, **kw):
    """One bounded read-only pass. `limit` is a per-source row cap. Returns the report dict (never raises on
    per-source errors; they are listed under report['errors'])."""
    return RealtimeSync(db_path=db_path, connect=connect, clock=clock).sync_once(limit=limit, days=days, sources=sources, **kw)


def effective_policy():
    """The sync window / caps currently in force (defaults plus env overrides), for status displays."""
    return dict(
        news_days=_env_num("FNP_RT_NEWS_DAYS", DEFAULTS["news_days"]),
        social_days=_env_num("FNP_RT_SOCIAL_DAYS", DEFAULTS["social_days"]),
        news_limit_per_source=_env_num("FNP_RT_NEWS_LIMIT", DEFAULTS["news_limit"], int),
        social_limit_per_table=_env_num("FNP_RT_SOCIAL_LIMIT", DEFAULTS["social_limit"], int),
        topic_refresh_per_cycle=_env_num("FNP_RT_REFRESH_BUDGET", DEFAULTS["refresh_budget"], int),
        overlap_hours=_env_num("FNP_RT_OVERLAP_HOURS", DEFAULTS["overlap_hours"]))


def status(db_path=None, now=None):
    """Store-only status (no database access): rows per source/kind, newest published_at, last sync, lag, errors."""
    if not os.path.exists(db_path or rt_store.DEFAULT_DB):
        return dict(total_rows=0, per_source=[], per_kind={}, newest_published_at=None, last_sync_at=None,
                    last_ok_sync_at=None, minutes_since_last_ok_sync=None, lag_minutes=None, recent_errors=[],
                    watermarks={}, store_exists=False, background=background_status())
    con = rt_store.open_store(db_path, create=False)
    try:
        st = rt_store.store_status(con, now)
    finally:
        con.close()
    st["store_exists"] = True
    st["background"] = background_status()
    return st


def near_duplicates(item_id, max_hamming=3, db_path=None):
    """[(other_item_id, hamming)] among stored items (in-memory simhash index, rebuilt when the store changes)."""
    con = rt_store.open_store(db_path, create=False)
    try:
        return rt_store.near_duplicates(con, item_id, max_hamming, path_key=db_path or rt_store.DEFAULT_DB)
    finally:
        con.close()


def search_ids(q, limit=50, offset=0, db_path=None):
    con = rt_store.open_store(db_path, create=False)
    try:
        return rt_store.search_ids(con, q, limit, offset)
    finally:
        con.close()


class Background:
    """Daemon thread that runs sync_once every `interval_s`, resuming from the stored watermarks."""

    def __init__(self, interval_s=300, **sync_kwargs):
        self.interval_s = max(1.0, float(interval_s))
        self.kwargs = sync_kwargs
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="fnp-rt-sync", daemon=True)
        self.runs = 0
        self.last_report = None
        self.last_error = None
        self.last_run_at = None

    def start(self):
        self.thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.last_report = sync_once(stop_event=self._stop, prune=True, **self.kwargs)
                self.last_error = None
            except BaseException as e:  # a failing pass must never kill the server
                self.last_error = redact("%s: %s" % (type(e).__name__, e))
                log.warning("background sync error: %s", self.last_error)
            self.runs += 1
            self.last_run_at = rt_store.utc_iso(datetime.now(timezone.utc))
            if self._stop.wait(self.interval_s):
                break

    def stop(self, timeout=5.0):
        self._stop.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout)

    def is_running(self):
        return self.thread.is_alive() and not self._stop.is_set()

    def info(self):
        return dict(running=self.is_running(), interval_s=self.interval_s, runs=self.runs,
                    last_run_at=self.last_run_at, last_error=self.last_error,
                    last_rows=(self.last_report or {}).get("rows_total"),
                    last_seconds=(self.last_report or {}).get("seconds"))


_background = None
_bg_lock = threading.Lock()


def start_background(interval_s=300, **sync_kwargs):
    """Start (once) the background sync thread; returns the Background handle (call .stop() to end it)."""
    global _background
    with _bg_lock:
        if _background is not None and _background.is_running():
            return _background
        _background = Background(interval_s, **sync_kwargs).start()
        return _background


def stop_background(timeout=5.0):
    global _background
    with _bg_lock:
        bg, _background = _background, None
    if bg is not None:
        bg.stop(timeout)


def background_status():
    bg = _background
    return bg.info() if bg is not None else dict(running=False)


# ---- CLI -------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Sync the realtime news/social databases into the local SQLite cache "
                                             "(read-only on the databases).")
    ap.add_argument("--once", action="store_true", help="run one sync pass")
    ap.add_argument("--status", action="store_true", help="print store status (no database access)")
    ap.add_argument("--limit", type=int, help="row cap PER SOURCE for this run (tests / dry runs)")
    ap.add_argument("--days", type=float, help="window in days for both kinds (default: news 7 / social 30, FNP_RT_NEWS_DAYS / FNP_RT_SOCIAL_DAYS)")
    ap.add_argument("--per-source-limit", type=int, help="row cap per source (default: news 400000 / social 100000, FNP_RT_NEWS_LIMIT / FNP_RT_SOCIAL_LIMIT)")
    ap.add_argument("--sources", help="comma list: news, social, a social table name, or a news source table")
    ap.add_argument("--db", help="SQLite path (default Interface/data/realtime.sqlite)")
    ap.add_argument("--prune", action="store_true", help="after the pass, delete stored items older than the window")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr)
    if not (a.once or a.status):
        ap.print_help()
        return 2
    if a.once:
        rep = sync_once(limit=a.limit, days=a.days, sources=a.sources, db_path=a.db,
                        per_source_limit=a.per_source_limit, prune=a.prune)
        print(json.dumps(rep, indent=2, default=str))
        if a.status:
            print(json.dumps(status(a.db), indent=2, default=str))
        return 1 if rep.get("errors") else 0
    print(json.dumps(status(a.db), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
