"""SQLite store for the realtime cache: schema, FTS5 search, near-duplicate index, status.

The file lives in Interface/data/ (git-ignored), is created with mode 600 and uses WAL. It only ever holds the
fields the UI needs (title, a <=1500-char excerpt, ids, classification, engagement); never raw HTML or full bodies.
Do not VACUUM this file: the FTS5 index is keyed on items.rowid (use rebuild_fts() after any such maintenance).
"""
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone

from rt_text import hamming, to_signed, to_unsigned

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.environ.get("FNP_RT_DB") or os.path.join(HERE, "data", "realtime.sqlite")

ITEM_COLUMNS = (
    "id", "kind", "platform", "source_table", "source_id", "url", "title", "text_excerpt", "source", "lang",
    "published_at", "db_topic_id", "db_category_id", "db_subtopic_id", "topic_name", "category_name",
    "cluster_id", "is_duplicate", "engagement", "simhash", "synced_at",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('news','social')),
  platform TEXT NOT NULL,
  source_table TEXT NOT NULL,
  source_id TEXT NOT NULL,
  url TEXT,
  title TEXT,
  text_excerpt TEXT,
  source TEXT,
  lang TEXT NOT NULL DEFAULT 'und',
  published_at TEXT NOT NULL,
  db_topic_id TEXT,
  db_category_id TEXT,
  db_subtopic_id TEXT,
  topic_name TEXT,
  category_name TEXT,
  cluster_id TEXT,
  is_duplicate INTEGER,
  engagement INTEGER,
  simhash INTEGER,
  synced_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_pub ON items(published_at DESC);
CREATE INDEX IF NOT EXISTS items_kind_platform ON items(kind, platform, published_at DESC);
CREATE INDEX IF NOT EXISTS items_table ON items(source_table, source_id);
CREATE INDEX IF NOT EXISTS items_lang ON items(lang);
CREATE INDEX IF NOT EXISTS items_topic ON items(db_topic_id);
CREATE INDEX IF NOT EXISTS items_source ON items(source);
CREATE INDEX IF NOT EXISTS items_stats ON items(published_at, kind, platform, source, lang, db_topic_id);
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
  title, text_excerpt, content='items', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
  INSERT INTO items_fts(rowid, title, text_excerpt) VALUES (new.rowid, new.title, new.text_excerpt);
END;
CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, title, text_excerpt) VALUES ('delete', old.rowid, old.title, old.text_excerpt);
END;
CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE ON items
WHEN old.title IS NOT new.title OR old.text_excerpt IS NOT new.text_excerpt BEGIN
  INSERT INTO items_fts(items_fts, rowid, title, text_excerpt) VALUES ('delete', old.rowid, old.title, old.text_excerpt);
  INSERT INTO items_fts(rowid, title, text_excerpt) VALUES (new.rowid, new.title, new.text_excerpt);
END;
CREATE TABLE IF NOT EXISTS watermarks (
  source_key TEXT PRIMARY KEY,
  last_published_at TEXT,
  last_id TEXT,
  floor_published_at TEXT
);
CREATE TABLE IF NOT EXISTS sync_log (
  ts TEXT NOT NULL,
  source_key TEXT NOT NULL,
  rows INTEGER NOT NULL DEFAULT 0,
  seconds REAL,
  error TEXT,
  note TEXT
);
CREATE INDEX IF NOT EXISTS sync_log_ts ON sync_log(ts DESC);
CREATE TABLE IF NOT EXISTS analyses (
  item_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  engine_version TEXT NOT NULL,
  result_json TEXT NOT NULL,
  PRIMARY KEY (item_id, engine_version)
);
"""

SCHEDULE_LOG_DDL = """
CREATE TABLE IF NOT EXISTS schedule_log (
  started_at TEXT NOT NULL,
  ended_at TEXT,
  picked INTEGER NOT NULL DEFAULT 0,
  analysed INTEGER NOT NULL DEFAULT 0,
  cached_dupes INTEGER NOT NULL DEFAULT 0,
  degraded INTEGER NOT NULL DEFAULT 0,
  errors INTEGER NOT NULL DEFAULT 0,
  skipped INTEGER NOT NULL DEFAULT 0,
  stop_reason TEXT,
  next_run_at TEXT,
  partial INTEGER NOT NULL DEFAULT 0,
  retried INTEGER NOT NULL DEFAULT 0,
  median_secs REAL
);
CREATE INDEX IF NOT EXISTS schedule_log_started ON schedule_log(started_at DESC);
"""

# Deep verdicts: the newest usable Claim Check result per item in `analyses` (a result without an engine verdict, e.g.
# a degraded one that was never cached or a malformed row, is not a deep verdict). Exposed as the temp view `deep` so
# the API can LEFT JOIN it like the FNP batch sidecar. d_pfm = fake_call P(FAKE) + P(MISLEADING).
DEEP_VIEW = """
CREATE TEMP VIEW IF NOT EXISTS deep AS
SELECT item_id AS id, max(created_at) AS deep_at, engine_version,
       json_extract(result_json, '$.engine.verdict.label') AS d_label,
       json_extract(result_json, '$.engine.verdict.probs.FAKE') AS d_fake,
       json_extract(result_json, '$.engine.verdict.probs.MISLEADING') AS d_mis,
       coalesce(json_extract(result_json, '$.engine.verdict.probs.FAKE'), 0)
         + coalesce(json_extract(result_json, '$.engine.verdict.probs.MISLEADING'), 0) AS d_pfm,
       json_extract(result_json, '$.dup_of') AS dup_of
FROM main.analyses
WHERE json_valid(result_json) AND json_extract(result_json, '$.engine.verdict.label') IS NOT NULL
GROUP BY item_id
"""


def utc_iso(dt):
    """Aware/naive datetime (naive = UTC) -> 'YYYY-MM-DDTHH:MM:SSZ'."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) if s else None


# Partial results (S2): a degraded deep reply (timeout / gemma_error) is kept with its propaganda signals and NO verdict
# (`partial: true`, `partial_info: {retries, last_try_at, first_at, reason}` inside result_json). `partial` lists the items
# whose only stored result is partial (a complete deep verdict always wins).
PARTIAL_VIEW = """
CREATE TEMP VIEW IF NOT EXISTS partial AS
SELECT item_id AS id, max(created_at) AS partial_at, engine_version,
       coalesce(json_extract(result_json, '$.partial_info.retries'), 0) AS retries,
       json_extract(result_json, '$.partial_info.last_try_at') AS last_try_at,
       json_extract(result_json, '$.partial_info.reason') AS reason
FROM main.analyses
WHERE json_valid(result_json) AND json_extract(result_json, '$.partial') = 1
  AND item_id NOT IN (SELECT id FROM deep)
GROUP BY item_id
"""

# columns added to schedule_log after its first release (old stores are migrated in place, old rows are never rewritten)
SCHEDULE_LOG_ADDED = (("partial", "INTEGER NOT NULL DEFAULT 0"), ("retried", "INTEGER NOT NULL DEFAULT 0"),
                      ("median_secs", "REAL"))


def ensure_deep_view(con):
    """Create the temp views `deep` and `partial` on this connection (call BEFORE PRAGMA query_only=ON; temp only)."""
    con.execute(DEEP_VIEW)
    con.execute(PARTIAL_VIEW)


def ensure_schedule_log(con):
    """Make sure the schedule_log table exists and has every column (stores created before the scheduler or before S2
    lack some: ALTER TABLE ADD COLUMN, guarded, old rows keep their values)."""
    con.executescript(SCHEDULE_LOG_DDL)
    have = {r[1] for r in con.execute("PRAGMA table_info(schedule_log)")}
    for name, decl in SCHEDULE_LOG_ADDED:
        if name not in have:
            con.execute("ALTER TABLE schedule_log ADD COLUMN %s %s" % (name, decl))


def open_store(path=None, create=True):
    """Open (and on first use create, mode 600, WAL) the realtime SQLite store. One connection per caller/thread."""
    path = path or DEFAULT_DB
    if create:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(path):
            os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    con = sqlite3.connect(path, timeout=30, isolation_level=None)  # explicit BEGIN/COMMIT below
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=30000")
    if create:
        con.executescript(SCHEMA)
        con.executescript(SCHEDULE_LOG_DDL)
        for suffix in ("-wal", "-shm"):
            try:
                os.chmod(path + suffix, 0o600)
            except OSError:
                pass
    return con


UPSERT_SQL = (
    "INSERT INTO items (%s) VALUES (%s) ON CONFLICT(id) DO UPDATE SET %s"
    % (",".join(ITEM_COLUMNS), ",".join("?" * len(ITEM_COLUMNS)),
       ",".join("%s=excluded.%s" % (c, c) for c in ITEM_COLUMNS if c != "id"))
)


def upsert_items(con, items):
    """Idempotent upsert of item dicts (keys = ITEM_COLUMNS; simhash is the UNSIGNED value or None)."""
    rows = []
    for it in items:
        r = dict(it)
        r["simhash"] = to_signed(r.get("simhash"))
        rows.append(tuple(r.get(c) for c in ITEM_COLUMNS))
    if not rows:
        return 0
    con.execute("BEGIN IMMEDIATE")
    try:
        con.executemany(UPSERT_SQL, rows)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)


def rebuild_fts(con):
    con.execute("INSERT INTO items_fts(items_fts) VALUES('rebuild')")


def prune(con, cutoff_iso, kind=None):
    """Delete items (of `kind`, or all) published before cutoff that have no stored analysis. Returns rows deleted."""
    con.execute("BEGIN IMMEDIATE")
    try:
        cur = con.execute(
            "DELETE FROM items WHERE published_at < ? AND id NOT IN (SELECT item_id FROM analyses)"
            + (" AND kind = ?" if kind else ""), (cutoff_iso, kind) if kind else (cutoff_iso,))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return cur.rowcount


# ---- full-text search ------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"\w+", re.U)


def fts_query(q, prefix_last=True):
    """Turn free text into a safe FTS5 query (every word quoted, AND-ed, last word as prefix). None if no words."""
    words = _TOKEN.findall(q or "")[:12]
    if not words:
        return None
    parts = ['"%s"' % w.replace('"', "") for w in words]
    if prefix_last:
        parts[-1] += "*"
    return " ".join(parts)


def search_ids(con, q, limit=50, offset=0):
    """Ids of items matching `q` in title/text, best match first."""
    m = fts_query(q)
    if not m:
        return []
    rows = con.execute(
        "SELECT i.id FROM items_fts f JOIN items i ON i.rowid = f.rowid WHERE items_fts MATCH ? "
        "ORDER BY bm25(items_fts), i.published_at DESC LIMIT ? OFFSET ?", (m, int(limit), int(offset))).fetchall()
    return [r[0] for r in rows]


# ---- near duplicates -------------------------------------------------------------------------------------------

class SimhashIndex:
    """In-memory simhash index. Bands (max_hamming+1 chunks of the 64 bits): by pigeonhole, two hashes within
    max_hamming bits agree exactly on at least one band, so candidate lookup is complete; candidates are then
    verified with the real hamming distance."""

    def __init__(self, max_hamming=3):
        self.max_hamming = int(max_hamming)
        n = self.max_hamming + 1
        base, extra = divmod(64, n)
        self.bands = []  # (shift, width)
        shift = 0
        for i in range(n):
            w = base + (1 if i < extra else 0)
            self.bands.append((shift, w))
            shift += w
        self.buckets = [dict() for _ in self.bands]
        self.hashes = {}

    def add(self, item_id, h):
        if h is None:
            return
        self.hashes[item_id] = h
        for b, (shift, w) in enumerate(self.bands):
            self.buckets[b].setdefault((h >> shift) & ((1 << w) - 1), []).append(item_id)

    def query(self, h, exclude=None):
        seen = set()
        out = []
        for b, (shift, w) in enumerate(self.bands):
            for cid in self.buckets[b].get((h >> shift) & ((1 << w) - 1), ()):
                if cid in seen or cid == exclude:
                    continue
                seen.add(cid)
                d = hamming(h, self.hashes[cid])
                if d <= self.max_hamming:
                    out.append((cid, d))
        out.sort(key=lambda t: (t[1], t[0]))
        return out


_index_cache = {}
_index_lock = threading.Lock()


def _fingerprint(con):
    r = con.execute("SELECT count(*), coalesce(max(synced_at),''), coalesce(sum(simhash IS NOT NULL),0) FROM items").fetchone()
    return (r[0], r[1], r[2])


def get_index(con, path_key, max_hamming=3):
    """Index built from the store, rebuilt only when the store changed (row count / newest synced_at)."""
    key = (path_key, int(max_hamming))
    fp = _fingerprint(con)
    with _index_lock:
        cached = _index_cache.get(key)
        if cached and cached[0] == fp:
            return cached[1]
    idx = SimhashIndex(max_hamming)
    for r in con.execute("SELECT id, simhash FROM items WHERE simhash IS NOT NULL"):
        idx.add(r[0], to_unsigned(r[1]))
    with _index_lock:
        _index_cache[key] = (fp, idx)
    return idx


def near_duplicates(con, item_id, max_hamming=3, path_key=None):
    """[(other_item_id, hamming_distance)] of stored items whose simhash is within max_hamming of item_id's."""
    r = con.execute("SELECT simhash FROM items WHERE id = ?", (item_id,)).fetchone()
    if r is None or r[0] is None:
        return []
    idx = get_index(con, path_key or id(con), max_hamming)
    return idx.query(to_unsigned(r[0]), exclude=item_id)


def near_dup_scan(con, item_id, h, max_hamming=3, limit=20000):
    """[(hamming, published_at, id, title, source)] of stored items whose simhash is within `max_hamming` bits of the
    UNSIGNED hash `h`, nearest first. Same pigeonhole banding as SimhashIndex, run as an indexless SQL scan so no
    per-process index (hundreds of MB at this scale) has to be held in memory."""
    bands = SimhashIndex(max_hamming).bands
    cond = " OR ".join("((simhash >> ?) & ?) = ?" for _ in bands)
    args = [item_id]
    for shift, w in bands:
        mask = (1 << w) - 1
        args += [shift, mask, (h >> shift) & mask]
    out = []
    for r in con.execute("SELECT id, simhash, title, published_at, source FROM items WHERE id != ? AND "
                         "simhash IS NOT NULL AND (%s) LIMIT %d" % (cond, int(limit)), args):
        d = hamming(h, to_unsigned(r[1]))
        if d <= max_hamming:
            out.append((d, r[3], r[0], r[2], r[4]))
    out.sort(key=lambda t: (t[0], t[2]))
    return out


# ---- status ----------------------------------------------------------------------------------------------------

def store_status(con, now=None):
    """Counts and freshness straight from the store (no database access)."""
    now = now or datetime.now(timezone.utc)
    per_source = [dict(source_key=r[0], kind=r[1], platform=r[2], rows=r[3], newest_published_at=r[4])
                  for r in con.execute(
                      "SELECT source_table, kind, platform, count(*), max(published_at) FROM items "
                      "GROUP BY source_table, kind, platform ORDER BY count(*) DESC")]
    per_kind = {}
    for r in con.execute("SELECT kind, count(*), max(published_at) FROM items GROUP BY kind"):
        newest = r[2]
        lag = None
        if newest:
            lag = round((now - parse_iso(newest)).total_seconds() / 60.0, 1)
        per_kind[r[0]] = dict(rows=r[1], newest_published_at=newest, lag_minutes=lag)
    last = con.execute("SELECT max(ts) FROM sync_log").fetchone()[0]
    last_ok = con.execute("SELECT max(ts) FROM sync_log WHERE error IS NULL").fetchone()[0]
    errors = [dict(ts=r[0], source_key=r[1], error=r[2]) for r in con.execute(
        "SELECT ts, source_key, error FROM sync_log WHERE error IS NOT NULL ORDER BY ts DESC, rowid DESC LIMIT 10")]
    total = con.execute("SELECT count(*) FROM items").fetchone()[0]
    newest_all = con.execute("SELECT max(published_at) FROM items").fetchone()[0]
    since_sync = None
    if last_ok:
        since_sync = round((now - parse_iso(last_ok)).total_seconds() / 60.0, 1)
    return dict(
        total_rows=total, per_source=per_source, per_kind=per_kind, newest_published_at=newest_all,
        last_sync_at=last, last_ok_sync_at=last_ok, minutes_since_last_ok_sync=since_sync,
        lag_minutes=round((now - parse_iso(newest_all)).total_seconds() / 60.0, 1) if newest_all else None,
        recent_errors=errors,
        watermarks={r[0]: r[1] for r in con.execute("SELECT source_key, last_published_at FROM watermarks")},
    )
