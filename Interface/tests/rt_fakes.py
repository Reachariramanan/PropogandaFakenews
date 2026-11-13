"""In-memory stand-in for the two PostgreSQL databases (offline tests for realtime_sync).

FakePG emulates, by pattern-matching the SQL realtime_sync sends, just enough of: a social table page query
(keyset on (ts, sid)), the news article_fact page / classification refresh / source-table summary queries, the body
tables and the taxonomy. It records every statement so tests can assert they are all SELECTs."""
import re
from datetime import datetime, timedelta, timezone

NOW = datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)
PASSWORD = "s3cr3t-pw-xyz"


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.description = None
        self._rows = []
        self._pos = 0

    def execute(self, sql, params=None):
        self.conn.pg.statements.append((self.conn.which, sql, params))
        cols, rows = self.conn.pg.answer(self.conn.which, sql, params or {})
        self.description = [(c, None, None, None, None, None, None) for c in cols]
        self._rows = [tuple(r[c] for c in cols) for r in rows]
        self._pos = 0

    def fetchall(self):
        out, self._pos = self._rows[self._pos:], len(self._rows)
        return out

    def fetchmany(self, n):
        out = self._rows[self._pos:self._pos + n]
        self._pos += len(out)
        return out

    def fetchone(self):
        r = self.fetchmany(1)
        return r[0] if r else None

    def close(self):
        pass


class FakeConn:
    def __init__(self, pg, which):
        self.pg, self.which, self.closed = pg, which, False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


class FakePG:
    def __init__(self, social=None, news=None, content=None, taxonomy=None, raise_on=(), connect_fail=()):
        self.social = social or {}      # table -> [row dict: sid, ts, title, body, author, url, lang, eng]
        self.news = news or {}          # source_table -> [row dict: sid, ts, topic_id, ..., title]
        self.content = content or {}    # source_table -> {article_id: {"content": str, "lang": str}}
        self.taxonomy = taxonomy or {}
        self.raise_on = set(raise_on)   # social tables / news source tables whose queries raise
        self.connect_fail = set(connect_fail)  # 'news' / 'social'
        self.statements = []
        self.connections = []

    def connect(self, which):
        if which in self.connect_fail:
            raise RuntimeError("could not connect: FATAL password authentication failed (password=%s)" % PASSWORD)
        c = FakeConn(self, which)
        self.connections.append(c)
        return c

    # ---- SQL emulation ----
    def answer(self, which, sql, p):
        m = re.search(r"FROM public\.(\w+)", sql)
        if m:
            t = m.group(1)
            if t in self.raise_on:
                raise RuntimeError("canceling statement due to statement timeout (password=%s)" % PASSWORD)
            return self._page(self.social[t], p, "sid")
        if "FROM news_engine.taxonomy_node" in sql:
            return ["code", "name"], [dict(code=k, name=v) for k, v in self.taxonomy.items()]
        if "GROUP BY source_table" in sql:
            out = {}
            for st, rows in self.news.items():
                n = sum(1 for r in rows if p["lo"] <= r["ts"] <= p["hi"])
                if n:
                    out[st] = n
            return ["source_table", "n"], [dict(source_table=k, n=v) for k, v in out.items()]
        if "FROM news_engine.article_fact" in sql and "ANY(" in sql:
            rows = [r for r in self.news.get(p["st"], []) if r["sid"] in set(p["ids"])]
            return ["sid", "topic_id", "category_id", "subtopic_id", "cluster_id", "is_duplicate"], rows
        if "FROM news_engine.article_fact" in sql:
            if p["st"] in self.raise_on:
                raise RuntimeError("news query failed (password=%s)" % PASSWORD)
            cols, rows = self._page(self.news.get(p["st"], []), p, "sid")
            return ["sid", "ts", "topic_id", "category_id", "subtopic_id", "cluster_id", "is_duplicate",
                    "source_domain", "url", "title"], rows
        m = re.search(r"FROM raw\.(\w+)", sql)
        if m:
            table = m.group(1)
            st = {"scraped_articles": "scraped_articles", "scraped_articles_trusted": "scraped_articles_trusted"}[table]
            got = [dict(id=i, **self.content[st][i]) for i in p["ids"] if i in self.content.get(st, {})]
            cols = ["id", "content"] + (["lang"] if "AS lang" in sql else [])
            for g in got:
                g.setdefault("lang", None)
            return cols, got
        raise AssertionError("fake DB cannot answer: %s" % sql[:120])

    @staticmethod
    def _page(rows, p, idcol):
        sel = [r for r in rows if p["lo"] <= r["ts"] <= p["hi"]]
        if p.get("kt") is not None:
            sel = [r for r in sel if (r["ts"], r[idcol]) < (p["kt"], p["kid"])]
        sel.sort(key=lambda r: (r["ts"], r[idcol]), reverse=True)
        sel = sel[:p["n"]]
        cols = ["sid", "ts", "title", "body", "author", "url", "lang", "eng"]
        return cols, [{c: r.get(c) for c in cols} | {k: v for k, v in r.items() if k not in cols} for r in sel]


def social_row(i, minutes_ago=None, hours_ago=None, **kw):
    ts = NOW - timedelta(minutes=minutes_ago if minutes_ago is not None else 0, hours=hours_ago or 0)
    r = dict(sid=i, ts=ts, title=None, body="Body text number %s about floods in the river valley" % i,
             author="user%s" % i, url="https://example.org/p/%s" % i, lang="en", eng=None)
    r.update(kw)
    return r


def news_row(i, minutes_ago=0, hours_ago=0, **kw):
    r = dict(sid=i, ts=NOW - timedelta(minutes=minutes_ago, hours=hours_ago), topic_id=None, category_id=None,
             subtopic_id=None, cluster_id=None, is_duplicate=False, source_domain="news.example.com",
             url="https://news.example.com/a/%s" % i, title="Headline %s | Example Times" % i)
    r.update(kw)
    return r
