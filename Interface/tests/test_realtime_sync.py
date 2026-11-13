import json
import logging
import os
import re
import stat
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from unittest import mock

import helpers  # noqa: F401  (sys.path / env setup)
import realtime_sync as rs
import rt_store
from rt_fakes import NOW, PASSWORD, FakePG, news_row, social_row
from rt_text import hamming, simhash64, to_unsigned


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "data", "realtime.sqlite")
        self.addCleanup(self.tmp.cleanup)

    def run_sync(self, pg, **kw):
        kw.setdefault("clock", lambda: NOW)
        return rs.sync_once(db_path=self.db, connect=pg.connect, **kw)

    def store(self):
        con = rt_store.open_store(self.db, create=False)
        self.addCleanup(con.close)
        return con

    def ids(self):
        return {r[0] for r in self.store().execute("SELECT id FROM items")}


class Mapping(Base):
    def test_every_social_table_maps_to_an_item(self):
        social = {t: [social_row(100 + n, minutes_ago=5 + n, title="T <b>x</b>" if cfg["title"] != "NULL::text" else None,
                                 body="<p>Hello &amp; welcome</p>", eng=7, lang="HI")
                      for n in range(2)] for t, cfg in rs.SOCIAL_TABLES.items()}
        pg = FakePG(social=social)
        rep = self.run_sync(pg, sources="social")
        self.assertEqual(rep["errors"], [])
        self.assertEqual(rep["rows_total"], 2 * len(rs.SOCIAL_TABLES))
        con = self.store()
        for t, cfg in rs.SOCIAL_TABLES.items():
            r = con.execute("SELECT * FROM items WHERE id = ?", ("%s:100" % t,)).fetchone()
            self.assertIsNotNone(r, t)
            self.assertEqual((r["kind"], r["platform"], r["source_table"], r["source_id"]),
                             ("social", cfg["platform"], t, "100"))
            self.assertEqual(r["text_excerpt"], "Hello & welcome")
            self.assertEqual(r["lang"], "hi")
            self.assertEqual(r["engagement"], 7)
            self.assertEqual(r["published_at"], "2026-11-06T11:55:00Z")
            self.assertEqual(r["source"], "user100")
            self.assertTrue(r["url"].startswith("https://"))
        self.assertEqual(con.execute("SELECT title FROM items WHERE id='reddit_posts:100'").fetchone()[0], "T x")

    def test_news_mapping_taxonomy_excerpt_and_lang(self):
        pg = FakePG(
            news={"scraped_articles": [news_row(1, 10, topic_id="D01.T02", category_id="D01.T02.C01", subtopic_id="s1",
                                                cluster_id="c9", is_duplicate=True),
                                       news_row(2, 20)],
                  "scraped_articles_trusted": [news_row(3, 30, title="Trusted story")]},
            content={"scraped_articles": {1: dict(content="<div>Full <i>body</i>\x00 text</div>", lang=None),
                                          2: dict(content="Second body", lang=None)},
                     "scraped_articles_trusted": {3: dict(content="Trusted body", lang="EN")}},
            taxonomy={"D01.T02": "Protest & Unrest", "D01.T02.C01": "Riots"})
        rep = self.run_sync(pg, sources="news")
        self.assertEqual(rep["errors"], [])
        con = self.store()
        r = con.execute("SELECT * FROM items WHERE id='news:scraped_articles:1'").fetchone()
        self.assertEqual((r["kind"], r["platform"], r["source_table"], r["source_id"]),
                         ("news", "news", "scraped_articles", "1"))
        self.assertEqual(r["text_excerpt"], "Full body text")
        self.assertEqual((r["topic_name"], r["category_name"]), ("Protest & Unrest", "Riots"))
        self.assertEqual((r["db_topic_id"], r["db_subtopic_id"], r["cluster_id"], r["is_duplicate"]), ("D01.T02", "s1", "c9", 1))
        self.assertEqual(r["source"], "news.example.com")
        self.assertEqual(r["lang"], "und")
        t = con.execute("SELECT lang, text_excerpt FROM items WHERE id='news:scraped_articles_trusted:3'").fetchone()
        self.assertEqual((t["lang"], t["text_excerpt"]), ("en", "Trusted body"))

    def test_unjoinable_news_source_table_is_reported_not_synced(self):
        pg = FakePG(news={"scraped_articles": [news_row(1, 10)], "mystery_table": [news_row(9, 5)]},
                    content={"scraped_articles": {1: dict(content="b", lang=None)}})
        rep = self.run_sync(pg, sources="scraped_articles")
        self.assertTrue(any("mystery_table" in s["table"] for s in rep["skipped_tables"]), rep["skipped_tables"])
        self.assertEqual(self.ids(), {"news:scraped_articles:1"})
        self.assertTrue(any(s["table"] == "youtube_comments" for s in rep["skipped_tables"]))

    def test_rows_without_text_or_with_bad_urls(self):
        pg = FakePG(social={"telegram_posts": [
            social_row(1, 1, body="   ", title=None), social_row(2, 2, body="ok text", url="javascript:alert(1)"),
            social_row(3, 3, body="has engagement", eng=0)]})
        rep = self.run_sync(pg, sources="telegram_posts")
        self.assertEqual(rep["sources"]["telegram_posts"]["skipped_empty"], 1)
        con = self.store()
        self.assertEqual(self.ids(), {"telegram_posts:2", "telegram_posts:3"})
        self.assertIsNone(con.execute("SELECT url FROM items WHERE id='telegram_posts:2'").fetchone()[0])
        self.assertEqual(con.execute("SELECT engagement FROM items WHERE id='telegram_posts:3'").fetchone()[0], 0)
        self.assertIsNone(con.execute("SELECT engagement FROM items WHERE id='telegram_posts:2'").fetchone()[0])

    def test_stored_text_is_clean_and_bounded(self):
        dirty = "<p>" + ("lorem ipsum &amp; dolor " * 300) + "\x00\x07</p>"
        pg = FakePG(social={"reddit_posts": [social_row(1, 1, title="<h1>Big   title</h1>", body=dirty)]})
        self.run_sync(pg, sources="reddit_posts")
        r = self.store().execute("SELECT title, text_excerpt FROM items").fetchone()
        self.assertEqual(r["title"], "Big title")
        self.assertLessEqual(len(r["text_excerpt"]), 1500)
        for bad in ("<", ">", "&amp;", "\x00", "\x07", "  "):
            self.assertNotIn(bad, r["text_excerpt"])

    def test_all_generated_sql_is_select_only(self):
        for t, cfg in rs.SOCIAL_TABLES.items():
            for ks in (False, True):
                rs.check_select(rs.social_sql(cfg, ks).replace("{table}", t))
        for sql in (rs.NEWS_SQL.format(keyset=""), rs.NEWS_SQL.format(keyset=rs.NEWS_KEYSET), rs.NEWS_TAXONOMY_SQL,
                    rs.NEWS_TABLES_SQL, rs.NEWS_REFRESH_SQL,
                    rs.NEWS_CONTENT_SQL.format(table="raw.scraped_articles", lang="")):
            rs.check_select(sql)

    def test_check_select_refuses_writes(self):
        for bad in ("INSERT INTO x VALUES (1)", "update t set a=1", "DELETE FROM t", "SELECT 1; DROP TABLE t",
                    "select * into newt from t", "CREATE TABLE a(b int)", "COPY t TO '/tmp/x'", "  drop table t",
                    "SELECT pg_terminate_backend(1)", "WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x",
                    "SET default_transaction_read_only=off", "SHOW all"):
            with self.assertRaises(rs.DbError, msg=bad):
                rs.check_select(bad)
        rs.check_select("select 1;")
        rs.check_select("WITH a AS (SELECT 1) SELECT * FROM a")


class Pagination(Base):
    def test_keyset_pagination_batches_without_offset_and_ties(self):
        rows = [social_row(i, minutes_ago=i // 3) for i in range(1, 26)]  # 3 rows per timestamp: ties across batches
        pg = FakePG(social={"reddit_posts": rows})
        rep = self.run_sync(pg, sources="reddit_posts", batch=10, overlap_hours=0)
        self.assertEqual(rep["rows_total"], 25)
        self.assertEqual(len(self.ids()), 25)
        pages = [s for s in pg.statements if "FROM public.reddit_posts" in s[1]]
        self.assertEqual(len(pages), 3)
        self.assertEqual([("kt" in s[2]) for s in pages], [False, True, True])
        self.assertTrue(all("OFFSET" not in s[1].upper() for s in pg.statements))
        self.assertTrue(all(s[2]["n"] <= 10 for s in pages))

    def test_limit_caps_rows_per_source_and_keeps_the_newest(self):
        rows = [social_row(i, minutes_ago=i) for i in range(1, 31)]
        pg = FakePG(social={"reddit_posts": rows, "telegram_posts": rows[:5]})
        rep = self.run_sync(pg, sources="reddit_posts,telegram_posts", limit=12, batch=5)
        self.assertEqual(rep["sources"]["reddit_posts"]["rows"], 12)
        self.assertEqual(rep["sources"]["telegram_posts"]["rows"], 5)
        self.assertTrue(rep["sources"]["reddit_posts"]["truncated"])
        got = {int(i.split(":")[1]) for i in self.ids() if i.startswith("reddit")}
        self.assertEqual(got, set(range(1, 13)))  # newest 12 (smallest minutes_ago)

    def test_window_excludes_old_and_future_rows(self):
        rows = [social_row(1, minutes_ago=5), social_row(2, hours_ago=24 * 40), social_row(3, minutes_ago=-3 * 24 * 60)]
        pg = FakePG(social={"reddit_posts": rows})
        self.run_sync(pg, sources="reddit_posts", days=30)
        self.assertEqual(self.ids(), {"reddit_posts:1"})


class Watermarks(Base):
    def test_resync_is_idempotent_and_incremental(self):
        rows = [social_row(i, hours_ago=i * 5) for i in range(1, 11)]
        pg = FakePG(social={"reddit_posts": rows})
        self.run_sync(pg, sources="reddit_posts", overlap_hours=1)
        con = self.store()
        before = con.execute("SELECT count(*), max(synced_at) FROM items").fetchone()
        wm = con.execute("SELECT last_published_at, last_id, floor_published_at FROM watermarks").fetchone()
        self.assertEqual(wm[0], rt_store.utc_iso(rows[0]["ts"]))
        self.assertEqual(wm[1], "1")
        self.assertEqual(wm[2], rt_store.utc_iso(NOW - timedelta(days=30)))
        n = len(pg.statements)
        self.run_sync(pg, sources="reddit_posts", overlap_hours=1)
        self.assertEqual(con.execute("SELECT count(*) FROM items").fetchone()[0], before[0])
        second = pg.statements[n:]
        self.assertEqual(len(second), 1)  # only the newest overlap window is re-read
        self.assertGreaterEqual(second[0][2]["lo"], rows[0]["ts"] - timedelta(hours=1, seconds=1))
        # new rows arrive
        pg.social["reddit_posts"].append(social_row(99, minutes_ago=1))
        pg.social["reddit_posts"].append(social_row(98, minutes_ago=2))
        rep = self.run_sync(pg, sources="reddit_posts", overlap_hours=1)
        self.assertEqual(rep["rows_total"], 3)  # 2 new + the overlapping newest row, upserted again
        self.assertEqual(con.execute("SELECT count(*) FROM items").fetchone()[0], 12)
        self.assertEqual(con.execute("SELECT last_id FROM watermarks").fetchone()[0], "99")

    def test_resync_updates_changed_fields_in_place(self):
        pg = FakePG(social={"reddit_posts": [social_row(1, 1, body="first version of the text here", eng=1)]})
        self.run_sync(pg, sources="reddit_posts")
        pg.social["reddit_posts"][0].update(body="edited version of the text here", eng=50)
        self.run_sync(pg, sources="reddit_posts")
        con = self.store()
        self.assertEqual(con.execute("SELECT count(*) FROM items").fetchone()[0], 1)
        r = con.execute("SELECT text_excerpt, engagement FROM items").fetchone()
        self.assertEqual((r[0], r[1]), ("edited version of the text here", 50))
        self.assertEqual(rt_store.search_ids(con, "edited"), ["reddit_posts:1"])
        self.assertEqual(rt_store.search_ids(con, "first"), [])  # FTS follows the update

    def test_truncated_run_is_backfilled_by_later_runs(self):
        rows = [social_row(i, hours_ago=i * 3) for i in range(1, 26)]
        pg = FakePG(social={"reddit_posts": rows})
        self.run_sync(pg, sources="reddit_posts", limit=10, batch=4, overlap_hours=1)
        self.assertEqual(len(self.ids()), 10)
        self.run_sync(pg, sources="reddit_posts", limit=10, batch=4, overlap_hours=1)
        self.assertEqual(len(self.ids()), 10)  # budget (cap - stored) is zero: no growth beyond the cap
        self.run_sync(pg, sources="reddit_posts", limit=20, batch=4, overlap_hours=1)
        self.assertGreaterEqual(len(self.ids()), 19)
        self.run_sync(pg, sources="reddit_posts", limit=30, batch=4, overlap_hours=1)
        self.run_sync(pg, sources="reddit_posts", limit=30, batch=4, overlap_hours=1)
        self.assertEqual(len(self.ids()), 25)
        floor = self.store().execute("SELECT floor_published_at FROM watermarks").fetchone()[0]
        self.assertEqual(floor, rt_store.utc_iso(NOW - timedelta(days=30)))

    def test_failed_source_keeps_its_old_watermark(self):
        pg = FakePG(social={"reddit_posts": [social_row(1, hours_ago=3)]})
        self.run_sync(pg, sources="reddit_posts")
        wm = self.store().execute("SELECT last_published_at FROM watermarks").fetchone()[0]
        pg.social["reddit_posts"].append(social_row(2, minutes_ago=1))
        pg.raise_on.add("reddit_posts")
        rep = self.run_sync(pg, sources="reddit_posts")
        self.assertTrue(rep["errors"])
        self.assertEqual(self.store().execute("SELECT last_published_at FROM watermarks").fetchone()[0], wm)

    def test_news_classification_refresh_fills_late_topics(self):
        pg = FakePG(news={"scraped_articles": [news_row(1, 10), news_row(2, 20)]},
                    content={"scraped_articles": {1: dict(content="body one", lang=None), 2: dict(content="body two", lang=None)}},
                    taxonomy={"D01.T02": "Protest"})
        self.run_sync(pg, sources="scraped_articles")
        con = self.store()
        self.assertEqual(con.execute("SELECT count(*) FROM items WHERE db_topic_id IS NOT NULL").fetchone()[0], 0)
        pg.news["scraped_articles"][0]["topic_id"] = "D01.T02"
        pg.news["scraped_articles"][0]["category_id"] = "D01.T02.C01"
        # article 1 is re-read by the overlap; article 2 only through the refresh; both must stay consistent
        self.run_sync(pg, sources="scraped_articles", overlap_hours=0)
        r = con.execute("SELECT topic_name, db_category_id FROM items WHERE id='news:scraped_articles:1'").fetchone()
        self.assertEqual((r[0], r[1]), ("Protest", "D01.T02.C01"))
        pg.news["scraped_articles"][1]["topic_id"] = "D01.T02"
        self.run_sync(pg, sources="scraped_articles", overlap_hours=0)
        self.assertEqual(con.execute("SELECT topic_name FROM items WHERE id='news:scraped_articles:2'").fetchone()[0], "Protest")

    def test_known_news_bodies_are_not_refetched(self):
        pg = FakePG(news={"scraped_articles": [news_row(1, 10)]},
                    content={"scraped_articles": {1: dict(content="body one", lang=None)}})
        self.run_sync(pg, sources="scraped_articles")
        body_queries = lambda: len([s for s in pg.statements if "FROM raw." in s[1]])
        self.assertEqual(body_queries(), 1)
        self.run_sync(pg, sources="scraped_articles")
        self.assertEqual(body_queries(), 1)
        self.assertEqual(self.store().execute("SELECT text_excerpt FROM items").fetchone()[0], "body one")


class SearchAndDuplicates(Base):
    def test_fts_search_and_query_sanitising(self):
        pg = FakePG(social={"reddit_posts": [
            social_row(1, 1, title="Flood warning issued", body="River levels rising near the old bridge"),
            social_row(2, 2, title=None, body="Cricket final tonight at the stadium"),
            social_row(3, 3, title="Bridge repairs", body="Council approves bridge budget")]})
        self.run_sync(pg, sources="reddit_posts")
        con = self.store()
        self.assertEqual(set(rt_store.search_ids(con, "bridge")), {"reddit_posts:1", "reddit_posts:3"})
        self.assertEqual(rt_store.search_ids(con, "flood bridge"), ["reddit_posts:1"])
        self.assertEqual(rt_store.search_ids(con, "crick"), ["reddit_posts:2"])  # prefix on the last word
        for evil in ('"', "bridge OR", "a AND (", "NEAR(", "title:flood", "* ^ -", "'; DROP TABLE items; --"):
            rt_store.search_ids(con, evil)  # must not raise
        self.assertEqual(rt_store.search_ids(con, ""), [])
        self.assertEqual(set(rs.search_ids("bridge", db_path=self.db)), {"reddit_posts:1", "reddit_posts:3"})

    def test_near_duplicates(self):
        base = ("Heavy floods hit the river valley on Monday and thousands of residents were evacuated from low lying "
                "villages while rescue teams worked through the night")
        pg = FakePG(social={"telegram_posts": [
            social_row(1, 1, body=base), social_row(2, 2, body=base), social_row(3, 3, body=base.upper() + "!!"),
            social_row(4, 4, body="The home side won the toss and elected to bat first on a dry pitch before scoring "
                                  "four hundred runs in the opening innings"),
            social_row(5, 5, body="short")]})
        self.run_sync(pg, sources="telegram_posts")
        d = dict(rs.near_duplicates("telegram_posts:1", db_path=self.db))
        self.assertEqual(d, {"telegram_posts:2": 0, "telegram_posts:3": 0})
        self.assertEqual(rs.near_duplicates("telegram_posts:4", db_path=self.db), [])
        self.assertEqual(rs.near_duplicates("telegram_posts:5", db_path=self.db), [])  # too short: no fingerprint
        self.assertEqual(rs.near_duplicates("nope:1", db_path=self.db), [])

    def test_near_duplicates_respects_hamming_and_sees_new_rows(self):
        pg = FakePG(social={"telegram_posts": [social_row(1, 1, body="alpha beta gamma delta epsilon zeta")]})
        self.run_sync(pg, sources="telegram_posts")
        con = self.store()
        h = to_unsigned(con.execute("SELECT simhash FROM items").fetchone()[0])
        now = rt_store.utc_iso(NOW)
        for i, flips in enumerate((2, 3, 4), start=10):
            hh = h
            for b in range(flips):
                hh ^= 1 << (b * 7)
            rt_store.upsert_items(con, [dict(
                id="telegram_posts:%d" % i, kind="social", platform="telegram", source_table="telegram_posts",
                source_id=str(i), title=None, text_excerpt="x", lang="und", published_at=now, simhash=hh, synced_at=now)])
        self.assertEqual(sorted(dict(rs.near_duplicates("telegram_posts:1", 3, db_path=self.db))),
                         ["telegram_posts:10", "telegram_posts:11"])
        self.assertEqual(sorted(dict(rs.near_duplicates("telegram_posts:1", 4, db_path=self.db))),
                         ["telegram_posts:10", "telegram_posts:11", "telegram_posts:12"])
        self.assertIsNone(simhash64("a b c", ""))
        self.assertEqual(hamming(h, h ^ 0b111), 3)


class Status(Base):
    def test_status_counts_lag_and_errors(self):
        pg = FakePG(social={"reddit_posts": [social_row(1, 30), social_row(2, 90)], "telegram_posts": [social_row(5, 10)]},
                    news={"scraped_articles": [news_row(1, 5)]},
                    content={"scraped_articles": {1: dict(content="body", lang=None)}})
        pg.raise_on.add("instagram_posts")
        pg.social["instagram_posts"] = []
        self.run_sync(pg, sources="reddit_posts,telegram_posts,scraped_articles,instagram_posts")
        st = rs.status(self.db, now=NOW + timedelta(minutes=10))
        self.assertEqual(st["total_rows"], 4)
        self.assertEqual(st["per_kind"]["social"]["rows"], 3)
        self.assertEqual(st["per_kind"]["news"]["rows"], 1)
        by = {p["source_key"]: p for p in st["per_source"]}
        self.assertEqual(by["reddit_posts"]["rows"], 2)
        self.assertEqual(by["scraped_articles"]["kind"], "news")
        self.assertEqual(st["newest_published_at"], rt_store.utc_iso(NOW - timedelta(minutes=5)))
        self.assertAlmostEqual(st["lag_minutes"], 15.0, places=1)
        self.assertAlmostEqual(st["per_kind"]["social"]["lag_minutes"], 20.0, places=1)
        self.assertEqual(st["last_sync_at"], rt_store.utc_iso(NOW))
        self.assertAlmostEqual(st["minutes_since_last_ok_sync"], 10.0, places=1)
        self.assertEqual([e["source_key"] for e in st["recent_errors"]], ["instagram_posts"])
        self.assertNotIn(PASSWORD, json.dumps(st))
        json.dumps(st, allow_nan=False)

    def test_status_without_a_store(self):
        st = rs.status(os.path.join(self.tmp.name, "absent.sqlite"))
        self.assertEqual(st["total_rows"], 0)
        self.assertFalse(st["store_exists"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "absent.sqlite")))


class ErrorIsolation(Base):
    def test_one_failing_source_does_not_stop_the_others(self):
        pg = FakePG(social={"reddit_posts": [social_row(1, 1)], "telegram_posts": [social_row(2, 1)],
                            "instagram_posts": [social_row(3, 1)]}, raise_on={"telegram_posts"})
        rep = self.run_sync(pg, sources="reddit_posts,telegram_posts,instagram_posts")
        self.assertEqual({e["source"] for e in rep["errors"]}, {"telegram_posts"})
        self.assertEqual(self.ids(), {"reddit_posts:1", "instagram_posts:3"})
        self.assertIsNotNone(rep["sources"]["telegram_posts"]["error"])
        self.assertIsNone(rep["sources"]["reddit_posts"]["error"])
        self.assertNotIn(PASSWORD, json.dumps(rep))
        errs = self.store().execute("SELECT source_key, error FROM sync_log WHERE error IS NOT NULL").fetchall()
        self.assertEqual([e[0] for e in errs], ["telegram_posts"])

    def test_unreachable_database_fails_its_sources_only(self):
        pg = FakePG(social={"reddit_posts": [social_row(1, 1)]}, news={"scraped_articles": [news_row(1, 1)]},
                    connect_fail={"news"})
        rep = self.run_sync(pg, sources="scraped_articles,scraped_articles_trusted,reddit_posts")
        self.assertEqual(len(rep["errors"]), 2)
        self.assertEqual(self.ids(), {"reddit_posts:1"})
        self.assertEqual(len([c for c in pg.connections if c.which == "news"]), 0)

    def test_unknown_source_name_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            rs.sync_once(db_path=self.db, connect=FakePG().connect, sources="nope_table")

    def test_connections_are_closed_one_per_database(self):
        pg = FakePG(social={"reddit_posts": [social_row(1, 1)], "telegram_posts": [social_row(2, 1)]},
                    news={"scraped_articles": [news_row(1, 1)]}, content={"scraped_articles": {1: dict(content="b", lang=None)}})
        self.run_sync(pg, sources="reddit_posts,telegram_posts,scraped_articles")
        self.assertEqual(sorted(c.which for c in pg.connections), ["news", "social"])
        self.assertTrue(all(c.closed for c in pg.connections))


class SafetyAndSecrets(Base):
    def test_every_statement_sent_is_a_select(self):
        pg = FakePG(social={t: [social_row(1, 1)] for t in rs.SOCIAL_TABLES},
                    news={"scraped_articles": [news_row(1, 1), news_row(2, 2)], "scraped_articles_trusted": [news_row(3, 3)]},
                    content={"scraped_articles": {1: dict(content="b", lang=None), 2: dict(content="c", lang=None)},
                             "scraped_articles_trusted": {3: dict(content="d", lang="en")}})
        self.run_sync(pg)
        self.run_sync(pg)
        self.assertGreater(len(pg.statements), 20)
        for which, sql, params in pg.statements:
            self.assertRegex(sql, r"^\s*(SELECT|WITH)\b", sql[:80])
            self.assertNotRegex(sql, r"(?i)\b(insert|update|delete|drop|alter|create|truncate|grant|copy|into)\b")
            self.assertNotIn(";", sql.strip().rstrip(";"))

    def test_default_connection_is_read_only_with_timeouts(self):
        calls = {}

        class Cur:
            def __init__(self, c): self.c = c
            def execute(self, sql, params=None):
                self.c.sql.append(sql)
            def fetchone(self): return ("on",)
            def close(self): pass

        class Conn:
            def __init__(self): self.sql, self.closed = [], False
            def set_session(self, **kw): calls["session"] = kw
            def cursor(self): return Cur(self)
            def close(self): self.closed = True

        conn = Conn()

        def fake_connect(**kw):
            calls["kw"] = kw
            return conn

        env = {"FNP_RT_HOST": "db.example", "FNP_RT_NEWS_PORT": "1", "FNP_RT_NEWS_DB": "n", "FNP_RT_NEWS_USER": "u",
               "FNP_RT_NEWS_PASSWORD": PASSWORD, "FNP_RT_SOCIAL_PORT": "2", "FNP_RT_SOCIAL_DB": "s",
               "FNP_RT_SOCIAL_USER": "u2", "FNP_RT_SOCIAL_PASSWORD": "other-pw-123"}
        settings = rs.load_settings(env, env_file="/nonexistent")
        with mock.patch.dict("sys.modules", {"psycopg2": mock.Mock(connect=fake_connect)}):
            got = rs.default_connect("news", settings[0])
        self.assertIs(got, conn)
        opts = calls["kw"]["options"]
        self.assertIn("default_transaction_read_only=on", opts)
        self.assertIn("statement_timeout=30000", opts)
        self.assertIn("idle_in_transaction_session_timeout=60000", opts)
        self.assertEqual(calls["session"], dict(readonly=True, autocommit=True))
        self.assertEqual(calls["kw"]["dbname"], "n")
        self.assertTrue(all(s.lstrip().upper().startswith("SELECT") for s in conn.sql))

    def test_default_connection_refuses_a_session_that_is_not_read_only(self):
        class Cur:
            def execute(self, *a): pass
            def fetchone(self): return ("off",)
            def close(self): pass

        class Conn:
            closed = False
            def set_session(self, **kw): pass
            def cursor(self): return Cur()
            def close(self): self.closed = True

        c = Conn()
        settings = rs.load_settings({"FNP_RT_HOST": "h", "FNP_RT_NEWS_PORT": "1", "FNP_RT_NEWS_DB": "n",
                                     "FNP_RT_NEWS_USER": "u", "FNP_RT_NEWS_PASSWORD": PASSWORD}, "/nonexistent")[0]
        with mock.patch.dict("sys.modules", {"psycopg2": mock.Mock(connect=lambda **kw: c)}):
            with self.assertRaises(rs.DbError):
                rs.default_connect("news", settings)
        self.assertTrue(c.closed)

    def test_connect_failure_message_never_contains_the_password(self):
        env = {"FNP_RT_HOST": "h", "FNP_RT_NEWS_PORT": "1", "FNP_RT_NEWS_DB": "n", "FNP_RT_NEWS_USER": "u",
               "FNP_RT_NEWS_PASSWORD": PASSWORD}
        settings = rs.load_settings(env, "/nonexistent")[0]

        def boom(**kw):
            raise RuntimeError('connection to server failed: FATAL password authentication failed, password=%s dsn="x"' % kw["password"])

        with mock.patch.dict("sys.modules", {"psycopg2": mock.Mock(connect=boom)}):
            with self.assertRaises(rs.DbError) as cm:
                rs.default_connect("news", settings)
        self.assertNotIn(PASSWORD, str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)
        self.assertTrue(cm.exception.__suppress_context__)

    def test_password_never_in_repr_logs_report_store_or_exceptions(self):
        env = {"FNP_RT_HOST": "h", "FNP_RT_NEWS_PORT": "1", "FNP_RT_NEWS_DB": "n", "FNP_RT_NEWS_USER": "u",
               "FNP_RT_NEWS_PASSWORD": PASSWORD, "FNP_RT_SOCIAL_PORT": "2", "FNP_RT_SOCIAL_DB": "s",
               "FNP_RT_SOCIAL_USER": "u2", "FNP_RT_SOCIAL_PASSWORD": "other-pw-123"}
        cfgs, missing = rs.load_settings(env, "/nonexistent")
        self.assertEqual(missing, {})
        for c in cfgs.values():
            for text in (repr(c), str(c), "%s" % c, "%r" % [c], json.dumps(repr(c))):
                self.assertNotIn(PASSWORD, text)
                self.assertNotIn("other-pw-123", text)
        with self.assertRaises(TypeError):
            __import__("pickle").dumps(cfgs["news"])
        pg = FakePG(social={"reddit_posts": [social_row(1, 1)], "telegram_posts": [social_row(2, 1)]},
                    raise_on={"telegram_posts"}, connect_fail={"news"})
        logs = []

        class H(logging.Handler):
            def emit(self, rec): logs.append(rec.getMessage())

        h = H()
        logging.getLogger("fnp.realtime").addHandler(h)
        self.addCleanup(logging.getLogger("fnp.realtime").removeHandler, h)
        rep = self.run_sync(pg, sources="reddit_posts,telegram_posts,scraped_articles")
        blob = json.dumps(rep) + "\n".join(logs)
        self.assertTrue(rep["errors"])
        self.assertNotIn(PASSWORD, blob)
        raw = b""
        for suffix in ("", "-wal"):
            if os.path.exists(self.db + suffix):
                with open(self.db + suffix, "rb") as fh:
                    raw += fh.read()
        self.assertNotIn(PASSWORD.encode(), raw)
        self.assertNotIn("password=%s" % PASSWORD, json.dumps(rs.status(self.db)))
        rs._SECRETS.add("zzz-registered-secret")
        self.assertEqual(rs.redact("x zzz-registered-secret y password=abc123 PASSWORD: 'q r'"), "x *** y password=*** PASSWORD: ***")

    def test_missing_credentials_error_names_variables_not_values(self):
        cfgs, missing = rs.load_settings({"FNP_RT_HOST": "h", "FNP_RT_NEWS_PORT": "1"}, "/nonexistent")
        self.assertEqual(cfgs, {})
        self.assertIn("FNP_RT_NEWS_PASSWORD", missing["news"])
        with self.assertRaises(rs.ConfigError) as cm:
            rs.default_connect("social", cfgs)
        self.assertIn("social", str(cm.exception))

    def test_env_vars_win_over_file_and_file_parsing(self):
        f = os.path.join(self.tmp.name, "rt.env")
        with open(f, "w") as fh:
            fh.write("# comment\nFNP_RT_HOST=filehost\nexport FNP_RT_NEWS_PORT='13355'\nFNP_RT_NEWS_DB=newsdb\n"
                     "FNP_RT_NEWS_USER=nu\nFNP_RT_NEWS_PASSWORD=filepw-1\nbadline\n\nFNP_RT_SOCIAL_PORT=2\n"
                     "FNP_RT_SOCIAL_DB=sdb\nFNP_RT_SOCIAL_USER=su\nFNP_RT_SOCIAL_PASSWORD=\"filepw-2\"\n")
        os.chmod(f, 0o600)
        cfgs, missing = rs.load_settings({"FNP_RT_HOST": "envhost"}, f)
        self.assertEqual(missing, {})
        self.assertEqual((cfgs["news"].host, cfgs["news"].port, cfgs["news"].dbname), ("envhost", 13355, "newsdb"))
        self.assertEqual(cfgs["social"].password, "filepw-2")
        cfgs, _ = rs.load_settings({"FNP_RT_NEWS_PASSWORD": "envpw-9"}, f)
        self.assertEqual(cfgs["news"].password, "envpw-9")

    def test_sqlite_file_is_private_wal_and_has_the_schema(self):
        self.run_sync(FakePG(social={"reddit_posts": [social_row(1, 1)]}), sources="reddit_posts")
        self.assertEqual(stat.S_IMODE(os.stat(self.db).st_mode), 0o600)
        con = self.store()
        self.assertEqual(con.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table')")}
        for t in ("items", "items_fts", "watermarks", "sync_log", "analyses"):
            self.assertIn(t, tables)
        cols = [r[1] for r in con.execute("PRAGMA table_info(items)")]
        self.assertEqual(tuple(cols), rt_store.ITEM_COLUMNS)
        self.assertNotIn("raw_html", cols)


class Prune(Base):
    def test_prune_keeps_analysed_items(self):
        pg = FakePG(social={"reddit_posts": [social_row(1, hours_ago=24 * 20), social_row(2, hours_ago=24 * 25),
                                             social_row(3, hours_ago=24 * 29)]})
        self.run_sync(pg, sources="reddit_posts")
        con = self.store()
        con.execute("INSERT INTO analyses VALUES ('reddit_posts:2', 'x', 'v1', '{}')")
        cut = rt_store.utc_iso(NOW - timedelta(days=22))
        self.assertEqual(rt_store.prune(con, cut), 1)
        self.assertEqual(self.ids(), {"reddit_posts:1", "reddit_posts:2"})
        self.assertEqual(len(rt_store.search_ids(con, "body")), 2)  # FTS rows removed with the item


class Background(Base):
    def test_background_runs_resumes_and_stops(self):
        pg = FakePG(social={"reddit_posts": [social_row(1, 1)]})
        try:
            bg = rs.start_background(interval_s=1.0, db_path=self.db, connect=pg.connect, clock=lambda: NOW,
                                     sources="reddit_posts")
            self.assertIs(rs.start_background(interval_s=1.0), bg)  # idempotent
            deadline = time.time() + 10
            while bg.runs < 2 and time.time() < deadline:
                time.sleep(0.05)
            self.assertGreaterEqual(bg.runs, 2)
            self.assertTrue(rs.background_status()["running"])
            self.assertEqual(rs.status(self.db)["background"]["runs"], bg.runs)
            self.assertEqual(self.ids(), {"reddit_posts:1"})
        finally:
            rs.stop_background()
        self.assertFalse(bg.thread.is_alive())
        self.assertFalse(rs.background_status()["running"])
        self.assertTrue(bg.thread.daemon)

    def test_background_survives_exceptions(self):
        calls = []

        def boom(**kw):
            calls.append(1)
            raise RuntimeError("explode password=%s" % PASSWORD)

        rs._SECRETS.add(PASSWORD)
        try:
            with mock.patch.object(rs, "sync_once", boom):
                bg = rs.start_background(interval_s=1.0)
                deadline = time.time() + 10
                while len(calls) < 2 and time.time() < deadline:
                    time.sleep(0.05)
                self.assertGreaterEqual(len(calls), 2)
                self.assertTrue(bg.is_running())
                self.assertNotIn(PASSWORD, bg.last_error)
        finally:
            rs.stop_background()
        self.assertFalse(bg.thread.is_alive())

    def test_overlapping_runs_are_refused_not_queued(self):
        pg = FakePG(social={"reddit_posts": [social_row(1, 1)]})
        gate, entered = threading.Event(), threading.Event()
        orig = pg.connect

        def slow(which):
            entered.set()
            gate.wait(5)
            return orig(which)

        out = []
        t = threading.Thread(target=lambda: out.append(rs.sync_once(db_path=self.db, connect=slow, sources="reddit_posts", clock=lambda: NOW)))
        t.start()
        self.assertTrue(entered.wait(5))
        rep = rs.sync_once(db_path=self.db, connect=pg.connect, sources="reddit_posts", clock=lambda: NOW)
        self.assertIn("skipped", rep)
        gate.set()
        t.join(10)
        self.assertEqual(out[0]["rows_total"], 1)


class Cli(Base):
    def test_cli_status_and_bad_args(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(rs.main(["--status", "--db", self.db]), 0)
        self.assertEqual(json.loads(buf.getvalue())["total_rows"], 0)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rs.main([]), 2)


if __name__ == "__main__":
    unittest.main()
