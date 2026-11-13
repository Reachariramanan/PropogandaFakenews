"""Evidence Base API (phase E2): stats, items, item detail, FNP enrichment join, auth gate, rate limit, sync policy.
Offline: a small SQLite store built through rt_store in a temp dir, fake export files, the real handler class on an
ephemeral 127.0.0.1 port. Nothing touches the real store, the network or the customer databases."""
import base64
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from unittest import mock

import helpers
import realtime_sync as rs
import rt_api
import rt_enrich
import rt_store
import server
from helpers import call, jcall, needs_index
from rt_fakes import NOW, FakePG, news_row, social_row
from rt_text import hamming, simhash64, to_unsigned

NEAR_TEXT = "Flood waters rise across the river valley as rescue teams evacuate hundreds of villagers tonight"


def iso(**delta):
    return rt_store.utc_iso(NOW - timedelta(**delta))


def item(i, kind="social", platform="reddit", hours=1, title=None, text=None, source="user1", lang="en", topic=None,
         topic_name=None, table=None, simhash=None, url=None):
    table = table or ("scraped_articles" if kind == "news" else "%s_posts" % platform)
    iid = ("news:%s:%s" if kind == "news" else "%s:%s") % (table, i)
    return dict(id=iid, kind=kind, platform=platform, source_table=table, source_id=str(i),
                url=url or "https://example.org/%s/%s" % (table, i), title=title, text_excerpt=text or "body %s" % i,
                source=source, lang=lang, published_at=iso(hours=hours), db_topic_id=topic, db_category_id=None,
                db_subtopic_id=None, topic_name=topic_name, category_name=None, cluster_id=None, is_duplicate=None,
                engagement=None, simhash=simhash, synced_at=iso(minutes=5))


def build_items():
    dup = simhash64("a", NEAR_TEXT)
    far = simhash64("b", "completely different text about cricket tournament final scores and records broken")
    return [
        item(1, "news", "news", hours=1, title="Flood alert for the river valley", source="a.com", topic="D01.T01",
             topic_name="Politics", text=NEAR_TEXT, simhash=dup),
        item(2, "news", "news", hours=2, title="Flood alert repeated", source="b.com", topic="D02.T01",
             topic_name="Economy", text=NEAR_TEXT, simhash=dup),
        item(3, "news", "news", hours=3, title="Cricket final tonight", source="a.com", lang="hi", simhash=far),
        item(4, "news", "news", hours=30, title="Budget debate in parliament", source="a.com", topic="D01.T01",
             topic_name="Politics"),
        item(5, "news", "news", hours=100, title="Old unclassified story", source="b.com"),
        item(6, "news", "news", hours=5, title="Trusted story", source="c.com", table="scraped_articles_trusted"),
        item(10, "social", "reddit", hours=1, title="Rumour about vaccine", text="they say the vaccine is a plot", source="u1"),
        item(11, "social", "reddit", hours=4, title=None, text="the river valley is flooding badly", source="u2"),
        item(12, "social", "telegram", hours=6, title=None, text="forward this to everyone now", source="chan1", lang="hi"),
        item(13, "social", "telegram", hours=26, title=None, text="quiet day no news", source="chan1"),
        item(14, "social", "reddit", hours=50, title="'; DROP TABLE items; --", text="injection \" OR 1=1 NEAR(", source="u3"),
        item(15, "social", "x", hours=2, title="x post", text="short post", source="u4"),
    ]


def rec(iid, p_prop=None, p_fake=None, p_mis=None, verdict=None, techs=(), built="2026-11-06T09:00:00Z", **kw):
    d = dict(id=iid, kind="news" if iid.startswith("news:") else "social", published_at=None, lang="en", p_prop=p_prop,
             techniques=[dict(technique=t, count=c, max_confidence=m) for t, c, m in techs],
             countries=dict(about=["IN"], origin=None), nearest_factcheck=None, p_fake=p_fake, p_misleading=p_mis,
             verdict=verdict, model_version="fnp-test-v1", built_at=built)
    d.update(kw)
    return d


def write_export(base, records_by_file, built_at="2026-11-06T09:00:00Z", raw_extra=None, sha_ok=True, meta_extra=None):
    os.makedirs(base, exist_ok=True)
    files = []
    for rel, recs in records_by_file.items():
        p = os.path.join(base, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        lines = [json.dumps(r) for r in recs] + list((raw_extra or {}).get(rel, []))
        data = ("\n".join(lines) + "\n").encode()
        with open(p, "wb") as f:
            f.write(data)
        files.append(dict(path=rel, rows=len(recs), sha256=hashlib.sha256(data).hexdigest() if sha_ok else "0" * 64))
    meta = dict(schema="fnp.realtime.v1", built_at=built_at, model_version="fnp-test-v1", files=files)
    meta.update(meta_extra or {})
    tmp = os.path.join(base, "meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, os.path.join(base, "meta.json"))
    return meta


def default_export(base, **kw):
    news = [
        rec("news:scraped_articles:1", p_prop=0.9, p_fake=0.7, p_mis=0.2, verdict="FAKE",
            techs=[("loaded_language", 3, 0.91), ("repetition", 1, 0.5)],
            nearest_factcheck=dict(id="kb2:af-1", similarity=0.71),
            spans=[dict(text="river valley", technique="loaded_language", confidence=0.9, field="title")]),
        rec("news:scraped_articles:2", p_prop=0.2, p_fake=0.1, p_mis=0.6, verdict="MISLEADING"),
        rec("news:scraped_articles:3", p_prop=0.05, p_fake=0.02, p_mis=0.01, verdict="LEGIT"),
        rec("news:scraped_articles:4", p_prop=0.55, techs=[("loaded_language", 1, 0.6)]),
        rec("news:scraped_articles:99999", p_prop=0.5),  # not in the cache: ignored by joins
    ]
    social = [
        rec("reddit_posts:10", p_prop=0.95, p_fake=0.8, p_mis=0.1, verdict="FAKE", techs=[("loaded_language", 2, 0.8)]),
        rec("reddit_posts:11", p_prop=0.35),
        dict(source_table="telegram_posts", source_id="12", kind="social", p_prop=0.7, p_fake=None, p_misleading=None,
             verdict=None, techniques=[], model_version="fnp-test-v1", built_at="2026-11-06T09:00:00Z"),  # no id field
    ]
    raw = {"social/part-0001.jsonl": ["{not json", '{"id": "telegram_posts:13", "p_prop": NaN, "verdict": null, "p_fake": null}',
                                      json.dumps(rec("reddit_posts:14", p_prop=0.5))]}
    return write_export(base, {"news/part-0001.jsonl": news, "social/part-0001.jsonl": social}, raw_extra=raw, **kw)


class Fixture(unittest.TestCase):
    """A temp store (items above), fake export dir and an RtApi on a fixed clock."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = os.path.join(self.tmp, "realtime.sqlite")
        self.export = os.path.join(self.tmp, "export")
        self.triage = os.path.join(self.tmp, "triage")
        con = rt_store.open_store(self.db)
        rt_store.upsert_items(con, build_items())
        con.execute("INSERT INTO sync_log(ts, source_key, rows, seconds, error, note) VALUES (?,?,?,?,?,?)",
                    (iso(minutes=5), "reddit_posts", 3, 0.1, None, None))
        con.execute("INSERT INTO sync_log(ts, source_key, rows, seconds, error, note) VALUES (?,?,?,?,?,?)",
                    (iso(minutes=5), "telegram_posts", 0, 0.1, "RuntimeError: boom", None))
        con.execute("INSERT INTO analyses(item_id, created_at, engine_version, result_json) VALUES (?,?,?,?)",
                    ("news:scraped_articles:1", iso(minutes=1), "check_v1", json.dumps({"final": "FAKE"})))
        con.close()
        self.enrich = rt_enrich.Enrich(export_dir=self.export, triage_dir=self.triage,
                                       sidecar=os.path.join(self.tmp, "enr.sqlite"), store_path=self.db)
        self.matches = []
        self.api = rt_api.RtApi(db_path=self.db, enrich=self.enrich, clock=lambda: NOW,
                                fn_matches=lambda text: dict(state="strong", nearest_similarity=0.71, matches=list(self.matches)))

    def q(self, **kw):
        return {k: [str(v)] for k, v in kw.items()}

    def sql(self, query, *args):
        con = sqlite3.connect(self.db)
        try:
            return con.execute(query, args).fetchall()
        finally:
            con.close()


class Stats(Fixture):
    def test_without_export_nothing_is_analysed(self):
        s = self.api.stats(self.q())
        self.assertEqual(s["total"], 12)
        self.assertEqual(s["coverage"]["analysed"], 0)
        self.assertEqual(s["coverage"]["pct"], 0.0)
        self.assertIsNone(s["likely_fake_misleading"]["share"])
        self.assertEqual(s["likely_fake_misleading"]["denominator"], 0)
        self.assertEqual(s["technique_mix"]["top"], [])

    def test_counts_unclassified_and_denominators(self):
        default_export(self.export)
        s = self.api.stats(self.q())
        self.assertEqual(s["by_kind"], {"news": 6, "social": 6})
        self.assertEqual({p["platform"]: p["count"] for p in s["by_platform"]},
                         {"news": 6, "reddit": 3, "telegram": 2, "x": 1})
        top = {t["topic_id"]: t for t in s["by_topic"]["top"]}
        self.assertEqual(top["D01.T01"]["count"], 2)
        self.assertEqual(top["D01.T01"]["name"], "Politics")
        self.assertEqual(s["by_topic"]["classified"], 3)
        un = s["by_topic"]["unclassified"]
        self.assertEqual((un["name"], un["count"]), ("Unclassified", 12 - 3))
        self.assertEqual(s["by_source"]["distinct"], len(self.sql("SELECT DISTINCT source FROM items")))
        self.assertEqual(s["by_source"]["top"][0]["count"], 3)  # a.com has 3 news items
        self.assertEqual({l["lang"]: l["count"] for l in s["by_language"]["top"]}, {"en": 10, "hi": 2})
        # analysed: records whose item is in the cache. 99999 is not; telegram_posts:13 (NaN line) was dropped.
        cov = s["coverage"]
        self.assertEqual(cov["items_in_scope"], 12)
        self.assertEqual(cov["analysed"], 4 + 3 + 1)  # news 1-4, reddit 10/11/14, telegram 12
        self.assertEqual(cov["pct"], round(100.0 * 8 / 12, 1))
        self.assertEqual(cov["by_kind"]["news"], dict(items=6, analysed=4, pct=round(100 * 4 / 6, 1), batch=4, deep=0, both=0))
        self.assertEqual(cov["verdicts"], {"FAKE": 2, "MISLEADING": 1, "LEGIT": 1})
        self.assertEqual(cov["with_verdict"], 4)
        lf = s["likely_fake_misleading"]
        self.assertEqual((lf["count"], lf["denominator"], lf["fake"], lf["misleading"]), (3, 4, 2, 1))
        self.assertAlmostEqual(lf["share"], 0.75)
        # p_prop buckets: only analysed items with p_prop
        self.assertEqual(s["p_prop_distribution"]["n"], 8)
        self.assertEqual(sum(b["count"] for b in s["p_prop_distribution"]["buckets"]), 8)
        self.assertEqual(s["p_prop_distribution"]["buckets"][9]["count"], 2)  # 0.9 and 0.95
        mix = {t["technique"]: t for t in s["technique_mix"]["top"]}
        self.assertEqual((mix["loaded_language"]["items"], mix["loaded_language"]["mentions"]), (3, 6))
        self.assertEqual(s["technique_mix"]["denominator_analysed"], 8)

    def test_scope_filters(self):
        default_export(self.export)
        s = self.api.stats(self.q(kind="news"))
        self.assertEqual(s["total"], 6)
        self.assertEqual(s["coverage"]["analysed"], 4)
        self.assertEqual(s["likely_fake_misleading"]["denominator"], 3)
        s = self.api.stats(self.q(platform="telegram"))
        self.assertEqual(s["total"], 2)
        s = self.api.stats(self.q(days=1))  # items newer than 24 h
        self.assertEqual(s["total"], len(self.sql("SELECT 1 FROM items WHERE published_at >= ?", iso(days=1))))
        self.assertEqual(sum(d["total"] for d in s["by_day"]), s["total"])
        for bad in ({"days": "abc"}, {"days": "9999"}, {"kind": "x"}, {"platform": "a;b"}):
            with self.assertRaises(rt_api.ApiError) as cm:
                self.api.stats(self.q(**bad))
            self.assertEqual(cm.exception.code, 400)

    def test_freshness_and_strict_json(self):
        default_export(self.export)
        s = self.api.stats(self.q())
        self.assertEqual(s["freshness"]["newest_item_at"], iso(hours=1))
        self.assertEqual(s["freshness"]["lag_minutes"], 60.0)
        self.assertEqual(s["freshness"]["synced_at"], iso(minutes=5))
        json.dumps(s, allow_nan=False)

    def test_store_without_the_stats_index_still_works(self):
        default_export(self.export)
        before = self.api.stats(self.q(days=2))
        self.api._cache.clear()
        con = sqlite3.connect(self.db)
        con.execute("DROP INDEX items_stats")
        con.commit()
        con.close()
        after = self.api.stats(self.q(days=2))
        for k in ("total", "by_kind", "by_source", "by_topic", "coverage", "likely_fake_misleading", "by_day"):
            self.assertEqual(before[k], after[k], k)

    def test_status_block(self):
        default_export(self.export)
        st = self.api.status()
        self.assertEqual(st["total_rows"], 12)
        self.assertEqual(st["enrichment"]["records"], 9)
        self.assertEqual(st["policy"]["news_days"], 7)
        self.assertEqual(st["policy"]["social_limit_per_table"], 100000)
        self.assertEqual(st["recent_errors"][0]["error"], "RuntimeError: boom")


class Items(Fixture):
    def ids(self, **kw):
        return [i["id"] for i in self.api.items(self.q(**kw))["items"]]

    def test_default_recent_and_total(self):
        r = self.api.items(self.q())
        self.assertEqual(r["total"], 12)
        self.assertEqual(r["sort"], "recent")
        times = [i["published_at"] for i in r["items"]]
        self.assertEqual(times, sorted(times, reverse=True))
        first = r["items"][0]
        self.assertIn("snippet", first)
        self.assertNotIn("text_excerpt", first)
        self.assertIsNone(first["inferred"])
        self.assertEqual(first["kind"] in ("news", "social"), True)

    def test_pagination_and_size_cap(self):
        a = self.api.items(self.q(size=5, page=1))
        b = self.api.items(self.q(size=5, page=2))
        c = self.api.items(self.q(size=5, page=3))
        self.assertEqual((len(a["items"]), len(b["items"]), len(c["items"]), a["pages"]), (5, 5, 2, 3))
        self.assertEqual(len({i["id"] for i in a["items"] + b["items"] + c["items"]}), 12)
        self.assertEqual(len(self.api.items(self.q(size=50))["items"]), 12)
        for bad in ({"size": 51}, {"size": 0}, {"page": 0}, {"size": "x"}, {"page": 99999}):
            with self.assertRaises(rt_api.ApiError):
                self.api.items(self.q(**bad))

    def test_filters(self):
        self.assertEqual(self.api.items(self.q(kind="news"))["total"], 6)
        self.assertEqual(self.api.items(self.q(platform="telegram"))["total"], 2)
        self.assertEqual(self.api.items(self.q(source="a.com"))["total"], 3)
        self.assertEqual(self.api.items(self.q(topic="D01.T01"))["total"], 2)
        self.assertEqual(self.api.items(self.q(topic="unclassified"))["total"], 9)
        self.assertEqual(self.api.items(self.q(lang="hi"))["total"], 2)
        r = self.api.items(self.q(source="a.com", kind="news", lang="hi"))
        self.assertEqual([i["id"] for i in r["items"]], ["news:scraped_articles:3"])
        day = NOW.strftime("%Y-%m-%d")
        self.assertEqual(self.api.items(self.q(**{"from": day}))["total"],
                         len(self.sql("SELECT 1 FROM items WHERE published_at >= ?", day + "T00:00:00Z")))
        self.assertEqual(self.api.items(self.q(**{"to": "2000-01-01"}))["total"], 0)
        full = self.api.items(self.q(**{"from": iso(hours=3), "to": iso(hours=1)}))
        self.assertEqual(full["total"], len(self.sql("SELECT 1 FROM items WHERE published_at BETWEEN ? AND ?",
                                                     iso(hours=3), iso(hours=1))))

    def test_item_shape_topic_database_fields(self):
        r = self.api.items(self.q(topic="D02.T01"))
        it = r["items"][0]
        self.assertEqual(it["topic"], {"id": "D02.T01", "name": "Economy"})
        self.assertEqual((it["platform"], it["source"], it["lang"], it["source_table"]), ("news", "b.com", "en", "scraped_articles"))
        self.assertIsNone(self.api.items(self.q(kind="social"))["items"][0]["topic"])

    def test_analysed_filter_and_inferred_block(self):
        default_export(self.export)
        yes = self.api.items(self.q(analysed="yes", size=50))
        self.assertEqual(yes["total"], 8)
        no = self.api.items(self.q(analysed="no", size=50))
        self.assertEqual(no["total"], 4)
        self.assertTrue(all(i["inferred"] is None for i in no["items"]))
        self.assertTrue(all(i["inferred"] is not None for i in yes["items"]))
        one = {i["id"]: i for i in yes["items"]}["news:scraped_articles:1"]
        inf = one["inferred"]
        self.assertEqual((inf["p_prop"], inf["p_fake"], inf["verdict"], inf["source"]), (0.9, 0.7, "FAKE", "batch"))
        self.assertEqual((inf["sources"], inf["export_source"], inf["deep"]), (["batch"], "realtime_export", None))
        self.assertEqual({t["technique"]: t["count"] for t in inf["techniques"]}, {"loaded_language": 3, "repetition": 1})
        # database-only fields never carry FNP data
        self.assertNotIn("p_fake", one)
        self.assertEqual(yes["total"] + no["total"], 12)

    def test_sort_by_p_fake_and_p_prop(self):
        default_export(self.export)
        r = self.api.items(self.q(sort="p_fake", size=50))
        pf = [i["inferred"]["p_fake"] if i["inferred"] else None for i in r["items"]]
        vals = [v for v in pf if v is not None]
        self.assertEqual(vals, sorted(vals, reverse=True))
        self.assertEqual(pf[:len(vals)], vals)  # items without p_fake always come last
        self.assertEqual(r["items"][0]["id"], "reddit_posts:10")
        r = self.api.items(self.q(sort="p_prop", size=50))
        self.assertEqual(r["items"][0]["id"], "reddit_posts:10")  # 0.95
        self.assertEqual(r["items"][1]["id"], "news:scraped_articles:1")  # 0.9

    def test_full_text_search(self):
        self.assertEqual(self.api.items(self.q(q="river"))["total"], 3)
        r = self.api.items(self.q(q="flood"))
        self.assertEqual({i["id"] for i in r["items"]}, {"news:scraped_articles:1", "news:scraped_articles:2", "reddit_posts:11"})
        self.assertEqual(r["sort"], "relevance")
        self.assertEqual(self.api.items(self.q(q="flo"))["total"], 3)  # last word is a prefix
        self.assertEqual(self.api.items(self.q(q="river", kind="social"))["total"], 1)
        self.assertEqual(self.api.items(self.q(q="river", sort="recent", kind="news"))["total"], 2)
        # FTS operators are plain words, never syntax
        for evil in ('river OR 1=1', 'river NOT flood', 'NEAR(river valley, 2)', '"river', 'river*', '* OR *', 'title:river',
                     "river'; DROP TABLE items; --", 'AND', '(river'):
            r = self.api.items(self.q(q=evil))
            self.assertIsInstance(r["total"], int, evil)
        self.assertEqual(self.api.items(self.q(q="river NOT flood"))["total"], 0)  # NOT is just a word
        with self.assertRaises(rt_api.ApiError):
            self.api.items(self.q(q="*** ((("))
        with self.assertRaises(rt_api.ApiError):
            self.api.items(self.q(q="x" * 301))
        with self.assertRaises(rt_api.ApiError):
            self.api.items(self.q(sort="relevance"))
        self.assertEqual(len(self.sql("SELECT 1 FROM items")), 12)

    def test_fulltext_total_is_capped_but_plain_totals_are_exact(self):
        with mock.patch.object(rt_api, "COUNT_CAP", 2):
            r = self.api.items(self.q(q="river"))  # 3 matches > cap
            self.assertEqual((r["total"], r["total_capped"]), (2, True))
            self.assertEqual(self.api.items(self.q(q="vaccine"))["total_capped"], False)
            r = self.api.items(self.q())
            self.assertEqual((r["total"], r["total_capped"]), (12, False))

    def test_injection_and_validation(self):
        inj = "' OR '1'='1"
        self.assertEqual(self.api.items(self.q(source=inj))["total"], 0)
        self.assertEqual(self.api.items(self.q(source="a.com' --"))["total"], 0)
        for bad in ({"kind": "news;DROP TABLE items"}, {"kind": "all"}, {"platform": "x' OR 1=1"}, {"lang": "en' OR '1'='1"},
                    {"topic": "x' OR 1=1 --"}, {"analysed": "maybe"}, {"sort": "published_at; DROP"},
                    {"from": "yesterday"}, {"to": "2026-13-45"}, {"from": "2026-11-06 12:00:00"}, {"source": "a\x00b"}):
            with self.assertRaises(rt_api.ApiError, msg=str(bad)) as cm:
                self.api.items(self.q(**bad))
            self.assertEqual(cm.exception.code, 400)
        self.assertEqual(len(self.sql("SELECT 1 FROM items")), 12)  # still intact
        self.assertEqual(self.api.items(self.q(topic="UNCLASSIFIED"))["total"], 9)

    def test_empty_store_and_missing_store(self):
        os.unlink(self.db)
        with self.assertRaises(rt_api.ApiError) as cm:
            self.api.items(self.q())
        self.assertEqual(cm.exception.code, 503)


class Detail(Fixture):
    def test_full_row_inferred_and_analysis(self):
        default_export(self.export)
        self.matches = [dict(id="fc1", title="Fact-check", similarity=0.71)]
        d = self.api.item("news:scraped_articles:1")
        self.assertEqual(d["title"], "Flood alert for the river valley")
        self.assertEqual(d["text_excerpt"], NEAR_TEXT)
        self.assertEqual(d["topic"]["name"], "Politics")
        inf = d["inferred"]
        self.assertEqual(inf["nearest_factcheck"], {"id": "kb2:af-1", "similarity": 0.71})
        self.assertEqual(inf["countries"], {"about": ["IN"], "origin": None})
        self.assertEqual(inf["spans"][0]["text"], "river valley")
        self.assertEqual(inf["model_version"], "fnp-test-v1")
        self.assertEqual(d["near_duplicates"]["count"], 1)
        self.assertEqual(d["near_duplicates"]["sample"][0]["id"], "news:scraped_articles:2")
        self.assertEqual(d["near_duplicates"]["sample"][0]["hamming"], 0)
        self.assertEqual(d["nearest_factchecks"]["matches"], self.matches)
        self.assertEqual(d["nearest_factchecks"]["nearest_similarity"], 0.71)
        self.assertEqual(d["analyses"][0]["engine_version"], "check_v1")
        self.assertEqual(d["analyses"][0]["result"], {"final": "FAKE"})
        json.dumps(d, allow_nan=False)

    def test_missing_inferred_is_null_not_a_guess(self):
        d = self.api.item("telegram_posts:13")
        self.assertIsNone(d["inferred"])
        self.assertEqual(d["analyses"], [])
        self.assertEqual(d["near_duplicates"]["count"], None)  # short text: no fingerprint
        default_export(self.export)
        self.assertIsNone(self.api.item("telegram_posts:13")["inferred"])  # its only record line was invalid

    def test_no_factcheck_backend(self):
        api = rt_api.RtApi(db_path=self.db, enrich=self.enrich, clock=lambda: NOW)
        self.assertEqual(api.item("reddit_posts:10")["nearest_factchecks"], {"available": False, "matches": []})
        api = rt_api.RtApi(db_path=self.db, enrich=self.enrich, clock=lambda: NOW,
                           fn_matches=mock.Mock(side_effect=RuntimeError("index down")))
        self.assertFalse(api.item("reddit_posts:10")["nearest_factchecks"]["available"])

    def test_unknown_and_malformed_ids_404(self):
        for bad in ("nope:1", "news:scraped_articles:424242", "x' OR 1=1", "", "a" * 300, "../etc/passwd"):
            with self.assertRaises(rt_api.ApiError, msg=bad) as cm:
                self.api.item(bad)
            self.assertEqual(cm.exception.code, 404)

    def test_near_dup_scan_agrees_with_simhash_index(self):
        con = rt_store.open_store(self.db, create=False)
        try:
            ref = rt_store.near_duplicates(con, "news:scraped_articles:1", 3, path_key="t")
            mine = rt_api.RtApi._near_dups(con, "news:scraped_articles:1", to_unsigned(
                con.execute("SELECT simhash FROM items WHERE id='news:scraped_articles:1'").fetchone()[0]))
        finally:
            con.close()
        self.assertEqual([(s["id"], s["hamming"]) for s in mine["sample"]], ref)
        self.assertEqual(mine["count"], len(ref))

    def test_near_dups_find_close_hashes_only(self):
        con = rt_store.open_store(self.db)
        base = to_unsigned(con.execute("SELECT simhash FROM items WHERE id='news:scraped_articles:1'").fetchone()[0])
        flip = lambda h, bits: h ^ sum(1 << b for b in bits)
        rt_store.upsert_items(con, [
            item(20, "social", "reddit", simhash=flip(base, [0, 17, 33]), text="c1"),    # distance 3: counts
            item(21, "social", "reddit", simhash=flip(base, [1, 20, 40, 63]), text="c2"),  # distance 4: does not
            item(22, "social", "reddit", simhash=flip(base, [60, 61, 62]), text="c3")])  # distance 3 in the top band
        con.close()
        d = self.api.item("news:scraped_articles:1")["near_duplicates"]
        self.assertEqual(d["count"], 3)  # item 2 (0) + 20 + 22
        self.assertEqual({s["id"] for s in d["sample"]}, {"news:scraped_articles:2", "reddit_posts:20", "reddit_posts:22"})


class Enrichment(Fixture):
    def test_absent_dirs_are_normal(self):
        e = self.enrich
        self.assertIsNone(e.lookup("news:scraped_articles:1"))
        st = e.status()
        self.assertFalse(st["present"])
        self.assertEqual(st["coverage"]["analysed"], 0)
        self.assertIsNone(st["error"])

    def test_join_validation_and_coverage(self):
        default_export(self.export)
        st = self.enrich.status()
        self.assertTrue(st["present"])
        self.assertEqual(st["built_at"], "2026-11-06T09:00:00Z")
        self.assertEqual(st["model_version"], "fnp-test-v1")
        self.assertEqual(st["records"], 5 + 4)  # 5 news, 4 social; the not-json and NaN lines are dropped
        self.assertEqual(st["invalid_lines"], 2)  # not json, NaN
        cov = st["coverage"]
        self.assertEqual(cov["by_kind"]["news"]["analysed"], 5)
        self.assertEqual(cov["by_kind"]["news"]["verdicts"], {"FAKE": 1, "LEGIT": 1, "MISLEADING": 1})
        self.assertEqual(cov["by_kind"]["social"]["analysed"], 4)
        self.assertEqual(cov["verdicts"], {"FAKE": 2, "MISLEADING": 1, "LEGIT": 1})
        # an id-less record joins on source_table + source_id and keeps a null verdict
        inf = self.enrich.lookup("telegram_posts:12")
        self.assertEqual((inf["p_prop"], inf["verdict"], inf["p_fake"]), (0.7, None, None))
        self.assertIsNone(self.enrich.lookup("telegram_posts:404"))

    def test_verdict_must_match_p_fake(self):
        bad = rec("reddit_posts:10", p_prop=0.5, p_fake=None, verdict="FAKE")
        bad2 = rec("reddit_posts:11", p_prop=1.5)
        bad3 = rec("reddit_posts:12", p_fake=0.5, verdict=None)
        ok = rec("reddit_posts:13", p_prop=0.5)
        write_export(self.export, {"social/p.jsonl": [bad, bad2, bad3, ok]})
        st = self.enrich.status()
        self.assertEqual((st["records"], st["invalid_lines"]), (1, 3))

    def test_latest_built_at_wins_ties_to_later_file(self):
        a = rec("reddit_posts:10", p_prop=0.1, built="2026-11-06T08:00:00Z", techs=[("old", 1, 0.1)])
        b = rec("reddit_posts:10", p_prop=0.9, built="2026-11-06T10:00:00Z", techs=[("new", 2, 0.9)])
        c = rec("reddit_posts:11", p_prop=0.1, built="2026-11-06T09:00:00Z")
        d = rec("reddit_posts:11", p_prop=0.2, built="2026-11-06T09:00:00Z")
        write_export(self.export, {"social/a.jsonl": [b, c], "social/b.jsonl": [a, d]})
        self.assertEqual(self.enrich.lookup("reddit_posts:10")["p_prop"], 0.9)
        self.assertEqual([t["technique"] for t in self.enrich.lookup("reddit_posts:10")["techniques"]], ["new"])
        self.assertEqual(self.enrich.lookup("reddit_posts:11")["p_prop"], 0.2)

    def test_reload_when_built_at_changes_only(self):
        write_export(self.export, {"social/p1.jsonl": [rec("reddit_posts:10", p_prop=0.1)]}, built_at="2026-11-06T09:00:00Z")
        self.assertEqual(self.enrich.lookup("reddit_posts:10")["p_prop"], 0.1)
        loaded = self.enrich.status()["loaded_at"]
        # rewrite the data file but keep built_at: not reloaded (the contract: built_at is the trigger)
        write_export(self.export, {"social/p1.jsonl": [rec("reddit_posts:10", p_prop=0.5)]}, built_at="2026-11-06T09:00:00Z")
        self.assertEqual(self.enrich.lookup("reddit_posts:10")["p_prop"], 0.1)
        self.assertEqual(self.enrich.status()["loaded_at"], loaded)
        write_export(self.export, {"social/p1.jsonl": [rec("reddit_posts:10", p_prop=0.5), rec("reddit_posts:11", p_prop=0.3)]},
                     built_at="2026-11-06T10:00:00Z")
        self.assertEqual(self.enrich.lookup("reddit_posts:10")["p_prop"], 0.5)
        self.assertEqual(self.enrich.lookup("reddit_posts:11")["p_prop"], 0.3)
        self.assertEqual(self.enrich.status()["built_at"], "2026-11-06T10:00:00Z")

    def test_bad_builds_keep_the_previous_one(self):
        write_export(self.export, {"social/p1.jsonl": [rec("reddit_posts:10", p_prop=0.1)]}, built_at="2026-11-06T09:00:00Z")
        self.assertEqual(self.enrich.lookup("reddit_posts:10")["p_prop"], 0.1)
        # sha256 mismatch
        write_export(self.export, {"social/p1.jsonl": [rec("reddit_posts:10", p_prop=0.9)]}, built_at="2026-11-06T10:00:00Z",
                     sha_ok=False)
        self.assertEqual(self.enrich.lookup("reddit_posts:10")["p_prop"], 0.1)
        self.assertIn("sha256", self.enrich.status()["error"])
        self.assertEqual(self.enrich.status()["built_at"], "2026-11-06T09:00:00Z")
        # unsafe / missing paths
        for rel in ("../outside.jsonl", "/etc/passwd", "social/missing.jsonl", "social/x.txt"):
            write_export(self.export, {"social/p1.jsonl": [rec("reddit_posts:10", p_prop=0.9)]},
                         built_at="2026-11-06T1%d:00:00Z" % (1 + len(rel) % 8))
            mp = os.path.join(self.export, "meta.json")
            with open(mp) as fh:
                m = json.load(fh)
            m["files"][0]["path"] = rel
            m["built_at"] = "2026-11-07T%02d:00:00Z" % (len(rel) % 24)
            with open(mp, "w") as fh:
                json.dump(m, fh)
            self.assertEqual(self.enrich.lookup("reddit_posts:10")["p_prop"], 0.1, rel)
            self.assertIsNotNone(self.enrich.status()["error"], rel)
        # unreadable meta
        with open(os.path.join(self.export, "meta.json"), "w") as f:
            f.write("{ nope")
        self.assertEqual(self.enrich.lookup("reddit_posts:10")["p_prop"], 0.1)

    def test_tmp_files_are_never_read(self):
        write_export(self.export, {"social/p1.jsonl": [rec("reddit_posts:10", p_prop=0.1)]})
        with open(os.path.join(self.export, "social", "p2.jsonl.tmp"), "w") as f:
            f.write(json.dumps(rec("reddit_posts:11", p_prop=0.9)) + "\n")
        self.assertIsNone(self.enrich.lookup("reddit_posts:11"))

    def test_sidecar_survives_restart_without_rebuild(self):
        default_export(self.export)
        self.enrich.status()
        loaded = self.enrich.status()["loaded_at"]
        e2 = rt_enrich.Enrich(export_dir=self.export, triage_dir=self.triage, sidecar=self.enrich.sidecar, store_path=self.db)
        self.assertEqual(e2.status()["loaded_at"], loaded)
        self.assertEqual(e2.status()["records"], 9)
        self.assertEqual(oct(os.stat(self.enrich.sidecar).st_mode & 0o777), "0o600")

    def test_triage_join_on_url_only(self):
        os.makedirs(self.triage)
        url = self.sql("SELECT url FROM items WHERE id='reddit_posts:11'")[0][0]
        lines = [
            dict(post_uid="a" * 40, source="social", platform="reddit", url=url, p_prop=0.6, p_fake=0.4, p_misleading=0.3,
                 verdict=dict(label="MISLEADING", basis="stacker"), techniques=["Loaded_Language"],
                 spans=[[0, 4, "Loaded_Language", 0.8], [5, 9, "Loaded_Language", 0.9]], countries=["IND"],
                 nearest_factcheck=dict(kb_id="kb:1", cos=0.7), tier="high", model_version="triage-v1",
                 built_at="2026-10-27T00:00:00Z"),
            dict(post_uid="b" * 40, source="social", platform="reddit", url="https://elsewhere.example/none", p_prop=0.9,
                 p_fake=None, p_misleading=None, verdict=None, techniques=[], countries=[], nearest_factcheck=None),
            dict(post_uid="c" * 40, source="social", platform="reddit", url=None)]
        with open(os.path.join(self.triage, "items.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(l) for l in lines) + "\n")
        with open(os.path.join(self.triage, "meta.json"), "w") as f:
            json.dump(dict(version="triage_v1", built_at="2026-10-27T00:00:00Z"), f)
        inf = self.enrich.lookup("reddit_posts:11")
        self.assertEqual((inf["source"], inf["verdict"], inf["p_fake"], inf["p_prop"]), ("triage", "MISLEADING", 0.4, 0.6))
        self.assertEqual(inf["techniques"], [dict(technique="Loaded_Language", count=2, max_confidence=0.9)])
        self.assertEqual(inf["countries"]["about"], ["IND"])
        self.assertEqual(inf["countries"]["format"], "iso3")
        self.assertEqual(inf["nearest_factcheck"], {"id": "kb:1", "similarity": 0.7})
        st = self.enrich.status()
        self.assertEqual(st["records"], 1)
        self.assertEqual(st["invalid_lines"], 1)
        # a realtime-export record for the same item wins over triage
        write_export(self.export, {"social/p.jsonl": [rec("reddit_posts:11", p_prop=0.25)]}, built_at="2026-11-06T09:00:00Z")
        self.assertEqual(self.enrich.lookup("reddit_posts:11")["source"], "realtime_export")


class SyncPolicy(unittest.TestCase):
    def run_sync(self, pg, env=None, **kw):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = os.path.join(tmp, "r.sqlite")
        with mock.patch.dict(os.environ, env or {}):
            rep = rs.sync_once(db_path=db, connect=pg.connect, clock=lambda: NOW, **kw)
        con = rt_store.open_store(db, create=False)
        self.addCleanup(con.close)
        return rep, {r[0] for r in con.execute("SELECT id FROM items")}

    def pg(self):
        return FakePG(
            social={"reddit_posts": [social_row(1, hours_ago=24 * 20), social_row(2, hours_ago=24 * 40), social_row(3, hours_ago=1)]},
            news={"scraped_articles": [news_row(1, hours_ago=24 * 3), news_row(2, hours_ago=24 * 10), news_row(3, hours_ago=2)]},
            content={"scraped_articles": {1: dict(content="body one", lang=None), 2: dict(content="body two", lang=None),
                                          3: dict(content="body three", lang=None)}})

    def test_default_windows_news_7_social_30(self):
        rep, ids = self.run_sync(self.pg(), sources="reddit_posts,scraped_articles")
        self.assertEqual(rep["window_days"], {"news": 7, "social": 30})
        self.assertEqual(rep["per_source_cap"], {"news": 400000, "social": 100000})
        self.assertEqual(ids, {"reddit_posts:1", "reddit_posts:3", "news:scraped_articles:1", "news:scraped_articles:3"})

    def test_env_overrides(self):
        rep, ids = self.run_sync(self.pg(), env={"FNP_RT_NEWS_DAYS": "14", "FNP_RT_SOCIAL_DAYS": "2",
                                                 "FNP_RT_NEWS_LIMIT": "2", "FNP_RT_SOCIAL_LIMIT": "5"},
                                 sources="reddit_posts,scraped_articles")
        self.assertEqual(rep["window_days"], {"news": 14.0, "social": 2.0})
        self.assertEqual(rep["per_source_cap"], {"news": 2, "social": 5})
        self.assertEqual(ids, {"reddit_posts:3", "news:scraped_articles:3", "news:scraped_articles:1"})  # news capped at 2 newest
        self.assertTrue(rep["sources"]["news:scraped_articles"]["truncated"])

    def test_cli_days_and_limit_override_both(self):
        rep, ids = self.run_sync(self.pg(), sources="reddit_posts,scraped_articles", days=3, limit=1)
        self.assertEqual(rep["window_days"], {"news": 3.0, "social": 3.0})
        self.assertEqual(ids, {"reddit_posts:3", "news:scraped_articles:3"})

    def test_prune_uses_the_window_of_each_kind(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = os.path.join(tmp, "r.sqlite")
        con = rt_store.open_store(db)
        rt_store.upsert_items(con, [item(1, "news", "news", hours=24 * 10), item(2, "social", "reddit", hours=24 * 10)])
        con.close()
        rep = rs.sync_once(db_path=db, connect=FakePG(social={"reddit_posts": []}, news={"scraped_articles": []}).connect,
                           clock=lambda: NOW, prune=True, sources="reddit_posts,scraped_articles")
        self.assertEqual(rep["pruned"], 1)
        con = rt_store.open_store(db, create=False)
        self.assertEqual([r[0] for r in con.execute("SELECT id FROM items")], ["reddit_posts:2"])
        con.close()

    def test_topic_refresh_budget_default(self):
        self.assertEqual(rs.DEFAULTS["refresh_budget"], 5000)

    def test_topic_refresh_fills_late_classification(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = os.path.join(tmp, "r.sqlite")
        pg = FakePG(news={"scraped_articles": [news_row(1, minutes_ago=30), news_row(2, minutes_ago=40)]},
                    content={"scraped_articles": {1: dict(content="b1", lang=None), 2: dict(content="b2", lang=None)}},
                    taxonomy={"D01.T01": "Politics"})
        rs.sync_once(db_path=db, connect=pg.connect, clock=lambda: NOW, sources="scraped_articles")
        pg.news["scraped_articles"][0].update(topic_id="D01.T01", category_id=None)
        rs.sync_once(db_path=db, connect=pg.connect, clock=lambda: NOW, sources="scraped_articles")
        con = rt_store.open_store(db, create=False)
        got = {r[0]: (r[1], r[2]) for r in con.execute("SELECT id, db_topic_id, topic_name FROM items")}
        con.close()
        self.assertEqual(got["news:scraped_articles:1"], ("D01.T01", "Politics"))
        self.assertEqual(got["news:scraped_articles:2"], (None, None))
        self.assertTrue(all(sql.lstrip().upper().startswith(("SELECT", "WITH")) for _, sql, _p in pg.statements))


# ---------------------------------------------------------------------------------------------- HTTP: gate, rate limit
class Http(Fixture):
    def setUp(self):
        super().setUp()
        default_export(self.export)
        self._saved = {k: getattr(server, k) for k in ("RT", "AUTH", "RT_RATE")}
        server.RT = self.api
        server.AUTH = None
        server._rt_hits.clear()
        server.H.log_message = lambda *a, **k: None
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.t = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.t.start()
        self._so, self._se = sys.stdout, sys.stderr
        self.addCleanup(self.close)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.t.join(5)
        for k, v in self._saved.items():
            setattr(server, k, v)
        server._rt_hits.clear()
        server._auth_fails.clear()

    def get(self, path, headers=None):
        return jcall(self.port, "GET", path, headers=headers)

    @staticmethod
    def basic(user, pw):
        return {"Authorization": "Basic " + base64.b64encode(("%s:%s" % (user, pw)).encode()).decode()}

    ROUTES = ("/api/rt/status", "/api/rt/stats", "/api/rt/items", "/api/rt/item/reddit_posts:10", "/api/rt/nope")

    @needs_index

    def test_closed_by_default_every_route_403(self):
        for path in self.ROUTES:
            st, hd, b = self.get(path)
            self.assertEqual(st, 403, path)
            self.assertEqual(b, {"error": "login_required", "hint": "Set FNP_AUTH_USER and FNP_AUTH_PASS to use the Evidence Base"})
            self.assertEqual(hd["cache-control"], "no-store")
        self.assertEqual(self.get("/api/rt/gate")[2], {"open": False, "auth": False})
        self.assertEqual(self.get("/healthz")[0], 200)

    def test_open_flag_allows(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            self.assertEqual(self.get("/api/rt/gate")[2], {"open": True, "auth": False})
            for path in self.ROUTES[:4]:
                st, hd, b = self.get(path)
                self.assertEqual(st, 200, path)
                self.assertEqual(hd["cache-control"], "no-store")
            self.assertEqual(self.get("/api/rt/nope")[0], 404)
        for val in ("0", "true", "yes", ""):
            with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": val}):
                self.assertEqual(self.get("/api/rt/items")[0], 403, val)

    def test_auth_configured_requires_credentials_then_serves(self):
        server.AUTH = (b"alice", b"pw-correct")
        st, hd, b = self.get("/api/rt/items")
        self.assertEqual(st, 401)  # the site-wide login comes first
        self.assertIn("www-authenticate", hd)
        self.assertEqual(self.get("/api/rt/items", self.basic("alice", "wrong"))[0], 401)
        st, hd, b = self.get("/api/rt/items?size=2", self.basic("alice", "pw-correct"))
        self.assertEqual((st, len(b["items"])), (200, 2))
        self.assertEqual(self.get("/api/rt/gate", self.basic("alice", "pw-correct"))[2], {"open": False, "auth": True})
        self.assertEqual(self.get("/api/rt/gate")[0], 401)

    def test_error_bodies_and_statuses(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            for path, code in (("/api/rt/items?size=500", 400), ("/api/rt/items?kind=x", 400), ("/api/rt/stats?days=-1", 400),
                               ("/api/rt/item/news:scraped_articles:424242", 404), ("/api/rt/item/", 404),
                               ("/api/rt/item/x%27%20OR%201%3D1", 404)):
                st, hd, b = self.get(path)
                self.assertEqual(st, code, path)
                self.assertEqual(list(b), ["error"], path)

    def test_every_open_response_is_strict_json(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            for path in ("/api/rt/status", "/api/rt/stats", "/api/rt/stats?kind=news&days=7", "/api/rt/items?analysed=yes",
                         "/api/rt/items?q=river", "/api/rt/item/news:scraped_articles:1"):
                st, hd, raw = call(self.port, "GET", path)
                self.assertEqual(st, 200, path)
                self.assertTrue(hd["content-type"].startswith("application/json"))
                json.loads(raw.decode(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))

    @needs_index

    def test_rate_limit_429(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}), mock.patch.object(server, "RT_RATE", 5):
            codes = [self.get("/api/rt/status")[0] for _ in range(7)]
            self.assertEqual(codes, [200] * 5 + [429] * 2)
            st, hd, b = self.get("/api/rt/items")
            self.assertEqual(st, 429)
            self.assertIn("retry-after", hd)
        self.assertEqual(server.RT_RATE, 120)
        # a closed gate does not spend the budget, and other routes are not limited
        server._rt_hits.clear()
        with mock.patch.object(server, "RT_RATE", 2):
            for _ in range(5):
                self.assertEqual(self.get("/api/rt/items")[0], 403)
            self.assertEqual(self.get("/healthz")[0], 200)

    @needs_index

    def test_healthz_realtime_block_with_sync_disabled(self):
        st, hd, b = self.get("/healthz")
        rt = b["realtime"]
        self.assertEqual(set(rt), {"enabled", "running", "last_sync", "lag_minutes", "rows", "errors", "analysis"})
        self.assertIs(rt["enabled"], False)
        self.assertIs(rt["running"], False)
        self.assertEqual(rt["rows"], 12)
        self.assertEqual(rt["errors"], 1)
        self.assertEqual(rt["last_sync"], iso(minutes=5))
        self.assertIsInstance(rt["lag_minutes"], float)

    @needs_index

    def test_healthz_survives_a_broken_realtime_block(self):
        broken = mock.Mock()
        broken.health.side_effect = RuntimeError("store exploded")
        server.RT = broken
        st, hd, b = self.get("/healthz")
        self.assertEqual(st, 200)
        self.assertIs(b["ok"], True)
        self.assertEqual(b["realtime"]["enabled"], False)
        os.unlink(self.db)  # a missing store also never fails healthz
        server.RT = self.api
        self.api._cache.clear()
        st, hd, b = self.get("/healthz")
        self.assertEqual((st, b["realtime"]["rows"]), (200, 0))


class Startup(unittest.TestCase):
    def test_sync_thread_only_when_enabled(self):
        with mock.patch.object(rs, "start_background") as sb:
            self.assertFalse(server.maybe_start_sync({}))
            self.assertFalse(server.maybe_start_sync({"FNP_RT_SYNC": "0"}))
            self.assertFalse(server.maybe_start_sync({"FNP_RT_SYNC": "true"}))
            sb.assert_not_called()
            self.assertTrue(server.maybe_start_sync({"FNP_RT_SYNC": "1"}))
            sb.assert_called_once_with(interval_s=300.0)
            sb.reset_mock()
            self.assertTrue(server.maybe_start_sync({"FNP_RT_SYNC": "1", "FNP_RT_INTERVAL": "600"}))
            sb.assert_called_once_with(interval_s=600.0)
            sb.reset_mock()
            server.maybe_start_sync({"FNP_RT_SYNC": "1", "FNP_RT_INTERVAL": "junk"})
            sb.assert_called_once_with(interval_s=300.0)

    def test_background_stops_cleanly(self):
        calls = []
        with mock.patch.object(rs, "sync_once", side_effect=lambda **kw: calls.append(1) or dict(rows_total=0)):
            bg = rs.start_background(interval_s=1)
            time.sleep(0.2)
            self.assertTrue(rs.background_status()["running"])
            rs.stop_background(timeout=5)
            self.assertFalse(bg.thread.is_alive())
            self.assertEqual(rs.background_status(), {"running": False})
        self.assertGreaterEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
