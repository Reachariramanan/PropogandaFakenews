"""S4: source freshness rows in /api/rt/status, the /api/rt/sources type-ahead, and script-based language repair in build_index.
Offline: temp store only (the fixtures of test_rt_api), synthetic strings."""
import os
import unittest
from unittest import mock

import helpers  # noqa: F401
import build_index as bi
import rt_api
import rt_store
import test_rt_api as T  # module import only: its TestCase classes are not re-collected here


class SourceFreshnessUnit(unittest.TestCase):
    NOW = T.NOW

    def st(self, rows, since=5.0, errors=0, total=None):
        last = "2026-11-06T10:00:00Z"
        return dict(total_rows=total if total is not None else sum(r["rows"] for r in rows), per_source=rows,
                    minutes_since_last_ok_sync=since, last_sync_at=last,
                    recent_errors=[dict(ts=last, source_key="x", error="e")] * errors)

    def row(self, table, kind, platform, n, hours):
        return dict(source_key=table, kind=kind, platform=platform, rows=n,
                    newest_published_at=rt_store.utc_iso(self.NOW - T.timedelta(hours=hours)))

    POL = dict(news_limit_per_source=400000, social_limit_per_table=100)

    def test_flags_and_share(self):
        rows = [self.row("instagram_posts", "social", "instagram", 100, 1), self.row("reddit_posts", "social", "reddit", 50, 100),
                self.row("scraped_articles", "news", "news", 50, 47)]
        out = rt_api.source_freshness(self.st(rows), self.POL, self.NOW)
        by = {r["source_table"]: r for r in out["rows"]}
        self.assertTrue(out["sync_healthy"])
        self.assertTrue(by["instagram_posts"]["capped"] and not by["instagram_posts"]["stale"])
        self.assertEqual(by["instagram_posts"]["cap"], 100)
        self.assertTrue(by["reddit_posts"]["stale"] and not by["reddit_posts"]["capped"])
        self.assertFalse(by["scraped_articles"]["stale"])  # 47 h < 48 h
        self.assertFalse(by["scraped_articles"]["capped"])  # 50 rows vs the news cap
        self.assertAlmostEqual(by["reddit_posts"]["share"], 0.25)
        self.assertEqual(by["reddit_posts"]["age_hours"], 100.0)

    def test_unhealthy_sync_never_marks_stale(self):
        rows = [self.row("reddit_posts", "social", "reddit", 5, 100)]
        for kw in (dict(since=45.0), dict(errors=1), dict(since=None)):
            out = rt_api.source_freshness(self.st(rows, **kw), self.POL, self.NOW)
            self.assertFalse(out["sync_healthy"])
            self.assertFalse(out["rows"][0]["stale"])

    def test_empty_store(self):
        out = rt_api.source_freshness(dict(total_rows=0, per_source=[]), self.POL, self.NOW)
        self.assertEqual(out["rows"], [])


class StatusAndSources(T.Fixture):
    def setUp(self):
        super().setUp()
        con = rt_store.open_store(self.db)
        con.execute("DELETE FROM sync_log WHERE error IS NOT NULL")
        rt_store.upsert_items(con, [T.item(40, "social", "discord", hours=200, source="old-chan"),
                                    T.item(41, "social", "reddit", hours=1, source="Reddit_User_%"),
                                    T.item(42, "social", "reddit", hours=1, source="plain")])
        con.close()
        self.api._cache.clear()

    def test_status_has_source_freshness(self):
        with mock.patch.dict(os.environ, {"FNP_RT_SOCIAL_LIMIT": "3"}):
            st = self.api.status()
        sf = st["source_freshness"]
        by = {r["source_table"]: r for r in sf["rows"]}
        self.assertTrue(sf["sync_healthy"])
        self.assertTrue(by["discord_posts"]["stale"])
        self.assertFalse(by["reddit_posts"]["stale"])
        self.assertTrue(by["reddit_posts"]["capped"])  # 5 reddit rows >= cap 3
        self.assertEqual(sum(r["rows"] for r in sf["rows"]), st["total_rows"])
        self.assertAlmostEqual(sum(r["share"] for r in sf["rows"]), 1.0, places=2)

    def test_sources_typeahead(self):
        r = self.api.sources(self.q(q="a.com"))
        self.assertEqual(r["sources"], [dict(source="a.com", count=3)])
        r = self.api.sources(self.q(kind="social", limit=2))
        self.assertEqual(len(r["sources"]), 2)
        counts = [s["count"] for s in r["sources"]]
        self.assertEqual(counts, sorted(counts, reverse=True))
        # LIKE wildcards in the text are literal
        self.assertEqual([s["source"] for s in self.api.sources(self.q(q="%"))["sources"]], ["Reddit_User_%"])
        self.assertEqual([s["source"] for s in self.api.sources(self.q(q="_"))["sources"]], ["Reddit_User_%"])
        self.assertEqual(self.api.sources(self.q(q="'; DROP TABLE items; --"))["sources"], [])
        self.assertEqual(self.sql("SELECT count(*) FROM items")[0][0], 15)
        self.assertEqual([s["source"] for s in self.api.sources(self.q(q="CHAN", platform="telegram"))["sources"]], ["chan1"])

    def test_sources_validation(self):
        for bad in ({"limit": "0"}, {"limit": "51"}, {"limit": "x"}, {"kind": "video"}, {"platform": "a;b"}, {"q": "x" * 81}):
            with self.assertRaises(rt_api.ApiError) as cm:
                self.api.sources(self.q(**bad))
            self.assertEqual(cm.exception.code, 400, bad)


class SourcesRoute(T.Http):
    def test_gate_and_route(self):
        self.assertEqual(self.get("/api/rt/sources?q=a")[0], 403)
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            st, hd, b = self.get("/api/rt/sources?q=a.com&limit=5")
            self.assertEqual((st, b["sources"][0]["source"]), (200, "a.com"))
            self.assertEqual(self.get("/api/rt/sources?limit=999")[0], 400)
            self.assertIn("source_freshness", self.get("/api/rt/status")[2])


class ScriptLang(unittest.TestCase):
    def test_overrides_only_en_und_empty(self):
        hi = "सरकार ने नई योजना की घोषणा की"
        self.assertEqual(bi.script_lang(hi, "en"), "hi")
        self.assertEqual(bi.script_lang(hi, "en-US"), "hi")
        self.assertEqual(bi.script_lang(hi, "und"), "hi")
        self.assertEqual(bi.script_lang(hi, ""), "hi")
        self.assertEqual(bi.script_lang(hi, None), "hi")
        self.assertEqual(bi.script_lang(hi, "mr-IN"), "mr-IN")  # a real non-English tag is left alone
        self.assertEqual(bi.script_lang(hi, "hi-IN"), "hi-IN")

    def test_each_script(self):
        cases = {"te": "ప్రభుత్వం కొత్త పథకం ప్రకటించింది", "kn": "ಸರ್ಕಾರ ಹೊಸ ಯೋಜನೆ ಘೋಷಿಸಿದೆ", "gu": "સરકારે નવી યોજના જાહેર કરી",
                 "ta": "அரசு புதிய திட்டத்தை அறிவித்தது", "bn": "সরকার নতুন প্রকল্প ঘোষণা করেছে", "ur": "حکومت نے نئی اسکیم کا اعلان کیا"}
        for code, text in cases.items():
            self.assertEqual(bi.script_lang(text, "en"), code, code)

    def test_threshold_and_latin(self):
        self.assertEqual(bi.script_lang("The government announced a new scheme", "en"), "en")
        self.assertEqual(bi.script_lang("Fact check: क्या यह सच है", "en"), "en")  # Devanagari letters are < 50%
        self.assertEqual(bi.script_lang("PM मोदी ने कहा 2026 में 100% सच्चाई", "en"), "hi")
        self.assertEqual(bi.script_lang("1234 !!! ???", "en"), "en")  # no letters
        self.assertEqual(bi.script_lang("", "en"), "en")
        self.assertEqual(bi.script_lang(None, None), None)


if __name__ == "__main__":
    unittest.main()
