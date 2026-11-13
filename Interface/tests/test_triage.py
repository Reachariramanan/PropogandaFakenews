"""S3 Triage tab: /api/triage/{meta,items,facets,item/<id>}. Offline and SYNTHETIC: every id, text and URL below is invented, in the
real triage / realtime-export schemas (never a copy of a real post). Temp store, temp export and temp triage dirs only; the real
reports/ directories are never read (the Triage object gets an explicit fnp_dir)."""
import hashlib
import itertools
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from http.server import ThreadingHTTPServer
from unittest import mock

import helpers  # noqa: F401  (sys.path + offline env)
import rt_api
import rt_enrich
import rt_store
import rt_triage
import server
from build_index import norm_url
from helpers import call, jcall, needs_index
from test_rt_api import NOW, iso
from test_rt_real_export import RealFixture, probs, real_row, write_real

STRICT = lambda c: (_ for _ in ()).throw(ValueError("non-strict constant %s" % c))  # noqa: E731
FC_URL = "https://factcheck.example.org/fc/invented-claim/"


def uid(n):
    return hashlib.sha1(("synthetic-post-%d" % n).encode()).hexdigest()


def fnp_item(n, tier="low", platform="telegram", lang="en", posted="2026-10-26T10:00:00+00:00", reach=100, fake=None, verdict=None,
             cos=0.7, rerank=None, label="FAKE", techs=("Loaded_Language",), spans=None, countries=("IND",), text=None, validated=False,
             p_prop=0.5, url="default", author="invented_channel"):
    text = text if text is not None else "Invented sample post number %d about a made-up flood in Exampleville. Share now!" % n
    return dict(post_uid=uid(n), source="social", platform=platform, url="https://example.org/p/%d" % n if url == "default" else url,
                author=author, text=text, lang=lang, posted_at=posted, reach=reach,
                p_fake=None if fake is None else round(fake * 0.8, 4), p_misleading=None if fake is None else round(fake * 0.2, 4),
                fake_score=fake, verdict=verdict, p_prop=p_prop, techniques=list(techs), spans=spans or [], countries=list(countries),
                tier=tier, validated=validated, top_evidence_ids=["kb:t1"],
                nearest_factcheck=dict(kb_id="kb:t1", claim="Invented fact-checked claim <b>x</b>", label=label, url=FC_URL, cos=cos,
                                       rerank=rerank, debunk_match=rerank is not None and rerank >= 0.8),
                model_version="triage_test_v1", built_at="2026-11-05T11:45:48Z")


def default_fnp_items():
    t = "Invented sample: shocking lies exposed everywhere, share before they delete this!"
    return [
        fnp_item(1, "high", fake=0.99, verdict=dict(label="FAKE", basis="gemma", source="live_scan", rationale="Invented <i>rationale</i>.",
                                                    at="2026-11-05T09:00:00Z"), rerank=0.93, cos=0.8, platform="telegram", reach=5000,
                 text=t, spans=[[0, 15, "Loaded_Language", 0.9], [16, 33, "Slogans", 0.7], [9000, 9005, "Doubt", 0.5], [5, 3, "Doubt", 0.5]],
                 techs=("Loaded_Language", "Slogans"), countries=("IND", "GBR"), posted="2026-10-27T08:00:00+00:00"),
        fnp_item(2, "high", fake=0.7, verdict=dict(label="MISLEADING", basis="stacker", source="prop_lab_sample", rationale=None, at=None),
                 platform="reddit", reach=900, lang="hi", posted="2026-10-20T08:00:00+00:00", countries=("IND",)),
        fnp_item(3, "medium", fake=0.55, verdict=dict(label="LEGIT", basis="gemma", source="live_scan", rationale="r", at=None),
                 platform="telegram", reach=10, posted="2026-10-10T08:00:00+00:00"),
        fnp_item(4, "medium", platform="youtube_video", reach=None, rerank=0.6, techs=("Doubt",), countries=("USA",),
                 posted="2026-09-01T08:00:00+00:00"),
        fnp_item(5, "low", platform="telegram", reach=300, lang="und", posted=None, countries=(), techs=(), p_prop=None),
        fnp_item(6, "low", platform="reddit", reach=1, posted="2026-10-27T07:00:00+00:00", cos=0.66, label="LEGIT"),
        fnp_item(7, "low", platform="telegram", reach=1, posted="2026-10-26T23:00:00+00:00", text="Plain words with 'quote and \"double\" and <script>x</script>"),
    ]


def write_triage(d, items=None, built_at="2026-11-05T11:45:48Z", meta_extra=None, raw_lines=(), tiers=None):
    items = default_fnp_items() if items is None else items
    os.makedirs(d, exist_ok=True)
    counts = {"high": 0, "medium": 0, "low": 0}
    for it in items:
        counts[it["tier"]] += 1
    meta = dict(version="triage_v1", built_at=built_at, model_version="triage_test_v1", n_items=len(items), tiers=tiers or counts,
                n_with_verdict=sum(1 for i in items if i["verdict"]), data_through="2026-10-27T08:00:00+00:00", validated=False,
                rules=dict(high="invented high rule", medium="invented medium rule", low="invented low rule"),
                caveats=["Invented caveat <b>one</b>.", "Invented caveat two."])
    meta.update(meta_extra or {})
    with open(os.path.join(d, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, allow_nan=False) + "\n")
        for ln in raw_lines:
            f.write(ln + "\n")
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


# ================================================================================================ tier rules (pure + SQL)
class TierRules(unittest.TestCase):
    T = staticmethod(rt_triage.rt_tier)

    def test_high_by_rerank_or_strong_propaganda_boundaries(self):
        self.assertEqual(self.T(None, None, 0.7, 0.8, None, False), ("high", "rerank_match"))
        self.assertEqual(self.T(0.0, 0, 0.0, 0.0, None, False), ("high", "rerank_match"))          # any rerank value counts
        self.assertEqual(self.T(0.8, 2, None, None, None, False), ("high", "strong_propaganda"))    # exactly the line
        self.assertEqual(self.T(0.7999, 2, None, None, None, False)[0], "medium")
        self.assertEqual(self.T(0.95, 1, None, None, None, False)[0], "medium")                     # a single strong span
        self.assertEqual(self.T(0.9, 0, None, None, None, False)[0], "medium")                      # no strong span
        self.assertEqual(self.T(0.9, None, None, None, None, False)[0], "medium")

    def test_medium_rules_and_low(self):
        self.assertEqual(self.T(0.4, 0, None, None, None, False), ("medium", "p_prop"))
        self.assertEqual(self.T(0.3999, 0, None, 0.6, None, False), ("medium", "cosine"))
        self.assertEqual(self.T(0.3999, 0, None, 0.5999, None, False), ("low", "below"))
        self.assertEqual(self.T(None, None, None, None, 0.5, True), ("medium", "deep"))
        self.assertEqual(self.T(None, None, None, None, 0.4999, True), ("low", "below"))
        self.assertEqual(self.T(None, None, None, None, None, False), ("low", "below"))

    def test_deep_verdict_demotes_a_high_item_only_below_the_line(self):
        self.assertEqual(self.T(0.9, 2, 0.8, 0.8, 0.2999, True), ("medium", "deep_demoted"))
        self.assertEqual(self.T(0.9, 2, 0.8, 0.8, 0.3, True), ("high", "rerank_match"))
        self.assertEqual(self.T(0.9, 2, None, None, 0.0, True), ("medium", "deep_demoted"))
        self.assertEqual(self.T(0.9, 2, None, None, 0.9, True), ("high", "strong_propaganda"))
        # a deep verdict that is low does not lower a MEDIUM item, and never raises a LOW one
        self.assertEqual(self.T(0.45, 0, None, None, 0.1, True), ("medium", "p_prop"))
        self.assertEqual(self.T(0.1, 0, None, None, 0.2, True), ("low", "below"))

    def test_sql_case_equals_python_on_a_boundary_grid(self):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE e(p_prop REAL, n_strong INT, fc_rerank REAL, fc_cos REAL)")
        con.execute("CREATE TABLE d(id INT, d_pfm REAL)")
        n = 0
        for pp, ns, rr, cs, dp in itertools.product((None, 0.0, 0.3999, 0.4, 0.5999, 0.6, 0.79, 0.8, 0.81, 0.95), (None, 0, 1, 2, 3), (None, 0.0, 0.8),
                                                    (None, 0.5999, 0.6, 0.9), (None, 0.0, 0.2999, 0.3, 0.4999, 0.5, 1.0)):
            con.execute("DELETE FROM e")
            con.execute("DELETE FROM d")
            con.execute("INSERT INTO e VALUES (?,?,?,?)", (pp, ns, rr, cs))
            if dp is not None:
                con.execute("INSERT INTO d VALUES (1, ?)", (dp,))
            got = con.execute("SELECT %s FROM e LEFT JOIN d ON 1 = 1" % rt_triage.TIER_SQL).fetchone()[0]
            want = self.T(pp, ns, rr, cs, dp, dp is not None)[0]
            self.assertEqual(rt_triage.TIER_NAME[got], want, (pp, ns, rr, cs, dp))
            n += 1
        self.assertEqual(n, 10 * 5 * 3 * 4 * 7)

    def test_explanations_carry_the_numbers(self):
        s = dict(p_prop=0.72, n_strong=2, rerank=None, cosine=0.3, deep_pfm=None, has_deep=False)
        txt = " ".join(rt_triage.explain_rt("high", "strong_propaganda", s))
        self.assertIn("2 strong propaganda spans", txt)
        self.assertIn("0.72", txt)
        s = dict(p_prop=0.1, n_strong=0, rerank=0.9, cosine=0.8, deep_pfm=0.12, has_deep=True)
        txt = " ".join(rt_triage.explain_rt("medium", "deep_demoted", s))
        self.assertIn("lowered to MEDIUM", txt)
        self.assertIn("0.12", txt)
        self.assertIn("rerank score of 0.90", txt)
        txt = " ".join(rt_triage.explain_rt("low", "below", dict(p_prop=0.2, n_strong=0, rerank=None, cosine=0.1, deep_pfm=None, has_deep=False)))
        self.assertIn("0.40", txt)


# ================================================================================================ the realtime queue
def deep_json(fake, mis=0.0, label="FAKE"):
    return json.dumps({"engine": {"verdict": {"label": label, "probs": {"FAKE": fake, "MISLEADING": mis}}}})


class QueueFixture(RealFixture):
    """The 12 fixture items scored on purpose to sit on the tier boundaries (see EXPECT)."""
    EXPECT = {  # item id -> (tier, rule)
        "news:scraped_articles:1": ("high", "rerank_match"),            # rerank 0.9
        "news:scraped_articles:2": ("high", "strong_propaganda"),       # 2 strong spans, p_prop exactly 0.8
        "news:scraped_articles:3": ("medium", "p_prop"),                # 2 strong but p_prop 0.7999
        "news:scraped_articles:4": ("medium", "p_prop"),                # p_prop exactly 0.4, no strong
        "news:scraped_articles:5": ("medium", "cosine"),                # cosine exactly 0.6
        "news:scraped_articles_trusted:6": ("low", "below"),            # cosine 0.5999, p_prop 0.3
        "reddit_posts:10": ("medium", "deep_demoted"),                  # rerank but deep 0.29
        "reddit_posts:11": ("high", "rerank_match"),                    # rerank and deep exactly 0.3: stays HIGH
        "reddit_posts:14": ("medium", "deep"),                          # deep-only, exactly 0.5
        "x_posts:15": ("low", "below"),                                 # deep-only 0.49
    }

    def setUp(self):
        super().setUp()
        nf = lambda cos, rr=None: dict(kb_id="kb:q-%s" % cos, label="FAKE", claim="Invented claim %s" % cos, url=FC_URL, cosine=cos, rerank=rr)  # noqa: E731
        rows = [
            real_row("news:scraped_articles:1", p_prop=0.2, strong=(), nf=nf(0.8, 0.9)),
            real_row("news:scraped_articles:2", p_prop=0.8, strong=("Loaded_Language", "Doubt"), nf=nf(0.1)),
            real_row("news:scraped_articles:3", p_prop=0.7999, strong=("Doubt", "Doubt"), nf=nf(0.1)),
            real_row("news:scraped_articles:4", p_prop=0.4, strong=(), nf=nf(0.1)),
            real_row("news:scraped_articles:5", p_prop=0.3999, strong=(), nf=nf(0.6)),
            real_row("news:scraped_articles_trusted:6", p_prop=0.3, strong=(), nf=nf(0.5999)),
            real_row("reddit_posts:10", p_prop=0.2, strong=(), nf=nf(0.8, 0.8)),
            real_row("reddit_posts:11", p_prop=0.2, strong=(), nf=nf(0.8, 0.8)),
            real_row("telegram_posts:12", too_short=True, p_prop=None, strong=(), nf=nf(0.9, 0.99)),
        ]
        write_real(self.export, {"news/part_0000.jsonl": rows[:6], "social/part_0000.jsonl": rows[6:]})
        con = rt_store.open_store(self.db)
        for iid, js in (("reddit_posts:10", deep_json(0.29)), ("reddit_posts:11", deep_json(0.3)), ("reddit_posts:14", deep_json(0.5)),
                        ("x_posts:15", deep_json(0.49))):
            con.execute("INSERT INTO analyses(item_id, created_at, engine_version, result_json) VALUES (?,?,?,?)",
                        (iid, iso(minutes=2), "check_v1", js))
        # telegram_posts:13 is unscored by the batch export but appears in the OLDER triage export (src='triage'): not in the queue
        con.close()
        write_triage(self.triage, [fnp_item(13, "low", url="https://example.org/telegram_posts/13")])
        self.tri = rt_triage.Triage(self.api, fnp_dir=self.triage, clock=lambda: NOW)

    def ids(self, **kw):
        args = dict(source="realtime", size=50)
        args.update(kw)
        out = self.tri.items({k: [str(v)] for k, v in args.items()})
        return out, [r["id"] for r in out["items"]]


class Queue(QueueFixture):
    def test_tiers_match_the_rules_and_python_agrees_with_sql(self):
        out, ids = self.ids()
        got = {r["id"]: (r["tier"], r["rule"]) for r in out["items"]}
        self.assertEqual(got, self.EXPECT)
        self.assertEqual(out["tiers"], {"high": 3, "medium": 5, "low": 2})
        self.assertEqual(out["total"], 10)
        # the tier the SQL assigned (the one used to sort and filter) is the one the Python rule reports for each row
        for t in ("high", "medium", "low"):
            _, tids = self.ids(tier=t)
            self.assertEqual(sorted(tids), sorted(i for i, v in self.EXPECT.items() if v[0] == t), t)

    def test_unscored_too_short_and_triage_rows_are_excluded_and_counted(self):
        out, ids = self.ids()
        self.assertNotIn("telegram_posts:12", ids)       # too short for the encoder
        self.assertNotIn("telegram_posts:13", ids)       # unscored (only the older triage export has it)
        m = self.tri.meta()["sources"]["realtime"]
        self.assertEqual((m["n_queue"], m["n_too_short"], m["n_items"], m["n_unscored"], m["n_deep"]), (10, 1, 12, 1, 4))
        self.assertEqual(m["tiers"], {"high": 3, "medium": 5, "low": 2})
        self.assertEqual(sum(m["tiers"].values()), m["n_queue"])
        self.assertEqual(m["share_scored"], round(11 / 12, 6))
        self.assertEqual(m["validated"], 0)
        self.assertEqual(m["rules"]["high"], rt_triage.RT_RULES["high"])
        self.assertEqual(m["thresholds"]["deep_demote"], 0.3)

    def test_priority_order_inside_and_across_tiers(self):
        _, ids = self.ids()
        # HIGH first: deep probability first (reddit:11 has 0.3), then rerank (news:1 0.9), then the strong-span item
        self.assertEqual(ids[:3], ["reddit_posts:11", "news:scraped_articles:1", "news:scraped_articles:2"])
        tiers = [self.EXPECT[i][0] for i in ids]
        self.assertEqual(tiers, sorted(tiers, key=["high", "medium", "low"].index))
        # MEDIUM: all five items present; inside the tier the freshness band never goes backwards (fresher first),
        # and inside one band the deep probability decides (deep 0.5 reddit:14 before deep 0.29 reddit:10)
        med = [i for i in ids if self.EXPECT[i][0] == "medium"]
        self.assertEqual(set(med), {"reddit_posts:14", "reddit_posts:10", "news:scraped_articles:3",
                                    "news:scraped_articles:4", "news:scraped_articles:5"})
        con = sqlite3.connect(self.db)
        ts = {i: rt_triage._parse_ts(p) for i, p in con.execute("SELECT id, published_at FROM items")}
        con.close()
        now = NOW.timestamp() if hasattr(NOW, "timestamp") else float(NOW)
        bands = [rt_triage.age_band(ts[i], now) for i in med]
        self.assertEqual(bands, sorted(bands))
        if bands[med.index("reddit_posts:14")] == bands[med.index("reddit_posts:10")]:
            self.assertLess(med.index("reddit_posts:14"), med.index("reddit_posts:10"))
        # recent sort ignores tiers: newest first
        _, rec = self.ids(sort="recent")
        con = sqlite3.connect(self.db)
        pub = {i: p for i, p in con.execute("SELECT id, published_at FROM items")}
        con.close()
        self.assertEqual([pub[i] for i in rec], sorted((pub[i] for i in rec), reverse=True))

    def test_reach_then_recency_break_ties(self):
        # two items with identical signals: more engagement first, then newer
        con = rt_store.open_store(self.db)
        con.execute("UPDATE items SET engagement = 10 WHERE id = 'news:scraped_articles:4'")
        con.execute("UPDATE items SET engagement = 500 WHERE id = 'news:scraped_articles:3'")
        con.close()
        write_real(self.export, {"social/part_0000.jsonl": [
            real_row("reddit_posts:10", p_prop=0.5, strong=(), nf=None), real_row("telegram_posts:13", p_prop=0.5, strong=(), nf=None),
            real_row("x_posts:15", p_prop=0.5, strong=(), nf=None)]}, built_at="2026-11-12T11:00:00+00:00")
        con = rt_store.open_store(self.db)
        con.execute("UPDATE items SET engagement = 40 WHERE id = 'telegram_posts:13'")
        con.execute("UPDATE items SET engagement = 40 WHERE id = 'x_posts:15'")
        con.close()
        _, ids = self.ids(tier="medium")
        pos = {i: ids.index(i) for i in ids}
        # same signal, engagement 40 vs 40: the newer published_at (x_posts:15 is 2 h old, telegram 26 h) comes first
        self.assertLess(pos["x_posts:15"], pos["telegram_posts:13"])

    def test_filters(self):
        _, ids = self.ids(platform="reddit")
        self.assertEqual(sorted(ids), ["reddit_posts:10", "reddit_posts:11", "reddit_posts:14"])
        _, ids = self.ids(lang="hi")
        self.assertEqual(ids, ["news:scraped_articles:3"])
        _, ids = self.ids(days=1)       # published within 24 h of NOW (the fixture clock)
        con = sqlite3.connect(self.db)
        want = {r[0] for r in con.execute("SELECT id FROM items WHERE published_at >= ?", (rt_store.utc_iso(NOW - timedelta(days=1)),))}
        con.close()
        self.assertEqual(set(ids), want & set(self.EXPECT))
        _, ids = self.ids(country="ind")
        self.assertEqual(set(ids), {i for i in self.EXPECT if i not in ("reddit_posts:14", "x_posts:15")})   # deep-only items have no countries
        _, ids = self.ids(country="GBR")
        self.assertEqual(ids, [])
        _, ids = self.ids(technique="Loaded_Language")
        self.assertEqual(set(ids), {"news:scraped_articles:2"})
        _, ids = self.ids(q="river")
        self.assertTrue(ids and set(ids) <= set(self.EXPECT))
        out, ids = self.ids(q="river", tier="high")
        self.assertEqual(out["total"], sum(out["tiers"][t] for t in ("high",)))
        # chips count the other filters but not the tier filter
        out, _ = self.ids(platform="reddit", tier="low")
        self.assertEqual(out["tiers"], {"high": 1, "medium": 2, "low": 0})
        self.assertEqual(out["total"], 0)

    def test_pagination_and_totals(self):
        out, ids = self.ids(size=4, page=1)
        self.assertEqual((out["total"], out["pages"], len(ids)), (10, 3, 4))
        _, p2 = self.ids(size=4, page=2)
        _, p3 = self.ids(size=4, page=3)
        _, allids = self.ids()
        self.assertEqual(ids + p2 + p3, allids)
        out, ids = self.ids(size=4, page=4)
        self.assertEqual((out["total"], ids), (10, []))

    def test_validation_and_hostile_values(self):
        for k, v in (("size", 51), ("size", 0), ("page", 0), ("page", "x"), ("tier", "urgent"), ("source", "other"), ("sort", "random"),
                     ("platform", "a b"), ("platform", "x'; DROP TABLE items; --"), ("lang", "e n"), ("country", "IN"), ("country", "IND' OR 1=1"),
                     ("technique", "Loaded_Language' OR '1'='1"), ("days", -1), ("days", "x"), ("q", "x" * 301), ("page", 10000)):
            with self.assertRaises(rt_api.ApiError, msg="%s=%r" % (k, v)) as cm:
                self.tri.items({k: [str(v)], "source": ["realtime"]} if k != "source" else {k: [str(v)]})
            self.assertEqual(cm.exception.code, 400, (k, v))
        # free text is data: quotes, operators and SQL never change the statement
        for q in ("'; DROP TABLE items; --", '" OR 1=1 NEAR(', "river*", "a AND NOT b", "\u0000‮", "%", "_"):
            try:
                out, _ = self.ids(q=q)
                self.assertGreaterEqual(out["total"], 0)
            except rt_api.ApiError as e:
                self.assertEqual(e.code, 400)
        con = sqlite3.connect(self.db)
        self.assertEqual(con.execute("SELECT count(*) FROM items").fetchone()[0], 12)
        con.close()
        with self.assertRaises(rt_api.ApiError):
            self.tri.items({"source": ["realtime"], "q": ["!!!"]})

    def test_item_detail_explains_with_numbers(self):
        d = self.tri.item("reddit_posts:10", {"source": ["realtime"]})
        self.assertEqual((d["source"], d["triage"]["tier"], d["triage"]["rule"]), ("realtime", "medium", "deep_demoted"))
        self.assertTrue(d["triage"]["in_queue"])
        self.assertIn("lowered to MEDIUM", " ".join(d["triage"]["why"]))
        self.assertEqual(d["item"]["id"], "reddit_posts:10")
        self.assertEqual(d["triage"]["signals"]["rerank"], 0.8)
        d = self.tri.item("news:scraped_articles:2")  # default source for a non-sha1 id is realtime
        self.assertEqual(d["triage"]["rule"], "strong_propaganda")
        self.assertIn("0.80", " ".join(d["triage"]["why"]))
        d = self.tri.item("telegram_posts:12")
        self.assertFalse(d["triage"]["in_queue"])
        self.assertIsNone(d["triage"]["tier"])
        self.assertIn("too short", " ".join(d["triage"]["why"]))
        d = self.tri.item("telegram_posts:13")
        self.assertFalse(d["triage"]["in_queue"])
        self.assertTrue("older FNP triage export" in " ".join(d["triage"]["why"]))
        for bad in ("nope:1", "x" * 300, "a b", "../etc/passwd"):
            with self.assertRaises(rt_api.ApiError) as cm:
                self.tri.item(bad, {"source": ["realtime"]})
            self.assertEqual(cm.exception.code, 404, bad)

    def test_facets(self):
        f = self.tri.facets({"source": ["realtime"]})
        self.assertEqual({p["value"]: p["count"] for p in f["platforms"]}, {"news": 6, "reddit": 3, "x": 1})
        self.assertEqual(f["countries"][0]["value"], "IND")
        self.assertEqual(f["countries"][0]["name"], None if self.api.country_names.get("IND") is None else "India")
        self.assertIn("Loaded_Language", {t["value"] for t in f["techniques"]})

    def test_no_store_and_no_batch_means_an_empty_queue_not_an_error(self):
        shutil.rmtree(self.export)
        self.enrich.refresh(force=True)
        con = rt_store.open_store(self.db)
        con.execute("DELETE FROM analyses")
        con.close()
        tri = rt_triage.Triage(self.api, fnp_dir=os.path.join(self.tmp, "none"), clock=lambda: NOW)
        out = tri.items({"source": ["realtime"]})
        self.assertEqual((out["total"], out["items"]), (0, []))
        os.unlink(self.db)
        with self.assertRaises(rt_api.ApiError) as cm:
            tri.items({"source": ["realtime"]})
        self.assertEqual(cm.exception.code, 503)
        m = tri.meta()["sources"]["realtime"]
        self.assertFalse(m["available"])
        self.assertFalse(os.path.exists(self.db), "a missing store must not be created by a read")


# ================================================================================================ the FNP export
class FnpSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.dir = os.path.join(self.tmp, "triage", "v1")
        self.desk = {norm_url(FC_URL): "db:desk-1"}
        self.names = rt_api.CountryNames(os.path.join(self.tmp, "none.json"))
        self.fx = rt_triage.FnpExport(self.dir, country_names=self.names, desk_url=lambda u: self.desk.get(norm_url(u)))

    def list(self, sort="priority", page=1, size=50, **flt):
        f = dict(tier="", platform="", lang="", country="", technique="", days=0, q="")
        f.update(flt)
        return self.fx.list(f, sort, page, size)

    def test_missing_directory_is_an_empty_state(self):
        out = self.list()
        self.assertFalse(out["available"])
        self.assertEqual((out["total"], out["items"]), (0, []))
        self.assertIn("no triage export", out["reason"])
        with self.assertRaises(rt_api.ApiError) as cm:
            self.fx.detail(uid(1))
        self.assertEqual(cm.exception.code, 404)
        os.makedirs(self.dir)  # directory without files
        self.assertFalse(self.list()["available"])

    def test_loads_counts_match_meta_and_order(self):
        write_triage(self.dir)
        out = self.list()
        st = self.fx.state
        self.assertTrue(st["available"])
        self.assertTrue(st["counts_match"])
        self.assertEqual(out["tiers"], {"high": 2, "medium": 2, "low": 3})
        self.assertEqual(out["total"], 7)
        # export order: tier, then fake_score, rerank, reach
        self.assertEqual([r["id"] for r in out["items"]][:4], [uid(1), uid(2), uid(3), uid(4)])
        self.assertEqual(st["n_with_verdict"], 3)
        self.assertEqual(st["n_validated"], 0)
        self.assertEqual(st["load_seconds"] < 5, True)

    def test_counts_mismatch_is_flagged_not_hidden(self):
        write_triage(self.dir, tiers={"high": 99, "medium": 2, "low": 3})
        self.list()
        self.assertFalse(self.fx.state["counts_match"])

    def test_filters_sort_and_pagination(self):
        write_triage(self.dir)
        self.assertEqual({r["id"] for r in self.list(platform="reddit")["items"]}, {uid(2), uid(6)})
        self.assertEqual([r["id"] for r in self.list(lang="hi")["items"]], [uid(2)])
        self.assertEqual({r["id"] for r in self.list(country="GBR")["items"]}, {uid(1)})
        self.assertEqual([r["id"] for r in self.list(technique="Doubt")["items"]], [uid(4)])
        self.assertLessEqual({uid(2), uid(3), uid(4)}, {r["id"] for r in self.list(q="FLOOD")["items"]})
        self.assertEqual([r["id"] for r in self.list(q="shocking LIES delete")["items"]], [uid(1)])
        # days count back from the newest post in the export (2026-10-27T08:00), undated posts never match a window
        got = {r["id"] for r in self.list(days=1)["items"]}
        self.assertEqual(got, {uid(1), uid(6), uid(7)})
        self.assertEqual({r["id"] for r in self.list(days=3650)["items"]}, {uid(i) for i in (1, 2, 3, 4, 6, 7)})
        out = self.list(tier="medium")
        self.assertEqual(out["total"], 2)
        self.assertEqual(out["tiers"], {"high": 2, "medium": 2, "low": 3})       # chips ignore the tier filter
        rec = [r["id"] for r in self.list(sort="recent")["items"]]
        self.assertEqual(rec[:3], [uid(1), uid(6), uid(7)])
        self.assertEqual(rec[-1], uid(5))                                         # undated last
        out = self.list(size=3, page=3)
        self.assertEqual((out["total"], out["pages"], len(out["items"])), (7, 3, 1))
        self.assertEqual(self.list(q="nothing at all matches this")["items"], [])

    def test_hostile_text_is_inert_data(self):
        write_triage(self.dir)
        for q in ("'; DROP TABLE x; --", "<script>alert(1)</script>", "\u0000", "%%", "a" * 300):
            self.assertIsInstance(self.list(q=q)["total"], int)
        r = [x for x in self.list()["items"] if x["id"] == uid(7)][0]
        self.assertIn("<script>", r["snippet"])        # carried as data; the UI escapes it (see test_ui_escapes)

    def test_detail_resolves_spans_countries_factcheck(self):
        write_triage(self.dir)
        d = self.fx.detail(uid(1))
        it, tr = d["item"], d["triage"]
        self.assertEqual(d["source"], "fnp")
        self.assertEqual(it["tier"], "high")
        self.assertFalse(it["validated"])
        # two valid spans kept; the out-of-range and the reversed one dropped and counted
        self.assertEqual([(s["start"], s["end"], s["technique"], s["text"]) for s in it["spans"]],
                         [(0, 15, "Loaded_Language", it["text"][0:15]), (16, 33, "Slogans", it["text"][16:33])])
        self.assertEqual((it["n_spans"], it["spans_dropped"]), (2, 2))
        self.assertEqual({t["technique"]: t["count"] for t in it["techniques"]}, {"Loaded_Language": 1, "Slogans": 1})
        self.assertEqual([c["iso3"] for c in it["countries"]], ["IND", "GBR"])
        self.assertEqual(it["nearest_factcheck"]["desk_id"], "db:desk-1")
        self.assertEqual((it["nearest_factcheck"]["similarity"], it["nearest_factcheck"]["rerank"]), (0.8, 0.93))
        self.assertEqual(it["verdict"]["label"], "FAKE")
        self.assertEqual(tr["rule_text"], "invented high rule")
        self.assertTrue(any("FAKE" in w.title().upper() or "Fake" in w for w in tr["why"]))
        self.assertTrue(any("debunk match" in w for w in tr["why"]))
        self.assertEqual(tr["caveats"][0], "Invented caveat <b>one</b>.")
        # a non-http fact-check link is dropped, a missing desk callable just means no desk_id
        bad = fnp_item(8, "low")
        bad["nearest_factcheck"]["url"] = "javascript:alert(1)"
        write_triage(self.dir, [bad], built_at="2026-11-06T00:00:00Z")
        d = self.fx.detail(uid(8))
        self.assertIsNone(d["item"]["nearest_factcheck"]["url"])
        self.assertIsNone(d["item"]["nearest_factcheck"]["desk_id"])
        with self.assertRaises(rt_api.ApiError):
            self.fx.detail("0" * 40)

    def test_why_for_every_tier_states_numbers_or_falls_back_to_the_export_rule(self):
        write_triage(self.dir)
        w2 = " ".join(self.fx.detail(uid(2))["triage"]["why"])
        self.assertIn("MISLEADING".title(), w2)
        w3 = " ".join(self.fx.detail(uid(3))["triage"]["why"])
        self.assertIn("0.55", w3)
        w4 = " ".join(self.fx.detail(uid(4))["triage"]["why"])
        self.assertIn("rerank 0.60", w4)
        w6 = " ".join(self.fx.detail(uid(6))["triage"]["why"])
        self.assertIn("0.66", w6)
        w5 = self.fx.detail(uid(5))["triage"]["why"]          # cos 0.7, label FAKE: low still explains the candidate rule
        self.assertTrue(w5)
        odd = fnp_item(9, "high", cos=None)
        odd["nearest_factcheck"] = None
        write_triage(self.dir, [odd], built_at="2026-11-06T00:00:00Z")
        w = self.fx.detail(uid(9))["triage"]["why"]
        self.assertEqual(w, ["The export placed it in the HIGH tier. Its stated rule: invented high rule"])

    def test_reload_when_the_export_changes_and_never_a_half_loaded_list(self):
        write_triage(self.dir)
        self.assertEqual(self.list()["total"], 7)
        first = self.fx.state
        self.assertIs(self.fx.refresh(), first)                    # unchanged: no reload, same object
        items = default_fnp_items() + [fnp_item(20, "high", fake=0.9, verdict=dict(label="FAKE", basis="gemma", source="x", rationale=None, at=None))]
        write_triage(self.dir, items, built_at="2026-11-07T00:00:00Z")
        out = self.list()
        self.assertEqual(out["total"], 8)
        self.assertEqual(out["built_at"], "2026-11-07T00:00:00Z")
        self.assertIsNot(self.fx.state, first)
        # the directory disappears during a rebuild: the last copy keeps being served and the problem is stated
        shutil.rmtree(self.dir)
        out = self.list()
        self.assertTrue(out["available"])
        self.assertEqual(out["total"], 8)
        self.assertIn("missing", self.fx.state["error"])

    def test_bad_lines_are_counted_and_strict_json_enforced(self):
        raw = ["{not json", '{"post_uid": "%s", "tier": "high", "text": "x", "p_prop": NaN}' % uid(50), json.dumps([1]),
               json.dumps(dict(fnp_item(51), tier="urgent")), json.dumps(dict(fnp_item(52), post_uid="bad id!")), "",
               json.dumps(fnp_item(1))]                           # duplicate of an existing uid
        write_triage(self.dir, raw_lines=raw)
        self.list()
        self.assertEqual(self.fx.state["invalid"], 6)
        self.assertEqual(len(self.fx.state["items"]), 7)
        write_triage(self.dir, built_at="2026-11-08T00:00:00Z")
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            f.write('{"built_at": "x", "tiers": {"high": NaN}}')
        out = self.list()
        self.assertIn("could not read", self.fx.state["error"])
        self.assertTrue(out["available"])                            # the last good copy stays

    def test_validated_flag_is_counted_not_assumed(self):
        items = default_fnp_items()
        items[0]["validated"] = True
        write_triage(self.dir, items)
        self.list()
        self.assertEqual(self.fx.state["n_validated"], 1)

    def test_tolerates_missing_optional_fields(self):
        sparse = dict(post_uid=uid(60), tier="low", text="Invented short text")
        write_triage(self.dir, [], raw_lines=[json.dumps(sparse)])
        out = self.list()
        self.assertEqual(out["total"], 1)
        r = out["items"][0]
        self.assertEqual((r["platform"], r["reach"], r["p_prop"], r["nearest_factcheck"], r["countries"], r["techniques"]), (None, None, None, None, [], []))
        d = self.fx.detail(uid(60))
        self.assertEqual((d["item"]["spans"], d["item"]["nearest_factcheck"], d["item"]["verdict"]), ([], None, None))

    def test_a_large_export_loads_quickly_and_stays_correct(self):
        items = [fnp_item(1000 + i, ("high", "medium", "low")[i % 3], platform=("telegram", "reddit")[i % 2], reach=i) for i in range(6000)]
        write_triage(self.dir, items)
        t0 = time.monotonic()
        out = self.list(size=50)
        self.assertLess(time.monotonic() - t0, 10)
        self.assertEqual((out["total"], out["tiers"]), (6000, {"high": 2000, "medium": 2000, "low": 2000}))
        self.assertTrue(self.fx.state["counts_match"])


# ================================================================================================ HTTP: gate, JSON, limits
class Http(QueueFixture):
    def setUp(self):
        super().setUp()
        write_triage(self.triage, default_fnp_items())
        self.tri = rt_triage.Triage(self.api, fnp_dir=self.triage, clock=lambda: NOW,)
        self.tri.fnp.desk_url = lambda u: "db:desk-1"
        self._saved = {k: getattr(server, k) for k in ("RT", "TRI", "AUTH", "RT_RATE")}
        server.RT, server.TRI, server.AUTH = self.api, self.tri, None
        server._rt_hits.clear()
        server.H.log_message = lambda *a, **k: None
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.t = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.t.start()
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

    ROUTES = ("/api/triage/meta", "/api/triage/items", "/api/triage/items?source=realtime", "/api/triage/facets",
              "/api/triage/item/%s" % uid(1), "/api/triage/item/reddit_posts:10", "/api/triage/nope")

    @needs_index

    def test_closed_by_default_every_route_403(self):
        for path in self.ROUTES:
            st, hd, b = self.get(path)
            self.assertEqual(st, 403, path)
            self.assertEqual(b["error"], "login_required")
            self.assertEqual(hd["cache-control"], "no-store")
        self.assertEqual(self.get("/api/rt/gate")[2], {"open": False, "auth": False})

    def test_open_flag_serves_and_other_values_do_not(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            for path in self.ROUTES[:-1]:
                st, hd, b = self.get(path)
                self.assertEqual(st, 200, path)
                self.assertEqual(hd["cache-control"], "no-store")
            self.assertEqual(self.get("/api/triage/nope")[0], 404)
        for val in ("0", "true", "yes", ""):
            with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": val}):
                self.assertEqual(self.get("/api/triage/items")[0], 403, val)

    def test_login_configured_requires_credentials_then_serves(self):
        server.AUTH = (b"alice", b"pw-correct")
        st, hd, b = self.get("/api/triage/items")
        self.assertEqual(st, 401)
        self.assertIn("www-authenticate", hd)
        import base64
        ok = {"Authorization": "Basic " + base64.b64encode(b"alice:pw-correct").decode()}
        bad = {"Authorization": "Basic " + base64.b64encode(b"alice:nope").decode()}
        self.assertEqual(self.get("/api/triage/items", bad)[0], 401)
        st, hd, b = self.get("/api/triage/items?size=2", ok)
        self.assertEqual((st, len(b["items"])), (200, 2))

    def test_counts_equal_meta_and_json_is_strict(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            st, hd, raw = call(self.port, "GET", "/api/triage/meta")
            self.assertEqual(st, 200)
            self.assertTrue(hd["content-type"].startswith("application/json"))
            meta = json.loads(raw.decode(), parse_constant=STRICT)
            fnp, rt = meta["sources"]["fnp"], meta["sources"]["realtime"]
            self.assertEqual(fnp["tiers"], fnp["tiers_in_meta"])
            self.assertTrue(fnp["counts_match"])
            self.assertEqual(fnp["validated"], 0)
            self.assertEqual(meta["validated"], 0)
            self.assertEqual(meta["caveats"], rt_triage.CAVEATS)
            for src, m in (("fnp", fnp), ("realtime", rt)):
                for tier in ("high", "medium", "low"):
                    st, _, b = self.get("/api/triage/items?source=%s&tier=%s&size=50" % (src, tier))
                    self.assertEqual((st, b["total"]), (200, m["tiers"][tier]), (src, tier))
                    self.assertEqual(len(b["items"]), m["tiers"][tier])
                st, _, b = self.get("/api/triage/items?source=%s" % src)
                self.assertEqual(b["tiers"], m["tiers"])
            for path in self.ROUTES[:-1] + ("/api/triage/items?q=flood&tier=high", "/api/triage/facets?source=realtime"):
                st, hd, raw = call(self.port, "GET", path)
                self.assertEqual(st, 200, path)
                json.loads(raw.decode(), parse_constant=STRICT)

    def test_default_source_is_fnp_and_item_routing(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            st, _, b = self.get("/api/triage/items")
            self.assertEqual((b["source"], b["total"]), ("fnp", 7))
            st, _, b = self.get("/api/triage/item/%s" % uid(1))
            self.assertEqual((st, b["source"], b["item"]["id"]), (200, "fnp", uid(1)))
            st, _, b = self.get("/api/triage/item/news:scraped_articles:2")
            self.assertEqual((st, b["source"]), (200, "realtime"))
            st, _, b = self.get("/api/triage/item/%s?source=realtime" % uid(1))
            self.assertEqual(st, 404)                      # a sha1 looked up in the realtime store: unknown

    def test_errors_are_json_with_the_right_status(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            for path, code in (("/api/triage/items?size=500", 400), ("/api/triage/items?tier=x", 400), ("/api/triage/items?source=x", 400),
                               ("/api/triage/items?technique=nope", 400), ("/api/triage/items?country=I", 400), ("/api/triage/items?days=-1", 400),
                               ("/api/triage/item/" + "0" * 40, 404), ("/api/triage/item/", 404), ("/api/triage/item/x%27%20OR%201%3D1", 404),
                               ("/api/triage/item/news:scraped_articles:424242", 404), ("/api/triage/nope", 404), ("/api/triage/", 404)):
                st, hd, b = self.get(path)
                self.assertEqual(st, code, path)
                self.assertEqual(list(b), ["error"], path)
            os.unlink(self.db)
            self.api._cache.clear()
            self.tri._cache.clear()
            st, hd, b = self.get("/api/triage/items?source=realtime")
            self.assertEqual(st, 503)
            self.assertEqual(self.get("/api/triage/items?source=fnp")[0], 200)     # the other source is unaffected
            st, hd, b = self.get("/api/triage/meta")
            self.assertEqual((st, b["sources"]["realtime"]["available"]), (200, False))

    @needs_index

    def test_rate_limit_shared_with_the_evidence_routes(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}), mock.patch.object(server, "RT_RATE", 4):
            codes = [self.get("/api/triage/meta")[0] for _ in range(3)] + [self.get("/api/rt/status")[0]] + [self.get("/api/triage/items")[0]]
            self.assertEqual(codes, [200, 200, 200, 200, 429])
            st, hd, b = self.get("/api/triage/items")
            self.assertEqual(st, 429)
            self.assertIn("retry-after", hd)
        server._rt_hits.clear()
        with mock.patch.object(server, "RT_RATE", 2):
            for _ in range(4):
                self.assertEqual(self.get("/api/triage/items")[0], 403)    # a closed gate does not spend the budget
            self.assertEqual(self.get("/healthz")[0], 200)

    def test_only_get(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            st, hd, raw = call(self.port, "POST", "/api/triage/items", body={})
            self.assertIn(st, (404, 405, 403, 400))
            self.assertNotEqual(st, 200)

    def test_no_writes_to_the_store(self):
        con = sqlite3.connect(self.db)
        n = con.execute("SELECT count(*) FROM analyses").fetchone()[0]
        con.close()
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            for path in self.ROUTES[:-1]:
                self.get(path)
        con = sqlite3.connect(self.db)
        self.assertEqual(con.execute("SELECT count(*) FROM analyses").fetchone()[0], n)
        con.close()


class UiGuards(unittest.TestCase):
    """The front end only ever reaches item text through Labels.esc; these checks read the shipped scripts as text."""
    WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

    def read(self, name):
        with open(os.path.join(self.WEB, name), encoding="utf-8") as f:
            return f.read()

    def test_triage_js_never_assigns_untrusted_text_as_html_without_esc(self):
        js = self.read("triage.js")
        self.assertTrue("Labels" in js)
        for banned in ("document.write", "eval(", "new Function"):
            self.assertFalse(banned in js, banned)
        # no request leaves the origin and nothing but the four read-only triage routes plus the gate is called
        import re
        urls = set(re.findall(r"['\"](/api/[a-z/]+)", js))
        self.assertTrue(urls <= {"/api/triage/meta", "/api/triage/items", "/api/triage/item/", "/api/triage/facets", "/api/rt/gate"}, urls)
        self.assertFalse("http://" in js.replace("http://www.w3.org", ""))
        self.assertFalse("POST" in js)

    def test_route_nav_and_glossary_are_registered(self):
        self.assertTrue("triage:" in self.read("shell.js"))
        html = self.read("index.html")
        self.assertTrue('data-route="triage"' in html)
        self.assertTrue("triage.js" in html)
        lab = self.read("labels.js")
        for key in ("'tier'", "'triage queue'", "'validated'"):
            self.assertTrue(key in lab, key)


if __name__ == "__main__":
    unittest.main()
