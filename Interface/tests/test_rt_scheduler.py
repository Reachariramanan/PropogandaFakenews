"""Scheduled deep analysis (rt_scheduler): selection, cycle execution, politeness, schedule_log, gating by env and by the
first-delay rule, and how cached deep results feed /api/rt/stats and /api/rt/items next to the FNP batch export.
Offline: temp SQLite stores, fake engine functions / tests/fake_engine.py, the real handler on an ephemeral port."""
import contextlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

import helpers
import claimcheck
import fake_engine
import rt_api
import rt_enrich
import rt_jobs
import rt_scheduler
import rt_store
import server
from helpers import call, jcall, needs_index
from rt_fakes import NOW
from rt_text import simhash64
from test_rt_api import Fixture, default_export, item as mk_item, iso, rec, write_export

LONG = "The river valley is flooding after the dam gates were opened overnight, officials say. "


def item(i, kind="news", hours=1, text=None, title=None, **kw):
    """A realtime item with >= 40 characters of text (unique per id so simhashes differ)."""
    plat = "news" if kind == "news" else kw.pop("platform", "reddit")
    return mk_item(i, kind, plat, hours=hours, title=title if title is not None else "Headline number %s" % i,
                   text=text or (LONG + "Unique marker %s %s." % (i, "zq" * (int(i) % 7 + 1))), **kw)


def deep_result(fake=0.7, mis=0.2, label="FAKE", techs=(("Doubt", 0.7, "some words"),), dup_of=None):
    res = {"engine": {"state": "ok", "verdict": {"label": label, "probs": {
        "FAKE": fake, "MISLEADING": mis, "LEGIT": round(max(0.0, 1 - fake - mis - 0.02), 4), "INCONCLUSIVE": 0.02}}},
           "step1": {"label": label, "classes": []},
           "step2": {"shown": True, "propaganda": {"techniques": [
               {"technique": t, "start": 0, "end": len(x), "text": x, "confidence": c} for t, c, x in techs],
               "classes": []}},
           "text_sha1": "abc"}
    if dup_of:
        res["dup_of"] = dup_of
    return res


def ok_body(label="FAKE"):
    return {"engine": {"state": "ok", "verdict": {"label": label, "probs": {"FAKE": 0.8, "MISLEADING": 0.1, "LEGIT": 0.08,
                                                                              "INCONCLUSIVE": 0.02}}},
            "step1": {"label": label, "classes": []}, "step2": {"shown": True, "propaganda": {"techniques": []}}}


def busy_body():
    return {"engine": {"state": "busy", "retry_after_s": 0, "reason": "busy"}}, 200, {}


def degraded_body():
    b = ok_body()
    b["engine"] = dict(b["engine"], state="degraded", reason="timeout after 55 s")
    return b, 200, {}


class Base(unittest.TestCase):
    """A temp store, an enrich sidecar over a temp export dir, and a scheduler whose engine is a python function."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = os.path.join(self.tmp, "realtime.sqlite")
        self.export = os.path.join(self.tmp, "export")
        self.enrich = rt_enrich.Enrich(export_dir=self.export, triage_dir=os.path.join(self.tmp, "triage"),
                                       sidecar=os.path.join(self.tmp, "enr.sqlite"), store_path=self.db)
        con = rt_store.open_store(self.db)
        con.close()
        self.calls, self.inflight, self.max_inflight = [], 0, 0
        self.script = None            # callable(text) -> (body, code, headers); None = ok_body
        self.jobs = rt_jobs.Jobs(db_path=self.db, analyse_fn=self.fake_engine)
        self._se = sys.stderr
        sys.stderr = self.err = io.StringIO()
        self.addCleanup(self._restore)

    def _restore(self):
        sys.stderr = self._se
        self.jobs.shutdown()

    def fake_engine(self, text, lang, force, ip, index_fn):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            self.calls.append(text)
            time.sleep(0.005)
            if self.script:
                return self.script(text)
            return ok_body(), 200, {}
        finally:
            self.inflight -= 1

    def add(self, items):
        con = rt_store.open_store(self.db)
        rt_store.upsert_items(con, items)
        con.close()

    def put_analysis(self, item_id, result, version="check_v1"):
        con = sqlite3.connect(self.db)
        con.execute("INSERT OR REPLACE INTO analyses VALUES (?,?,?,?)", (item_id, iso(minutes=1), version, json.dumps(result)))
        con.commit()
        con.close()

    def sql(self, q, *a):
        con = sqlite3.connect(self.db)
        try:
            return con.execute(q, a).fetchall()
        finally:
            con.close()

    def sched(self, **kw):
        kw.setdefault("budget", 10)
        kw.setdefault("sleep_cap", 0)
        s = rt_scheduler.Scheduler(db_path=self.db, enrich=self.enrich, jobs=self.jobs, clock=lambda: NOW,
                                   sync_runs_fn=lambda: 1, **kw)
        s.ensure_table()
        return s

    def write_batch(self, recs):
        news = [r for r in recs if r["id"].startswith("news:")]
        social = [r for r in recs if not r["id"].startswith("news:")]
        write_export(self.export, {"news/p1.jsonl": news, "social/p1.jsonl": social})

    def ids(self, cands):
        return [c[0] for c in cands]


class Selection(Base):
    def test_fnp_signal_items_come_first(self):
        self.add([item(1, hours=1), item(2, hours=2), item(3, hours=3, kind="news"), item(10, "social", hours=1)])
        nid = lambda i: "news:scraped_articles:%s" % i
        self.write_batch([rec(nid(3), p_prop=0.1, nearest_factcheck=dict(id="kb2:a", similarity=0.8)),
                          rec(nid(2), p_prop=0.9), rec("reddit_posts:10")])
        s = self.sched(budget=4)
        ids = self.ids(s.select_candidates())
        self.assertEqual(ids[0], nid(3))              # the fact-check match outranks everything (+3)
        self.assertEqual(ids.index(nid(2)) < ids.index(nid(1)), True)   # p_prop breaks the tie over recency
        self.assertEqual(len(ids), 4)

    def test_low_similarity_gets_no_boost(self):
        self.add([item(1, hours=1), item(3, hours=60)])
        nid = lambda i: "news:scraped_articles:%s" % i
        self.write_batch([rec(nid(3), nearest_factcheck=dict(id="kb2:a", similarity=0.64))])
        ids = self.ids(self.sched().select_candidates())
        self.assertEqual(ids, [nid(1), nid(3)])       # newer first; 0.64 is under the 0.72 threshold

    def test_without_an_export_recency_and_engagement_decide(self):
        a, b = item(1, hours=5), item(2, hours=5)
        b["engagement"] = 100000
        c = item(3, hours=1)
        self.add([a, b, c])
        ids = self.ids(self.sched().select_candidates())
        pos = {i: ids.index("news:scraped_articles:%d" % i) for i in (1, 2, 3)}
        self.assertLess(pos[3], pos[1])                              # newer first
        self.assertLess(pos[2], pos[1])                              # same age: the engagement bonus breaks the tie

    def test_kind_balance_and_budget_cap(self):
        self.add([item(i, "news", hours=1) for i in range(1, 21)] +
                 [item(100 + i, "social", hours=40) for i in range(1, 11)])
        picks = self.sched(budget=10).select_candidates()
        self.assertEqual(len(picks), 10)
        kinds = [k for _, k, _, _ in picks]
        self.assertGreaterEqual(kinds.count("social"), 3)     # 30 % of 10 even though social is much older
        self.assertGreaterEqual(kinds.count("news"), 3)
        self.assertEqual(len(self.sched(budget=3).select_candidates()), 3)

    def test_a_kind_with_few_items_leaves_room(self):
        self.add([item(i, "news") for i in range(1, 9)] + [item(100, "social")])
        kinds = [k for _, k, _, _ in self.sched(budget=6).select_candidates()]
        self.assertEqual((len(kinds), kinds.count("social")), (6, 1))

    def test_already_analysed_short_duplicate_and_old_are_skipped(self):
        dup = item(4)
        dup["is_duplicate"] = 1
        self.add([item(1), item(2), item(3, text="tiny", title="x"), dup, item(5, hours=24 * 9),   # news window is 7 days
                  item(6, "social", hours=24 * 9)])                                                 # social window 30 days
        self.put_analysis("news:scraped_articles:1", deep_result())
        self.put_analysis("news:scraped_articles:2", {"final": "FAKE"})                           # no engine verdict: not deep
        ids = set(self.ids(self.sched().select_candidates()))
        self.assertEqual(ids, {"news:scraped_articles:2", "reddit_posts:6"})

    def test_any_engine_version_counts_as_analysed(self):
        self.add([item(1), item(2)])
        self.put_analysis("news:scraped_articles:1", deep_result(), version="predict_v0")
        self.assertEqual(self.ids(self.sched().select_candidates()), ["news:scraped_articles:2"])

    def test_empty_store_selects_nothing(self):
        self.assertEqual(self.sched().select_candidates(), [])

    def test_priority_function(self):
        base = rt_scheduler.priority(None, None, 10, None)
        self.assertAlmostEqual(rt_scheduler.priority(0.72, None, 10, None) - base, 3.0)
        self.assertEqual(rt_scheduler.priority(0.719, None, 10, None), base)
        self.assertAlmostEqual(rt_scheduler.priority(None, 0.5, 10, None) - base, 0.5)
        self.assertGreater(rt_scheduler.priority(None, None, 1, None), rt_scheduler.priority(None, None, 100, None))
        self.assertLessEqual(rt_scheduler.priority(None, None, 1, 10 ** 12) - rt_scheduler.priority(None, None, 1, None), 0.5)


class Cycle(Base):
    def test_sequential_run_cache_and_log_row(self):
        self.add([item(i, "news", hours=i) for i in range(1, 5)] + [item(20, "social")])
        s = self.sched(budget=10)
        rec_ = s.run_cycle(next_run_in=10800)
        self.assertEqual((rec_["picked"], rec_["analysed"], rec_["errors"], rec_["degraded"]), (5, 5, 0, 0))
        self.assertEqual(rec_["stop_reason"], "done")
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(self.max_inflight, 1)                       # strictly one engine call at a time
        self.assertEqual(self.sql("SELECT count(*) FROM analyses")[0][0], 5)
        rows = self.sql("SELECT started_at, ended_at, picked, analysed, cached_dupes, degraded, errors, skipped, "
                        "stop_reason, next_run_at FROM schedule_log")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2:9], (5, 5, 0, 0, 0, 0, "done"))
        self.assertEqual(rows[0][9], rt_store.utc_iso(NOW.replace(microsecond=0) + __import__("datetime").timedelta(seconds=10800)))
        # a second cycle finds nothing left and logs that
        r2 = s.run_cycle()
        self.assertEqual((r2["picked"], r2["stop_reason"]), (0, "nothing to analyse"))
        self.assertEqual(len(self.calls), 5)

    def test_logs_never_contain_item_text(self):
        self.add([item(1, title="SECRETTITLE zebra")])
        self.sched().run_cycle()
        out = self.err.getvalue()
        self.assertIn("news:scraped_articles:1", out)
        for bad in ("SECRETTITLE", "zebra", "river valley"):
            self.assertNotIn(bad, out)
        for row in self.sql("SELECT * FROM schedule_log"):
            self.assertFalse(any(isinstance(v, str) and ("SECRETTITLE" in v or "river" in v) for v in row))

    def test_time_cap(self):
        self.add([item(i, hours=i) for i in range(1, 9)])
        clock = {"t": 0.0}

        def script(text):
            clock["t"] += 100.0
            return ok_body(), 200, {}
        self.script = script
        s = rt_scheduler.Scheduler(db_path=self.db, enrich=self.enrich, jobs=self.jobs, clock=lambda: NOW, budget=8,
                                   max_secs=250, monotonic=lambda: clock["t"], sync_runs_fn=lambda: 1)
        s.ensure_table()
        r = s.run_cycle()
        self.assertEqual((r["analysed"], len(self.calls)), (3, 3))
        self.assertTrue(r["stop_reason"].startswith("time cap"))
        self.assertEqual(r["picked"], 8)

    def test_short_text_is_skipped_by_the_job_path_not_counted_as_failure(self):
        # 41 characters in the store, but the engine text (title + excerpt) is what compose_text yields
        self.add([item(1)])
        with mock.patch.object(rt_jobs, "MIN_TEXT_CHARS", 10 ** 6):
            r = self.sched().run_cycle()
        self.assertEqual((r["skipped"], r["errors"], r["analysed"]), (1, 0, 1 - 1))

    def test_near_duplicates_reuse_the_result_and_never_rerun(self):
        h = simhash64("a", LONG + "dup")
        near = simhash64("a", LONG + "dup") ^ 0b101                   # 2 bits away
        a, b, c = item(1, simhash=h), item(2, simhash=near, hours=2), item(3, hours=3)
        d = item(4, hours=4, simhash=simhash64("zzz", "completely different words " * 6))
        self.add([a, b, c, d])
        self.put_analysis("news:scraped_articles:4", deep_result(label="LEGIT", fake=0.05, mis=0.05))
        s = self.sched()
        r = s.run_cycle()
        # a (newest) is analysed, b is its near-duplicate: copied, not run; c runs
        self.assertEqual((r["analysed"], r["cached_dupes"], len(self.calls)), (2, 1, 2))
        rows = dict(self.sql("SELECT item_id, result_json FROM analyses"))
        self.assertEqual(json.loads(rows["news:scraped_articles:2"])["dup_of"], "news:scraped_articles:1")
        self.assertNotIn("dup_of", json.loads(rows["news:scraped_articles:1"]))

    def test_duplicate_of_an_already_analysed_item_is_not_run(self):
        h = simhash64("a", LONG + "old")
        self.add([item(1, simhash=h, hours=5), item(2, simhash=h, hours=1)])
        self.put_analysis("news:scraped_articles:1", deep_result(label="FAKE"))
        r = self.sched().run_cycle()
        self.assertEqual((r["analysed"], r["cached_dupes"], len(self.calls)), (0, 1, 0))
        self.assertEqual(self.sql("SELECT json_extract(result_json,'$.dup_of') FROM analyses WHERE item_id = ?",
                                  "news:scraped_articles:2")[0][0], "news:scraped_articles:1")

    def test_a_crashing_engine_or_selection_never_kills_the_scheduler(self):
        self.add([item(i) for i in range(1, 6)])

        def boom(text):
            raise RuntimeError("engine exploded with SECRET text")
        self.script = boom
        s = self.sched()
        r = s.run_cycle()
        self.assertEqual((r["errors"], len(self.calls)), (5, 5))     # each item caught; 5 replies < the 6 the stop rule needs
        self.assertNotIn("SECRET", self.err.getvalue())
        with mock.patch.object(s, "select_candidates", side_effect=sqlite3.OperationalError("locked")):
            r = s.run_cycle()
        self.assertTrue(r["stop_reason"].startswith("selection failed"))
        with mock.patch.object(s, "_execute", side_effect=ValueError("bug")):
            r = s.run_cycle()
        self.assertTrue(r["stop_reason"].startswith("cycle error"))
        self.script = None
        self.assertEqual(s.run_cycle()["analysed"], 5)               # still alive; nothing was cached by the failures
        self.assertEqual(self.sql("SELECT count(*) FROM schedule_log")[0][0], 4)

    def test_no_overlap(self):
        self.add([item(i) for i in range(1, 4)])
        gate, started = threading.Event(), threading.Event()

        def slow(text):
            started.set()
            gate.wait(5)
            return ok_body(), 200, {}
        self.script = slow
        s = self.sched()
        res = {}
        t = threading.Thread(target=lambda: res.update(r=s.run_cycle()))
        t.start()
        self.assertTrue(started.wait(5))
        self.assertTrue(s.info()["running_now"])
        self.assertIsNone(s.run_cycle())                              # a second cycle is refused, not queued
        self.assertEqual(rt_scheduler.schedule_status(self.db)["running_now"], False)   # module status: no singleton
        gate.set()
        t.join(10)
        self.assertEqual(res["r"]["analysed"], 3)
        self.assertEqual(self.max_inflight, 1)
        self.assertEqual(self.sql("SELECT count(*) FROM schedule_log")[0][0], 1)
        self.assertFalse(s.info()["running_now"])

    def test_stop_ends_a_running_cycle_early(self):
        self.add([item(i) for i in range(1, 30)])
        s = self.sched(budget=30)

        def script(text):
            if len(self.calls) == 2:
                s._stop.set()
            return ok_body(), 200, {}
        self.script = script
        r = s.run_cycle()
        self.assertEqual(r["stop_reason"], "server stopping")
        self.assertLessEqual(len(self.calls), 3)


class Loop(Base):
    def setUp(self):
        super().setUp()
        self.add([item(1)])
        p = mock.patch.object(rt_scheduler, "READY_RECHECK_S", 0.05)
        p.start()
        self.addCleanup(p.stop)

    def wait_rows(self, n=1, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.sql("SELECT count(*) FROM schedule_log WHERE ended_at IS NOT NULL")[0][0] >= n:
                return True
            time.sleep(0.02)
        return False

    def test_first_delay_then_waits_for_a_sync_or_fresh_data(self):
        # store data is 1 hour old relative to NOW but the scheduler clock is NOW: make it stale (> 24 h)
        con = sqlite3.connect(self.db)
        con.execute("UPDATE items SET published_at = ?", (iso(hours=48),))
        con.commit()
        con.close()
        runs = {"n": 0}
        s = rt_scheduler.Scheduler(db_path=self.db, enrich=self.enrich, jobs=self.jobs, clock=lambda: NOW, budget=5,
                                   first_delay_s=0.3, interval_s=3600, sync_runs_fn=lambda: runs["n"])
        t0 = time.monotonic()
        s.start()
        self.addCleanup(s.stop)
        time.sleep(0.15)
        self.assertFalse(self.calls)                                  # inside the first delay: nothing yet
        self.assertEqual(s.info()["waiting"], "first delay")
        self.assertIsNotNone(s.info()["next_run_at"])
        time.sleep(0.5)
        self.assertFalse(self.calls)                                  # delay over, but no sync cycle and stale data
        self.assertEqual(s.info()["waiting"], "waiting for the first sync")
        runs["n"] = 1                                                 # the realtime sync completed a cycle
        self.assertTrue(self.wait_rows(1))
        self.assertGreaterEqual(time.monotonic() - t0, 0.3)
        self.assertEqual(len(self.calls), 1)

    def test_fresh_store_data_is_enough(self):
        s = rt_scheduler.Scheduler(db_path=self.db, enrich=self.enrich, jobs=self.jobs, clock=lambda: NOW,
                                   first_delay_s=0, interval_s=3600, sync_runs_fn=lambda: 0)
        self.assertTrue(s.ready())                                    # item newer than 24 h
        s.start()
        self.addCleanup(s.stop)
        self.assertTrue(self.wait_rows(1))

    def test_interval_and_clean_stop(self):
        s = rt_scheduler.Scheduler(db_path=self.db, enrich=self.enrich, jobs=self.jobs, clock=lambda: NOW,
                                   first_delay_s=0, interval_s=3600, sync_runs_fn=lambda: 1)
        s.start()
        self.assertTrue(self.wait_rows(1))
        time.sleep(0.1)
        info = s.info()
        self.assertTrue(info["running"])
        self.assertEqual(info["next_run_at"], rt_store.utc_iso(NOW + __import__("datetime").timedelta(seconds=3600)))
        t0 = time.monotonic()
        s.stop(5)
        self.assertLess(time.monotonic() - t0, 3)
        self.assertFalse(s.is_running())
        self.assertEqual(self.sql("SELECT count(*) FROM schedule_log")[0][0], 1)   # no second cycle

    def test_open_rows_of_a_dead_process_are_closed_as_interrupted(self):
        con = sqlite3.connect(self.db)
        con.execute("INSERT INTO schedule_log(started_at) VALUES (?)", (iso(hours=5),))
        con.commit()
        con.close()
        st = rt_scheduler.schedule_status(self.db)
        self.assertEqual(st["last_run"]["stop_reason"], "interrupted")
        self.sched()                                                  # ensure_table closes it for good
        self.assertEqual(self.sql("SELECT stop_reason FROM schedule_log")[0][0], "interrupted")


class Config(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(rt_scheduler.enabled_from_env({}))
        for v in ("0", "true", "yes", ""):
            self.assertFalse(rt_scheduler.enabled_from_env({"FNP_RT_ANALYSE": v}), v)
        self.assertTrue(rt_scheduler.enabled_from_env({"FNP_RT_ANALYSE": "1"}))
        self.assertFalse(server.maybe_start_analysis({}))
        self.assertFalse(rt_scheduler.schedule_status("/nonexistent/x.sqlite", env={})["enabled"])

    def test_defaults_and_bounds(self):
        c = rt_scheduler.config_from_env({})
        self.assertEqual((c["interval_s"], c["budget"], c["max_secs"], c["first_delay_s"]), (10800.0, 40, 5400.0, 600.0))
        c = rt_scheduler.config_from_env({"FNP_RT_ANALYSE_INTERVAL": "60", "FNP_RT_ANALYSE_BUDGET": "7",
                                          "FNP_RT_ANALYSE_MAX_SECS": "99", "FNP_RT_ANALYSE_FIRST_DELAY": "5"})
        self.assertEqual((c["interval_s"], c["budget"], c["max_secs"], c["first_delay_s"]), (60.0, 7, 99.0, 5.0))
        c = rt_scheduler.config_from_env({"FNP_RT_ANALYSE_INTERVAL": "1", "FNP_RT_ANALYSE_BUDGET": "100000",
                                          "FNP_RT_ANALYSE_FIRST_DELAY": "abc"})
        self.assertEqual((c["interval_s"], c["budget"], c["first_delay_s"]), (30.0, rt_scheduler.MAX_BUDGET, 600.0))

    def test_env_enables_the_thread_and_stop_background_ends_it(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = os.path.join(tmp, "r.sqlite")
        rt_store.open_store(db).close()
        env = {"FNP_RT_ANALYSE": "1", "FNP_RT_ANALYSE_FIRST_DELAY": "3600"}
        sch = rt_scheduler.start_background(db_path=db, env=env)
        try:
            time.sleep(0.1)
            self.assertIs(rt_scheduler.start_background(db_path=db, env=env), sch)   # once
            st = rt_scheduler.schedule_status(db, env=env)
            self.assertTrue(st["enabled"])
            self.assertEqual(st["state"], "waiting_first_delay")
            self.assertIsNotNone(st["next_run_at"])
        finally:
            rt_scheduler.stop_background()
        self.assertFalse(sch.is_running())
        self.assertFalse(rt_scheduler.schedule_status(db, env={})["enabled"])


class DeepStats(Fixture):
    """The batch export (default_export) plus stored deep results: two coverage numbers, one verdict per item."""

    def put(self, item_id, result, version="check_v1"):
        con = sqlite3.connect(self.db)
        con.execute("INSERT OR REPLACE INTO analyses VALUES (?,?,?,?)", (item_id, iso(minutes=2), version, json.dumps(result)))
        con.commit()
        con.close()
        self.api._cache.clear()

    def stats(self, **kw):
        return self.api.stats(self.q(**kw))

    def test_batch_only_matches_the_old_numbers(self):
        default_export(self.export)
        s = self.stats()
        cov = s["coverage"]
        self.assertEqual((cov["analysed"], cov["batch"]["analysed"], cov["deep"]["analysed"], cov["both"]), (8, 8, 0, 0))
        self.assertEqual((cov["batch"]["items_in_scope"], cov["deep"]["items_in_scope"]), (12, 12))
        self.assertEqual(cov["batch"]["pct"], round(100 * 8 / 12, 1))
        self.assertEqual(cov["deep"]["pct"], 0.0)
        lf = s["likely_fake_misleading"]
        self.assertEqual((lf["count"], lf["denominator"]), (3, 4))
        self.assertEqual(lf["by_source"], dict(batch=dict(count=3, denominator=4), deep=dict(count=0, denominator=0)))
        self.assertEqual(s["p_prop_distribution"]["n"], 8)

    def test_bogus_stored_rows_are_not_deep_verdicts(self):
        default_export(self.export)
        # the fixture already stores {"final": "FAKE"} for news 1: that is not a deep verdict
        cov = self.stats()["coverage"]
        self.assertEqual(cov["deep"]["analysed"], 0)
        self.put("news:scraped_articles:5", {"engine": {"state": "ok"}})     # engine but no verdict
        self.put("telegram_posts:13", "not even an object")
        self.assertEqual(self.stats()["coverage"]["deep"]["analysed"], 0)

    def test_deep_only(self):
        self.put("news:scraped_articles:5", deep_result(0.8, 0.1, techs=(("Doubt", 0.7, "x y"), ("Slogans", 0.5, "z"))))
        self.put("telegram_posts:13", deep_result(0.05, 0.05, label="LEGIT", techs=()))
        s = self.stats()
        cov = s["coverage"]
        self.assertEqual((cov["analysed"], cov["batch"]["analysed"], cov["deep"]["analysed"], cov["both"], cov["deep_only"]),
                         (2, 0, 2, 0, 2))
        self.assertEqual(cov["with_verdict"], 2)
        self.assertEqual(cov["verdicts"], {"FAKE": 1, "LEGIT": 1})
        lf = s["likely_fake_misleading"]
        self.assertEqual((lf["count"], lf["denominator"], lf["share"]), (1, 2, 0.5))   # 0.8+0.1 >= 0.5; 0.05+0.05 < 0.5
        self.assertEqual(lf["by_source"]["deep"], dict(count=1, denominator=2))
        self.assertEqual(cov["by_kind"]["news"]["deep"], 1)
        # technique mix from the deep technique classes; p_prop distribution has no deep-only items in it
        mix = {t["technique"]: t["items"] for t in s["technique_mix"]["top"]}
        self.assertEqual(mix, {"Doubt": 1, "Slogans": 1})
        self.assertEqual(s["technique_mix"]["basis"], "batch techniques, else deep")
        self.assertEqual((s["p_prop_distribution"]["n"], s["p_prop_distribution"]["deep_only_without_p_prop"]), (0, 2))
        self.assertEqual(s["p_prop_distribution"]["basis"], "batch p_prop")

    def test_both_sources_count_once_and_deep_wins(self):
        default_export(self.export)
        # news 3 is LEGIT in the batch (not flagged); the deep result says FAKE. reddit 14 has a batch record, no verdict.
        self.put("news:scraped_articles:3", deep_result(0.9, 0.05))
        self.put("reddit_posts:14", deep_result(0.1, 0.1, label="LEGIT", techs=(("Doubt", 0.9, "q"),)))
        self.put("news:scraped_articles:5", deep_result(0.6, 0.2, techs=()))                 # deep only
        s = self.stats()
        cov = s["coverage"]
        self.assertEqual((cov["batch"]["analysed"], cov["deep"]["analysed"], cov["both"], cov["deep_only"]), (8, 3, 2, 1))
        self.assertEqual(cov["analysed"], 9)                                       # 8 batch + 1 deep-only, not 11
        # verdict items: batch verdicts of news1 FAKE, news2 MIS, reddit10 FAKE (news3's batch LEGIT is replaced by deep)
        # + deep: news3 FAKE, reddit14 LEGIT, news5 FAKE  => 6 items, each once
        self.assertEqual(cov["with_verdict"], 6)
        self.assertEqual(cov["verdicts"], {"FAKE": 4, "MISLEADING": 1, "LEGIT": 1})
        lf = s["likely_fake_misleading"]
        self.assertEqual((lf["count"], lf["denominator"]), (5, 6))
        self.assertEqual(lf["by_source"], dict(batch=dict(count=3, denominator=3), deep=dict(count=2, denominator=3)))
        # technique mix: batch lists where a batch record exists (reddit 14 has none, so its deep "Doubt" is NOT added),
        # deep-only news5 has no techniques here
        mix = {t["technique"]: t["items"] for t in s["technique_mix"]["top"]}
        self.assertNotIn("Doubt", mix)
        self.assertEqual(mix["loaded_language"], 3)                                # unchanged batch numbers
        # p_prop distribution is batch p_prop only
        self.assertEqual(s["p_prop_distribution"]["n"], 8)

    def test_scope_filters_apply_to_deep_too(self):
        self.put("news:scraped_articles:5", deep_result())
        self.put("telegram_posts:13", deep_result())
        self.assertEqual(self.stats(kind="news")["coverage"]["deep"]["analysed"], 1)
        self.assertEqual(self.stats(kind="social")["coverage"]["deep"]["analysed"], 1)
        self.assertEqual(self.stats(kind="social", platform="telegram")["coverage"]["deep"]["analysed"], 1)
        self.assertEqual(self.stats(kind="social", platform="reddit")["coverage"]["deep"]["analysed"], 0)
        self.assertEqual(self.stats(days=2)["coverage"]["deep"]["analysed"], 1)       # news 5 is 100 h old, telegram 13 is 26 h

    def test_newest_engine_version_wins_and_counts_once(self):
        self.put("news:scraped_articles:5", deep_result(0.1, 0.1, label="LEGIT"), version="predict_v0")
        con = sqlite3.connect(self.db)
        con.execute("INSERT INTO analyses VALUES (?,?,?,?)", ("news:scraped_articles:5", iso(minutes=1), "check_v1",
                                                              json.dumps(deep_result(0.9, 0.05))))
        con.commit()
        con.close()
        self.api._cache.clear()
        s = self.stats()
        self.assertEqual(s["coverage"]["deep"]["analysed"], 1)
        self.assertEqual(s["coverage"]["verdicts"], {"FAKE": 1})   # created 1 minute ago beats 2 minutes ago

    def test_analysed_filter_includes_either_source(self):
        default_export(self.export)
        self.put("news:scraped_articles:5", deep_result())          # deep only
        self.put("news:scraped_articles:1", deep_result())          # both
        yes = {i["id"] for i in self.api.items(self.q(analysed="yes", size=50))["items"]}
        no = {i["id"] for i in self.api.items(self.q(analysed="no", size=50))["items"]}
        self.assertIn("news:scraped_articles:5", yes)
        self.assertEqual(len(yes), 9)
        self.assertEqual(len(no), 3)
        self.assertFalse(yes & no)
        self.assertEqual(self.api.items(self.q(analysed="yes"))["total"], 9)
        self.assertEqual(self.api.items(self.q(analysed="no"))["total"], 3)

    def test_items_and_case_file_expose_the_source(self):
        default_export(self.export)
        self.put("news:scraped_articles:5", deep_result(0.8, 0.1))
        self.put("news:scraped_articles:1", deep_result(0.2, 0.1, label="LEGIT"))
        by = {i["id"]: i for i in self.api.items(self.q(size=50))["items"]}
        deep_only, both, batch_only, none = (by["news:scraped_articles:5"]["inferred"], by["news:scraped_articles:1"]["inferred"],
                                             by["news:scraped_articles:2"]["inferred"], by["telegram_posts:13"]["inferred"])
        self.assertEqual((deep_only["source"], deep_only["sources"]), ("deep", ["deep"]))
        self.assertEqual(deep_only["deep"]["verdict"], "FAKE")
        self.assertAlmostEqual(deep_only["deep"]["p_fake_misleading"], 0.9)
        self.assertIsNone(deep_only["batch"])
        self.assertEqual((both["source"], both["sources"]), ("deep", ["batch", "deep"]))
        self.assertEqual(both["verdict"], "LEGIT")                      # deep preferred at the top level
        self.assertEqual(both["batch"]["verdict"], "FAKE")              # the batch verdict is still shown, apart
        self.assertEqual(both["p_prop"], 0.9)                           # p_prop stays the batch one
        self.assertEqual((batch_only["source"], batch_only["sources"], batch_only["deep"]), ("batch", ["batch"], None))
        self.assertIsNone(none)
        d = self.api.item("news:scraped_articles:5")
        self.assertEqual((d["inferred"]["source"], d["inferred"]["sources"]), ("deep", ["deep"]))
        self.assertEqual([t["technique"] for t in d["inferred"]["techniques"]], ["Doubt"])
        self.assertEqual(d["inferred"]["spans"][0]["text"], "some words")
        self.assertEqual(d["inferred"]["deep"]["engine_version"], "check_v1")
        d1 = self.api.item("news:scraped_articles:1")                   # both: batch techniques kept, no deep ones merged
        self.assertEqual(d1["inferred"]["sources"], ["batch", "deep"])
        self.assertEqual({t["technique"] for t in d1["inferred"]["techniques"]}, {"loaded_language", "repetition"})
        self.assertEqual(self.api.item("telegram_posts:13")["inferred"], None)

    def test_dup_copies_are_marked(self):
        self.put("news:scraped_articles:5", deep_result(dup_of="news:scraped_articles:2"))
        inf = self.api.item("news:scraped_articles:5")["inferred"]
        self.assertEqual(inf["deep"]["dup_of"], "news:scraped_articles:2")
        row = [i for i in self.api.items(self.q(size=50))["items"] if i["id"] == "news:scraped_articles:5"][0]
        self.assertEqual(row["inferred"]["deep"]["dup_of"], "news:scraped_articles:2")

    def test_sort_by_p_fake_uses_the_deep_score_when_present(self):
        default_export(self.export)
        self.put("news:scraped_articles:5", deep_result(0.99, 0.0))
        first = self.api.items(self.q(sort="p_fake", size=3))["items"]
        self.assertEqual(first[0]["id"], "news:scraped_articles:5")

    def test_strict_json(self):
        default_export(self.export)
        self.put("news:scraped_articles:5", deep_result())
        for obj in (self.stats(), self.api.items(self.q(size=50)), self.api.item("news:scraped_articles:5"),
                    self.api.schedule()):
            json.dumps(obj, allow_nan=False)


class ScheduleApi(Fixture):
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

    def log_cycles(self, n):
        con = rt_store.open_store(self.db)
        for k in range(n):
            con.execute("INSERT INTO schedule_log(started_at, ended_at, picked, analysed, cached_dupes, degraded, errors, "
                        "skipped, stop_reason, next_run_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (iso(hours=30 - k), iso(hours=30 - k, minutes=-20), 10, 8 - k % 3, 1, 0, 1, 0, "done", iso(hours=-3)))
        con.close()

    def test_gate_matrix(self):
        st, hd, b = self.get("/api/rt/schedule")
        self.assertEqual((st, b["error"]), (403, "login_required"))
        self.assertEqual(hd["cache-control"], "no-store")
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            st, hd, b = self.get("/api/rt/schedule")
            self.assertEqual(st, 200)
            self.assertEqual(hd["cache-control"], "no-store")
        for val in ("0", "true", ""):
            with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": val}):
                self.assertEqual(self.get("/api/rt/schedule")[0], 403, val)
        server.AUTH = (b"alice", b"pw")
        self.assertEqual(self.get("/api/rt/schedule")[0], 401)
        ok = {"Authorization": "Basic YWxpY2U6cHc="}
        self.assertEqual(self.get("/api/rt/schedule", ok)[0], 200)
        self.assertEqual(jcall(self.port, "POST", "/api/rt/schedule", body={}, headers=ok)[0], 404)  # read-only route
        server.AUTH = None

    def test_rate_limit_shared_with_the_other_rt_routes(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}), mock.patch.object(server, "RT_RATE", 3):
            self.assertEqual([self.get("/api/rt/schedule")[0] for _ in range(4)], [200, 200, 200, 429])

    def test_shape_off_and_strict_json(self):
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            st, hd, raw = call(self.port, "GET", "/api/rt/schedule")
            b = json.loads(raw.decode(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        self.assertEqual(st, 200)
        for k in ("enabled", "running_now", "state", "interval_s", "budget", "max_secs", "first_delay_s", "last_run",
                  "next_run_at", "totals", "log", "generated_at"):
            self.assertIn(k, b)
        self.assertIs(b["enabled"], False)
        self.assertEqual(b["state"], "off")
        self.assertIsNone(b["next_run_at"])
        self.assertEqual((b["interval_s"], b["budget"]), (10800.0, 40))
        self.assertEqual(b["totals"], dict(analysed_deep=0, partial_items=0, cycles=0, scheduled_analysed=0))
        self.assertEqual((b["last_run"], b["log"]), (None, []))

    def test_log_rows_last_ten_and_totals(self):
        self.log_cycles(13)
        con = sqlite3.connect(self.db)
        con.execute("INSERT OR REPLACE INTO analyses VALUES (?,?,?,?)", ("news:scraped_articles:5", iso(minutes=1), "check_v1",
                                                                         json.dumps(deep_result())))
        con.commit()
        con.close()
        with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"}):
            st, hd, b = self.get("/api/rt/schedule")
        self.assertEqual(len(b["log"]), 10)
        self.assertEqual(b["log"][0], b["last_run"])
        self.assertGreater(b["log"][0]["started_at"], b["log"][1]["started_at"])      # newest first
        self.assertEqual(b["totals"]["cycles"], 13)
        self.assertEqual(b["totals"]["analysed_deep"], 1)
        self.assertEqual(set(b["last_run"]), {"started_at", "ended_at", "picked", "analysed", "cached_dupes", "degraded",
                                              "errors", "skipped", "stop_reason", "next_run_at", "duration_s", "partial",
                                              "retried", "median_secs"})
        self.assertEqual(b["last_run"]["duration_s"], 1200)

    @needs_index

    def test_enabled_state_and_healthz_block(self):
        env = {"FNP_RT_ANALYSE": "1", "FNP_RT_ALLOW_OPEN": "1", "FNP_RT_ANALYSE_FIRST_DELAY": "3600"}
        with mock.patch.dict(os.environ, env):
            sch = rt_scheduler.start_background(db_path=self.db, enrich=self.enrich, jobs=rt_jobs.Jobs(db_path=self.db))
            try:
                time.sleep(0.1)
                st, hd, b = self.get("/api/rt/schedule")
                self.assertEqual((st, b["enabled"], b["state"]), (200, True, "waiting_first_delay"))
                self.assertIsNotNone(b["next_run_at"])
                st, hd, h = self.get("/healthz")
                self.assertEqual(st, 200)
                a = h["realtime"]["analysis"]
                self.assertEqual(set(a), {"enabled", "last_run", "next_run_at", "analysed_deep", "partial_items",
                                    "rolling"})
                self.assertTrue(a["enabled"])
                self.assertEqual(a["next_run_at"], b["next_run_at"])
            finally:
                rt_scheduler.stop_background()

    @needs_index

    def test_healthz_analysis_block_never_fails_the_probe(self):
        with mock.patch.object(rt_scheduler, "schedule_status", side_effect=RuntimeError("boom")):
            self.api._cache.clear()
            st, hd, h = self.get("/healthz")
        self.assertEqual(st, 200)
        self.assertEqual(h["realtime"]["analysis"], dict(enabled=False, last_run=None, next_run_at=None, analysed_deep=0,
                                                              partial_items=0, rolling=None))
        os.unlink(self.db)
        self.api._cache.clear()
        st, hd, h = self.get("/healthz")
        self.assertEqual(st, 200)
        self.assertIn("analysis", h["realtime"])


class RealEnginePath(unittest.TestCase):
    """One cycle through claimcheck.analyse against tests/fake_engine.py (no python stub): the whole Claim Check path."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = os.path.join(self.tmp, "r.sqlite")
        con = rt_store.open_store(self.db)
        rt_store.upsert_items(con, [item(1), item(2, "social", hours=2)])
        con.close()
        claimcheck.reset_state()
        self._cc = (claimcheck.ENGINE_URL, claimcheck.ENGINE_TIMEOUT, claimcheck.QUEUE_WAIT)
        self._se = sys.stderr
        sys.stderr = io.StringIO()
        self.addCleanup(self._restore)

    def _restore(self):
        sys.stderr = self._se
        claimcheck.ENGINE_URL, claimcheck.ENGINE_TIMEOUT, claimcheck.QUEUE_WAIT = self._cc
        claimcheck.reset_state()

    def test_cycle_through_check_v1_stores_deep_results(self):
        with fake_engine.FakeEngine(make=lambda t: fake_engine.v1_response(), path="/api/check_v1") as e:
            claimcheck.ENGINE_URL = e.url
            jobs = rt_jobs.Jobs(db_path=self.db)
            self.addCleanup(jobs.shutdown)
            s = rt_scheduler.Scheduler(db_path=self.db, jobs=jobs, clock=lambda: NOW, budget=5, sync_runs_fn=lambda: 1)
            r = s.run_cycle()
            self.assertEqual((r["analysed"], e.count), (2, 2))
            r2 = s.run_cycle()
            self.assertEqual((r2["picked"], e.count), (0, 2))              # cached: never run twice
        api = rt_api.RtApi(db_path=self.db, enrich=rt_enrich.Enrich(export_dir="/nonexistent", triage_dir="/nonexistent",
                                                                    sidecar=os.path.join(self.tmp, "e.sqlite"),
                                                                    store_path=self.db), clock=lambda: NOW)
        st = api.stats({})
        self.assertEqual((st["coverage"]["deep"]["analysed"], st["coverage"]["batch"]["analysed"]), (2, 0))
        self.assertEqual(st["coverage"]["verdicts"], {"FAKE": 2})

    def test_degraded_check_v1_is_not_cached(self):
        with fake_engine.FakeEngine(make=lambda t: fake_engine.v1_response(degraded=True), path="/api/check_v1") as e:
            claimcheck.ENGINE_URL = e.url
            jobs = rt_jobs.Jobs(db_path=self.db)
            self.addCleanup(jobs.shutdown)
            s = rt_scheduler.Scheduler(db_path=self.db, jobs=jobs, clock=lambda: NOW, budget=5, sync_runs_fn=lambda: 1)
            r = s.run_cycle()
        self.assertEqual((r["degraded"], r["analysed"]), (2, 0))
        con = sqlite3.connect(self.db)   # kept only as PARTIAL rows: never a normal (cacheable, verdict-bearing) result
        self.assertEqual(con.execute("SELECT count(*), sum(json_extract(result_json, '$.partial') = 1), "
                                     "sum(json_extract(result_json, '$.engine.verdict.label') IS NOT NULL) FROM analyses"
                                     ).fetchone(), (2, 2, 0))
        con.close()


if __name__ == "__main__":
    unittest.main()
