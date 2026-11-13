"""S2: the deep scheduler made useful. Partial results (degraded replies kept as propaganda-only rows) and their retries,
the rolling stop rule, a single Retry-After wait, adaptive pacing, priority from the FNP export, kind balance, no_claims as
a complete result, the schedule_log migration. Offline: temp store, fake engine functions / tests/fake_engine.py."""
import json
import os
import sqlite3
import time
import unittest
from datetime import timedelta
from unittest import mock

import helpers  # noqa: F401  (sys.path)
import claimcheck
import fake_engine
import rt_api
import rt_enrich
import rt_jobs
import rt_scheduler
import rt_store
from rt_fakes import NOW
from test_rt_real_export import real_row, write_real
from test_rt_scheduler import Base, item, ok_body, busy_body, deep_result
from test_rt_api import iso


def nid(i):
    return "news:scraped_articles:%s" % i


def partial_body(why="timeout"):
    """What claimcheck returns for a degraded check_v1 reply: no verdict, propaganda step present."""
    return ({"engine": {"state": "degraded", "degraded": True, "degraded_why": why, "verdict": None, "fake_call": None,
                        "reason": "the deep verdict timed out"},
             "step1": {"source": "model estimate", "lead": "engine", "label": "FAKE", "plausibility": 0.8, "classes": [{"x": 1}]},
             "step2": {"shown": True, "propaganda": {"techniques": [{"technique": "Doubt", "start": 0, "end": 3, "text": "The",
                                                                      "confidence": 0.7}],
                                                     "classes": [{"technique": "Doubt", "p": 0.7}],
                                                     "technique_probs": None}}}, 200, {})


class S2Base(Base):
    def setUp(self):
        super().setUp()
        self.now = NOW
        self.jobs = rt_jobs.Jobs(db_path=self.db, analyse_fn=self.fake_engine, clock=lambda: self.now)
        self.waits = []

    def sched2(self, **kw):
        kw.setdefault("budget", 10)
        kw.setdefault("sleep_cap", rt_scheduler.BUSY_WAIT_CAP)
        kw.setdefault("wait_fn", lambda s: self.waits.append(s) or False)
        s = rt_scheduler.Scheduler(db_path=self.db, enrich=self.enrich, jobs=self.jobs, clock=lambda: self.now,
                                   sync_runs_fn=lambda: 1, **kw)
        s.ensure_table()
        return s

    def analyses(self):
        return {r[0]: json.loads(r[1]) for r in self.sql("SELECT item_id, result_json FROM analyses")}


class Partial(S2Base):
    def test_degraded_reply_is_stored_as_partial_never_a_verdict(self):
        self.add([item(1)])
        self.script = lambda t: partial_body()
        r = self.sched2().run_cycle()
        self.assertEqual((r["analysed"], r["degraded"], r["partial"], r["retried"]), (0, 1, 1, 0))
        res = self.analyses()[nid(1)]
        self.assertIs(res["partial"], True)
        self.assertEqual(res["partial_info"]["retries"], 0)
        self.assertEqual(res["partial_info"]["reason"], "timeout")
        self.assertIsNone(res["engine"]["verdict"])
        self.assertIsNone(res["step1"]["label"])                 # the engine's lead label is dropped too
        self.assertTrue(res["step2"]["propaganda"]["classes"])   # the propaganda part is kept

    def test_degraded_without_propaganda_is_not_stored(self):
        self.add([item(1)])
        b = partial_body()[0]
        b["step2"] = {"shown": False, "propaganda": None}
        self.script = lambda t: (b, 200, {})
        r = self.sched2().run_cycle()
        self.assertEqual((r["degraded"], r["partial"]), (1, 0))
        self.assertEqual(self.sql("SELECT count(*) FROM analyses")[0][0], 0)

    def test_partial_is_not_a_cache_hit_and_retry_spacing_limit(self):
        self.add([item(1)])
        self.script = lambda t: partial_body()
        s = self.sched2()
        s.run_cycle()
        self.assertEqual(len(self.calls), 1)
        # same moment: not due again (needs one cycle between tries)
        self.assertEqual(s.select_retries(), [])
        self.assertEqual(s.select_candidates(), [])              # partial items are not "fresh" either
        self.assertEqual(s.run_cycle()["stop_reason"], "nothing to analyse")
        retries_seen = []
        for k in range(1, 5):
            self.now = self.now + timedelta(seconds=s.interval_s + 5)
            due = s.select_retries()
            retries_seen.append(len(due))
            r = s.run_cycle()
            if due:
                self.assertEqual((r["retried"], r["partial"], r["degraded"]), (1, 1, 1))
        self.assertEqual(retries_seen, [1, 1, 1, 0])              # 3 retries, then never again
        self.assertEqual(self.analyses()[nid(1)]["partial_info"]["retries"], 3)
        self.assertEqual(len(self.calls), 4)                      # first try + 3 retries

    def test_retry_that_succeeds_becomes_a_verdict_and_runs_after_fresh_items(self):
        self.add([item(1, hours=40), item(2, hours=2), item(3, hours=3)])
        order = []
        self.script = lambda t: partial_body() if "Unique marker 1 " in t else ok_body() and (order.append(t), (ok_body(), 200, {}))[1]
        s = self.sched2()
        s.run_cycle()
        self.assertEqual(set(self.analyses()), {nid(1), nid(2), nid(3)})
        self.assertTrue(self.analyses()[nid(1)].get("partial"))
        self.now = self.now + timedelta(seconds=s.interval_s + 5)
        self.add([item(9, hours=1)])                              # a fresh item must go before the retry
        calls_before = len(self.calls)
        self.script = None
        r = s.run_cycle()
        self.assertEqual((r["picked"], r["analysed"], r["retried"]), (2, 2, 1))
        self.assertIn("Unique marker 9", self.calls[calls_before])       # fresh first
        self.assertIn("Unique marker 1 ", self.calls[calls_before + 1])  # retry after
        self.assertNotIn("partial", self.analyses()[nid(1)])
        self.assertIsNotNone(self.analyses()[nid(1)]["engine"]["verdict"])

    def test_retries_take_at_most_a_quarter_of_the_budget(self):
        self.add([item(i, hours=i) for i in range(1, 9)])
        self.script = lambda t: partial_body()
        s = self.sched2(budget=8)
        s.run_cycle()
        self.now = self.now + timedelta(seconds=s.interval_s + 5)
        self.assertEqual(len(s.select_retries()), 2)              # ceil(0.25 * 8)

    def test_no_claims_is_a_complete_cacheable_result(self):
        self.add([item(1), item(2)])
        with fake_engine.FakeEngine(make=lambda t: fake_engine.v1_response(degraded_reason="no_claims"),
                                    path="/api/check_v1") as e:
            old = claimcheck.ENGINE_URL
            claimcheck.ENGINE_URL = e.url
            self.addCleanup(setattr, claimcheck, "ENGINE_URL", old)
            claimcheck.CACHE.clear() if hasattr(claimcheck.CACHE, "clear") else None
            jobs = rt_jobs.Jobs(db_path=self.db, clock=lambda: self.now)
            self.addCleanup(jobs.shutdown)
            s = rt_scheduler.Scheduler(db_path=self.db, enrich=self.enrich, jobs=jobs, clock=lambda: self.now, budget=5,
                                       sync_runs_fn=lambda: 1)
            s.ensure_table()
            r = s.run_cycle()
            self.assertEqual((r["analysed"], r["partial"], r["degraded"], e.count), (2, 0, 0, 2))
            self.assertEqual(s.run_cycle()["picked"], 0)           # not picked again
            self.assertEqual(e.count, 2)
            res = list(self.analyses().values())[0]
            self.assertNotIn("partial", res)
            self.assertEqual(res["engine"]["verdict"]["label"], "INCONCLUSIVE")
            self.assertEqual(res["engine"]["degraded_why"], "no_claims")

    def test_old_engine_no_claims_shape_is_complete_too(self):
        self.add([item(1)])
        with fake_engine.FakeEngine(make=lambda t: fake_engine.v1_response(degraded_reason="no_claims_old"),
                                    path="/api/check_v1") as e:
            old = claimcheck.ENGINE_URL
            claimcheck.ENGINE_URL = e.url
            self.addCleanup(setattr, claimcheck, "ENGINE_URL", old)
            jobs = rt_jobs.Jobs(db_path=self.db, clock=lambda: self.now)
            self.addCleanup(jobs.shutdown)
            s = rt_scheduler.Scheduler(db_path=self.db, enrich=self.enrich, jobs=jobs, clock=lambda: self.now,
                                       sync_runs_fn=lambda: 1)
            s.ensure_table()
            r = s.run_cycle()
            self.assertEqual((r["analysed"], r["partial"]), (1, 0))
            self.assertEqual(s.run_cycle()["picked"], 0)
            self.assertEqual(e.count, 1)

    def test_timeout_reply_from_the_real_adapter_path_is_partial(self):
        self.add([item(1)])
        tp = {c: 0.1 for c in __import__("engine_adapter").TECHNIQUE_CLASSES} if hasattr(
            __import__("engine_adapter"), "TECHNIQUE_CLASSES") else None
        with fake_engine.FakeEngine(make=lambda t: fake_engine.v1_response(degraded_reason="timeout", technique_probs=tp),
                                    path="/api/check_v1") as e:
            old = claimcheck.ENGINE_URL
            claimcheck.ENGINE_URL = e.url
            self.addCleanup(setattr, claimcheck, "ENGINE_URL", old)
            jobs = rt_jobs.Jobs(db_path=self.db, clock=lambda: self.now)
            self.addCleanup(jobs.shutdown)
            s = rt_scheduler.Scheduler(db_path=self.db, enrich=self.enrich, jobs=jobs, clock=lambda: self.now,
                                       sync_runs_fn=lambda: 1)
            s.ensure_table()
            r = s.run_cycle()
        self.assertEqual((r["degraded"], r["partial"], r["analysed"]), (1, 1, 0))
        res = self.analyses()[nid(1)]
        self.assertIsNone(res["engine"]["verdict"])
        self.assertIsNone(res["step1"]["label"])


class PartialInApi(S2Base):
    def api(self):
        return rt_api.RtApi(db_path=self.db, enrich=self.enrich, clock=lambda: NOW)

    def test_coverage_counts_partial_apart_and_verdict_share_ignores_it(self):
        self.add([item(1), item(2), item(3), item(4, "social")])
        self.put_analysis(nid(1), deep_result(label="FAKE", fake=0.9, mis=0.05))
        s = self.sched2()
        self.script = lambda t: partial_body()
        con = None
        # items 2 and 4 become partial through a real cycle
        with mock.patch.object(s, "select_candidates", return_value=[(nid(2), "news", 1.0, None), ("reddit_posts:4", "social", 1.0, None)]):
            s.run_cycle()
        st = self.api().stats({})
        cv = st["coverage"]
        self.assertEqual(cv["deep"]["analysed"], 1)
        self.assertEqual(cv["deep"]["partial_n"], 2)
        self.assertEqual(cv["deep"]["analysed_n"], 1)
        self.assertEqual(cv["partial_n"], 2)
        self.assertEqual(cv["analysed"], 1)                       # partial is not "analysed" with a verdict
        self.assertEqual(cv["verdicts"], {"FAKE": 1})
        lf = st["likely_fake_misleading"]
        self.assertEqual((lf["count"], lf["denominator"]), (1, 1))
        # a complete verdict replaces the partial one
        self.put_analysis(nid(2), deep_result(label="LEGIT", fake=0.1, mis=0.1))
        cv2 = self.api().stats({"_": "1"})["coverage"]
        self.assertEqual((cv2["deep"]["analysed"], cv2["deep"]["partial_n"]), (2, 1))

    def test_items_and_case_file_show_partial_state_and_never_a_verdict(self):
        self.add([item(1), item(2)])
        s = self.sched2()
        self.script = lambda t: partial_body()
        with mock.patch.object(s, "select_candidates", return_value=[(nid(1), "news", 1.0, None)]):
            s.run_cycle()
        api = self.api()
        lst = api.items({})
        by = {i["id"]: i for i in lst["items"]}
        inf = by[nid(1)]["inferred"]
        self.assertEqual((inf["source"], inf["verdict"], inf["deep"]), ("partial", None, None))
        self.assertEqual(inf["partial"]["state"], "partial")
        self.assertEqual(inf["partial"]["retries"], 0)
        self.assertIn("propaganda signals only", inf["partial"]["note"])
        self.assertIsNone(by[nid(2)]["inferred"])
        yes = {i["id"] for i in api.items({"analysed": ["yes"]})["items"]}
        self.assertEqual(yes, {nid(1)})
        self.assertEqual({i["id"] for i in api.items({"analysed": ["no"]})["items"]}, {nid(2)})
        det = api.item(nid(1))["inferred"]
        self.assertEqual((det["source"], det["verdict"]), ("partial", None))
        self.assertTrue(det["techniques"])                        # from the partial result's propaganda
        json.dumps(det, allow_nan=False)


class StopRule(S2Base):
    def go(self, seq, n=12, **kw):
        self.add([item(i, hours=i) for i in range(1, n + 1)])
        it = iter(seq)
        mk = {"ok": lambda: (ok_body(), 200, {}), "bad": lambda: partial_body(), "busy": busy_body,
              "err": lambda: ({"error": "boom"}, 500, {})}
        self.script = lambda t: mk[next(it, "ok")]()
        return self.sched2(budget=n, **kw).run_cycle()

    def test_five_bad_replies_do_not_stop_it(self):
        r = self.go(["bad"] * 5, n=5)                              # fewer than the 6 replies the rule needs
        self.assertEqual(len(self.calls), 5)
        self.assertEqual((r["degraded"], r["analysed"], r["stop_reason"]), (5, 0, "done"))

    def test_six_bad_replies_stop_it(self):
        r = self.go(["bad"] * 20)
        self.assertEqual(len(self.calls), 6)                      # the minimum is 6 replies
        self.assertIn("6 of the last 6", r["stop_reason"])
        self.assertEqual(r["degraded"], 6)

    def test_seventy_percent_of_the_last_ten(self):
        # 4 ok first then bad: 7 bad of the last 10 replies is reached on reply 11? last 10 = ok*3 + bad*7 -> 70 %
        r = self.go(["ok"] * 4 + ["bad"] * 20)
        self.assertEqual(len(self.calls), 11)                     # replies 2..11 = 7 bad + 3 ok; at 10 replies it was 6
        self.assertIn("7 of the last 10", r["stop_reason"])

    def test_sixty_percent_never_stops(self):
        r = self.go((["bad", "ok", "bad", "ok", "bad"] * 4), n=20)   # 60 % bad
        self.assertEqual(len(self.calls), 20)
        self.assertEqual(r["stop_reason"], "done")

    def test_mixed_busy_degraded_errors_count_together(self):
        r = self.go(["busy", "bad", "err", "busy", "bad", "err"] * 3, sleep_cap=0)
        self.assertIn("of the last", r["stop_reason"])
        self.assertLessEqual(len(self.calls), 12)

    def test_unreachable_stops_immediately(self):
        self.add([item(i) for i in range(1, 6)])
        self.script = lambda t: ({"engine": {"state": "unavailable", "reason": "cannot connect"}}, 200, {})
        r = self.sched2().run_cycle()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(r["stop_reason"], "engine unreachable")
        self.assertEqual(r["errors"], 1)


class BusyWait(S2Base):
    def test_busy_waits_retry_after_once_then_the_item_is_tried_again(self):
        self.add([item(1)])
        seq = iter([({"engine": {"state": "busy", "retry_after_s": 7, "reason": "busy"}}, 200, {}), (ok_body(), 200, {})])
        self.script = lambda t: next(seq)
        r = self.sched2().run_cycle()
        self.assertEqual(self.waits, [7.0])
        self.assertEqual((r["analysed"], r["errors"], len(self.calls)), (1, 1, 2))

    def test_a_second_busy_on_the_same_item_does_not_wait_again(self):
        self.add([item(1), item(2)])
        self.script = lambda t: ({"engine": {"state": "busy", "retry_after_s": 5, "reason": "busy"}}, 200, {})
        self.sched2().run_cycle()
        self.assertEqual(self.waits, [5.0, 5.0])                  # one wait per busy item, not per reply
        self.assertEqual(len(self.calls), 4)

    def test_wait_is_capped_at_sixty_seconds(self):
        self.add([item(1)])
        seq = iter([({"engine": {"state": "busy", "retry_after_s": 500, "reason": "busy"}}, 200, {}), (ok_body(), 200, {})])
        self.script = lambda t: next(seq)
        self.sched2().run_cycle()
        self.assertEqual(self.waits, [60.0])

    def test_stop_during_the_wait_ends_the_cycle(self):
        self.add([item(1), item(2)])
        self.script = lambda t: busy_body()
        r = self.sched2(wait_fn=lambda s: True).run_cycle()
        self.assertEqual(r["stop_reason"], "server stopping")
        self.assertEqual(len(self.calls), 1)


class Pacing(S2Base):
    def test_slow_engine_halves_the_remaining_budget(self):
        self.add([item(i, hours=i) for i in range(1, 21)])
        clock = {"t": 0.0}

        def script(text):
            clock["t"] += 50.0
            return ok_body(), 200, {}
        self.script = script
        s = self.sched2(budget=20, monotonic=lambda: clock["t"], max_secs=10 ** 6)
        r = s.run_cycle()
        # after 5 calls the median is 50 s (> 40): 15 remain, half (7) are kept; the check repeats after 5 more calls
        self.assertLess(len(self.calls), 20)
        self.assertEqual(r["picked"], 20)
        self.assertEqual(r["median_secs"], 50.0)
        self.assertIn("engine slow", r["stop_reason"])
        self.assertEqual(r["analysed"], len(self.calls))

    def test_fast_engine_is_not_cut(self):
        self.add([item(i, hours=i) for i in range(1, 13)])
        clock = {"t": 0.0}

        def script(text):
            clock["t"] += 39.0
            return ok_body(), 200, {}
        self.script = script
        r = self.sched2(budget=12, monotonic=lambda: clock["t"], max_secs=10 ** 6).run_cycle()
        self.assertEqual((len(self.calls), r["stop_reason"]), (12, "done"))

    def test_default_budget_is_forty(self):
        self.assertEqual(rt_scheduler.DEFAULT_BUDGET, 40)
        self.assertEqual(rt_scheduler.config_from_env({})["budget"], 40)
        self.assertEqual(rt_scheduler.config_from_env({"FNP_RT_ANALYSE_BUDGET": "60"})["budget"], 60)


class Priority(S2Base):
    def rows(self, spec):
        """spec: {id: dict(row overrides)} -> real-format export for these ids"""
        news = [real_row(i, **kw) for i, kw in spec.items() if i.startswith("news:")]
        social = [real_row(i, **kw) for i, kw in spec.items() if not i.startswith("news:")]
        write_real(self.export, {"news/part_0000.jsonl": news, "social/part_0000.jsonl": social})

    def nf(self, cos=0.8, rerank=0.9):
        return dict(kb_id="kb:x", label="FAKE", claim="c", url="https://f.example/x", cosine=cos, rerank=rerank)

    def test_order_rerank_then_strong_propaganda_then_newest_and_too_short_skipped(self):
        self.add([item(i, hours=h) for i, h in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6))])
        self.rows({nid(5): dict(nf=self.nf(), p_prop=0.2, strong=()),                 # rerank present
                   nid(4): dict(nf=None, p_prop=0.7, strong=("Loaded_Language",)),    # strong propaganda
                   nid(3): dict(nf=None, p_prop=0.9, strong=()),                      # high p_prop but n_strong 0: no boost
                   nid(2): dict(too_short=True, p_prop=None, strong=(), nf=None),     # skipped
                   nid(6): dict(nf=dict(self.nf(0.65, None)), p_prop=0.5, strong=())})  # cosine below the floor, no rerank
        ids = self.ids(self.sched2().select_candidates())
        self.assertEqual(ids[:2], [nid(5), nid(4)])
        self.assertNotIn(nid(2), ids)
        self.assertGreater(len(ids), 3)
        self.assertEqual(set(ids), {nid(i) for i in (1, 3, 4, 5, 6)})

    def test_too_short_is_skipped_even_when_it_is_the_only_flagged_item(self):
        self.add([item(1), item(2)])
        self.rows({nid(1): dict(too_short=True, p_prop=None, strong=(), nf=self.nf())})
        self.assertEqual(self.ids(self.sched2().select_candidates()), [nid(2)])

    def test_old_items_with_a_signal_beat_newer_unflagged_ones(self):
        self.add([item(1, hours=1), item(2, hours=2), item(3, hours=24 * 5)])
        self.rows({nid(3): dict(nf=self.nf(), p_prop=0.1, strong=())})
        self.assertEqual(self.ids(self.sched2().select_candidates())[0], nid(3))

    def test_kind_balance_with_signals(self):
        self.add([item(i, "news", hours=1) for i in range(1, 21)] + [item(100 + i, "social", hours=30) for i in range(1, 6)])
        self.rows({nid(i): dict(nf=self.nf(), p_prop=0.9) for i in range(1, 21)})
        kinds = [k for _, k, _, _ in self.sched2(budget=10).select_candidates()]
        self.assertEqual(len(kinds), 10)
        self.assertGreaterEqual(kinds.count("social"), 3)

    def test_recency_only_when_no_export_row_matches(self):
        self.add([item(1, hours=3), item(2, hours=1), item(3, hours=2)])
        self.rows({nid(99): dict(nf=self.nf())})                                       # an id that is not a candidate
        self.assertEqual(self.ids(self.sched2().select_candidates()), [nid(2), nid(3), nid(1)])
        # and with no export directory at all
        import shutil
        shutil.rmtree(self.export)
        self.assertEqual(self.ids(self.sched2().select_candidates()), [nid(2), nid(3), nid(1)])

    def test_priority_function_tiers(self):
        pr = rt_scheduler.priority
        base = pr(None, None, 10, None)
        self.assertAlmostEqual(pr(None, None, 10, None, rerank=0.8) - base, 3.0)
        self.assertAlmostEqual(pr(None, 0.7, 10, None, n_strong=1) - base, 2.7)
        self.assertAlmostEqual(pr(None, 0.7, 10, None, n_strong=0) - base, 0.7)      # no strong technique: no boost
        self.assertAlmostEqual(pr(None, 0.59, 10, None, n_strong=2) - base, 0.59)    # under 0.6: no boost


class Migration(unittest.TestCase):
    OLD = """CREATE TABLE schedule_log (started_at TEXT NOT NULL, ended_at TEXT, picked INTEGER NOT NULL DEFAULT 0,
      analysed INTEGER NOT NULL DEFAULT 0, cached_dupes INTEGER NOT NULL DEFAULT 0, degraded INTEGER NOT NULL DEFAULT 0,
      errors INTEGER NOT NULL DEFAULT 0, skipped INTEGER NOT NULL DEFAULT 0, stop_reason TEXT, next_run_at TEXT);"""

    def setUp(self):
        import tempfile, shutil
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.db = os.path.join(self.tmp, "realtime.sqlite")
        con = rt_store.open_store(self.db)
        con.execute("DROP TABLE IF EXISTS schedule_log")
        con.execute(self.OLD)
        con.execute("INSERT INTO schedule_log VALUES ('2026-11-12T01:00:00Z','2026-11-12T02:00:00Z',200,9,0,3,3,0,"
                    "'engine busy 3 times in a row',NULL)")
        con.close()

    def test_status_reads_an_unmigrated_store_without_writing(self):
        st = rt_scheduler.schedule_status(self.db, env={}, log_rows=10)
        self.assertEqual(st["last_run"]["analysed"], 9)
        self.assertEqual((st["last_run"]["partial"], st["last_run"]["retried"], st["last_run"]["median_secs"]), (0, 0, None))
        cols = {r[1] for r in sqlite3.connect(self.db).execute("PRAGMA table_info(schedule_log)")}
        self.assertNotIn("partial", cols)                                              # read-only: nothing migrated
        self.assertEqual(st["rolling"]["replies"], 15)
        self.assertAlmostEqual(st["rolling"]["success_rate"], 9 / 15)

    def test_ensure_adds_columns_once_and_keeps_old_rows(self):
        for _ in range(2):                                                              # idempotent
            con = rt_store.open_store(self.db, create=False)
            rt_store.ensure_schedule_log(con)
            con.close()
        con = sqlite3.connect(self.db)
        cols = [r[1] for r in con.execute("PRAGMA table_info(schedule_log)")]
        self.assertEqual(cols[-3:], ["partial", "retried", "median_secs"])
        row = con.execute("SELECT picked, analysed, partial, retried, median_secs, stop_reason FROM schedule_log").fetchone()
        self.assertEqual(row, (200, 9, 0, 0, None, "engine busy 3 times in a row"))    # old row untouched
        con.close()

    def test_scheduler_start_migrates_and_new_cycle_logs_the_new_columns(self):
        con = rt_store.open_store(self.db, create=False)
        rt_store.upsert_items(con, [item(1)])
        con.close()
        jobs = rt_jobs.Jobs(db_path=self.db, analyse_fn=lambda *a: (ok_body(), 200, {}))
        self.addCleanup(jobs.shutdown)
        s = rt_scheduler.Scheduler(db_path=self.db, jobs=jobs, clock=lambda: NOW, sync_runs_fn=lambda: 1, budget=3)
        s.ensure_table()
        r = s.run_cycle()
        con = sqlite3.connect(self.db)
        row = con.execute("SELECT partial, retried, median_secs FROM schedule_log ORDER BY rowid DESC").fetchone()
        con.close()
        self.assertEqual(row[:2], (0, 0))
        self.assertIsNotNone(row[2])
        self.assertEqual(r["analysed"], 1)


if __name__ == "__main__":
    unittest.main()
