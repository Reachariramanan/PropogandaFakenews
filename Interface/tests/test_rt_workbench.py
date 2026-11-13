"""Analysis Workbench (jobs) and grounded insights (phase E4). Offline: temp SQLite store from test_rt_api's fixture,
fake check_v1 engine and fake insight_v1 service on ephemeral 127.0.0.1 ports, the real handler class."""
import base64
import contextlib
import io
import json
import os
import re
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import helpers
import claimcheck
import fake_engine
import rt_api
import rt_insights
import rt_jobs
import server
from helpers import call, jcall
from test_rt_api import Fixture, default_export, iso, item as mk_item
from rt_fakes import NOW

NEWS1, NEWS2, SOC11 = "news:scraped_articles:1", "news:scraped_articles:2", "reddit_posts:11"
SHORT1, SHORT2 = "telegram_posts:13", "x_posts:15"   # fewer than 20 characters of text


def until(fn, timeout=12.0, step=0.02):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = fn()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("timed out waiting for a condition")


class JobBase(Fixture):
    def setUp(self):
        super().setUp()
        default_export(self.export)
        claimcheck.reset_state()
        self._cc = (claimcheck.ENGINE_URL, claimcheck.ENGINE_TIMEOUT, claimcheck.QUEUE_WAIT)
        self._se = sys.stderr
        sys.stderr = self.err = io.StringIO()
        self.jobs = rt_jobs.Jobs(db_path=self.db)
        self.addCleanup(self._restore)

    def _restore(self):
        self.jobs.shutdown()
        sys.stderr = self._se
        claimcheck.ENGINE_URL, claimcheck.ENGINE_TIMEOUT, claimcheck.QUEUE_WAIT = self._cc
        claimcheck.reset_state()

    @contextlib.contextmanager
    def engine(self, make=None, **kw):
        with fake_engine.FakeEngine(make=make or (lambda t: fake_engine.v1_response()), path="/api/check_v1", **kw) as e:
            claimcheck.ENGINE_URL = e.url
            yield e

    def finish(self, jid, jobs=None):
        jobs = jobs or self.jobs
        return until(lambda: (lambda v: v if v["state"] in ("done", "cancelled", "error") else None)(jobs.get(jid)))

    def analyses(self):
        return self.sql("SELECT item_id, engine_version, result_json FROM analyses ORDER BY item_id")


class Lifecycle(JobBase):
    def test_queued_running_done_and_results_cached(self):
        with self.engine(delay=0.3) as e:
            v = self.jobs.submit([NEWS2, SOC11])
            self.assertEqual((v["state"], v["total"], v["done"]), ("queued", 2, 0))
            self.assertEqual([i["state"] for i in v["items"]], ["queued", "queued"])
            run = until(lambda: (lambda x: x if x["items"][0]["state"] == "running" else None)(self.jobs.get(v["job_id"])))
            self.assertEqual((run["state"], run["queue_position"]), ("running", 0))
            self.assertEqual(run["items"][0]["state"], "running")
            self.assertTrue(run["items"][0]["started_at"])
            done = self.finish(v["job_id"])
            self.assertEqual((done["state"], done["done"], done["queue_position"]), ("done", 2, None))
            for it in done["items"]:
                self.assertEqual(it["state"], "done")
                self.assertGreater(it["secs"], 0.25)
                r = it["result"]
                self.assertEqual(r["step1"]["label"], "FAKE")  # same shape as POST /api/analyse
                self.assertEqual(r["engine"]["state"], "ok")
                self.assertIn("step2", r)
            self.assertEqual(e.count, 2)
        rows = self.analyses()
        ids = {r[0] for r in rows if r[0] in (NEWS2, SOC11)}
        self.assertEqual(ids, {NEWS2, SOC11})
        self.assertTrue(all(r[1] == "check_v1" for r in rows if r[0] in ids))
        self.assertEqual(json.loads(next(r[2] for r in rows if r[0] == SOC11))["step1"]["label"], "FAKE")

    def test_sequential_in_order_with_one_engine_call_in_flight(self):
        seen, state = [], {"now": 0, "max": 0}
        lock = threading.Lock()

        def make(text):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
                seen.append(text)
            time.sleep(0.1)
            with lock:
                state["now"] -= 1
            return fake_engine.v1_response()
        with self.engine(make=make):
            a = self.jobs.submit([SOC11, NEWS2, NEWS1])
            b = self.jobs.submit([NEWS1])  # a second job waits for the first: never two engine calls from jobs
            self.assertEqual(b["queue_position"], 2)
            self.finish(a["job_id"])
            self.finish(b["job_id"])
        self.assertEqual(state["max"], 1)
        self.assertTrue(seen[0].startswith("the river valley is flooding"))
        self.assertTrue(seen[1].startswith("Flood alert repeated\n\n"))   # title + blank line + excerpt
        self.assertTrue(seen[2].startswith("Flood alert for the river valley\n\n"))
        self.assertEqual(len(seen), 3)  # the second job's NEWS1 was a cache hit

    def test_text_is_title_plus_excerpt_trimmed_to_8000(self):
        self.assertEqual(rt_jobs.compose_text("Title", "Body text"), "Title\n\nBody text")
        self.assertEqual(rt_jobs.compose_text(None, "only body"), "only body")
        self.assertEqual(rt_jobs.compose_text("only title", ""), "only title")
        self.assertEqual(len(rt_jobs.compose_text("T" * 5000, "b" * 9000)), 8000)

    def test_cache_hit_is_instant_then_force_reruns(self):
        with self.engine(delay=0.2) as e:
            first = self.finish(self.jobs.submit([NEWS2])["job_id"])
            self.assertEqual(first["items"][0]["state"], "done")
            second = self.finish(self.jobs.submit([NEWS2])["job_id"])
            it = second["items"][0]
            self.assertEqual(it["state"], "cached")
            self.assertLess(it["secs"], 0.15)
            self.assertTrue(it["cached_at"])
            self.assertEqual(it["result"]["step1"], first["items"][0]["result"]["step1"])
            self.assertEqual(e.count, 1)
            third = self.finish(self.jobs.submit([NEWS2], force=True)["job_id"])
            self.assertEqual(third["items"][0]["state"], "done")
            self.assertEqual(e.count, 2)
            self.assertTrue(third["force"])

    def test_changed_text_makes_the_cached_result_stale(self):
        with self.engine() as e:
            self.finish(self.jobs.submit([SOC11])["job_id"])
            con = rt_jobs.rt_store.open_store(self.db, create=False)
            con.execute("UPDATE items SET text_excerpt = 'the river valley flooding got much worse overnight' WHERE id = ?",
                        (SOC11,))
            con.close()
            again = self.finish(self.jobs.submit([SOC11])["job_id"])
            self.assertEqual(again["items"][0]["state"], "done")
            self.assertEqual(e.count, 2)

    def test_degraded_is_marked_and_never_cached(self):
        with self.engine(make=lambda t: fake_engine.v1_response(degraded=True)) as e:
            v = self.finish(self.jobs.submit([NEWS2])["job_id"])
            it = v["items"][0]
            self.assertEqual(it["state"], "degraded")
            self.assertEqual(it["result"]["engine"]["state"], "degraded")
            self.assertIn("timed out", it["reason"])
            self.assertEqual(self.analyses(), [(NEWS1, "check_v1", json.dumps({"final": "FAKE"}))])  # only the fixture row
            again = self.finish(self.jobs.submit([NEWS2])["job_id"])
            self.assertEqual(again["items"][0]["state"], "degraded")  # ran the engine again, not served from cache
            self.assertEqual(e.count, 2)

    def test_busy_honours_retry_after_then_reports_error(self):
        with mock.patch.object(rt_jobs, "BUSY_RETRIES", 1), self.engine(status=503, headers={"Retry-After": "1"}) as e:
            t0 = time.monotonic()
            v = self.finish(self.jobs.submit([NEWS2])["job_id"])
            took = time.monotonic() - t0
            it = v["items"][0]
            self.assertEqual(it["state"], "error")
            self.assertIsNone(it["result"])
            self.assertIn("busy", it["reason"])
            self.assertEqual(it["retry_after_s"], 1)
            self.assertEqual(e.count, 2)        # the first call plus one re-try after waiting
            self.assertGreaterEqual(took, 0.9)  # waited for Retry-After
        self.assertEqual(v["state"], "done")     # an item error does not fail the job
        self.assertNotIn(NEWS2, [r[0] for r in self.analyses()])

    def test_busy_then_recovers(self):
        with mock.patch.object(rt_jobs, "BUSY_RETRIES", 2), self.engine(status=503, headers={"Retry-After": "1"}) as e:
            def flip():
                time.sleep(0.4)
                e.status, e.headers = 200, {}
            threading.Thread(target=flip, daemon=True).start()
            v = self.finish(self.jobs.submit([NEWS2])["job_id"])
        self.assertEqual(v["items"][0]["state"], "done")

    def test_engine_down_is_an_item_error_not_a_crash(self):
        claimcheck.ENGINE_URL = helpers.DEAD_ENGINE_URL.replace("predict", "check_v1")
        v = self.finish(self.jobs.submit([NEWS2, SOC11])["job_id"])
        self.assertEqual([i["state"] for i in v["items"]], ["error", "error"])
        self.assertEqual(v["items"][0]["engine_state"], "unavailable")
        self.assertNotIn("Traceback", json.dumps(v))

    def test_skips_items_without_enough_text(self):
        with self.engine() as e:
            v = self.finish(self.jobs.submit([SHORT1, NEWS2, SHORT2])["job_id"])
        self.assertEqual([i["state"] for i in v["items"]], ["skipped", "done", "skipped"])
        self.assertIn("20 characters", v["items"][0]["reason"])
        self.assertIsNone(v["items"][0]["result"])
        self.assertEqual(e.count, 1)
        self.assertEqual(v["done"], 3)

    def test_item_that_vanished_is_an_error(self):
        with self.engine(delay=0.3):
            v = self.jobs.submit([NEWS2, SOC11])
            con = rt_jobs.rt_store.open_store(self.db, create=False)
            con.execute("DELETE FROM items WHERE id = ?", (SOC11,))
            con.close()
            done = self.finish(v["job_id"])
        self.assertEqual([i["state"] for i in done["items"]], ["done", "error"])

    def test_lang_is_forwarded_and_text_is_never_logged(self):
        with self.engine() as e:
            self.finish(self.jobs.submit([SOC11])["job_id"])
        log = self.err.getvalue()
        self.assertIn("rt-job job=", log)
        self.assertIn("item=" + SOC11, log)
        self.assertNotIn("river valley", log)
        self.assertNotIn("flooding", log)


class Cancel(JobBase):
    def test_cancel_running_job_stops_after_current_call(self):
        with self.engine(delay=0.5) as e:
            v = self.jobs.submit([NEWS2, SOC11, NEWS1])
            until(lambda: self.jobs.get(v["job_id"])["items"][0]["state"] == "running")
            c = self.jobs.cancel(v["job_id"])
            self.assertTrue(c["cancelling"])
            self.assertEqual(c["state"], "running")  # the call in flight finishes first
            done = self.finish(v["job_id"])
            self.assertEqual(done["state"], "cancelled")
            self.assertEqual([i["state"] for i in done["items"]], ["done", "skipped", "skipped"])
            self.assertEqual(done["items"][1]["reason"], "cancelled")
            self.assertEqual(e.count, 1)

    def test_cancel_queued_job_is_immediate(self):
        with self.engine(delay=0.4):
            a = self.jobs.submit([NEWS2])
            b = self.jobs.submit([SOC11])
            self.assertEqual(b["queue_position"], 2)
            c = self.jobs.cancel(b["job_id"])
            self.assertEqual(c["state"], "cancelled")
            self.assertEqual(c["items"][0]["state"], "skipped")
            self.assertEqual(self.finish(a["job_id"])["state"], "done")
            self.assertEqual(self.jobs.get(b["job_id"])["state"], "cancelled")

    def test_cancel_finished_or_unknown(self):
        with self.engine():
            v = self.finish(self.jobs.submit([NEWS2])["job_id"])
        self.assertEqual(self.jobs.cancel(v["job_id"])["state"], "done")
        with self.assertRaises(rt_jobs.JobError) as cm:
            self.jobs.cancel("0" * 16)
        self.assertEqual(cm.exception.code, 404)

    def test_cancel_interrupts_the_busy_wait(self):
        with mock.patch.object(rt_jobs, "BUSY_RETRIES", 2), self.engine(status=503, headers={"Retry-After": "30"}):
            v = self.jobs.submit([NEWS2, SOC11])
            until(lambda: self.jobs.get(v["job_id"])["items"][0].get("note"))
            t0 = time.monotonic()
            self.jobs.cancel(v["job_id"])
            done = self.finish(v["job_id"])
            self.assertLess(time.monotonic() - t0, 5)
            self.assertEqual(done["state"], "cancelled")


class Validation(JobBase):
    def test_ids(self):
        bad = [None, [], "x", {}, [1], [""], ["a b"], ["x" * 201], ["a;DROP"], [None]]
        for ids in bad:
            with self.assertRaises(rt_jobs.JobError, msg=repr(ids)) as cm:
                self.jobs.submit(ids)
            self.assertEqual(cm.exception.code, 400)

    def test_cap_of_25_and_dedupe(self):
        self.assertEqual(len(rt_jobs.parse_ids(["id%d" % i for i in range(25)])), 25)
        with self.assertRaises(rt_jobs.JobError) as cm:
            rt_jobs.parse_ids(["id%d" % i for i in range(26)])
        self.assertEqual(cm.exception.code, 400)
        self.assertEqual(rt_jobs.parse_ids([NEWS1, NEWS1, NEWS2]), [NEWS1, NEWS2])
        with self.assertRaises(rt_jobs.JobError):
            rt_jobs.parse_ids(["id%d" % i for i in range(150)])

    def test_unknown_ids_are_404_and_create_no_job(self):
        with self.assertRaises(rt_jobs.JobError) as cm:
            self.jobs.submit([NEWS1, "news:scraped_articles:424242"])
        self.assertEqual(cm.exception.code, 404)
        self.assertIn("424242", cm.exception.msg)
        self.assertEqual(len(self.jobs.jobs), 0)

    def test_too_many_queued_jobs(self):
        jobs = rt_jobs.Jobs(db_path=self.db, max_queued=2)
        self.addCleanup(jobs.shutdown)
        with self.engine(delay=0.5):
            jobs.submit([NEWS2])
            until(lambda: jobs.running)
            jobs.submit([SOC11])
            jobs.submit([NEWS1])
            with self.assertRaises(rt_jobs.JobError) as cm:
                jobs.submit([NEWS2])
            self.assertEqual(cm.exception.code, 429)
            self.assertIn("Retry-After", cm.exception.headers)

    def test_eviction_keeps_the_last_jobs_only(self):
        fast = rt_jobs.Jobs(db_path=self.db, keep=3,
                            analyse_fn=lambda *a: ({"engine": {"state": "ok"}, "step1": {"label": "LEGIT"}}, 200, {}))
        self.addCleanup(fast.shutdown)
        ids = []
        for _ in range(5):
            v = fast.submit([NEWS2], force=True)
            self.finish(v["job_id"], fast)
            ids.append(v["job_id"])
        self.assertEqual(list(fast.jobs), ids[-3:])
        with self.assertRaises(rt_jobs.JobError) as cm:
            fast.get(ids[0])
        self.assertEqual(cm.exception.code, 404)

    def test_active_jobs_are_never_evicted(self):
        slow = rt_jobs.Jobs(db_path=self.db, keep=1, max_queued=5)
        self.addCleanup(slow.shutdown)
        with self.engine(delay=0.3):
            v = [slow.submit([NEWS2], force=True) for _ in range(3)]
            self.assertEqual(len(slow.jobs), 3)
            self.assertTrue(all(x["job_id"] in slow.jobs for x in v))  # all three are active: none was evicted
            until(lambda: slow.jobs[v[2]["job_id"]].state == "done")
        self.assertLessEqual(len(slow.jobs), 3)
        self.assertIn(v[2]["job_id"], slow.jobs)

    def test_a_crashing_analyser_is_an_item_error(self):
        def boom(*a):
            raise RuntimeError("secret internals river valley")
        j = rt_jobs.Jobs(db_path=self.db, analyse_fn=boom)
        self.addCleanup(j.shutdown)
        v = self.finish(j.submit([NEWS2])["job_id"], j)
        self.assertEqual(v["items"][0]["state"], "error")
        self.assertNotIn("secret", json.dumps(v))
        self.assertEqual(v["state"], "done")


# ---------------------------------------------------------------------------------------------------- insights
class FakeInsight:
    """A fake POST /api/insight_v1 on an ephemeral port: make(payload) -> (status, body dict)."""

    def __init__(self, make=None, delay=0.0):
        self.make, self.delay, self.hits = make or (lambda p: (200, {"text": "x", "citations": []})), delay, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n))
                outer.hits.append(payload)
                time.sleep(outer.delay)
                status, body = outer.make(payload)
                out = json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                except OSError:
                    pass
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = "http://127.0.0.1:%d/api/insight_v1" % self.httpd.server_address[1]
        helpers.assert_not_real_flag(self.url)
        self.t = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.t.join(5)


DEAD_INSIGHT_URL = "http://127.0.0.1:%d/api/insight_v1" % helpers.free_port()


class InsightBase(Fixture):
    def setUp(self):
        super().setUp()
        default_export(self.export)
        self.ins = rt_insights.Insights(self.api, url=DEAD_INSIGHT_URL)

    def run_ins(self, scope=None, phrase=False, question=None, ins=None):
        body = {"scope": scope or {}, "phrase": phrase}
        if question:
            body["question"] = question
        return (ins or self.ins).run(body)


class Facts(InsightBase):
    def test_whole_store_matches_the_stats_endpoint(self):
        r = self.run_ins()
        f, st = r["facts"], self.api.stats({})
        self.assertEqual(f["n_items"], st["total"])
        self.assertEqual((f["n_news"], f["n_social"]), (6, 6))
        self.assertEqual({p["platform"]: p["count"] for p in f["platforms"]}, {p["platform"]: p["count"] for p in st["by_platform"]})
        self.assertEqual(f["n_unclassified"], st["by_topic"]["unclassified"]["count"])
        self.assertEqual(f["n_classified"], st["by_topic"]["classified"])
        self.assertEqual([(t["topic_id"], t["count"]) for t in f["topics"]], [(t["topic_id"], t["count"]) for t in st["by_topic"]["top"]])
        self.assertEqual({l["lang"]: l["count"] for l in f["languages"]}, {l["lang"]: l["count"] for l in st["by_language"]["top"]})
        self.assertEqual(f["n_analysed"], st["coverage"]["analysed"])
        self.assertEqual(f["n_with_verdict"], st["coverage"]["with_verdict"])
        self.assertEqual(f["verdicts"], st["coverage"]["verdicts"])
        lf = st["likely_fake_misleading"]
        self.assertEqual((f["n_likely_fake_misleading"], f["n_with_verdict"]), (lf["count"], lf["denominator"]))
        self.assertEqual(f["n_with_p_prop"], st["p_prop_distribution"]["n"])
        self.assertEqual([b["count"] for b in f["p_prop_buckets"]], [b["count"] for b in st["p_prop_distribution"]["buckets"]])
        self.assertEqual(f["n_p_prop_ge_07"], 3)   # p_prop 0.9, 0.95, 0.7 (telegram_posts:12)
        self.assertEqual({t["technique"]: t["items"] for t in f["techniques"]},
                         {t["technique"]: t["items"] for t in st["technique_mix"]["top"]})
        self.assertEqual(f["techniques"][0]["pct_of_analysed"], round(100.0 * 3 / 8, 1))
        self.assertEqual(sum(d["count"] for d in f["daily"]), 12)
        self.assertEqual(f["busiest_day"]["count"], max(d["count"] for d in f["daily"]))
        self.assertEqual(f["pct_social"], 50.0)
        self.assertIsNone(f["workbench"])

    def test_sample_rows_are_deterministic_and_bounded(self):
        a, b = self.run_ins()["facts"]["rows"], self.run_ins()["facts"]["rows"]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)
        self.assertEqual(set(a[0]), {"id", "text", "platform", "source", "published_at"})
        times = [r["published_at"] for r in a]
        self.assertEqual(times, sorted(times, reverse=True))
        self.assertTrue(all(len(r["text"]) <= 300 for r in a))
        con = rt_jobs.rt_store.open_store(self.db, create=False)
        rt_jobs.rt_store.upsert_items(con, [mk_item(i, "social", "reddit", hours=1, text="long " * 200, title="T")
                                            for i in range(100, 140)])
        con.close()
        self.api._cache.clear()
        fresh = rt_insights.Insights(self.api, url=DEAD_INSIGHT_URL).run({"scope": {}})["facts"]
        self.assertEqual(len(fresh["rows"]), 20)
        self.assertEqual(fresh["n_sample_rows"], 20)
        self.assertTrue(all(len(r["text"]) <= 300 for r in fresh["rows"]))

    def test_scope_filters_and_denominators(self):
        f = self.run_ins({"kind": "news"})["facts"]
        self.assertEqual((f["n_items"], f["n_news"], f["n_social"]), (6, 6, 0))
        self.assertEqual((f["n_analysed"], f["n_with_verdict"], f["n_likely_fake_misleading"]), (4, 3, 2))
        f = self.run_ins({"platform": "telegram"})["facts"]
        self.assertEqual((f["n_items"], f["n_analysed"]), (2, 1))
        f = self.run_ins({"q": "river"})["facts"]
        self.assertEqual(f["n_items"], len(self.sql("SELECT 1 FROM items_fts WHERE items_fts MATCH 'river*'")))
        f = self.run_ins({"days": 1})["facts"]
        self.assertEqual(f["n_items"], len(self.sql("SELECT 1 FROM items WHERE published_at >= ?", iso(days=1))))
        f = self.run_ins({"analysed": "yes", "topic": "unclassified"})["facts"]
        self.assertEqual((f["n_items"], f["n_analysed"]), (5, 5))  # news 3, reddit 10/11/14, telegram 12
        f = self.run_ins({"analysed": "no"})["facts"]
        self.assertEqual((f["n_items"], f["n_analysed"], f["n_with_verdict"]), (4, 0, 0))

    def test_selection_scope_has_workbench_counts_and_rows_in_order(self):
        con = rt_jobs.rt_store.open_store(self.db, create=False)
        con.execute("DELETE FROM analyses")
        for iid, lab in ((NEWS1, "FAKE"), (NEWS2, "LEGIT"), (SOC11, "MISLEADING")):
            con.execute("INSERT INTO analyses VALUES (?,?,?,?)", (iid, iso(minutes=1), "check_v1",
                                                                 json.dumps({"step1": {"label": lab}})))
        con.execute("INSERT INTO analyses VALUES (?,?,?,?)", ("reddit_posts:10", iso(minutes=1), "check_v1", json.dumps({"step1": {}})))
        con.close()
        ids = [SOC11, NEWS1, NEWS2, "reddit_posts:10", "reddit_posts:14"]
        r = self.run_ins({"ids": ids})
        f, w = r["facts"], r["facts"]["workbench"]
        self.assertEqual(f["n_items"], 5)
        self.assertEqual([x["id"] for x in f["rows"]], ids)
        self.assertEqual((w["n_selected"], w["n_stored"], w["n_with_label"]), (5, 4, 3))
        self.assertEqual((w["n_likely_fake_misleading"], w["pct_likely_fake_misleading"]), (2, 66.7))
        self.assertEqual(w["labels"], {"FAKE": 1, "LEGIT": 1, "MISLEADING": 1})
        txt = " ".join(t["text"] for t in r["template_insights"])
        self.assertIn("4 of the 5 selected items have a stored Claim Check result", txt)
        self.assertIn("Of the 3 with a label, 2 (67%) are FAKE or MISLEADING", txt)
        self.assertEqual(r["scope_desc"], "5 selected items")

    def test_empty_scope(self):
        r = self.run_ins({"kind": "news", "platform": "reddit"})
        self.assertEqual(r["facts"]["n_items"], 0)
        self.assertEqual(r["template_insights"], [])
        self.assertEqual(r["facts"]["rows"], [])
        self.assertTrue(r["scope_line"].startswith("Based on 0 items from the realtime database, synced "))
        r = self.run_ins({"kind": "news", "platform": "reddit"}, phrase=True)
        self.assertIsNone(r["phrased"])
        self.assertIn("no items", r["phrasing"]["reason"])

    def test_scope_validation(self):
        bad = [{"kind": "x"}, {"platform": "a;b"}, {"days": 0.5}, {"days": 9999}, {"days": True}, {"ids": []},
               {"ids": ["a b"]}, {"ids": ["i%d" % i for i in range(26)]}, {"from": "yesterday"}, {"nope": 1},
               {"q": "***"}, {"analysed": "maybe"}, {"topic": "a b"}, {"lang": "english!!"}, {"source": 5}]
        for sc in bad:
            with self.assertRaises(rt_api.ApiError, msg=repr(sc)) as cm:
                self.ins.run({"scope": sc})
            self.assertEqual(cm.exception.code, 400)
        for body in ([], {"scope": []}, {"scope": {}, "question": 5}, {"scope": {}, "question": "x" * 301},
                     {"scope": {}, "phrase": "yes"}, {"scope": {}, "extra": 1}):
            with self.assertRaises(rt_api.ApiError):
                self.ins.run(body)

    def test_scope_line_format_and_injection_safety(self):
        r = self.run_ins({"source": "'; DROP TABLE items; --"})
        self.assertEqual(r["facts"]["n_items"], 0)
        self.assertEqual(self.sql("SELECT count(*) FROM items")[0][0], 12)
        r = self.run_ins()
        self.assertEqual(r["scope_line"], "Based on 12 items from the realtime database, synced %s UTC. No outside knowledge used."
                         % iso(minutes=5)[:16].replace("T", " "))


class Template(InsightBase):
    SCOPES = ({}, {"kind": "news"}, {"kind": "social"}, {"platform": "telegram"}, {"platform": "x"}, {"days": 1},
              {"analysed": "no"}, {"analysed": "yes"}, {"lang": "hi"}, {"ids": [NEWS1]}, {"ids": ["reddit_posts:10", SOC11]},
              {"source": "a.com"}, {"kind": "news", "platform": "reddit"})

    def test_example_sentence_and_denominators(self):
        f = dict(n_items=17917, n_news=1998, n_social=15919, platforms=[], sources=[], topics=[], n_unclassified=17917,
                 languages=[], busiest_day=None, daily=[], n_analysed=0, workbench=None)
        t = rt_insights.template_insights(f)
        self.assertEqual(t[0]["text"], "Social posts are 89% of the 17,917 items in scope (15,919 of 17,917); news articles are the other 1,998.")
        self.assertEqual(t[0]["cites"], ["fact:n_social", "fact:n_news", "fact:n_items"])
        self.assertFalse(any("analysed" in x["text"] for x in t))  # nothing analysed: no analysed-coverage claim
        self.assertEqual(rt_insights.template_insights(dict(f, n_items=0)), [])

    def test_no_zero_denominator_claims_and_every_number_is_in_the_facts(self):
        zero = re.compile(r"\bof 0\b|/ ?0\b|\(0 of 0|of 0\)")
        for sc in self.SCOPES:
            r = self.run_ins(sc)
            f = r["facts"]
            for t in r["template_insights"]:
                self.assertIsNone(zero.search(t["text"]), (sc, t["text"]))
                self.assertTrue(t["cites"], t)
                for c in t["cites"]:
                    self.assertTrue(c.startswith("fact:") or c in {x["id"] for x in f["rows"]} or c.split(":")[0] in
                                    ("news", "reddit_posts", "telegram_posts", "x"), c)
                    if c.startswith("fact:"):
                        self.assertIn(c[5:], f["keys"], c)
                ok, why = rt_insights.validate_phrased(t["text"], [f["rows"][0]["id"]], f)
                self.assertTrue(ok, (sc, t["text"], why))
            if f["n_items"] == 0:
                self.assertEqual(r["template_insights"], [])
            if f["n_analysed"] == 0:
                self.assertFalse(any("analysed" in t["text"] for t in r["template_insights"]), sc)
            if f["n_with_verdict"] == 0 and f["n_analysed"]:
                self.assertFalse(any("FAKE or MISLEADING" in t["text"] for t in r["template_insights"]), sc)

    def test_never_claims_causes(self):
        for sc in self.SCOPES:
            for t in self.run_ins(sc)["template_insights"]:
                self.assertIsNone(re.search(r"\b(because|due to|caused|leads? to|driven by|as a result)\b",
                                                           t["text"], re.I), t["text"])

    def test_whole_store_content(self):
        txt = [t["text"] for t in self.run_ins()["template_insights"]]
        self.assertEqual(txt[0], "Social posts are 50% of the 12 items in scope (6 of 12); news articles are the other 6.")
        joined = " ".join(txt)
        self.assertIn("FNP has analysed 8 of 12 items (67%)", joined)
        self.assertIn("Of the 8 analysed items, 4 have a verdict. Of those 4, 3 (75%) are FAKE or MISLEADING.", joined)
        self.assertIn("score 0.7 or higher", joined)
        self.assertIn("loaded_language: found in 3 of 8 analysed items (38%)", joined)
        self.assertIn("9 of 12 items (75%) have no database topic (Unclassified)", joined)

    def test_keys_carry_value_denominator_and_pct(self):
        k = self.run_ins()["facts"]["keys"]
        self.assertEqual((k["n_social"]["value"], k["n_social"]["of"], k["n_social"]["pct"]), (6, 12, 50.0))
        self.assertEqual((k["n_likely_fake_misleading"]["value"], k["n_likely_fake_misleading"]["of"]), (3, 4))
        self.assertIsNone(k["n_items"]["of"])


class Validation2(unittest.TestCase):
    F = dict(n_items=17917, n_social=15919, n_news=1998, pct_social=88.85, pct_news=11.15, windows=[7, 30],
             keys={"x": dict(value=15919, of=17917, pct=88.9)}, daily=[dict(day="2026-11-06", count=500)],
             scope={"days": 7}, rows=[dict(id="reddit_posts:10", text="they say 5000 things 2024"),
                                      dict(id="news:scraped_articles:1", text="t")])

    def v(self, text, cites=("reddit_posts:10",)):
        return rt_insights.validate_phrased(text, list(cites), self.F)

    def test_accepts_numbers_from_the_facts_with_rounding(self):
        for t in ("There are 17,917 items, 15,919 of them social (89%).", "About 88.9% are social.", "89 percent... 11% news.",
                  "17917 items", "On 2026-11-06 there were 500 items.", "Last 7 days: 1,998 news items."):
            ok, why = self.v(t)
            self.assertTrue(ok, (t, why))

    def test_refuses_an_invented_number(self):
        for t in ("There are 17,918 items.", "About 90% are social.", "Exactly 15,000 posts.", "12 items", "88.7% are social.",
                  "5000 things were said", "growth of 250%"):
            ok, why = self.v(t)
            self.assertFalse(ok, t)
            self.assertIn("number", why)

    def test_refuses_forged_or_missing_citations(self):
        for cites in (["reddit_posts:999"], ["reddit_posts:10", "fake:1"], [], "reddit_posts:10", [5], None):
            ok, why = rt_insights.validate_phrased("17,917 items", cites, self.F)
            self.assertFalse(ok, repr(cites))
        ok, _ = rt_insights.validate_phrased("", ["reddit_posts:10"], self.F)
        self.assertFalse(ok)
        ok, _ = rt_insights.validate_phrased("x" * 5000, ["reddit_posts:10"], self.F)
        self.assertFalse(ok)

    def test_ids_written_in_the_text_do_not_count_as_numbers(self):
        ok, why = self.v("See reddit_posts:10 and news:scraped_articles:1 for 17,917 items.", ("reddit_posts:10", "news:scraped_articles:1"))
        self.assertTrue(ok, why)


class Phrasing(InsightBase):
    OK = staticmethod(lambda p: (200, {"text": "Social posts make up 50% of the 12 items in scope; see the first row.",
                                       "citations": [p["facts"]["rows"][0]["id"]]}))

    def phrased(self, make, **kw):
        with FakeInsight(make=make, **kw) as svc:
            ins = rt_insights.Insights(self.api, url=svc.url)
            return self.run_ins(phrase=True, question="what stands out?", ins=ins), svc, ins

    def test_valid_answer_is_shown_with_its_citations(self):
        r, svc, _ = self.phrased(self.OK)
        self.assertEqual(r["phrased"]["source"], "fnp-insight_v1")
        self.assertEqual(r["phrased"]["citations"], [r["facts"]["rows"][0]["id"]])
        self.assertIn("50%", r["phrased"]["text"])
        self.assertTrue(r["template_insights"])
        self.assertEqual(r["phrasing"], dict(available=True, reason=None))
        # the service got exactly the facts and the rows: ids and text only, nothing else per row
        p = svc.hits[0]
        self.assertEqual(p["question"], "what stands out?")
        self.assertEqual(set(p["facts"]["rows"][0]), {"id", "text"})
        self.assertEqual(p["facts"]["n_items"], 12)
        self.assertNotIn("scope", p["facts"])

    def test_forged_citation_is_refused_and_the_text_is_not_shown(self):
        r, _, _ = self.phrased(lambda p: (200, {"text": "SECRET-FORGED-TEXT about 12 items.", "citations": ["reddit_posts:424242"]}))
        self.assertTrue(r["phrased"]["refused"])
        self.assertIn("cited a row that is not in the data", r["phrased"]["reason"])
        self.assertNotIn("text", r["phrased"])
        self.assertNotIn("SECRET-FORGED-TEXT", json.dumps(r))
        self.assertTrue(r["template_insights"])

    def test_invented_number_is_refused_and_the_text_is_not_shown(self):
        r, _, _ = self.phrased(lambda p: (200, {"text": "SECRET-INVENTED: 4,321 items were flagged.", "citations": [p["facts"]["rows"][0]["id"]]}))
        self.assertTrue(r["phrased"]["refused"])
        self.assertIn("4,321", r["phrased"]["reason"])
        self.assertNotIn("SECRET-INVENTED", json.dumps(r))

    def test_uncited_answer_is_refused(self):
        r, _, _ = self.phrased(lambda p: (200, {"text": "12 items.", "citations": []}))
        self.assertTrue(r["phrased"]["refused"])

    def test_service_rejection_and_garbage_are_refusals(self):
        for status, body in ((422, {"error": "uncited number"}), (400, {"error": "x"}), (200, {"refused": True}),
                             (200, {"nothing": 1}), (200, {"text": 5, "citations": []})):
            r, _, _ = self.phrased(lambda p, s=status, b=body: (s, b))
            self.assertTrue(r["phrased"]["refused"], (status, body))

    def test_service_down_falls_back_to_template(self):
        r = self.run_ins(phrase=True)
        self.assertIsNone(r["phrased"])
        self.assertEqual(r["phrasing"], dict(available=False, reason="phrasing service not available"))
        self.assertTrue(r["template_insights"])
        self.assertEqual(self.run_ins()["phrasing"]["available"], False)  # remembered for a minute: the UI disables the button

    def test_404_503_500_and_timeout_mean_not_available(self):
        for status in (404, 503, 500, 502):
            r, svc, _ = self.phrased(lambda p, s=status: (s, {"error": "x"}))
            self.assertIsNone(r["phrased"], status)
            self.assertEqual(r["phrasing"]["reason"], "phrasing service not available")
            self.assertEqual(len(svc.hits), 1)
        with mock.patch.object(rt_insights, "PHRASE_TIMEOUT", 0.3):
            r, svc, ins = self.phrased(self.OK, delay=1.0)
        self.assertIsNone(r["phrased"])
        self.assertEqual(r["phrasing"]["available"], False)

    def test_down_state_is_not_retried_until_it_expires(self):
        with FakeInsight(make=lambda p: (503, {})) as svc:
            ins = rt_insights.Insights(self.api, url=svc.url)
            self.run_ins(phrase=True, ins=ins)
            self.run_ins(phrase=True, ins=ins, question="again")
            self.assertEqual(len(svc.hits), 1)
            svc.make = self.OK
            ins._down_until = 0
            r = self.run_ins(phrase=True, ins=ins, question="again")
            self.assertEqual(len(svc.hits), 2)
            self.assertIsNotNone(r["phrased"]["text"])

    def test_env_url_is_honoured_and_default_is_the_documented_one(self):
        self.assertEqual(rt_insights.Insights(self.api)._endpoint(), "http://127.0.0.1:8879/api/insight_v1")
        with FakeInsight(make=self.OK) as svc, mock.patch.dict(os.environ, {"FNP_INSIGHT_URL": svc.url}):
            r = self.run_ins(phrase=True, ins=rt_insights.Insights(self.api))
            self.assertIsNotNone(r["phrased"]["text"])

    def test_template_and_phrased_are_cached_per_scope_and_sync(self):
        with FakeInsight(make=self.OK) as svc:
            ins = rt_insights.Insights(self.api, url=svc.url)
            a = self.run_ins(phrase=True, ins=ins)
            b = self.run_ins(phrase=True, ins=ins)
            self.assertFalse(a["cached"])
            self.assertTrue(b["cached"])
            self.assertEqual(len(svc.hits), 1)
            self.assertEqual(b["phrased"], a["phrased"])
            self.run_ins(phrase=True, ins=ins, question="a different question")
            self.assertEqual(len(svc.hits), 2)
            other = self.run_ins({"kind": "news"}, ins=ins)
            self.assertFalse(other["cached"])
            # a newer sync invalidates the entry
            con = rt_jobs.rt_store.open_store(self.db, create=False)
            con.execute("INSERT INTO sync_log(ts, source_key, rows, seconds, error, note) VALUES (?,?,?,?,?,?)",
                        (iso(minutes=1), "reddit_posts", 1, 0.1, None, None))
            con.close()
            self.assertFalse(self.run_ins(ins=ins)["cached"])
            with mock.patch.object(rt_insights, "CACHE_TTL", -1):
                self.assertFalse(self.run_ins(ins=ins)["cached"])


# ----------------------------------------------------------------------------------------------------------- HTTP
class WbHttp(JobBase):
    def setUp(self):
        super().setUp()
        self._saved = {k: getattr(server, k) for k in ("RT", "AUTH", "JOBS", "INSIGHTS", "RT_ANALYSE_LIMIT", "RT_INSIGHT_LIMIT")}
        server.RT, server.AUTH, server.JOBS = self.api, None, None
        server.INSIGHTS = rt_insights.Insights(self.api, url=DEAD_INSIGHT_URL)
        server.RT_ANALYSE_LIMIT = claimcheck.RateLimit(server.RT_ANALYSE_RATE)
        server.RT_INSIGHT_LIMIT = claimcheck.RateLimit(server.RT_INSIGHT_RATE)
        server._rt_hits.clear()
        self._log = server.H.log_message
        server.H.log_message = lambda *a, **k: None
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.t = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.t.start()
        self.env = mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": "1"})
        self.env.start()
        self.addCleanup(self.close)

    def close(self):
        self.env.stop()
        if server.JOBS:
            server.JOBS.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.t.join(5)
        server.H.log_message = self._log
        for k, v in self._saved.items():
            setattr(server, k, v)
        server._rt_hits.clear()
        server._auth_fails.clear()

    def post(self, path, body=None, headers=None, **kw):
        return jcall(self.port, "POST", path, body=body, headers=headers, **kw)

    def get(self, path, headers=None):
        return jcall(self.port, "GET", path, headers=headers)

    @staticmethod
    def basic(user, pw):
        return {"Authorization": "Basic " + base64.b64encode(("%s:%s" % (user, pw)).encode()).decode()}

    def poll(self, jid, light=False):
        return until(lambda: (lambda r: r[2] if r[2]["state"] in ("done", "cancelled", "error") else None)(
            self.get("/api/rt/jobs/%s%s" % (jid, "?results=0" if light else ""))))


class HttpGate(WbHttp):
    NEW = (("POST", "/api/rt/analyse", {"ids": [NEWS2]}), ("POST", "/api/rt/insights", {"scope": {}}),
           ("POST", "/api/rt/jobs/" + "a" * 16 + "/cancel", {}), ("GET", "/api/rt/jobs/" + "a" * 16, None))

    def test_closed_by_default(self):
        self.env.stop()
        try:
            for m, path, body in self.NEW:
                st, hd, b = jcall(self.port, m, path, body=body)
                self.assertEqual(st, 403, (m, path))
                self.assertEqual(b["error"], "login_required")
                self.assertEqual(hd["cache-control"], "no-store")
            self.assertEqual(self.get("/api/rt/gate")[2], {"open": False, "auth": False})
        finally:
            self.env.start()
        self.assertEqual(len(self.jobs.jobs), 0)
        self.assertIsNone(server.JOBS)  # a refused request never even builds the job manager

    def test_open_flag_and_other_values(self):
        with self.engine():
            self.assertEqual(self.post("/api/rt/analyse", {"ids": [NEWS2]})[0], 202)
        for val in ("0", "true", ""):
            with mock.patch.dict(os.environ, {"FNP_RT_ALLOW_OPEN": val}):
                self.assertEqual(self.post("/api/rt/analyse", {"ids": [NEWS2]})[0], 403, val)
                self.assertEqual(self.post("/api/rt/insights", {})[0], 403, val)

    def test_auth_configured(self):
        self.env.stop()
        server.AUTH = (b"alice", b"pw-correct")
        try:
            for m, path, body in self.NEW:
                self.assertEqual(jcall(self.port, m, path, body=body)[0], 401, (m, path))
                self.assertEqual(jcall(self.port, m, path, body=body, headers=self.basic("alice", "wrong"))[0], 401)
            good = self.basic("alice", "pw-correct")
            with self.engine():
                st, hd, b = self.post("/api/rt/analyse", {"ids": [NEWS2]}, headers=good)
                self.assertEqual(st, 202)
                self.assertEqual(self.get("/api/rt/jobs/" + b["job_id"], good)[0], 200)
                self.assertEqual(self.get("/api/rt/jobs/" + b["job_id"])[0], 401)
                self.assertEqual(self.post("/api/rt/insights", {"scope": {}}, headers=good)[0], 200)
                self.assertEqual(self.post("/api/rt/jobs/%s/cancel" % b["job_id"], {}, headers=good)[0], 200)
                self.finish(b["job_id"], server.JOBS)
        finally:
            self.env.start()

    def test_rate_limits(self):
        server.RT_ANALYSE_LIMIT = claimcheck.RateLimit(2)
        with self.engine():
            codes = [self.post("/api/rt/analyse", {"ids": [NEWS2]})[0] for _ in range(4)]
        self.assertEqual(codes, [202, 202, 429, 429])
        st, hd, b = self.post("/api/rt/analyse", {"ids": [NEWS2]})
        self.assertEqual(st, 429)
        self.assertIn("retry-after", hd)
        server.RT_INSIGHT_LIMIT = claimcheck.RateLimit(3)
        codes = [self.post("/api/rt/insights", {"scope": {}})[0] for _ in range(5)]
        self.assertEqual(codes, [200, 200, 200, 429, 429])
        self.assertEqual(server.RT_INSIGHT_RATE, 10)
        self.assertEqual(self.get("/api/rt/status")[0], 200)  # reads keep their own budget
        with mock.patch.object(server, "RT_RATE", 3):
            server._rt_hits.clear()
            server.RT_INSIGHT_LIMIT = claimcheck.RateLimit(100)
            codes = [self.post("/api/rt/insights", {})[0] for _ in range(5)]
            self.assertEqual(codes.count(429), 2)  # the general /api/rt limit also applies to POST


class HttpJobs(WbHttp):
    def test_end_to_end_and_light_polling(self):
        with self.engine(delay=0.2) as e:
            st, hd, b = self.post("/api/rt/analyse", {"ids": [NEWS2, SOC11, SHORT1]})
            self.assertEqual(st, 202)
            self.assertEqual(hd["cache-control"], "no-store")
            self.assertRegex(b["job_id"], r"^[0-9a-f]{16}$")
            self.assertEqual((b["total"], b["state"]), (3, "queued"))
            light = self.poll(b["job_id"], light=True)
            self.assertTrue(all(i["result"] is None and i["has_result"] in (True, False) for i in light["items"]))
            full = self.get("/api/rt/jobs/" + b["job_id"])[2]
            self.assertEqual([i["state"] for i in full["items"]], ["done", "done", "skipped"])
            self.assertEqual(full["items"][0]["result"]["step1"]["label"], "FAKE")
            self.assertEqual((full["total"], full["done"], full["state"]), (3, 3, "done"))
            self.assertEqual(set(full), {"job_id", "state", "total", "done", "cancelling", "queue_position", "force",
                                         "created_at", "started_at", "finished_at", "server_time", "engine_version", "items"})
            self.assertEqual(e.count, 2)
            # the stored results show up in the item detail
            d = self.get("/api/rt/item/" + NEWS2)[2]
            self.assertEqual(d["analyses"][0]["engine_version"], "check_v1")
            self.assertEqual(d["analyses"][0]["result"]["step1"]["label"], "FAKE")

    def test_cancel_over_http(self):
        with self.engine(delay=0.4):
            st, _, b = self.post("/api/rt/analyse", {"ids": [NEWS2, SOC11]})
            until(lambda: self.get("/api/rt/jobs/" + b["job_id"])[2]["items"][0]["state"] == "running")
            st, _, c = self.post("/api/rt/jobs/%s/cancel" % b["job_id"])  # empty body is fine
            self.assertEqual((st, c["cancelling"]), (200, True))
            self.assertEqual(self.poll(b["job_id"])["state"], "cancelled")

    def test_errors(self):
        for body, code in (({"ids": []}, 400), ({"ids": "x"}, 400), ({}, 400), ({"ids": [NEWS2], "force": "yes"}, 400),
                           ({"ids": [NEWS2], "extra": 1}, 400), ({"ids": ["i%d" % i for i in range(26)]}, 400),
                           ({"ids": ["nope:1"]}, 404)):
            st, hd, b = self.post("/api/rt/analyse", body)
            self.assertEqual((st, list(b)), (code, ["error"]), body)
        self.assertEqual(self.get("/api/rt/jobs/" + "f" * 16)[0], 404)
        self.assertEqual(self.get("/api/rt/jobs/xyz")[0], 404)
        self.assertEqual(self.get("/api/rt/jobs/../status")[0], 404)
        self.assertEqual(self.post("/api/rt/jobs/" + "f" * 16 + "/cancel")[0], 404)
        self.assertEqual(self.post("/api/rt/nope", {})[0], 404)
        self.assertEqual(self.post("/api/rt/jobs/xyz/cancel", {})[0], 404)

    def test_strict_json_bodies(self):
        for raw in ('{"ids": [NaN]}', '{"ids": [Infinity]}', "not json", "[1]", '"s"', "", "{", '{"ids": ["\ud800"]}'):
            st, hd, raw_b = call(self.port, "POST", "/api/rt/analyse", body=raw.encode("utf-8", "surrogatepass"),
                                 headers={"Content-Type": "application/json"})
            self.assertEqual(st, 400, raw)
            self.assertEqual(list(json.loads(raw_b)), ["error"])
        big = json.dumps({"ids": ["a" * 100] * 200})
        self.assertEqual(call(self.port, "POST", "/api/rt/analyse", body=big)[0], 413)
        st, _, _ = call(self.port, "POST", "/api/rt/analyse", raw_headers={"Content-Length": "abc"})
        self.assertEqual(st, 400)
        st, _, raw = call(self.port, "POST", "/api/rt/insights", body='{"scope": {"days": NaN}}')
        self.assertEqual(st, 400)

    def test_responses_are_strict_json(self):
        with self.engine():
            st, hd, raw = call(self.port, "POST", "/api/rt/analyse", body={"ids": [NEWS2]})
            jid = json.loads(raw)["job_id"]
            self.poll(jid)
            for path in ("/api/rt/jobs/" + jid, "/api/rt/jobs/%s?results=0" % jid):
                st, hd, raw = call(self.port, "GET", path)
                self.assertTrue(hd["content-type"].startswith("application/json"))
                json.loads(raw.decode(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        st, hd, raw = call(self.port, "POST", "/api/rt/insights", body={"scope": {"kind": "news"}})
        json.loads(raw.decode(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


class HttpInsights(WbHttp):
    def test_template_shape(self):
        st, hd, r = self.post("/api/rt/insights", {"scope": {"kind": "social"}})
        self.assertEqual(st, 200)
        self.assertEqual(set(r), {"scope", "scope_desc", "scope_line", "synced_at", "facts", "template_insights", "phrased",
                                  "phrasing", "cached", "generated_at"})
        self.assertIsNone(r["phrased"])
        self.assertEqual(r["facts"]["n_items"], 6)
        self.assertTrue(r["scope_line"].startswith("Based on 6 items from the realtime database, synced "))
        self.assertTrue(r["scope_line"].endswith("No outside knowledge used."))
        self.assertEqual(r["scope_desc"], "kind social")

    def test_phrased_through_http_and_refusal(self):
        ok = lambda p: (200, {"text": "There are 12 items in scope.", "citations": [p["facts"]["rows"][0]["id"]]})
        with FakeInsight(make=ok) as svc:
            server.INSIGHTS = rt_insights.Insights(self.api, url=svc.url)
            st, hd, r = self.post("/api/rt/insights", {"scope": {}, "phrase": True, "question": "what?"})
            self.assertEqual(r["phrased"]["text"], "There are 12 items in scope.")
            svc.make = lambda p: (200, {"text": "There are 99 items.", "citations": ["x:1"]})
            st, hd, r = self.post("/api/rt/insights", {"scope": {}, "phrase": True, "question": "other"})
            self.assertTrue(r["phrased"]["refused"])
            self.assertTrue(r["template_insights"])

    def test_service_down_over_http(self):
        st, hd, r = self.post("/api/rt/insights", {"scope": {"ids": [NEWS1, SOC11]}, "phrase": True})
        self.assertEqual(st, 200)
        self.assertIsNone(r["phrased"])
        self.assertEqual(r["phrasing"]["reason"], "phrasing service not available")

    def test_bad_requests(self):
        for body in ({"scope": {"kind": "x"}}, {"scope": {"ids": ["a b"]}}, {"scope": []}, {"phrase": 1}, {"zzz": 1}):
            st, hd, b = self.post("/api/rt/insights", body)
            self.assertEqual((st, list(b)), (400, ["error"]), body)


if __name__ == "__main__":
    unittest.main()
