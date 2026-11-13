"""POST /api/analyse and GET /api/examples on the real handler (temp build/, fake engine on an ephemeral port)."""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

import helpers
import fixture
import fake_engine
import server
import claimcheck
from helpers import call, jcall, needs_index

TMP = None
FLOOD = "PM announces free electricity for all households"
LEGIT = "New highway opens in Kerala"
NOVEL = "Mayor of Zorbleton bans all umbrellas at the harvest parade"


def setUpModule():
    global TMP
    TMP = tempfile.mkdtemp()
    fixture.write_inputs(TMP)
    fixture.run_build(TMP)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved = {k: getattr(server, k) for k in ("ITEMS_P", "INDEX_P", "AUTH", "S")}
        server.ITEMS_P = os.path.join(TMP, "build", "items.json")
        server.INDEX_P = os.path.join(TMP, "build", "index.pkl")
        server.AUTH = None
        server.S = server.Store()
        cls._log = server.H.log_message
        server.H.log_message = lambda *a, **k: None
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        cls.httpd.daemon_threads = True
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(5)
        server.H.log_message = cls._log
        for k, v in cls._saved.items():
            setattr(server, k, v)

    def setUp(self):
        claimcheck.reset_state()
        self._saved_cc = (claimcheck.ENGINE_URL, claimcheck.ENGINE_TIMEOUT, claimcheck.QUEUE_WAIT)
        claimcheck.ENGINE_URL = helpers.DEAD_ENGINE_URL
        self._se = sys.stderr
        sys.stderr = io.StringIO()  # one engine log line per call goes to stderr
        self.addCleanup(self._restore)

    def _restore(self):
        sys.stderr = self._se
        claimcheck.ENGINE_URL, claimcheck.ENGINE_TIMEOUT, claimcheck.QUEUE_WAIT = self._saved_cc
        claimcheck.reset_state()

    def analyse(self, text, **kw):
        helpers.assert_not_real_flag(claimcheck.ENGINE_URL)
        return jcall(self.port, "POST", "/api/analyse", body={"text": text, **kw})

    @contextlib.contextmanager
    def engine(self, **kw):
        with fake_engine.FakeEngine(**kw) as e:
            claimcheck.ENGINE_URL = e.url
            yield e


class Analyse(Base):
    def test_strong_match_fake_leads_and_shows_step2(self):
        spans = [{"technique": "Loaded_Language", "text": "free electricity"}]
        with self.engine(make=lambda t: fake_engine.v0_response("LEGIT", spans=spans)) as e:
            st, _, b = self.analyse(FLOOD)
        self.assertEqual(st, 200)
        s1, s2 = b["step1"], b["step2"]
        self.assertEqual((s1["source"], s1["lead"], s1["label"]), ("fact-check", "index", "MISLEADING"))
        self.assertGreaterEqual(b["index"]["nearest_similarity"], 0.5)
        self.assertEqual(s1["second_opinion"]["label"], "LEGIT")
        self.assertFalse(s1["second_opinion"]["agrees"])
        self.assertTrue(s2["shown"])
        self.assertFalse(s2["gated_out"])
        self.assertEqual(s2["propaganda"]["techniques"][0]["text"], "free electricity")
        self.assertEqual(b["engine"]["state"], "ok")
        self.assertNotIn("propaganda", b["engine"])
        self.assertEqual((b["truncated"], b["text_chars"]), (False, len(FLOOD)))
        self.assertEqual(e.count, 1)

    def test_legit_still_shows_propaganda_and_force_is_ignored(self):
        spans = [{"technique": "Doubt", "text": "Kerala"}]
        with self.engine(make=lambda t: fake_engine.v0_response("LEGIT", spans=spans)) as e:
            _, _, b = self.analyse(LEGIT)
            self.assertEqual(b["step1"]["label"], "LEGIT")
            self.assertTrue(b["step2"]["shown"])
            self.assertFalse(b["step2"]["gated_out"])
            self.assertFalse(b["step2"]["forced"])
            self.assertEqual(b["step2"]["propaganda"]["techniques"][0]["technique"], "Doubt")
            _, _, f = self.analyse(LEGIT, force_propaganda=True)  # old clients may still send it: no effect
            self.assertTrue(f["step2"]["shown"])
            self.assertFalse(f["step2"]["gated_out"])
            self.assertFalse(f["step2"]["forced"])
            self.assertTrue(f["engine"]["cached"])
            self.assertEqual(e.count, 1)

    def test_no_match_engine_leads_model_estimate(self):
        with self.engine(make=lambda t: fake_engine.v0_response("FAKE")) as e:
            _, _, b = self.analyse(NOVEL)
        self.assertEqual((b["step1"]["source"], b["step1"]["lead"], b["step1"]["note"]),
                         ("model estimate", "engine", "no matching fact-check"))
        self.assertEqual(b["step1"]["label"], "FAKE")
        self.assertAlmostEqual(b["step1"]["plausibility"], 0.85)
        self.assertTrue(b["step2"]["shown"])

    def test_engine_leads_legit_still_shows_both(self):
        probs = {"FAKE": 0.05, "MISLEADING": 0.05, "LEGIT": 0.9, "PROPAGANDA": 0, "INCONCLUSIVE": 0}
        with self.engine(make=lambda t: fake_engine.v0_response("LEGIT", probs=probs)):
            _, _, b = self.analyse(NOVEL)
        self.assertEqual(b["step1"]["label"], "LEGIT")
        self.assertTrue(b["step2"]["shown"])
        self.assertFalse(b["step2"]["gated_out"])

    def test_engine_down_gives_index_only(self):
        st, hd, b = self.analyse(FLOOD)
        self.assertEqual(st, 200)
        self.assertEqual(b["engine"]["state"], "unavailable")
        self.assertIn("reason", b["engine"])
        self.assertNotIn("Traceback", str(b))
        self.assertEqual((b["step1"]["source"], b["step1"]["label"]), ("fact-check", "MISLEADING"))  # strong match still leads
        self.assertIsNotNone(b["step1"]["plausibility"])
        self.assertEqual((b["step2"]["shown"], b["step2"]["available"], b["step2"]["propaganda"]), (False, False, None))
        self.assertEqual(b["index"]["matches"][0]["id"], "fc2")

    def test_engine_down_no_match(self):
        st, _, b = self.analyse(NOVEL)
        self.assertEqual((st, b["step1"]["source"], b["step1"]["plausibility"]), (200, "none", None))

    def test_engine_error_states(self):
        with self.engine(status=500):
            self.assertEqual(self.analyse(FLOOD)[2]["engine"]["state"], "error")
        with self.engine(make=lambda t: {"nonsense": 1}):
            self.assertEqual(self.analyse(NOVEL)[2]["engine"]["state"], "error")

    def test_timeout(self):
        claimcheck.ENGINE_TIMEOUT = 0.3
        with self.engine(delay=1.0):
            t0 = time.time()
            st, _, b = self.analyse(FLOOD)
        self.assertLess(time.time() - t0, 3)
        self.assertEqual((st, b["engine"]["state"]), (200, "timeout"))
        self.assertEqual(b["step1"]["source"], "fact-check")

    def test_busy_when_queue_full_has_retry_after(self):
        claimcheck.GATE = claimcheck.Gate(max_waiting=0)
        res = {}
        with self.engine(delay=1.0) as e:
            t = threading.Thread(target=lambda: res.update(first=self.analyse(FLOOD)))
            t.start()
            time.sleep(0.4)
            st, hd, b = self.analyse(NOVEL)
            t.join(10)
        self.assertEqual(st, 200)
        self.assertEqual(b["engine"]["state"], "busy")
        self.assertGreaterEqual(int(hd["retry-after"]), 5)
        self.assertEqual(b["step1"]["source"], "none")
        self.assertEqual(res["first"][2]["engine"]["state"], "ok")
        self.assertEqual(e.count, 1)  # the busy request never reached the engine

    def test_second_request_queues_then_runs(self):
        res = {}
        with self.engine(delay=0.6) as e:
            t = threading.Thread(target=lambda: res.update(first=self.analyse(FLOOD)))
            t.start()
            time.sleep(0.2)
            _, _, b = self.analyse(NOVEL)
            t.join(10)
        self.assertEqual(b["engine"]["state"], "ok")
        self.assertEqual(b["engine"]["queue_position"], 1)
        self.assertGreater(b["engine"]["queue_wait_s"], 0.2)
        self.assertEqual(res["first"][2]["engine"]["queue_position"], 0)
        self.assertEqual(e.count, 2)

    def test_identical_call_is_cached(self):
        with self.engine() as e:
            _, _, a = self.analyse(NOVEL)
            _, _, b = self.analyse(NOVEL)
            _, _, c = self.analyse("  " + NOVEL + "  ")  # whitespace around the text does not matter
            self.assertEqual(e.count, 1)
        self.assertFalse(a["engine"]["cached"])
        self.assertTrue(b["engine"]["cached"] and c["engine"]["cached"])
        self.assertEqual(a["step1"], b["step1"])

    def test_cache_is_lru_and_ttl(self):
        c = claimcheck.Cache(max_entries=2, ttl=3600)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")
        c.put("c", 3)
        self.assertEqual((c.get("a"), c.get("b"), c.get("c")), (1, None, 3))
        c = claimcheck.Cache(max_entries=2, ttl=0.05)
        c.put("a", 1)
        time.sleep(0.1)
        self.assertIsNone(c.get("a"))
        self.assertEqual((claimcheck.CACHE_MAX, claimcheck.CACHE_TTL), (200, 3600))

    def test_failures_are_not_cached(self):
        self.analyse(NOVEL)  # engine down
        with self.engine() as e:
            self.assertEqual(self.analyse(NOVEL)[2]["engine"]["state"], "ok")
            self.assertEqual(e.count, 1)

    @needs_index

    def test_rate_limit_429(self):
        claimcheck.LIMIT = claimcheck.RateLimit(per_min=3)
        with self.engine() as e:
            for i in range(3):
                self.assertEqual(self.analyse("distinct claim number %d about Zorbleton" % i)[0], 200)
            st, hd, b = self.analyse("distinct claim number 9 about Zorbleton")
            self.assertEqual(st, 429)
            self.assertGreaterEqual(int(hd["retry-after"]), 1)
            self.assertIn("error", b)
            self.assertEqual(self.analyse("distinct claim number 0 about Zorbleton")[0], 200)  # cache hits are free
            self.assertEqual(e.count, 3)
        self.assertEqual(claimcheck.ENGINE_RATE, 10)

    def test_truncation(self):
        long = ("Zorbleton news " * 700)
        seen = []
        with self.engine(make=lambda t: seen.append(len(t)) or fake_engine.v0_response()):
            _, _, b = self.analyse(long)
        self.assertTrue(b["truncated"])
        self.assertEqual(seen, [b["engine_chars"]])
        self.assertLessEqual(seen[0], 8000)
        self.assertEqual(b["text_chars"], len(long.strip()))

    def test_validation(self):
        post = lambda body, **kw: call(self.port, "POST", "/api/analyse", body=body, **kw)
        self.assertEqual(post({"text": "ab"})[0], 400)
        self.assertEqual(post({})[0], 400)
        self.assertEqual(post("not json")[0], 400)
        self.assertEqual(post("[1,2]")[0], 400)
        self.assertEqual(post({"text": 5})[0], 400)
        self.assertEqual(post(None, raw_headers={"Content-Length": "100001"})[0], 413)
        self.assertEqual(claimcheck.GATE.waiting(), 0)

    def test_lang_is_validated_and_forwarded(self):
        seen = []
        with self.engine(make=lambda t: fake_engine.v0_response()) as e:
            self.analyse(NOVEL, lang="hi")
            self.analyse(NOVEL + " again", lang="<script>")  # junk lang is ignored, not forwarded
        self.assertEqual(e.count, 2)

    def test_strict_json_with_nan_from_engine(self):
        probs = {"FAKE": float("nan"), "MISLEADING": 0.6, "LEGIT": 0.4}
        with self.engine(make=lambda t: fake_engine.v0_response("MISLEADING", probs=probs, secs=float("nan"))):
            st, _, b = self.analyse(NOVEL)  # jcall parses with parse_constant that rejects NaN
        self.assertEqual(st, 200)
        self.assertNotIn("FAKE", b["engine"]["verdict"]["probs"])

    def test_no_claim_text_in_log(self):
        with self.engine():
            self.analyse(NOVEL)
        log = sys.stderr.getvalue()
        self.assertIn("engine ip=127.0.0.1", log)
        self.assertNotIn("Zorbleton", log)

    def test_503_when_index_not_built(self):
        old = server.S
        server.S = server.Store()
        server.ITEMS_P = os.path.join(TMP, "build", "gone.json")
        try:
            self.assertEqual(call(self.port, "POST", "/api/analyse", body={"text": "hello world"})[0], 503)
        finally:
            server.ITEMS_P = os.path.join(TMP, "build", "items.json")
            server.S = old


class Combine(unittest.TestCase):
    def test_plausibility_rules(self):
        f = claimcheck.engine_plausibility
        self.assertAlmostEqual(f({"label": "X", "probs": {"FAKE": 0.3, "MISLEADING": 0.3}}), 0.6)
        self.assertEqual(f({"label": "FAKE", "probs": {}}), 1.0)
        self.assertEqual(f({"label": "LEGIT", "probs": {}}), 0.0)
        self.assertEqual(f({"label": "LEGIT", "probs": {"FAKE": 0.1}, "plausibly_fake": True}), 0.5)

    def test_weak_index_match_and_engine_down_is_index_only(self):
        idx = {"matches": [{"verdict_norm": "FAKE", "site": "x"}], "score": 0.7, "nearest_similarity": 0.4}
        s1, s2 = claimcheck.combine(idx, {"state": "unavailable", "reason": "r"}, True)
        self.assertEqual((s1["source"], s1["label"], s1["plausibility"]), ("index only", None, 0.7))
        self.assertEqual((s2["shown"], s2["gated_out"], s2["available"]), (False, False, False))

    def test_weak_index_match_does_not_lead(self):
        idx = {"matches": [{"verdict_norm": "FAKE", "site": "x"}], "score": 0.7, "nearest_similarity": 0.45}
        eng = {"state": "ok", "verdict": {"label": "LEGIT", "probs": {"LEGIT": 1.0}}, "propaganda": {"techniques": []}}
        s1, s2 = claimcheck.combine(idx, eng, False)
        self.assertEqual((s1["source"], s1["label"]), ("model estimate", "LEGIT"))
        self.assertTrue(s2["shown"])
        self.assertFalse(s2["gated_out"])


class Examples(Base):
    def get(self, path="/api/examples"):
        return jcall(self.port, "GET", path)

    def test_shape_and_balance_on_fixture(self):
        st, _, b = self.get("/api/examples?n=3")
        self.assertEqual(st, 200)
        self.assertEqual(sorted(i["verdict_norm"] for i in b["items"]), ["FAKE", "LEGIT", "MISLEADING"])
        for it in b["items"]:
            self.assertEqual(set(it), {"id", "title", "claim", "verdict_norm", "site", "lang"})
        self.assertEqual({i["id"] for i in b["items"]}, {"fc1-dup", "fc2", "fc3"})  # BOGUS (no verdict) never appears

    def test_n_clamped(self):
        self.assertEqual(len(self.get("/api/examples?n=99")[2]["items"]), 3)  # pool of 3 < 8
        self.assertEqual(len(self.get("/api/examples?n=0")[2]["items"]), 1)
        self.assertEqual(len(self.get("/api/examples?n=abc")[2]["items"]), 3)  # default 4, pool of 3

    def test_pool_built_once_per_index(self):
        built = []
        orig = claimcheck.build_pool
        claimcheck.build_pool = lambda items: built.append(1) or orig(items)
        try:
            claimcheck.EXAMPLES = claimcheck.ExamplePool()
            for _ in range(4):
                self.get()
        finally:
            claimcheck.build_pool = orig
        self.assertEqual(len(built), 1)


def synth(n_sites=10, per=30):
    items = []
    langs = ["en", "hi-IN", "ta", "te", "ur"]
    for v in ("FAKE", "MISLEADING", "LEGIT", "INCONCLUSIVE", "UNRATED"):
        for s in range(n_sites):
            for k in range(per):
                items.append({"id": "%s-%d-%d" % (v, s, k), "kind": "factcheck", "site": "site%d" % s,
                              "verdict_norm": v, "lang": langs[(s + k) % 5],
                              "title": "A reasonably long title %d" % k, "claim": "A claim that is long enough %d" % k})
    items += [
        {"id": "cand", "kind": "candidate", "site": "reddit_posts", "verdict_norm": "FAKE", "lang": "en",
         "title": "A reasonably long title x", "claim": "A claim that is long enough x"},
        {"id": "caps", "kind": "factcheck", "site": "s", "verdict_norm": "FAKE", "lang": "en",
         "title": "SHOUTING TITLE THAT IS LONG ENOUGH", "claim": "A claim that is long enough"},
        {"id": "short", "kind": "factcheck", "site": "s", "verdict_norm": "FAKE", "lang": "en",
         "title": "Too short", "claim": "A claim that is long enough"},
    ]
    return items


class Sampling(unittest.TestCase):
    def test_filters(self):
        pool = claimcheck.build_pool(synth())
        self.assertEqual(set(pool), {"FAKE", "MISLEADING", "LEGIT"})
        ids = {it["id"] for sites in pool.values() for its in sites.values() for it in its}
        self.assertFalse(ids & {"cand", "caps", "short"})

    def test_balance_site_spread_and_randomness(self):
        pool = claimcheck.build_pool(synth())
        seen = set()
        for _ in range(40):
            got = claimcheck.sample(pool, 4)
            self.assertEqual(len(got), 4)
            self.assertEqual(len({g["id"] for g in got}), 4)
            self.assertEqual(len({g["site"] for g in got}), 4)  # never 2 from one site while sites remain
            self.assertEqual({g["verdict_norm"] for g in got}, {"FAKE", "MISLEADING", "LEGIT"})
            seen.add(tuple(sorted(g["id"] for g in got)))
        self.assertGreater(len(seen), 30)  # different on (almost) every call

    def test_n_three_has_one_of_each(self):
        pool = claimcheck.build_pool(synth())
        for _ in range(10):
            self.assertEqual(sorted(g["verdict_norm"] for g in claimcheck.sample(pool, 3)),
                             ["FAKE", "LEGIT", "MISLEADING"])

    def test_empty_pool(self):
        self.assertEqual(claimcheck.sample({}, 4), [])


if __name__ == "__main__":
    unittest.main()
