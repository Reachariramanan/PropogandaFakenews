"""Route tests: the real handler class on an ephemeral 127.0.0.1 port, with a temp build/ and Propaganda Lab dir.
The flag proxy only ever points at a closed port or at an in-process stub on another ephemeral port."""
import base64
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import helpers
import fixture
import rt_api
import rt_enrich
import server
from helpers import call, jcall

TMP = None
PL = None
REAL_LOG = server.H.log_message  # the real access-log method, to restore after quieting it


def setUpModule():
    global TMP, PL
    TMP = tempfile.mkdtemp()
    fixture.write_inputs(TMP)
    fixture.run_build(TMP)  # real build_index.main() on the tiny corpus -> TMP/build/{items,index}
    PL = os.path.join(TMP, "pl")
    os.makedirs(os.path.join(PL, "country"))
    os.makedirs(os.path.join(PL, "posts", "about"))
    os.makedirs(os.path.join(PL, "posts", "origin"))

    def w(rel, txt):
        with open(os.path.join(PL, rel), "w", encoding="utf-8") as f:
            f.write(txt)
    w("meta.json", json.dumps({"version": "prop_lab_v1", "built_at": "2026-11-05T09:17:25Z", "n_posts": 3}))
    w("heatmap.json", json.dumps({"techniques": ["Loaded_Language", "Repetition", 7], "views": {}}))
    w("countries.json", "{}")
    w("world_paths.json", "{}")
    w("country/IND.json", '{"iso": "IND", "n": 3, "score": NaN, "nested": {"x": [Infinity, 1.5]}}')
    w("country/BAD.json", "{ not json")
    w("posts/about/IND.jsonl", "\n".join([json.dumps({"post_uid": "p%d" % i, "text": "t%d" % i}) for i in range(5)])
      + "\n\nnot json\n")


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class ServerCase(unittest.TestCase):
    """Starts H on 127.0.0.1:0 and points server.py's module globals at the temp fixture."""
    ITEMS = PLDIR = None

    @classmethod
    def setUpClass(cls):
        cls._saved = {k: getattr(server, k) for k in ("ITEMS_P", "INDEX_P", "PL_DIR", "FLAG_URL", "AUTH", "S", "PL", "RT")}
        server.ITEMS_P = os.path.join(TMP, "build", "items.json")
        server.INDEX_P = os.path.join(TMP, "build", "index.pkl")
        server.PL_DIR = PL
        server.FLAG_URL = helpers.DEAD_FLAG_URL
        server.AUTH = None
        server.S = server.Store()
        server.PL = server.PropLabCache()
        nort = os.path.join(TMP, "no_realtime")  # never the real store: a path that does not exist
        server.RT = rt_api.RtApi(db_path=os.path.join(nort, "realtime.sqlite"), enrich=rt_enrich.Enrich(
            export_dir=os.path.join(nort, "x"), triage_dir=os.path.join(nort, "t"),
            sidecar=os.path.join(nort, "enr.sqlite")))
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
        server.H.log_message = REAL_LOG
        for k, v in cls._saved.items():
            setattr(server, k, v)

    def setUp(self):
        helpers.assert_not_real_flag(server.FLAG_URL)
        server._flag_hits.clear()
        server._auth_fails.clear()
        server._flag_tech.update(mtime=None, set=frozenset())
        # the proxy prints an audit line per accepted flag (stdout), the gate one per failed login (stderr)
        self._so, self._se = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()

    def tearDown(self):
        sys.stdout, sys.stderr = self._so, self._se
        server._flag_hits.clear()
        server._auth_fails.clear()
        server.AUTH = None
        server.FLAG_URL = helpers.DEAD_FLAG_URL

    @contextlib.contextmanager
    def patched(self, **kw):
        old = {k: getattr(server, k) for k in kw}
        for k, v in kw.items():
            setattr(server, k, v)
        try:
            yield
        finally:
            for k, v in old.items():
                setattr(server, k, v)

    def get(self, path, **kw):
        return call(self.port, "GET", path, **kw)

    def jget(self, path, **kw):
        return jcall(self.port, "GET", path, **kw)

    def post(self, path, body=None, **kw):
        return call(self.port, "POST", path, body=body, **kw)


class FlagStub:
    """In-process stand-in for the flag service on an ephemeral port; records what it was sent."""

    def __init__(self, status=200, payload=b'{"ok": true}'):
        self.requests, self.status, self.payload = [], status, payload
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.requests.append(json.loads(self.rfile.read(n)))
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(outer.payload)))
                self.end_headers()
                self.wfile.write(outer.payload)

            def log_message(self, *a):
                pass
        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d/api/prop_flag" % self.httpd.server_address[1]
        helpers.assert_not_real_flag(self.url)
        self.t = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.t.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.t.join(5)


# --------------------------------------------------------------------------------------------- /healthz
class Healthz(ServerCase):
    def test_ok_shape(self):
        st, hd, b = self.jget("/healthz")
        self.assertEqual(st, 200)
        self.assertEqual(set(b), {"ok", "build_built_at", "n_items", "proplab_built_at", "uptime_s",
                                 "realtime"})
        self.assertEqual(b["realtime"], {"enabled": False, "running": False, "last_sync": None,
                                         "lag_minutes": None, "rows": 0, "errors": 0,
                                         "analysis": {"enabled": False, "last_run": None, "next_run_at": None,
                                                      "analysed_deep": 0, "partial_items": 0,
                                                      "rolling": {"cycles": 0, "analysed": 0, "degraded": 0,
                                                                  "errors": 0, "replies": 0, "success_rate": None}}})
        self.assertIs(b["ok"], True)
        self.assertEqual(b["n_items"], 7)
        with open(server.ITEMS_P, encoding="utf-8") as f:
            self.assertEqual(b["build_built_at"], json.load(f)["meta"]["built"])
        self.assertEqual(b["proplab_built_at"], "2026-11-05T09:17:25Z")
        self.assertIsInstance(b["uptime_s"], (int, float))
        self.assertGreaterEqual(b["uptime_s"], 0)
        self.assertEqual(hd["cache-control"], "no-store")

    def test_uptime_increases(self):
        a = self.jget("/healthz")[2]["uptime_s"]
        time.sleep(0.25)
        self.assertGreater(self.jget("/healthz")[2]["uptime_s"], a)

    def test_query_string_ignored(self):
        self.assertEqual(self.get("/healthz?x=1")[0], 200)

    def test_proplab_not_built_gives_null_but_still_ok(self):
        with self.patched(PL_DIR=os.path.join(TMP, "no-such-dir"), PL=server.PropLabCache()):
            st, _, b = self.jget("/healthz")
        self.assertEqual(st, 200)
        self.assertIsNone(b["proplab_built_at"])

    def test_missing_items_json_is_503(self):
        with self.patched(ITEMS_P=os.path.join(TMP, "build", "gone.json")):
            st, _, b = self.jget("/healthz")
        self.assertEqual(st, 503)
        self.assertIs(b["ok"], False)
        self.assertIn("error", b)
        self.assertNotIn(TMP, json.dumps(b))  # no filesystem paths leaked

    def test_unreadable_items_json_is_503_then_recovers(self):
        bad = os.path.join(TMP, "build", "bad_items.json")
        with open(bad, "w") as f:
            f.write("{ truncated")
        with self.patched(ITEMS_P=bad, S=server.Store()):
            self.assertEqual(self.jget("/healthz")[0], 503)
            self.assertEqual(self.jget("/healthz")[0], 503)  # cached failure, same answer
            with open(bad, "w") as f:  # a rebuild replaces the file -> new signature -> re-read
                json.dump({"meta": {"built": "2026-01-01T00:00:00+00:00"}, "items": [{"id": "x"}]}, f)
            st, _, b = self.jget("/healthz")
        self.assertEqual((st, b["n_items"], b["build_built_at"]), (200, 1, "2026-01-01T00:00:00+00:00"))

    def test_wrong_shape_items_json_is_503(self):
        bad = os.path.join(TMP, "build", "shape_items.json")
        with open(bad, "w") as f:
            json.dump({"meta": {}, "items": "nope"}, f)
        with self.patched(ITEMS_P=bad, S=server.Store()):
            self.assertEqual(self.jget("/healthz")[0], 503)

    def test_uses_loaded_store_without_reparsing(self):
        self.assertEqual(self.jget("/api/stats")[0], 200)  # loads the Store
        bad = os.path.join(TMP, "build", "x.json")
        calls = []
        real_open = open

        def spy(p, *a, **k):
            calls.append(p)
            return real_open(p, *a, **k)
        with unittest.mock.patch("builtins.open", spy):
            self.assertEqual(self.jget("/healthz")[0], 200)
        self.assertNotIn(server.ITEMS_P, calls)


# ------------------------------------------------------------------------------------------- /api/*
class Api(ServerCase):
    def test_items_default(self):
        st, hd, b = self.jget("/api/items")
        self.assertEqual(st, 200)
        self.assertEqual(b["total"], 7)
        self.assertEqual((b["page"], b["pages"]), (1, 1))
        self.assertEqual(len(b["items"]), 7)
        self.assertTrue({"id", "claim", "verdict_norm", "plaus", "summary"} <= set(b["items"][0]))
        self.assertNotIn("body", b["items"][0])
        self.assertEqual(hd["cache-control"], "no-store")

    def test_items_filters_and_paging(self):
        b = self.jget("/api/items?verdict=FAKE")[2]
        self.assertTrue(b["total"] >= 2 and all(i["verdict_norm"] == "FAKE" for i in b["items"]))
        self.assertEqual(self.jget("/api/items?site=altnews")[2]["total"], 1)
        self.assertEqual(self.jget("/api/items?lang=hi")[2]["total"], 1)
        self.assertEqual(self.jget("/api/items?kind=candidate")[2]["total"], 1)
        self.assertEqual(self.jget("/api/items?q=electricity")[2]["total"], 2)  # fc2 + candidate c1
        self.assertEqual(self.jget("/api/items?q=electricity+nonexistentword")[2]["total"], 0)
        p = self.jget("/api/items?page_size=3&page=2")[2]
        self.assertEqual((p["pages"], p["page"], len(p["items"])), (3, 2, 3))
        self.assertEqual(self.jget("/api/items?page=99&page_size=3")[2]["page"], 3)  # clamped
        self.assertEqual(self.jget("/api/items?page_size=100000")[2]["page_size"], 100)
        self.assertEqual(self.jget("/api/items?page_size=abc&page=x")[2]["page_size"], 24)
        self.assertEqual(self.jget("/api/items?min_plaus=zzz")[2]["total"], 7)  # bad number ignored
        hi = self.jget("/api/items?min_plaus=0.9")[2]
        self.assertTrue(hi["total"] and all(i["plaus"] >= 0.9 for i in hi["items"]))
        d = self.jget("/api/items?sort=date")[2]["items"]
        dates = [i["published"] or "" for i in d]
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_items_response_is_strict_json_with_unscored_items(self):
        st, _, raw = self.get("/api/items?kind=candidate")
        self.assertNotIn(b"NaN", raw)
        self.assertIsNone(json.loads(raw)["items"][0]["plaus"])

    def test_stats(self):
        st, _, b = self.jget("/api/stats")
        self.assertEqual((st, b["total"]), (200, 7))
        self.assertEqual(sum(b["verdicts"].values()), 7)
        self.assertEqual(b["unscored"], 1)
        self.assertEqual(len(b["score_histogram"]), 10)

    def test_item_detail_and_404(self):
        st, _, b = self.jget("/api/item/fc2")
        self.assertEqual((st, b["id"]), (200, "fc2"))
        self.assertIn("body", b)
        self.assertEqual(self.jget("/api/item/does-not-exist")[0], 404)

    def test_unknown_api_endpoint(self):
        self.assertEqual(self.jget("/api/nope")[0], 404)
        self.assertEqual(self.post("/api/nope", {})[0], 404)

    def test_check_post(self):
        st, _, b = self.post_json("/api/check", {"text": "PM announces free electricity for all households"})
        self.assertEqual(st, 200)
        self.assertIn(b["state"], ("weak", "moderate", "strong"))
        self.assertIn("signals", b)
        self.assertEqual(b["matches"][0]["id"], "fc2")
        st, _, b = self.post_json("/api/check", {"text": "zzzzqqqq wwwwxxxx"})
        self.assertEqual((st, b["state"], b["score"]), (200, "no_close_match", None))

    def post_json(self, path, body, **kw):
        return jcall(self.port, "POST", path, body=body, **kw)

    def test_check_validation(self):
        self.assertEqual(self.post_json("/api/check", {"text": "ab"})[0], 400)  # too short
        self.assertEqual(self.post_json("/api/check", {})[0], 400)
        self.assertEqual(self.post("/api/check", "not json")[0], 400)
        self.assertEqual(self.post("/api/check", "[1,2]")[0], 400)
        st, _, _ = self.post("/api/check", raw_headers={"Content-Length": "100001"})
        self.assertEqual(st, 413)

    def test_503_when_index_not_built(self):
        with self.patched(ITEMS_P=os.path.join(TMP, "build", "gone.json"), S=server.Store()):
            st, _, b = self.jget("/api/items")
            self.assertEqual(st, 503)
            self.assertEqual(self.post("/api/check", {"text": "hello world"})[0], 503)


class Static(ServerCase):
    def test_index_and_assets(self):
        st, hd, b = self.get("/")
        self.assertEqual(st, 200)
        self.assertIn("text/html", hd["content-type"])
        self.assertIn(b"<html", b.lower())
        self.assertEqual(self.get("/app.js")[0], 200)
        self.assertIn("javascript", self.get("/app.js")[1]["content-type"])
        self.assertEqual(self.get("/styles.css")[0], 200)
        self.assertEqual(self.get("/fonts/Inter-latin.woff2")[0], 200)

    def test_missing_file_404(self):
        self.assertEqual(self.get("/nope.js")[0], 404)

    def test_path_traversal_blocked(self):
        for p in ("/../server.py", "/..%2fserver.py", "/%2e%2e/server.py", "/%2e%2e%2fserver.py",
                  "/fonts/../../server.py", "/..%2f..%2f..%2fetc/passwd"):
            st, _, b = self.get(p)
            self.assertEqual(st, 404, p)
            self.assertNotIn(b"import ", b)
            self.assertNotIn(b"root:", b)


# ---------------------------------------------------------------------------------- Propaganda Lab
class PropLab(ServerCase):
    def test_static_documents_and_etag(self):
        st, hd, b = self.jget("/api/proplab/meta")
        self.assertEqual((st, b["version"]), (200, "prop_lab_v1"))
        self.assertTrue(hd["etag"])
        st, hd2, raw = self.get("/api/proplab/meta", headers={"If-None-Match": hd["etag"]})
        self.assertEqual((st, raw), (304, b""))
        for ep in ("world", "countries", "heatmap"):
            self.assertEqual(self.get("/api/proplab/" + ep)[0], 200, ep)

    def test_country_nan_becomes_null(self):
        st, _, b = self.jget("/api/proplab/country/IND")
        self.assertEqual(st, 200)
        self.assertEqual((b["n"], b["score"], b["nested"]["x"]), (3, None, [None, 1.5]))

    def test_bad_iso3_is_400(self):
        for iso in ("in", "IN", "INDX", "ind", "I1D", "IN%20", "%E0%A4%85%E0%A4%86%E0%A4%87", "I.D"):
            st, _, b = self.jget("/api/proplab/country/" + iso)
            self.assertEqual(st, 400, iso)
            self.assertIn("ISO3", b["error"])

    def test_country_traversal_never_serves_files(self):
        for p in ("/api/proplab/country/..%2fmeta", "/api/proplab/country/../../meta", "/api/proplab/country/I%2FD",
                  "/api/proplab/country/%2e%2e%2f%2e%2e%2fmeta.json"):
            self.assertIn(self.get(p)[0], (400, 404), p)

    def test_unknown_country_404_and_corrupt_file_502(self):
        self.assertEqual(self.jget("/api/proplab/country/ZZZ")[0], 404)
        st, _, b = self.jget("/api/proplab/country/BAD")
        self.assertEqual(st, 502)
        self.assertIn("unreadable", b["error"])

    def test_posts_validation_and_paging(self):
        self.assertEqual(self.jget("/api/proplab/posts?mode=bad&iso=IND")[0], 400)
        self.assertEqual(self.jget("/api/proplab/posts?iso=IND")[0], 400)
        self.assertEqual(self.jget("/api/proplab/posts?mode=about&iso=ind")[0], 400)
        self.assertEqual(self.jget("/api/proplab/posts?mode=about&iso=..%2f..")[0], 400)
        self.assertEqual(self.jget("/api/proplab/posts?mode=about&iso=ZZZ")[0], 404)
        st, _, b = self.jget("/api/proplab/posts?mode=about&iso=IND&limit=2")
        self.assertEqual((st, b["count"], b["has_more"], b["next_offset"]), (200, 2, True, 2))
        b = self.jget("/api/proplab/posts?mode=about&iso=IND&offset=2&limit=50")[2]
        self.assertEqual((b["count"], b["has_more"]), (3, False))  # blank and non-JSON lines skipped
        b = self.jget("/api/proplab/posts?mode=about&iso=IND&offset=999")[2]
        self.assertEqual(b["count"], 0)
        b = self.jget("/api/proplab/posts?mode=about&iso=IND&limit=zzz")[2]
        self.assertEqual((b["limit"], b["offset"]), (20, 0))
        self.assertEqual(self.jget("/api/proplab/posts?mode=about&iso=IND&limit=9999")[2]["limit"], 50)

    def test_unknown_endpoint_404(self):
        self.assertEqual(self.jget("/api/proplab/nothing")[0], 404)
        self.assertEqual(self.jget("/api/proplab/")[0], 404)

    def test_not_built_503(self):
        with self.patched(PL_DIR=os.path.join(TMP, "no-such-dir")):
            st, _, b = self.jget("/api/proplab/meta")
        self.assertEqual((st, b["code"]), (503, "not_built"))


# ---------------------------------------------------------------------------------- flag proxy
VALID = {"post_uid": "abc:1_2.x-3", "action": "confirm"}


class FlagValidation(ServerCase):
    """Every case here is rejected before any upstream call, or hits a closed port. Nothing real is contacted."""

    def flag(self, body, **kw):
        helpers.assert_not_real_flag(server.FLAG_URL)
        return jcall(self.port, "POST", "/api/proplab/flag", body=body, **kw)

    def test_target_is_a_closed_port(self):
        self.assertEqual(server.FLAG_URL, helpers.DEAD_FLAG_URL)
        import socket
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", int(server.FLAG_URL.split(":")[2].split("/")[0])), timeout=1)

    def test_bad_json_and_shapes(self):
        for raw in ("not json", "", "{", "[1]", "\"str\"", "null", "123", b"\xff\xfe\x00"):
            st, _, b = self.flag(raw)
            self.assertEqual(st, 400, raw)
            self.assertIn("error", b)

    def test_bad_post_uid(self):
        for uid in (None, 123, ["a"], "", "has space", "../x", "a/b", "a;b", "x" * 129, "ünï", "a\nb"):
            st, _, b = self.flag({"post_uid": uid, "action": "confirm"})
            self.assertEqual((st, b["error"]), (400, "invalid post_uid"), repr(uid))
        st, _, b = self.flag({"action": "confirm"})
        self.assertEqual((st, b["error"]), (400, "invalid post_uid"))

    def test_bad_action(self):
        for a in (None, "delete", "CONFIRM", "", 5, ["confirm"], {"a": 1}):
            st, _, b = self.flag({"post_uid": "u1", "action": a})
            self.assertEqual((st, b["error"]), (400, "invalid action"), repr(a))
        self.assertEqual(self.flag({"post_uid": "u1"})[0], 400)

    def test_bad_technique(self):
        for t in ("Nope", "", "loaded_language", 7, ["Repetition"], {"x": 1}):
            st, _, b = self.flag({**VALID, "technique": t})
            self.assertEqual((st, b["error"]), (400, "invalid technique"), repr(t))

    def test_technique_check_survives_missing_heatmap(self):
        with self.patched(PL_DIR=os.path.join(TMP, "no-such-dir")):
            server._flag_tech.update(mtime=None, set=frozenset())
            st, _, b = self.flag({**VALID, "technique": "Repetition"})
        self.assertEqual((st, b["error"]), (400, "invalid technique"))

    def test_oversize_and_bad_content_length(self):
        self.assertEqual(self.flag(None, raw_headers={"Content-Length": "2049"})[0], 413)
        self.assertEqual(self.flag(None, raw_headers={"Content-Length": "-1"})[0], 413)
        self.assertEqual(self.flag(None, raw_headers={"Content-Length": "abc"})[0], 400)
        st, _, b = self.flag({"post_uid": "u", "action": "confirm", "pad": "x" * 3000})
        self.assertEqual(st, 413)

    def test_upstream_down_is_502(self):
        for body in (VALID, {**VALID, "technique": "Repetition"}, {"post_uid": "u9", "action": "clear"},
                     {"post_uid": "u9", "action": "not_propaganda", "technique": "Loaded_Language"}):
            st, _, b = self.flag(body)
            self.assertEqual((st, b["error"]), (502, "flag service unavailable"), body)
        self.assertEqual(sys.stdout.getvalue(), "")  # no audit line: nothing was recorded upstream

    def test_wrong_path_and_method(self):
        self.assertEqual(self.jget("/api/proplab/flag")[0], 404)  # GET is not the flag route
        self.assertEqual(jcall(self.port, "POST", "/api/proplab/flag/x", body=VALID)[0], 404)

    def test_rate_limit_30_per_minute(self):
        codes = [self.flag({"post_uid": "bad uid", "action": "x"})[0] for _ in range(32)]
        self.assertEqual(codes[:30], [400] * 30)
        self.assertEqual(codes[30:], [429, 429])
        server._flag_hits.clear()
        self.assertEqual(self.flag({"post_uid": "bad uid", "action": "x"})[0], 400)


class FlagProxyAgainstStub(ServerCase):
    def setUp(self):
        super().setUp()
        self.stub = FlagStub()
        server.FLAG_URL = self.stub.url

    def tearDown(self):
        self.stub.close()
        super().tearDown()

    def test_valid_flag_forwarded_exactly_and_audited(self):
        body = {**VALID, "technique": "Repetition", "extra": "dropped"}
        st, _, b = jcall(self.port, "POST", "/api/proplab/flag", body=body)
        self.assertEqual((st, b), (200, {"ok": True}))
        self.assertEqual(self.stub.requests, [{"post_uid": VALID["post_uid"], "action": "confirm",
                                               "technique": "Repetition"}])
        line = sys.stdout.getvalue()
        self.assertIn("flag ip=127.0.0.1 action=confirm post_uid=%s" % VALID["post_uid"], line)

    def test_technique_optional(self):
        self.assertEqual(jcall(self.port, "POST", "/api/proplab/flag", body=VALID)[0], 200)
        self.assertEqual(self.stub.requests, [VALID])

    def test_upstream_error_passed_through_and_not_audited(self):
        self.stub.status, self.stub.payload = 404, b'{"error": "unknown post"}'
        st, _, b = jcall(self.port, "POST", "/api/proplab/flag", body=VALID)
        self.assertEqual((st, b), (404, {"error": "unknown post"}))
        self.assertEqual(sys.stdout.getvalue(), "")

    def test_garbage_upstream_is_502(self):
        self.stub.payload = b"<html>oops</html>"
        st, _, b = jcall(self.port, "POST", "/api/proplab/flag", body=VALID)
        self.assertEqual(st, 502)
        self.assertIn("bad response", b["error"])


# --------------------------------------------------------------------------------------------- auth
USER, PASS = "alice", "s3cr3t:with:colons"


def basic(user=USER, pw=PASS):
    return {"Authorization": "Basic " + base64.b64encode(("%s:%s" % (user, pw)).encode()).decode()}


class AuthUnit(unittest.TestCase):
    def test_creds_from_env(self):
        c = server.auth_creds
        self.assertIsNone(c({}))
        self.assertIsNone(c({"FNP_AUTH_USER": "u"}))
        self.assertIsNone(c({"FNP_AUTH_PASS": "p"}))
        self.assertIsNone(c({"FNP_AUTH_USER": "", "FNP_AUTH_PASS": "p"}))
        self.assertIsNone(c({"FNP_AUTH_USER": "u", "FNP_AUTH_PASS": ""}))
        self.assertEqual(c({"FNP_AUTH_USER": "u", "FNP_AUTH_PASS": "p"}), (b"u", b"p"))
        self.assertEqual(c({"FNP_AUTH_USER": "ü", "FNP_AUTH_PASS": "pä"}), ("ü".encode(), "pä".encode()))

    def test_default_is_off(self):
        self.assertIsNone(server.AUTH)  # helpers cleared the env before server was imported

    def test_header_parsing(self):
        creds = (USER.encode(), PASS.encode())
        ok = server.auth_header_ok
        self.assertTrue(ok(basic()["Authorization"], creds))
        self.assertTrue(ok(basic()["Authorization"].replace("Basic", "basic"), creds))
        self.assertTrue(ok(basic("ü", "pä")["Authorization"], ("ü".encode(), "pä".encode())))
        for bad in (None, "", "Basic", "Basic ", "Bearer abc", "Basic !!!notbase64!!!",
                    basic("alice", "wrong")["Authorization"], basic("bob", PASS)["Authorization"],
                    basic("alice", PASS + "x")["Authorization"], basic("alic", PASS)["Authorization"],
                    "Basic " + base64.b64encode(b"alice").decode(),  # no colon
                    "Basic " + base64.b64encode(b"\xff\xfe:\x00").decode()):
            self.assertFalse(ok(bad, creds), bad)
        self.assertFalse(ok(basic()["Authorization"], None))


class AuthOff(ServerCase):
    def test_everything_open_and_no_challenge(self):
        for path in ("/", "/app.js", "/api/items", "/api/stats", "/api/proplab/meta", "/healthz"):
            st, hd, _ = self.get(path)
            self.assertEqual(st, 200, path)
            self.assertNotIn("www-authenticate", hd)
        self.assertEqual(self.post("/api/check", {"text": "free electricity households"})[0], 200)

    def test_credentials_are_ignored_when_off(self):
        self.assertEqual(self.get("/api/items", headers=basic("x", "y"))[0], 200)


class AuthOn(ServerCase):
    def setUp(self):
        super().setUp()
        server.AUTH = server.auth_creds({"FNP_AUTH_USER": USER, "FNP_AUTH_PASS": PASS})
        self.stub = FlagStub()
        server.FLAG_URL = self.stub.url

    def tearDown(self):
        self.stub.close()
        super().tearDown()

    def assert_challenge(self, resp):
        st, hd, raw = resp
        self.assertEqual(st, 401)
        self.assertEqual(hd["www-authenticate"], 'Basic realm="Evidence Desk"')
        self.assertEqual(hd["cache-control"], "no-store")
        self.assertNotIn(PASS.encode(), raw)

    def test_every_route_needs_auth_except_healthz(self):
        for path in ("/", "/index.html", "/app.js", "/proplab.js", "/styles.css", "/fonts/Inter-latin.woff2",
                     "/nonexistent.png", "/../server.py", "/api/items", "/api/stats", "/api/item/fc2",
                     "/api/nope", "/api/proplab/meta", "/api/proplab/country/IND", "/api/proplab/country/in",
                     "/api/proplab/posts?mode=about&iso=IND", "/healthz/", "/healthz/x"):
            self.assert_challenge(self.get(path))
        for path, body in (("/api/check", {"text": "free electricity"}), ("/api/proplab/flag", VALID),
                           ("/api/proplab/flag", "garbage"), ("/api/nope", {}), ("/healthz", {})):
            self.assert_challenge(self.post(path, body))

    def test_healthz_is_open(self):
        st, hd, raw = self.get("/healthz")
        self.assertEqual(st, 200)
        self.assertNotIn("www-authenticate", hd)
        self.assertIs(json.loads(raw)["ok"], True)
        self.assertEqual(self.get("/healthz?x=1")[0], 200)

    def test_healthz_503_is_also_open(self):
        with self.patched(ITEMS_P=os.path.join(TMP, "build", "gone.json")):
            self.assertEqual(self.get("/healthz")[0], 503)

    def test_valid_credentials_pass_everywhere(self):
        h = basic()
        for path in ("/", "/app.js", "/api/items", "/api/stats", "/api/item/fc2", "/api/proplab/meta",
                     "/api/proplab/country/IND"):
            self.assertEqual(self.get(path, headers=h)[0], 200, path)
        self.assertEqual(self.get("/nonexistent.png", headers=h)[0], 404)
        self.assertEqual(self.get("/api/proplab/country/in", headers=h)[0], 400)
        self.assertEqual(self.post("/api/check", {"text": "free electricity households"}, headers=h)[0], 200)
        self.assertEqual(self.post("/api/check", {"text": "x"}, headers=h)[0], 400)

    def test_flag_route_gated_before_any_upstream_call(self):
        self.assert_challenge(self.post("/api/proplab/flag", VALID))
        self.assert_challenge(self.post("/api/proplab/flag", VALID, headers=basic(pw="wrong")))
        self.assertEqual(self.stub.requests, [])
        self.assertEqual(self.post("/api/proplab/flag", VALID, headers=basic())[0], 200)
        self.assertEqual(self.stub.requests, [VALID])

    def test_wrong_credentials_rejected(self):
        for h in (basic(pw="nope"), basic(user="bob"), basic(pw=PASS.upper()), {"Authorization": "Bearer " + PASS},
                  {"Authorization": "Basic %%%"}, {"Authorization": "Basic " + base64.b64encode(b"alice").decode()}):
            self.assert_challenge(self.get("/api/items", headers=h))

    def test_throttle_after_ten_failures(self):
        for i in range(10):
            self.assert_challenge(self.get("/api/items", headers=basic(pw="bad%d" % i)))
        st, hd, raw = self.get("/api/items", headers=basic(pw="bad-again"))
        self.assertEqual(st, 429)
        self.assertGreater(int(hd["retry-after"]), 0)
        self.assertLessEqual(int(hd["retry-after"]), 301)
        # locked out even with the right password, on every route, but /healthz stays reachable
        self.assertEqual(self.get("/api/items", headers=basic())[0], 429)
        self.assertEqual(self.get("/")[0], 429)
        self.assertEqual(self.post("/api/proplab/flag", VALID, headers=basic())[0], 429)
        self.assertEqual(self.stub.requests, [])
        self.assertEqual(self.get("/healthz")[0], 200)
        # failures age out after 5 minutes
        ip = "127.0.0.1"
        server._auth_fails[ip] = [t - server.AUTH_FAIL_WINDOW - 1 for t in server._auth_fails[ip]]
        self.assertEqual(self.get("/api/items", headers=basic())[0], 200)

    def test_nine_failures_then_success_is_fine(self):
        for i in range(9):
            self.assert_challenge(self.get("/api/items", headers=basic(pw="bad%d" % i)))
        self.assertEqual(self.get("/api/items", headers=basic())[0], 200)

    def test_anonymous_requests_do_not_count_as_failures(self):
        for _ in range(25):
            self.assert_challenge(self.get("/"))
        self.assertEqual(server._auth_fails, {})
        self.assertEqual(self.get("/", headers=basic())[0], 200)

    def test_throttle_is_per_ip(self):
        server._auth_fails["<INTERNAL_IP>"] = [time.monotonic()] * server.AUTH_FAIL_MAX
        self.assertEqual(server.auth_locked("<INTERNAL_IP>") > 0, True)
        self.assertEqual(server.auth_locked("127.0.0.1"), 0)
        self.assertEqual(self.get("/api/items", headers=basic())[0], 200)

    def test_failure_table_is_bounded(self):
        now = time.monotonic()
        for i in range(2500):
            server._auth_fails["old-%d" % i] = [now - server.AUTH_FAIL_WINDOW - 5]
        server.auth_record_failure("1.2.3.4")
        self.assertLess(len(server._auth_fails), 2000)
        for _ in range(50):
            server.auth_record_failure("1.2.3.4")
        self.assertEqual(len(server._auth_fails["1.2.3.4"]), server.AUTH_FAIL_MAX)

    def test_credentials_never_logged(self):
        secret = "hunter2-DISTINCT-secret"
        wrong = basic(pw=secret)["Authorization"]
        good = basic()["Authorization"]
        err, out = io.StringIO(), io.StringIO()
        server.H.log_message = REAL_LOG  # real access log on for this test
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                self.get("/api/items", headers={"Authorization": wrong})
                self.get("/api/items", headers={"Authorization": good})
                self.get("/api/items")
        finally:
            server.H.log_message = lambda *a, **k: None
        logged = err.getvalue() + out.getvalue()
        self.assertIn("auth failure ip=127.0.0.1", logged)
        for needle in (secret, PASS, USER, wrong.split()[1], good.split()[1], "Authorization"):
            self.assertNotIn(needle, logged)

    def test_unicode_credentials(self):
        server.AUTH = server.auth_creds({"FNP_AUTH_USER": "ünï", "FNP_AUTH_PASS": "pässwörd"})
        self.assertEqual(self.get("/api/items", headers=basic("ünï", "pässwörd"))[0], 200)
        self.assertEqual(self.get("/api/items", headers=basic("ünï", "passwörd"))[0], 401)


if __name__ == "__main__":
    unittest.main()
