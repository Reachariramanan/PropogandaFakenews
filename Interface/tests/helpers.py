"""Shared test setup (imported first by every test module).

Offline and side-effect free: Interface/ goes on sys.path, the flag proxy is pointed at a closed local port BEFORE
`server` is imported, and auth env vars are cleared. No test ever talks to the real flag service on :8879."""
import http.client
import json
import os
import socket
import sys
from urllib.parse import urlsplit

INTERFACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if INTERFACE not in sys.path:
    sys.path.insert(0, INTERFACE)
REAL_FLAG_PORT = 8879


import unittest  # noqa: E402

# The generated search index (Interface/build/items.json) is built from the scraped fact-check data, which is not part of the
# repository; the tests that drive the real `server` module (healthz, rate limits, closed-gate) need it.
HAS_INDEX = os.path.exists(os.path.join(INTERFACE, "build", "items.json"))
needs_index = unittest.skipUnless(HAS_INDEX, "needs the generated index: run Interface/build_index.py with the fact-check data files")


def free_port():
    """A currently unused 127.0.0.1 port (bound then released), so nothing is listening on it."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def assert_not_real_flag(url):
    p = urlsplit(url)
    if p.port == REAL_FLAG_PORT or p.hostname not in ("127.0.0.1", "localhost"):
        raise AssertionError("refusing to use %r: tests must never reach the real flag service" % url)


DEAD_FLAG_URL = "http://127.0.0.1:%d/api/prop_flag" % free_port()
assert_not_real_flag(DEAD_FLAG_URL)
os.environ["FNP_FLAG_URL"] = DEAD_FLAG_URL
DEAD_ENGINE_URL = "http://127.0.0.1:%d/api/predict" % free_port()  # the engine is never the real :8879 in tests
assert_not_real_flag(DEAD_ENGINE_URL)
os.environ["FNP_ENGINE_URL"] = DEAD_ENGINE_URL
for _k in ("FNP_AUTH_USER", "FNP_AUTH_PASS", "FNP_PROPLAB_DIR", "FNP_RT_SYNC", "FNP_RT_ALLOW_OPEN"):
    os.environ.pop(_k, None)


def call(port, method, path, body=None, headers=None, raw_headers=None):
    """One request to 127.0.0.1:port -> (status, lower-cased header dict, body bytes).
    `body` may be bytes/str/dict (dict -> JSON). `raw_headers` are added verbatim, which lets a test send a
    Content-Length that does not match the (absent) body."""
    h = dict(headers or {})
    if isinstance(body, dict):
        body = json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    elif isinstance(body, str):
        body = body.encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        if raw_headers:
            conn.putrequest(method, path)
            for k, v in {**h, **raw_headers}.items():
                conn.putheader(k, v)
            conn.endheaders()
        else:
            conn.request(method, path, body=body, headers=h)
        r = conn.getresponse()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()
    finally:
        conn.close()


def jcall(port, method, path, **kw):
    """Like call() but parses the body as strict JSON (NaN/Infinity are rejected)."""
    st, hd, raw = call(port, method, path, **kw)

    def bad(c):
        raise ValueError("non-strict JSON constant %s" % c)
    return st, hd, json.loads(raw.decode("utf-8"), parse_constant=bad)
