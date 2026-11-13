"""A fake FNP engine on an ephemeral 127.0.0.1 port (never the real :8879)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import helpers


def v0_response(label="FAKE", probs=None, spans=None, secs=1.2):
    probs = probs or {"FAKE": 0.7, "PROPAGANDA": 0.05, "LEGIT": 0.1, "MISLEADING": 0.15, "INCONCLUSIVE": 0.0}
    return {"doc": {"label": label, "probs": probs, "plausibly_fake": False, "secondary": [], "decider": "rollup",
                    "rationale": "1 considered claim(s)"},
            "claims": [{"verdict": {"label": label, "rationale": "Gemma says so.", "decider": "stacker"}}],
            "spans": spans if spans is not None else [], "spans_summary": {"n": 0, "techniques": [], "one_sided": False},
            "secs": secs}


_UNSET = object()


def v1_response(fc_label="FAKE", fake_score=0.93, basis="stacker", raw_label=None, degraded=False, spans=None,
                laya=("FAKE", 0.52), error=None, technique_probs=None, laya_is_propaganda_raw=0.0, one_sided=1.0,
                claims=_UNSET, degraded_reason=None, laya_reason=None, p_prop=_UNSET, too_short=_UNSET,
                n_tokens=_UNSET):
    """A POST /api/check_v1 body (docs/CHECK_V1_CONTRACT.md). `degraded_reason`: 'no_claims' is the current engine's
    reply for a text with no extracted claim (degraded false, basis none, claims [], laya null, laya_reason);
    'no_claims_old' is the older shape (same but degraded true, no reason fields); 'timeout' / 'gemma_error' degrade.
    p_prop / too_short / n_tokens / laya_reason are only sent when given (omitted = an older engine)."""
    if degraded_reason in ("no_claims", "no_claims_old"):
        degraded, error, claims, laya, basis = degraded_reason == "no_claims_old", None, [], None, "none"
        if degraded_reason == "no_claims":
            laya_reason = "no_claims"
    elif degraded_reason in ("timeout", "gemma_error"):
        degraded = True
        error = error or ("timeout after 55 s" if degraded_reason == "timeout" else "gemma error")
    fc_probs = {"FAKE": 0.9, "MISLEADING": 0.03, "LEGIT": 0.05, "INCONCLUSIVE": 0.02}
    raw = None if degraded else {
        "label": raw_label or fc_label,
        "probs": {"FAKE": 0.2, "PROPAGANDA": 0.6, "LEGIT": 0.05, "MISLEADING": 0.1, "INCONCLUSIVE": 0.05},
        "plausibly_fake": True, "prediction_set": ["FAKE"], "rationale": "stacker rationale", "decider": "stacker"}
    return {"verdict": raw,
            "fake_call": {"label": fc_label, "probs": fc_probs, "fake_score": fake_score, "basis": basis},
            "laya": None if laya is None else {"label": laya[0], "probs": {laya[0]: laya[1], "LEGIT": 1 - laya[1]},
                                               "confidence": laya[1]},
            "claims": [] if claims is _UNSET else claims, "degraded": degraded, "error": error or ("timeout after 55 s" if degraded and degraded_reason not in ("no_claims", "no_claims_old") else None),
            "propaganda": {"is_propaganda_p": None, "laya_is_propaganda_raw": laya_is_propaganda_raw, "one_sided": one_sided,
                           "techniques": spans if spans is not None else [],
                           **({"technique_probs": technique_probs} if technique_probs is not None else {}),
                           **({"p_prop": p_prop} if p_prop is not _UNSET else {}),
                           **({"too_short": too_short} if too_short is not _UNSET else {}),
                           **({"n_tokens": n_tokens} if n_tokens is not _UNSET else {})},
            **({"laya_reason": laya_reason} if laya_reason else {}),
            **({"degraded_reason": degraded_reason} if degraded_reason in ("no_claims", "timeout", "gemma_error") else {}),
            "model_versions": {"fake_call": "fake_call_v1"}, "pipeline_s": 14.9, "secs": 15.5, "queue_wait_s": 0.0}


class FakeEngine:
    def __init__(self, make=None, delay=0.0, status=200, headers=None, path="/api/predict"):
        self.make, self.delay, self.status = make or (lambda text: v0_response()), delay, status
        self.count = 0
        self.headers = headers or {}
        self.lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                with outer.lock:
                    outer.count += 1
                time.sleep(outer.delay)
                out = json.dumps(outer.make(body["text"])).encode()
                try:
                    self.send_response(outer.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    for k, v in outer.headers.items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(out)
                except OSError:
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = "http://127.0.0.1:%d%s" % (self.port, path)
        helpers.assert_not_real_flag(self.url)
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)
