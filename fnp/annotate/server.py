"""Annotation server: serves the UI assets (UI/) + the /api/annot/* JSON API. Separate process from the
dashboard (UI/server.py), same look (UI/assets/app.css).

  .venv/bin/python scripts/29_annot_server.py [--host 0.0.0.0] [--port 8890]

Identity: username + password (auth.py) -> bearer session token. Roles live in the DB (annotator | adjudicator | admin);
create the first admin with  scripts/31_annot_users.py create NAME --role admin  and manage everyone else in the UI
(Admin > Users). Accounts flagged must_change_password can only change their password until they do.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..config import FNP_ROOT
from . import adjudicate, auth, batching, metrics, quality
from . import service as S
from .db import tx

UI_DIR = FNP_ROOT / "UI"
LOGIN_PER_MINUTE = 20                                   # per client IP, on top of the per-account lock-out

_job = {"running": False, "kind": None, "started": None, "result": None, "error": None, "progress": None}
_job_lock = threading.Lock()
_login_hits: dict[str, deque] = defaultdict(deque)


def start_job(kind: str, fn) -> dict:
    with _job_lock:
        if _job["running"]:
            raise S.AnnotError(f"a {_job['kind']} job is already running")
        _job.update(running=True, kind=kind, started=time.strftime("%Y-%m-%dT%H:%M:%S"), result=None, error=None, progress=None)

    def work():
        try:
            _job["result"] = fn()
        except Exception as e:  # noqa: BLE001
            _job["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            traceback.print_exc()
        finally:
            _job["running"] = False
    threading.Thread(target=work, daemon=True).start()
    return dict(_job)


def _bearer(h) -> str:
    v = h.headers.get("Authorization", "")
    return v[7:].strip() if v.lower().startswith("bearer ") else ""


def _ip(h) -> str:
    return h.client_address[0] if h.client_address else ""


def _throttled(ip: str) -> bool:
    now = time.time()
    q = _login_hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= LOGIN_PER_MINUTE:
        return True
    q.append(now)
    return False


ROUTES: list[tuple[str, re.Pattern, str]] = []


def route(method: str, pattern: str):
    rx = re.compile("^" + re.sub(r"<(\w+)>", r"(?P<\1>[^/]+)", pattern) + "$")

    def deco(fn):
        ROUTES.append((method, rx, fn))
        return fn
    return deco


def _int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        raise S.AnnotError("bad id") from None


# ---------------------------------------------------------------- routes (a = annotator, b = json body, q = query dict)
@route("GET", "/api/annot/taxonomy")
def r_tax(a, b, q): return S.taxonomy()


PUBLIC = {("POST", "/api/annot/auth/login")}                       # reachable without a session
ALLOWED_WHEN_PASSWORD_MUST_CHANGE = {"/api/annot/auth/password", "/api/annot/auth/logout", "/api/annot/me", "/api/annot/heartbeat", "/api/annot/progress"}


@route("POST", "/api/annot/auth/login")
def r_login(a, b, q):
    ip = q["_ip"]
    if _throttled(ip):
        raise PermissionError("too many sign-in attempts from this address: wait a minute")
    r = auth.login(str(b.get("username", "")), str(b.get("password", "")), ip, q["_ua"])
    if "error" in r:
        raise auth.AuthError(r["error"])
    me = auth.resolve(r["token"])
    return {"token": r["token"], "annotator": r["annotator"], "taxonomy": S.taxonomy(), "progress": metrics.my_progress(me["annotator_id"]),
            "idle_seconds": int(auth.C("ann_idle_seconds"))}


@route("POST", "/api/annot/auth/logout")
def r_logout(a, b, q):
    auth.logout(q["_token"]); return {"ok": True}


@route("POST", "/api/annot/auth/password")
def r_password(a, b, q):
    auth.change_password(a["annotator_id"], str(b.get("old", "")), str(b.get("new", "")), keep_session=a["session_id"]); return {"ok": True}


@route("GET", "/api/annot/me")
def r_me(a, b, q):
    return {"annotator": {k: a[k] for k in ("annotator_id", "username", "display_name", "role", "must_change_password")},
            "taxonomy": S.taxonomy(), "progress": metrics.my_progress(a["annotator_id"]), "idle_seconds": int(auth.C("ann_idle_seconds"))}


@route("GET", "/api/annot/progress")
def r_progress(a, b, q):
    return metrics.my_progress(a["annotator_id"])


@route("POST", "/api/annot/heartbeat")
def r_heartbeat(a, b, q):
    auth.heartbeat(a, bool(b.get("active"))); return {"ok": True}


@route("GET", "/api/annot/next")
def r_next(a, b, q):
    iid = S.next_item(a["annotator_id"])
    return {"item_id": iid, "item": S.get_item(iid, a["annotator_id"]) if iid else None}


@route("GET", "/api/annot/item/<iid>")
def r_item(a, b, q, iid): return S.get_item(_int(iid), a["annotator_id"])


@route("POST", "/api/annot/item/<iid>/claims")
def r_claim_add(a, b, q, iid):
    return S.add_claim(_int(iid), a["annotator_id"], b.get("start"), b.get("end"), b.get("claim_type"),
                       b.get("check_worthy", "unsure"), b.get("normalised_claim"), b.get("speaker"), b.get("suggestion_id"))


@route("PUT", "/api/annot/claims/<sid>")
def r_claim_upd(a, b, q, sid):
    f = {k: b[k] for k in ("start", "end", "claim_type", "check_worthy", "normalised_claim", "speaker") if k in b}
    return S.update_claim(_int(sid), a["annotator_id"], **f)


@route("DELETE", "/api/annot/claims/<sid>")
def r_claim_del(a, b, q, sid):
    S.delete_claim(_int(sid), a["annotator_id"]); return {"ok": True}


@route("POST", "/api/annot/item/<iid>/propaganda")
def r_prop_add(a, b, q, iid):
    return S.add_propaganda(_int(iid), a["annotator_id"], b.get("start"), b.get("end"), b.get("techniques"), b.get("suggestion_id"))


@route("PUT", "/api/annot/propaganda/<sid>")
def r_prop_upd(a, b, q, sid):
    f = {k: b[k] for k in ("start", "end", "techniques") if k in b}
    return S.update_propaganda(_int(sid), a["annotator_id"], **f)


@route("DELETE", "/api/annot/propaganda/<sid>")
def r_prop_del(a, b, q, sid):
    S.delete_propaganda(_int(sid), a["annotator_id"]); return {"ok": True}


@route("POST", "/api/annot/item/<iid>/suggestions/<sid>/reject")
def r_sugg_rej(a, b, q, iid, sid):
    S.reject_suggestion(_int(iid), a["annotator_id"], _int(sid)); return {"ok": True}


@route("POST", "/api/annot/item/<iid>/submit")
def r_submit(a, b, q, iid):
    return S.submit_layer(_int(iid), a["annotator_id"], b.get("layer"), b.get("outcome"), b.get("active_ms", 0))


@route("POST", "/api/annot/item/<iid>/skip")
def r_skip(a, b, q, iid):
    S.skip_item(_int(iid), a["annotator_id"], b.get("reason")); return {"ok": True}


@route("POST", "/api/annot/item/<iid>/reopen")
def r_reopen(a, b, q, iid):
    S.reopen_item(_int(iid), a["annotator_id"]); return S.get_item(_int(iid), a["annotator_id"])


@route("GET", "/api/annot/mine")
def r_mine(a, b, q): return {"items": S.mine(a["annotator_id"], int(q.get("limit", 100)))}


@route("GET", "/api/annot/history/<iid>")
def r_hist(a, b, q, iid): return {"events": S.history(_int(iid), a["annotator_id"])}


def _need_admin(a):
    if a["role"] != "admin":
        raise PermissionError("admin only")


@route("GET", "/api/annot/stats")
def r_stats(a, b, q):
    _need_admin(a)
    st = S.stats()
    st["job"] = dict(_job)
    return st


@route("GET", "/api/annot/admin/job")
def r_job(a, b, q):
    _need_admin(a); return dict(_job)


@route("GET", "/api/annot/admin/team")
def r_team(a, b, q):
    _need_admin(a); return metrics.team(int(q.get("days", 30)))


@route("GET", "/api/annot/admin/user/<uid>")
def r_user(a, b, q, uid):
    _need_admin(a)
    d = metrics.user_detail(_int(uid), int(q.get("days", 30)))
    if not d:
        raise S.AnnotError("no such annotator")
    return d


@route("GET", "/api/annot/admin/coverage")
def r_cov(a, b, q):
    _need_admin(a); return metrics.coverage()


@route("GET", "/api/annot/admin/logins")
def r_logins(a, b, q):
    _need_admin(a); return {"events": metrics.login_events(int(q.get("limit", 100)), q.get("problems") == "1")}


@route("POST", "/api/annot/admin/users")
def r_user_create(a, b, q):
    _need_admin(a)
    pw = b.get("password") or None
    return auth.create_user(str(b.get("username", "")), pw, str(b.get("role", "annotator")), b.get("display_name"),
                            must_change=True, actor=a["username"])


@route("POST", "/api/annot/admin/users/<uid>")
def r_user_update(a, b, q, uid):
    """Any of: active, role, batch_size, reset_password (true -> a generated temporary password is returned once),
    new_password (+ must_change, default true) to set a password the admin typed."""
    _need_admin(a)
    uid = _int(uid)
    out: dict = {}
    with tx() as cur:
        cur.execute("SELECT annotator_id, username, role, active FROM annotator WHERE annotator_id=%s", (uid,))
        u = cur.fetchone()
        if not u:
            raise S.AnnotError("no such annotator")
        cur.execute("SELECT count(*) n FROM annotator WHERE role='admin' AND active AND annotator_id <> %s", (uid,))
        other_admins = cur.fetchone()["n"]
    losing_admin = (b.get("active") is False and u["role"] == "admin") or (b.get("role") and b["role"] != "admin" and u["role"] == "admin")
    if losing_admin and other_admins == 0:
        raise S.AnnotError("this is the only active admin: make another admin first")
    if "role" in b:
        if b["role"] not in auth.ROLES:
            raise S.AnnotError(f"role must be one of {auth.ROLES}")
        with tx() as cur:
            cur.execute("UPDATE annotator SET role=%s WHERE annotator_id=%s", (b["role"], uid))
    if "batch_size" in b:
        bs = b["batch_size"]
        if bs not in (None, "") and not (isinstance(bs, int) and 1 <= bs <= 5000):
            raise S.AnnotError("batch_size must be 1-5000, or empty for the default")
        with tx() as cur:
            cur.execute("UPDATE annotator SET batch_size=%s WHERE annotator_id=%s", (bs or None, uid))
    if "active" in b:
        auth.set_active(uid, bool(b["active"]))
        if not b["active"]:
            out["released_items"] = batching.release_unstarted(uid)
    if b.get("new_password"):
        # admin chose the password; must_change defaults to True so the user replaces it at first sign-in
        auth.set_password(uid, str(b["new_password"]), bool(b.get("must_change", True)), actor=a["username"])
        out["password_set"] = True
    elif b.get("reset_password"):
        out["password"] = auth.set_password(uid, None, True, actor=a["username"])
    if b.get("unlock"):
        with tx() as cur:
            cur.execute("UPDATE annotator SET failed_logins=0, locked_until=NULL WHERE annotator_id=%s", (uid,))
    out["ok"] = True
    return out


@route("POST", "/api/annot/admin/users/<uid>/release")
def r_release(a, b, q, uid):
    _need_admin(a); return {"released_items": batching.release_unstarted(_int(uid))}


@route("POST", "/api/annot/ingest")
def r_ingest(a, b, q):
    _need_admin(a)
    from . import ingest
    mode = b.get("mode", "newest")
    if mode not in ("newest", "backfill"):
        raise S.AnnotError("mode must be newest|backfill")
    n = max(1, min(int(b.get("per_source", 30)), 500))
    plats = set(b.get("platforms") or [])
    return start_job("ingest", lambda: ingest.run(mode, n, True, plats, b.get("since") or None, b.get("until") or None))


@route("POST", "/api/annot/admin/fill")
def r_fill(a, b, q):
    _need_admin(a)
    from . import ingest
    total = max(1, min(int(b.get("total", int(auth.C("ann_target_items")))), 500_000))
    per = max(10, min(int(b.get("per_source", 250)), 500))
    return start_job("fill pool", lambda: ingest.fill(total, per, progress=lambda r: _job.update(progress=r)))


@route("POST", "/api/annot/preannotate")
def r_pre(a, b, q):
    _need_admin(a)
    from . import preannotate
    n = max(1, min(int(b.get("limit", 100)), 5000))
    return start_job("preannotate", lambda: preannotate.run(n, True, bool(b.get("encoder", True)), bool(b.get("gemma", False))))


# ---------------------------------------------------------------- quality, adjudication, gold
def _need_adjudicator(a):
    if a["role"] not in ("admin", "adjudicator"):
        raise PermissionError("admin or adjudicator only")


@route("GET", "/api/annot/admin/quality")
def r_quality(a, b, q):
    _need_admin(a); return quality.team_quality()


@route("GET", "/api/annot/admin/quality/<uid>")
def r_user_quality(a, b, q, uid):
    _need_admin(a); return quality.user_quality(_int(uid))


@route("GET", "/api/annot/admin/disputes")
def r_disputes(a, b, q):
    _need_adjudicator(a); return adjudicate.disputes(int(q.get("limit", 300)))


@route("POST", "/api/annot/admin/adjudicate")
def r_adj_start(a, b, q):
    _need_adjudicator(a); return adjudicate.start(_int(b.get("item_id")), a["annotator_id"])


@route("POST", "/api/annot/admin/adjudicate/copy")
def r_adj_copy(a, b, q):
    _need_adjudicator(a)
    adjudicate.copy_from(_int(b.get("item_id")), a["annotator_id"], _int(b.get("from_annotator_id")), str(b.get("layer", "")))
    return {"ok": True}


@route("POST", "/api/annot/admin/gold")
def r_gold(a, b, q):
    _need_adjudicator(a)
    if b.get("remove"):
        adjudicate.unmark_gold(_int(b.get("item_id")), a["annotator_id"]); return {"ok": True}
    return adjudicate.mark_gold(_int(b.get("item_id")), a["annotator_id"], (b.get("note") or None))


@route("GET", "/api/annot/admin/gold")
def r_gold_list(a, b, q):
    _need_adjudicator(a); return {"items": adjudicate.gold_list()}


# ---------------------------------------------------------------- http
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(UI_DIR), **kw)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _dispatch(self, method: str):
        u = urlparse(self.path)
        if u.path == "/healthz" and method == "GET":      # liveness for systemd/monitoring; no auth, no data
            try:
                with tx() as cur:
                    cur.execute("SELECT 1 AS ok")
                    cur.fetchone()
                return self._json({"ok": True, "service": "fnp-annotation", "db": True})
            except Exception as e:  # noqa: BLE001
                return self._json({"ok": False, "service": "fnp-annotation", "db": False,
                                   "error": type(e).__name__}, 503)
        if not u.path.startswith("/api/annot/"):
            if method != "GET":
                return self._json({"error": "not found"}, 404)
            if u.path in ("", "/"):
                self.path = "/annotate.html"
            return SimpleHTTPRequestHandler.do_GET(self)
        path = unquote(u.path).rstrip("/")
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        q.update(_ip=_ip(self), _ua=self.headers.get("User-Agent", ""), _token=_bearer(self))
        body = {}
        if method in ("POST", "PUT"):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > 1_000_000:
                    raise ValueError
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except Exception:
                return self._json({"error": "bad json"}, 400)
        for m, rx, fn in ROUTES:
            mt = rx.match(path)
            if m == method and mt:
                try:
                    a = None
                    if (m, path) not in PUBLIC:
                        a = auth.resolve(q["_token"])
                        if a["must_change_password"] and path not in ALLOWED_WHEN_PASSWORD_MUST_CHANGE:
                            return self._json({"error": "you must change your temporary password first", "code": "must_change_password"}, 403)
                        if method != "GET" and path != "/api/annot/heartbeat":
                            auth.mark_activity(a["annotator_id"], a["session_id"])        # a label write is work by definition
                    return self._json(fn(a, body, q, **mt.groupdict()))
                except auth.Unauthorized as e:
                    return self._json({"error": str(e), "code": "unauthorized"}, 401)
                except (S.AnnotError, auth.AuthError) as e:
                    return self._json({"error": str(e)}, 400)
                except PermissionError as e:
                    return self._json({"error": str(e)}, 403)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    return self._json({"error": f"server error: {type(e).__name__}"}, 500)
        return self._json({"error": "unknown endpoint"}, 404)

    def do_GET(self): self._dispatch("GET")
    def do_POST(self): self._dispatch("POST")
    def do_PUT(self): self._dispatch("PUT")
    def do_DELETE(self): self._dispatch("DELETE")


def serve(host: str, port: int) -> None:
    srv = ThreadingHTTPServer((host, port), Handler)
    with tx() as cur:
        cur.execute("SELECT count(*) n FROM annotator WHERE role='admin' AND active AND password_hash IS NOT NULL")
        n_admin = cur.fetchone()["n"]
    print(f"FNP annotation server on http://{host}:{port} (UI from {UI_DIR}); password login; active admins: {n_admin}"
          + ("" if n_admin else "  ** none: run scripts/31_annot_users.py create NAME --role admin **"), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
