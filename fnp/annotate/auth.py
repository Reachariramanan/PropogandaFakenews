"""Accounts, password login, sessions and work-time tracking for the annotation tool.

  * passwords: scrypt (stdlib) with a per-user random salt; compared in constant time. Plain passwords are never
    stored or logged. Unknown usernames cost the same time as wrong passwords (no user enumeration by timing).
  * sessions: a random 256-bit bearer token, stored only as a sha256 hash, rolling expiry (ann_session_hours).
  * lock-out: ann_max_failed_logins consecutive bad passwords lock the account for ann_lockout_minutes.
  * every attempt lands in login_event; every session in auth_session; every minute of real work in activity_minute.
  * work time = number of distinct minutes with a heartbeat that reported recent input (ann_idle_seconds) or a label
    write. A minute is counted once however many tabs / sessions the annotator has open.

Transport is plain HTTP unless the server is put behind TLS: on an untrusted network terminate HTTPS in front of it.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from typing import Optional

from ..config import C
from .db import tx

USERNAME_RE = re.compile(r"[A-Za-z0-9_.@-]{2,64}")
ROLES = ("annotator", "adjudicator", "admin")
TZ = os.environ.get("FNP_ANNOT_TZ", "Asia/Kolkata")   # time zone used to bucket "today" and the daily series
_SCRYPT = dict(n=2 ** 14, r=8, p=1)


class AuthError(ValueError):
    """Bad credentials / disabled / locked (HTTP 401 or 400 depending on the case)."""


class Unauthorized(Exception):
    """No valid session (HTTP 401)."""


# ------------------------------------------------------------------ passwords
def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(pw.encode(), salt=salt, dklen=32, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt.hex()}${dk.hex()}"


def verify_password(pw: str, stored: Optional[str]) -> bool:
    try:
        kind, n, r, p, salt, dk = (stored or "").split("$")
        if kind != "scrypt":
            raise ValueError
        got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=len(dk) // 2)
        return hmac.compare_digest(got.hex(), dk)
    except (ValueError, TypeError):
        hashlib.scrypt(b"x", salt=b"0" * 16, dklen=32, **_SCRYPT)       # keep timing similar for malformed/missing hashes
        return False


def gen_password(n: int = 12) -> str:
    alphabet = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no look-alikes (0/O, 1/l/I)
    return "".join(secrets.choice(alphabet) for _ in range(n))


def check_password_policy(pw: str, username: str = "") -> None:
    mn = int(C("ann_min_password_len"))
    if not isinstance(pw, str) or len(pw) < mn:
        raise AuthError(f"password must be at least {mn} characters")
    if username and pw.lower() == username.lower():
        raise AuthError("password must differ from the username")


def _tok_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ------------------------------------------------------------------ accounts
def _public(a: dict) -> dict:
    return {k: a[k] for k in ("annotator_id", "username", "display_name", "role", "must_change_password") if k in a}


def create_user(username: str, password: Optional[str] = None, role: str = "annotator", display_name: Optional[str] = None,
                must_change: bool = True, actor: Optional[str] = None) -> dict:
    """-> {annotator, password}; `password` is the generated one when none was given (show it once, it is not recoverable)."""
    username = (username or "").strip()
    if not USERNAME_RE.fullmatch(username):
        raise AuthError("username must be 2-64 characters: letters, digits, _ . @ -")
    if role not in ROLES:
        raise AuthError(f"role must be one of {ROLES}")
    generated = password is None
    password = password or gen_password()
    check_password_policy(password, username)
    with tx() as cur:
        cur.execute("SELECT 1 FROM annotator WHERE lower(username)=lower(%s)", (username,))
        if cur.fetchone():
            raise AuthError(f"username {username!r} already exists")
        cur.execute("INSERT INTO annotator (username, display_name, role, password_hash, must_change_password) "
                    "VALUES (%s,%s,%s,%s,%s) RETURNING *", (username, display_name or username, role, hash_password(password), must_change))
        a = dict(cur.fetchone())
        cur.execute("INSERT INTO login_event (username, annotator_id, outcome, detail) VALUES (%s,%s,'user_created',%s)",
                    (username, a["annotator_id"], f"role={role} by={actor or 'cli'}"))
    return {"annotator": _public(a), "password": password if generated else None}


def set_password(annotator_id: int, password: Optional[str] = None, must_change: bool = True, actor: Optional[str] = None) -> Optional[str]:
    """Admin / CLI reset. Ends the user's open sessions. Returns the generated password if none was given."""
    generated = password is None
    password = password or gen_password()
    with tx() as cur:
        cur.execute("SELECT username FROM annotator WHERE annotator_id=%s", (annotator_id,))
        a = cur.fetchone()
        if not a:
            raise AuthError("no such annotator")
        check_password_policy(password, a["username"])
        cur.execute("UPDATE annotator SET password_hash=%s, must_change_password=%s, failed_logins=0, locked_until=NULL "
                    "WHERE annotator_id=%s", (hash_password(password), must_change, annotator_id))
        _end_sessions(cur, annotator_id, "password_changed")
        cur.execute("INSERT INTO login_event (username, annotator_id, outcome, detail) VALUES (%s,%s,'password_reset',%s)",
                    (a["username"], annotator_id, f"by={actor or 'cli'}"))
    return password if generated else None


def change_password(annotator_id: int, old: str, new: str, keep_session: Optional[int] = None) -> None:
    with tx() as cur:
        cur.execute("SELECT username, password_hash FROM annotator WHERE annotator_id=%s FOR UPDATE", (annotator_id,))
        a = cur.fetchone()
        if not a or not verify_password(old or "", a["password_hash"]):
            raise AuthError("current password is wrong")
        check_password_policy(new, a["username"])
        if new == old:
            raise AuthError("the new password must differ from the current one")
        cur.execute("UPDATE annotator SET password_hash=%s, must_change_password=false WHERE annotator_id=%s",
                    (hash_password(new), annotator_id))
        _end_sessions(cur, annotator_id, "password_changed", except_session=keep_session)
        cur.execute("INSERT INTO login_event (username, annotator_id, outcome) VALUES (%s,%s,'password_changed')",
                    (a["username"], annotator_id))


def set_active(annotator_id: int, active: bool) -> None:
    with tx() as cur:
        cur.execute("UPDATE annotator SET active=%s WHERE annotator_id=%s RETURNING annotator_id", (active, annotator_id))
        if not cur.fetchone():
            raise AuthError("no such annotator")
        if not active:
            _end_sessions(cur, annotator_id, "disabled")


def _end_sessions(cur, annotator_id: int, reason: str, except_session: Optional[int] = None) -> None:
    cur.execute("UPDATE auth_session SET ended_at=now(), end_reason=%s WHERE annotator_id=%s AND ended_at IS NULL "
                "AND (%s::bigint IS NULL OR session_id <> %s)", (reason, annotator_id, except_session, except_session))


# ------------------------------------------------------------------ login / sessions
def login(username: str, password: str, ip: str = "", user_agent: str = "") -> dict:
    username = (username or "").strip()
    ip = (ip or "")[:64]
    with tx() as cur:
        cur.execute("SELECT * FROM annotator WHERE lower(username)=lower(%s) FOR UPDATE", (username,))
        a = cur.fetchone()

        def log(outcome, detail=None):
            cur.execute("INSERT INTO login_event (username, annotator_id, outcome, ip, detail) VALUES (%s,%s,%s,%s,%s)",
                        (username[:64], a["annotator_id"] if a else None, outcome, ip, detail))

        if not a:
            verify_password(password or "", None)
            log("unknown_user")
            return {"error": "wrong username or password"}
        cur.execute("SELECT locked_until > now() AS locked FROM annotator WHERE annotator_id=%s", (a["annotator_id"],))
        if cur.fetchone()["locked"]:
            log("locked")
            return {"error": f"account locked after repeated wrong passwords; try again in {int(C('ann_lockout_minutes'))} minutes or ask an admin"}
        if not a["password_hash"]:
            verify_password(password or "", None)
            log("no_password")
            return {"error": "this account has no password yet: ask an admin to set one"}
        if not verify_password(password or "", a["password_hash"]):
            n = a["failed_logins"] + 1
            lock = n >= int(C("ann_max_failed_logins"))
            cur.execute("UPDATE annotator SET failed_logins=%s, locked_until = CASE WHEN %s THEN now() + make_interval(mins => %s) "
                        "ELSE locked_until END WHERE annotator_id=%s", (0 if lock else n, lock, int(C("ann_lockout_minutes")), a["annotator_id"]))
            log("bad_password", "locked" if lock else None)
            return {"error": "wrong username or password"}
        if not a["active"]:
            log("disabled")
            return {"error": "this account is disabled: ask an admin"}
        token = secrets.token_urlsafe(32)
        cur.execute("INSERT INTO auth_session (annotator_id, token_hash, expires_at, ip, user_agent) "
                    "VALUES (%s,%s, now() + %s * interval '1 hour', %s, %s) RETURNING session_id",
                    (a["annotator_id"], _tok_hash(token), float(C("ann_session_hours")), ip, (user_agent or "")[:200]))
        sid = cur.fetchone()["session_id"]
        cur.execute("UPDATE annotator SET failed_logins=0, locked_until=NULL, last_login_at=now(), last_seen_at=now() "
                    "WHERE annotator_id=%s", (a["annotator_id"],))
        log("ok")
        return {"token": token, "session_id": sid, "annotator": _public(a)}


def resolve(token: str) -> dict:
    """Bearer token -> annotator dict (+ session_id). Raises Unauthorized. Slides the expiry and last_seen forward."""
    if not token:
        raise Unauthorized("sign in required")
    with tx() as cur:
        cur.execute("""SELECT s.session_id, s.last_seen_at, a.* FROM auth_session s JOIN annotator a USING (annotator_id)
                       WHERE s.token_hash=%s AND s.ended_at IS NULL AND s.expires_at > now()""", (_tok_hash(token),))
        r = cur.fetchone()
        if not r:
            raise Unauthorized("session expired: sign in again")
        if not r["active"]:
            raise Unauthorized("this account is disabled")
        cur.execute("UPDATE auth_session SET last_seen_at=now(), expires_at = now() + %s * interval '1 hour' "
                    "WHERE session_id=%s AND last_seen_at < now() - interval '15 seconds'", (float(C("ann_session_hours")), r["session_id"]))
        if cur.rowcount:
            cur.execute("UPDATE annotator SET last_seen_at=now() WHERE annotator_id=%s", (r["annotator_id"],))
        return dict(r)


def logout(token: str) -> None:
    with tx() as cur:
        cur.execute("UPDATE auth_session SET ended_at=now(), end_reason='logout' WHERE token_hash=%s AND ended_at IS NULL "
                    "RETURNING annotator_id", (_tok_hash(token or ""),))
        r = cur.fetchone()
        if r:
            cur.execute("INSERT INTO login_event (username, annotator_id, outcome) "
                        "SELECT username, annotator_id, 'logout' FROM annotator WHERE annotator_id=%s", (r["annotator_id"],))


# ------------------------------------------------------------------ work time
def mark_activity(annotator_id: int, session_id: Optional[int]) -> None:
    with tx() as cur:
        cur.execute("INSERT INTO activity_minute (annotator_id, minute, session_id) VALUES (%s, date_trunc('minute', now()), %s) "
                    "ON CONFLICT DO NOTHING", (annotator_id, session_id))


def heartbeat(a: dict, active: bool) -> dict:
    """Client heartbeat (every ~30 s). `active` = the annotator gave input within ann_idle_seconds with the tab visible."""
    if active:
        mark_activity(a["annotator_id"], a.get("session_id"))
    return {"ok": True}


def worked_minutes(annotator_id: int) -> dict:
    with tx() as cur:
        cur.execute("""SELECT count(*) FILTER (WHERE (minute AT TIME ZONE %(tz)s)::date = (now() AT TIME ZONE %(tz)s)::date) today,
                              count(*) total FROM activity_minute WHERE annotator_id=%(a)s""", {"tz": TZ, "a": annotator_id})
        return dict(cur.fetchone())
