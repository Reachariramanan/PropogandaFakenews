#!/usr/bin/env python3
"""Stdlib HTTP server: static UI + JSON API (FNP_HOST/FNP_PORT, default 127.0.0.1:8765). No extra installs.

Optional HTTP Basic auth: set both FNP_AUTH_USER and FNP_AUTH_PASS to enable it (off otherwise). When on, every
route except GET /healthz needs it. The only outbound call is the Propaganda Lab flag proxy (FNP_FLAG_URL)."""
import base64
import hmac
import json
import mimetypes
import os
import pickle
import re
import signal
import sys
import threading
import urllib.error
import urllib.request
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if __name__ == "__main__":  # settings file first: the imports below read FNP_* variables at import time
    import settings
    settings.load_settings()
import fnlib  # noqa: E402
import claimcheck  # noqa: E402
import realtime_sync  # noqa: E402
import rt_api  # noqa: E402
import rt_insights  # noqa: E402
import rt_jobs  # noqa: E402
import rt_scheduler  # noqa: E402
import rt_triage  # noqa: E402

WEB = os.path.join(HERE, "web")
ITEMS_P = os.path.join(HERE, "build", "items.json")
INDEX_P = os.path.join(HERE, "build", "index.pkl")
HOST, PORT = os.environ.get("FNP_HOST", "127.0.0.1"), int(os.environ.get("FNP_PORT", "8765"))
NO_MATCH, WEAK, STRONG = 0.25, 0.40, 0.60

# Propaganda Lab: read-only view of the FNP aggregate build (never written from here).
ROOT = os.path.dirname(HERE)
PL_DIR = os.environ.get("FNP_PROPLAB_DIR") or os.path.join(ROOT, "reports", "prop_lab", "v1")
PL_ISO = re.compile(r"^[A-Z]{3}$")
PL_MODES = ("about", "origin")
PL_NOT_BUILT = {"error": "Propaganda Lab data not built yet", "code": "not_built"}
PL_LIMIT_MAX = 50
FLAG_URL = os.environ.get("FNP_FLAG_URL", "http://127.0.0.1:8879/api/prop_flag")
FLAG_TIMEOUT = 3.0
FLAG_MAX_BODY = 2048
FLAG_RATE = 30  # requests per client IP per minute
FLAG_UID = re.compile(r"^[A-Za-z0-9:_.\-]{1,128}$")
FLAG_ACTIONS = ("not_propaganda", "confirm", "clear")
_flag_hits = defaultdict(list)
_flag_lock = threading.Lock()
_flag_tech = {"mtime": None, "set": frozenset()}

# Optional HTTP Basic auth (off unless FNP_AUTH_USER and FNP_AUTH_PASS are both set and non-empty).
AUTH_REALM = "Evidence Desk"
AUTH_FAIL_MAX = 10       # failed logins per client IP ...
AUTH_FAIL_WINDOW = 300   # ... within this many seconds -> 429 until the oldest failure ages out
START_MONO = time.monotonic()
_auth_fails = {}         # ip -> monotonic timestamps of recent failed logins (never any credential text)
_auth_lock = threading.Lock()
# Evidence Base (/api/rt/*): refused unless login is configured or FNP_RT_ALLOW_OPEN=1; per-IP rate limit.
RT_RATE = 120            # requests per client IP per minute on /api/rt/*
RT_HINT = "Set FNP_AUTH_USER and FNP_AUTH_PASS to use the Evidence Base"
_rt_hits = defaultdict(list)
_rt_lock = threading.Lock()
RT = None                # rt_api.RtApi, created lazily (needs the fact-check index for nearest matches)
TRI = None               # rt_triage.Triage (Triage tab), built over whatever RT is current
JOBS = None              # rt_jobs.Jobs (Analysis Workbench), created lazily with RT
INSIGHTS = None          # rt_insights.Insights, created lazily with RT
_rt_init_lock = threading.Lock()
RT_ANALYSE_RATE = 6      # new analysis jobs per client IP per minute
RT_INSIGHT_RATE = 10     # insight requests per client IP per minute
RT_POST_MAX_BODY = 16384
RT_ANALYSE_LIMIT = claimcheck.RateLimit(RT_ANALYSE_RATE)
RT_INSIGHT_LIMIT = claimcheck.RateLimit(RT_INSIGHT_RATE)
_health = {"sig": None, "val": None}
_health_lock = threading.Lock()


def auth_creds(env=None):
    """(user, password) as bytes when both env vars are set and non-empty, else None (auth off)."""
    env = os.environ if env is None else env
    u, p = env.get("FNP_AUTH_USER", ""), env.get("FNP_AUTH_PASS", "")
    return (u.encode("utf-8"), p.encode("utf-8")) if u and p else None


AUTH = auth_creds()


def auth_header_ok(header, creds):
    """True when an 'Authorization: Basic ...' header carries exactly the configured credentials."""
    if not header or creds is None:
        return False
    scheme, _, rest = header.strip().partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        raw = base64.b64decode(rest.strip(), validate=True)
    except Exception:  # noqa: BLE001
        return False
    user, sep, pw = raw.partition(b":")
    ok_user = hmac.compare_digest(user, creds[0])  # evaluate both: no early exit on a wrong user name
    ok_pass = hmac.compare_digest(pw, creds[1])
    return bool(sep) and ok_user and ok_pass


def auth_locked(ip):
    """Seconds until a throttled IP may try again, or 0 when it is not throttled."""
    now = time.monotonic()
    with _auth_lock:
        hits = [t for t in _auth_fails.get(ip, ()) if now - t < AUTH_FAIL_WINDOW]
        if hits:
            _auth_fails[ip] = hits
        else:
            _auth_fails.pop(ip, None)
        if len(hits) >= AUTH_FAIL_MAX:
            return int(AUTH_FAIL_WINDOW - (now - hits[0])) + 1
    return 0


def auth_record_failure(ip):
    now = time.monotonic()
    with _auth_lock:
        hits = [t for t in _auth_fails.get(ip, ()) if now - t < AUTH_FAIL_WINDOW]
        hits.append(now)
        _auth_fails[ip] = hits[-AUTH_FAIL_MAX:]
        if len(_auth_fails) > 2000:
            for k in [k for k, v in _auth_fails.items() if not v or now - v[-1] >= AUTH_FAIL_WINDOW]:
                del _auth_fails[k]
        return len(_auth_fails[ip])


def build_health():
    """(built_at, n_items) of build/items.json. Raises if it is missing or unreadable. The 188 MB file is parsed
    at most once per (inode, mtime, size); a loaded Store is reused when it is current."""
    st = os.stat(ITEMS_P)
    if S.items and S.mtime is not None and S.mtime[0] == st.st_mtime:
        return S.meta.get("built"), len(S.items)
    sig = (st.st_ino, st.st_mtime_ns, st.st_size)
    with _health_lock:
        if _health["sig"] != sig:
            val = None
            try:
                with open(ITEMS_P, encoding="utf-8") as f:
                    d = json.load(f)
                if not isinstance(d["items"], list):
                    raise ValueError("items is not a list")
                val = (d["meta"].get("built"), len(d["items"]))
            except Exception:  # noqa: BLE001
                val = None
            _health["sig"], _health["val"] = sig, val
        if _health["val"] is None:
            raise ValueError("items.json unreadable")
        return _health["val"]


def flag_techniques():
    p = os.path.join(PL_DIR, "heatmap.json")
    try:
        mt = os.stat(p).st_mtime
        if _flag_tech["mtime"] != mt:
            with open(p, encoding="utf-8") as f:
                _flag_tech["set"] = frozenset(t for t in json.load(f).get("techniques", []) if isinstance(t, str))
            _flag_tech["mtime"] = mt
    except Exception:  # noqa: BLE001
        pass
    return _flag_tech["set"]


def flag_rate_ok(ip):
    now = time.time()
    with _flag_lock:
        hits = [t for t in _flag_hits[ip] if now - t < 60]
        ok = len(hits) < FLAG_RATE
        if ok:
            hits.append(now)
        _flag_hits[ip] = hits
        if len(_flag_hits) > 2000:
            for k in [k for k, v in _flag_hits.items() if not v or now - v[-1] >= 60]:
                del _flag_hits[k]
    return ok


def desk_url_map(items):
    """{normalised URL: Desk item id}; a fact-check beats a claim candidate that shares the URL."""
    from build_index import norm_url
    out = {}
    for it in items:
        u = it.get("url")
        if not isinstance(u, str) or not u.strip():
            continue
        k = norm_url(u)
        if k and (k not in out or (it.get("kind") == "factcheck" and out[k][1] != "factcheck")):
            out[k] = (it.get("id"), it.get("kind"))
    return {k: v[0] for k, v in out.items()}


class Store:
    """Loads build artefacts; reloads automatically when a rebuild replaces them."""

    def __init__(self):
        self.lock = threading.Lock()
        self.mtime = None
        self.items, self.by_id, self.meta, self.index, self.fc = [], {}, {}, None, []
        self.url_id = {}  # normalised fact-check URL -> Desk item id (built once per index load; FNP nearest fact-check links)

    def refresh(self):
        try:
            m = (os.path.getmtime(ITEMS_P), os.path.getmtime(INDEX_P))
        except OSError:
            return False
        if m == self.mtime:
            return True
        with self.lock:
            if m == self.mtime:
                return True
            with open(ITEMS_P, encoding="utf-8") as f:
                d = json.load(f)
            with open(INDEX_P, "rb") as f:
                idx = pickle.load(f)
            self.items, self.meta = d["items"], d["meta"]
            self.by_id = {it["id"]: it for it in self.items}
            self.url_id = desk_url_map(self.items)
            self.fc = [self.by_id[i] for i in self.meta.get("index_ids", []) if i in self.by_id]
            self.index = idx
            self.mtime = m
        return True


S = Store()


class PropLabCache:
    """Encoded-JSON cache keyed on each file's (inode, mtime, size), so an atomic v1.building -> v1 swap is
    picked up on the next request. Files are opened, read and closed per call; no handle is kept."""

    MAX = 64

    def __init__(self):
        self.lock = threading.Lock()
        self.d = {}  # rel -> (sig, body, etag); insertion order doubles as LRU order

    def get(self, rel):
        fp = os.path.join(PL_DIR, rel)
        for attempt in (0, 1):
            try:
                st = os.stat(fp)
                sig = (st.st_ino, st.st_mtime_ns, st.st_size)
                with self.lock:
                    hit = self.d.pop(rel, None)
                    if hit and hit[0] == sig:
                        self.d[rel] = hit
                        return hit[1], hit[2]
                with open(fp, "rb") as f:
                    raw = f.read()
                body = json.dumps(fnlib.clean_json(json.loads(raw)), ensure_ascii=False, allow_nan=False).encode()
                etag = '"%x-%x-%x"' % sig
                with self.lock:
                    self.d[rel] = (sig, body, etag)
                    while len(self.d) > self.MAX:
                        self.d.pop(next(iter(self.d)))
                return body, etag
            except FileNotFoundError:
                if attempt:
                    raise
                time.sleep(0.05)  # a rebuild may be mid-swap


PL = PropLabCache()


def api_proplab_posts(mode, iso, q):
    """Stream the jsonl line by line and keep only the requested page."""
    g = lambda k, d="": (q.get(k) or [d])[0]
    try:
        offset = min(max(int(g("offset", "0")), 0), 100000)
        limit = min(max(int(g("limit", "20")), 1), PL_LIMIT_MAX)
    except ValueError:
        offset, limit = 0, 20
    posts, has_more, nxt = [], False, offset
    with open(os.path.join(PL_DIR, "posts", mode, iso + ".jsonl"), "rb") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(posts) >= limit:
                has_more = True
                break
            nxt = i + 1
            line = line.strip()
            if not line:
                continue
            try:
                posts.append(json.loads(line))
            except ValueError:
                continue
    return {"mode": mode, "iso": iso, "offset": offset, "limit": limit, "count": len(posts),
            "has_more": has_more, "next_offset": nxt, "posts": posts}


def slim(it):
    summ = it.get("description") or (it.get("body") or "")[:260]
    summ = re.sub(r"\s+", " ", summ).strip()
    return {k: it.get(k) for k in ("id", "kind", "site", "url", "claim", "title", "verdict_norm", "verdict_raw",
                                   "published", "lang", "plaus", "score_source", "cross_checker_agreement",
                                   "org", "country", "source_tier", "verdict_basis", "channel")} | {
        "summary": summ[:260]}


def parse_date(s):
    return s if s and re.match(r"\d{4}-\d{2}-\d{2}", s) else None


def api_items(q):
    g = lambda k, d="": (q.get(k) or [d])[0]
    sites = {s for s in g("site").lower().split(",") if s}
    verdicts = {v for v in g("verdict").upper().split(",") if v}
    langs = {v for v in g("lang").lower().split(",") if v}
    countries = {v for v in g("country").upper().split(",") if v}
    tiers = {v for v in g("tier").lower().split(",") if v}
    channels = {v for v in g("channel").lower().split(",") if v}
    kind = g("kind")
    dfrom, dto = parse_date(g("from")), parse_date(g("to"))
    try:
        lo = float(g("min_plaus", "0") or 0)
        hi = float(g("max_plaus", "1") or 1)
    except ValueError:
        lo, hi = 0.0, 1.0
    terms = [t for t in re.findall(r"[\wऀ-෿]+", g("q").lower()) if t]
    out = []
    for it in S.items:
        if sites and it["site"] not in sites:
            continue
        if verdicts and it["verdict_norm"] not in verdicts:
            continue
        if langs and it["lang"].lower() not in langs:
            continue
        if countries and (it.get("country") or "").upper() not in countries:
            continue
        if tiers and (it.get("source_tier") or "").lower() not in tiers:
            continue
        if channels and (it.get("channel") or "web").lower() not in channels:
            continue
        if kind and it["kind"] != kind:
            continue
        p = it["plaus"]
        if (lo > 0 or hi < 1) and (p is None or p < lo or p > hi):
            continue
        pub = (it["published"] or "")[:10]
        if dfrom and (not pub or pub < dfrom):
            continue
        if dto and (not pub or pub > dto):
            continue
        if terms:
            hay = (it["title"] + " " + it["claim"] + " " + (it.get("description") or "") + " " + it["body"][:2000]).lower()
            if not all(t in hay for t in terms):
                continue
        out.append(it)
    if g("sort", "plaus") == "date":
        out.sort(key=lambda x: x["published"] or "", reverse=True)
    else:
        out.sort(key=lambda x: (x["plaus"] is None, -(x["plaus"] or 0), x["published"] or ""))
    try:
        size = min(max(int(g("page_size", "24")), 1), 100)
        page = max(int(g("page", "1")), 1)
    except ValueError:
        size, page = 24, 1
    pages = max(1, -(-len(out) // size))
    page = min(page, pages)
    return {"total": len(out), "page": page, "pages": pages, "page_size": size,
            "items": [slim(i) for i in out[(page - 1) * size: page * size]]}


def api_stats():
    its = S.items
    sites, verdicts, months, langs = Counter(), Counter(), Counter(), Counter()
    countries, tiers, channels = Counter(), Counter(), Counter()
    hist = [0] * 10
    unscored = 0
    for it in its:
        sites[it["site"]] += 1
        verdicts[it["verdict_norm"]] += 1
        langs[it["lang"]] += 1
        if it.get("country"):
            countries[it["country"]] += 1
        channels[it.get("channel") or "web"] += 1
        if it.get("source_tier"):
            tiers[it["source_tier"]] += 1
        if it["published"]:
            months[it["published"][:7]] += 1
        if it["plaus"] is None:
            unscored += 1
        else:
            hist[min(int(it["plaus"] * 10), 9)] += 1
    mix = defaultdict(Counter)
    for it in its:
        mix[it["site"]][it["verdict_norm"]] += 1
    return {"total": len(its), "sites": dict(sites.most_common()), "verdicts": dict(verdicts),
            "languages": dict(langs.most_common()), "countries": dict(countries.most_common()),
            "source_tiers": dict(tiers.most_common()), "channels": dict(channels.most_common()), "site_verdict_mix": {s: dict(c) for s, c in mix.items()},
            "months": dict(sorted(months.items())),
            "score_histogram": [{"lo": i / 10, "hi": (i + 1) / 10, "count": c} for i, c in enumerate(hist)],
            "unscored": unscored, "meta": {k: S.meta.get(k) for k in
                                          ("built", "n_factchecks", "n_candidates", "engine", "tfidf_backend")}}


def api_check(text):
    sig = fnlib.signals(text)
    res = {"signals": sig, "thresholds": {"no_close_match": NO_MATCH, "weak": WEAK, "strong": STRONG}}
    hits = [(S.fc[j], s) for j, s in S.index.query(text, 5) if j < len(S.fc)] if S.index and S.fc else []
    best = hits[0][1] if hits else 0.0
    res["nearest_similarity"] = round(best, 3)
    if best < NO_MATCH:
        res.update(state="no_close_match", score=None, confidence=None, matches=[],
                   message="No close precedent: nothing in the fact-check database resembles this text "
                           "(best cosine %.2f, below %.2f). No plausibility score is given; only the "
                           "linguistic signals below are reported. Absence of a match does not mean the "
                           "claim is true." % (best, NO_MATCH))
        return res
    used = [(m, s) for m, s in hits if s >= NO_MATCH and m["plaus"] is not None]
    w = [s * s for _, s in used]
    raw = sum(wi * m["plaus"] for wi, (m, _) in zip(w, used)) / sum(w)
    conf = min(1.0, best / STRONG)  # shrink toward the 0.5 prior when the match is weak
    score = 0.5 + (raw - 0.5) * conf
    level = "strong" if best >= STRONG else "weak" if best < WEAK else "moderate"
    msgs = {"strong": "Close match to previously fact-checked claims; the score reflects how those were rated.",
            "moderate": "Partial overlap with previously fact-checked claims. Read the evidence cards before trusting the score.",
            "weak": "Only a weak resemblance to existing fact-checks. Treat the score as a faint hint, not a finding."}
    res.update(state=level, score=round(score, 3), raw_score=round(raw, 3),
               confidence={"level": level, "max_similarity": round(best, 3), "shrink": round(conf, 3),
                           "matches_used": len(used)},
               message=msgs[level] + " The score is a similarity-weighted average of the matched fact-checks "
                                     "(weights = cosine squared), not a ruling on the pasted text.",
               matches=[{**{k: m.get(k) for k in ("id", "site", "url", "title", "claim", "verdict_norm",
                                                  "verdict_raw", "published", "plaus")},
                         "similarity": round(s, 3), "summary": re.sub(r"\s+", " ", m.get("description") or m["body"][:240]).strip()[:240]}
                        for m, s in hits if s >= NO_MATCH])
    return res


def rt_open_allowed(env=None):
    return (os.environ if env is None else env).get("FNP_RT_ALLOW_OPEN") == "1"


def rt_rate_ok(ip):
    now = time.time()
    with _rt_lock:
        hits = [t for t in _rt_hits[ip] if now - t < 60]
        ok = len(hits) < RT_RATE
        if ok:
            hits.append(now)
        _rt_hits[ip] = hits
        if len(_rt_hits) > 2000:
            for k in [k for k, v in _rt_hits.items() if not v or now - v[-1] >= 60]:
                del _rt_hits[k]
    return ok


def fn_matches(text):
    """Nearest Fake News Database matches for a text (same TF-IDF index and thresholds as Claim Check step 1)."""
    if not S.refresh() or not S.index:
        return None
    return api_check(text)


def desk_id_for_url(url):
    """Desk item id for a fact-check URL (normalised as the index builder does), or None."""
    if not isinstance(url, str) or not S.refresh():
        return None
    from build_index import norm_url
    return S.url_id.get(norm_url(url))


def rt():
    global RT
    if RT is None:
        with _rt_init_lock:
            if RT is None:
                RT = rt_api.RtApi(fn_matches=fn_matches, desk_url=desk_id_for_url)
    return RT


def triage():
    """The Triage API over the current RtApi (rebuilt if RT was replaced, e.g. by a test)."""
    global TRI
    api = rt()
    if TRI is None or TRI.api is not api:
        with _rt_init_lock:
            if TRI is None or TRI.api is not api:
                TRI = rt_triage.Triage(api)
    return TRI


def job_index_fn(text):
    """Fact-check index match for a workbench item; an unbuilt index just means no match."""
    try:
        r = fn_matches(text)
    except Exception:  # noqa: BLE001
        r = None
    return r or {"state": "no_close_match", "score": None, "matches": [], "nearest_similarity": 0.0}


def jobs():
    global JOBS
    if JOBS is None:
        with _rt_init_lock:
            if JOBS is None:
                JOBS = rt_jobs.Jobs(db_path=rt().db_path, index_fn=job_index_fn)
    return JOBS


def insights():
    global INSIGHTS
    if INSIGHTS is None:
        with _rt_init_lock:
            if INSIGHTS is None:
                INSIGHTS = rt_insights.Insights(rt())
    return INSIGHTS


def maybe_start_sync(env=None):
    """Start the in-process background sync only when FNP_RT_SYNC=1 (thread, no OS persistence). True if started."""
    env = os.environ if env is None else env
    if env.get("FNP_RT_SYNC") != "1":
        return False
    try:
        interval = float(env.get("FNP_RT_INTERVAL") or 300)
    except ValueError:
        interval = 300.0
    realtime_sync.start_background(interval_s=max(30.0, interval))
    return True


def maybe_start_analysis(env=None):
    """Start the in-process scheduled deep analysis only when FNP_RT_ANALYSE=1 (thread, no OS persistence)."""
    env = os.environ if env is None else env
    if not rt_scheduler.enabled_from_env(env):
        return False
    api = rt()
    rt_scheduler.start_background(db_path=api.db_path, enrich=api.enrich, jobs=jobs(), env=env)
    return True


class H(BaseHTTPRequestHandler):
    server_version = "FNPInterface/1.0"

    def log_message(self, fmt, *a):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))

    def send_json(self, obj, code=200, headers=None):
        b = json.dumps(fnlib.clean_json(obj), ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def send_cached(self, body, etag, code=200):
        if etag and self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        if etag:
            self.send_header("ETag", etag)
        self.end_headers()
        self.wfile.write(body)

    def proplab(self, path, q):
        if not os.path.isdir(PL_DIR):
            return self.send_json(PL_NOT_BUILT, 503)
        parts = path[len("/api/proplab/"):].split("/")
        static = {"meta": "meta.json", "world": "world_paths.json", "countries": "countries.json",
                  "heatmap": "heatmap.json"}
        try:
            if len(parts) == 1 and parts[0] in static:
                return self.send_cached(*PL.get(static[parts[0]]))
            if len(parts) == 2 and parts[0] == "country":
                if not PL_ISO.match(parts[1]):
                    return self.send_json({"error": "bad country code; expected ISO3 like IND"}, 400)
                return self.send_cached(*PL.get("country/%s.json" % parts[1]))
            if len(parts) == 1 and parts[0] == "posts":
                mode, iso = (q.get("mode") or [""])[0], (q.get("iso") or [""])[0]
                if mode not in PL_MODES:
                    return self.send_json({"error": "mode must be one of: about, origin"}, 400)
                if not PL_ISO.match(iso):
                    return self.send_json({"error": "bad country code; expected ISO3 like IND"}, 400)
                return self.send_json(api_proplab_posts(mode, iso, q))
        except FileNotFoundError:
            return self.send_json({"error": "not found"}, 404)
        except ValueError:
            return self.send_json({"error": "data file unreadable; a build may be in progress"}, 502)
        return self.send_json({"error": "unknown endpoint"}, 404)

    def gate(self):
        """True when the request may proceed. With auth on and no valid credentials a 401 (or 429 when this IP is
        throttled) has already been sent. Only a presented-but-wrong login counts as a failure; a request with no
        Authorization header (a browser's first, unauthenticated try) just gets the 401 challenge."""
        if AUTH is None:
            return True
        ip = self.client_address[0]
        wait = auth_locked(ip)
        if wait:
            self.send_json({"error": "too many failed logins; try again later"}, 429, {"Retry-After": str(wait)})
            return False
        hdr = self.headers.get("Authorization")
        if hdr and auth_header_ok(hdr, AUTH):
            return True
        if hdr:
            n = auth_record_failure(ip)
            print("%s auth failure ip=%s (%d/%d)" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), ip, n, AUTH_FAIL_MAX),
                  file=sys.stderr, flush=True)
        self.send_json({"error": "authentication required"}, 401,
                       {"WWW-Authenticate": 'Basic realm="%s"' % AUTH_REALM})
        return False

    def healthz(self):
        up = round(time.monotonic() - START_MONO, 1)
        try:
            built, n = build_health()
        except Exception:  # noqa: BLE001
            return self.send_json({"ok": False, "error": "build/items.json missing or unreadable",
                                   "uptime_s": up}, 503)
        try:
            pl = json.loads(PL.get("meta.json")[0]).get("built_at")
        except Exception:  # noqa: BLE001
            pl = None
        try:
            rtb = rt().health()
        except Exception:  # noqa: BLE001
            rtb = {"enabled": os.environ.get("FNP_RT_SYNC") == "1", "running": False, "last_sync": None,
                   "lag_minutes": None, "rows": 0, "errors": None,
                   "analysis": {"enabled": rt_scheduler.enabled_from_env(), "last_run": None, "next_run_at": None,
                                "analysed_deep": 0}}
        self.send_json({"ok": True, "build_built_at": built, "n_items": n, "proplab_built_at": pl,
                        "uptime_s": up, "realtime": rtb})

    def rt_guard(self):
        """Evidence Base gate shared by GET and POST: login configured or FNP_RT_ALLOW_OPEN=1 (else 403), then the
        per-IP rate limit (429). True when the request may proceed (otherwise the response has been sent)."""
        if AUTH is None and not rt_open_allowed():
            self.send_json({"error": "login_required", "hint": RT_HINT}, 403)
            return False
        if not rt_rate_ok(self.client_address[0]):
            self.send_json({"error": "rate limit: %d requests per minute" % RT_RATE}, 429, {"Retry-After": "30"})
            return False
        return True

    def evidence(self, path, q):
        """GET /api/rt/*: login gate (403), per-IP rate limit (429), then the read-only handlers in rt_api."""
        if path == "/api/rt/gate":  # no data: lets the UI explain why the Evidence Base is closed
            return self.send_json({"open": rt_open_allowed(), "auth": AUTH is not None})
        if not self.rt_guard():
            return
        try:
            api = rt()
            if path == "/api/rt/status":
                return self.send_json(api.status())
            if path == "/api/rt/schedule":
                return self.send_json(api.schedule())
            if path == "/api/rt/stats":
                return self.send_json(api.stats(q))
            if path == "/api/rt/items":
                return self.send_json(api.items(q))
            if path == "/api/rt/sources":
                return self.send_json(api.sources(q))
            if path.startswith("/api/rt/item/"):
                return self.send_json(api.item(path[len("/api/rt/item/"):]))
            m = re.fullmatch(r"/api/rt/jobs/([0-9a-f]{16})", path)
            if m:
                return self.send_json(jobs().get(m.group(1), results=(q.get("results") or ["1"])[0] != "0"))
            return self.send_json({"error": "unknown endpoint"}, 404)
        except rt_api.ApiError as e:
            return self.send_json({"error": e.msg}, e.code)
        except rt_jobs.JobError as e:
            return self.send_json({"error": e.msg}, e.code, e.headers)

    def triage_get(self, path, q):
        """GET /api/triage/*: same gate (403) and per-IP rate limit (429) as /api/rt/*, then the read-only handlers in rt_triage."""
        if not self.rt_guard():
            return
        try:
            api = triage()
            if path == "/api/triage/meta":
                return self.send_json(api.meta())
            if path == "/api/triage/items":
                return self.send_json(api.items(q))
            if path == "/api/triage/facets":
                return self.send_json(api.facets(q))
            if path.startswith("/api/triage/item/"):
                return self.send_json(api.item(path[len("/api/triage/item/"):], q))
            return self.send_json({"error": "unknown endpoint"}, 404)
        except rt_api.ApiError as e:
            return self.send_json({"error": e.msg}, e.code)

    def read_json_body(self, required=True):
        """Strict JSON object from the request body (no NaN/Infinity), or None after sending 400/413."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.send_json({"error": "bad Content-Length"}, 400)
            return None
        if n < 0 or n > RT_POST_MAX_BODY:
            self.send_json({"error": "payload too large (max %d KB)" % (RT_POST_MAX_BODY // 1024)}, 413)
            return None
        raw = self.rfile.read(n) if n else b""
        if not raw.strip():
            if required:
                self.send_json({"error": "expected a JSON body"}, 400)
                return None
            return {}

        def bad(c):
            raise ValueError("non-strict JSON constant %s" % c)
        try:
            body = json.loads(raw.decode("utf-8"), parse_constant=bad)
        except Exception:  # noqa: BLE001
            self.send_json({"error": "expected a JSON body"}, 400)
            return None
        if not isinstance(body, dict):
            self.send_json({"error": "expected a JSON object"}, 400)
            return None
        return body

    def evidence_post(self, path):
        """POST /api/rt/{analyse,insights,jobs/<id>/cancel}: same gate and rate limit as the GET routes."""
        if not self.rt_guard():
            return
        ip = self.client_address[0]
        try:
            if path == "/api/rt/analyse":
                body = self.read_json_body()
                if body is None:
                    return
                extra = set(body) - {"ids", "force"}
                if extra:
                    return self.send_json({"error": "unknown field: %s" % sorted(extra)[0]}, 400)
                if not isinstance(body.get("force", False), bool):
                    return self.send_json({"error": "force must be true or false"}, 400)
                wait = RT_ANALYSE_LIMIT.check(ip)
                if wait:
                    return self.send_json({"error": "rate limit: %d analysis jobs per minute" % RT_ANALYSE_RATE}, 429,
                                          {"Retry-After": str(wait)})
                return self.send_json(jobs().submit(body.get("ids"), body.get("force", False)), 202)
            if path == "/api/rt/insights":
                body = self.read_json_body()
                if body is None:
                    return
                wait = RT_INSIGHT_LIMIT.check(ip)
                if wait:
                    return self.send_json({"error": "rate limit: %d insight requests per minute" % RT_INSIGHT_RATE}, 429,
                                          {"Retry-After": str(wait)})
                return self.send_json(insights().run(body))
            m = re.fullmatch(r"/api/rt/jobs/([0-9a-f]{16})/cancel", path)
            if m:
                if self.read_json_body(required=False) is None:
                    return
                return self.send_json(jobs().cancel(m.group(1)))
            return self.send_json({"error": "unknown endpoint"}, 404)
        except rt_api.ApiError as e:
            return self.send_json({"error": e.msg}, e.code)
        except rt_jobs.JobError as e:
            return self.send_json({"error": e.msg}, e.code, e.headers)
        except Exception:  # noqa: BLE001 - never leak internals, never drop the connection
            return self.send_json({"error": "server error"}, 500)

    def need_data(self):
        if S.refresh():
            return True
        self.send_json({"error": "index not built; run: python3 Interface/build_index.py"}, 503)
        return False

    def do_GET(self):
        u = urlsplit(self.path)
        path = unquote(u.path)
        try:
            if path == "/healthz":  # liveness probe: the only route that never needs auth
                return self.healthz()
            if not self.gate():
                return
            if path.startswith("/api/proplab/"):
                return self.proplab(path, parse_qs(u.query))
            if path.startswith("/api/rt/"):
                return self.evidence(path, parse_qs(u.query))
            if path.startswith("/api/triage/"):
                return self.triage_get(path, parse_qs(u.query))
            if path.startswith("/api/"):
                if not self.need_data():
                    return
                if path == "/api/items":
                    return self.send_json(api_items(parse_qs(u.query)))
                if path == "/api/stats":
                    return self.send_json(api_stats())
                if path == "/api/examples":
                    return self.send_json(claimcheck.api_examples(S.items, S.mtime, parse_qs(u.query)))
                if path.startswith("/api/item/"):
                    it = S.by_id.get(path[len("/api/item/"):])
                    return self.send_json(it, 200) if it else self.send_json({"error": "not found"}, 404)
                return self.send_json({"error": "unknown endpoint"}, 404)
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            fp = os.path.realpath(os.path.join(WEB, rel))
            if not fp.startswith(WEB + os.sep) or not os.path.isfile(fp):
                self.send_response(404)
                self.end_headers()
                return
            with open(fp, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", (mimetypes.guess_type(fp)[0] or "application/octet-stream") + "; charset=utf-8"
                             if fp.endswith((".html", ".js", ".css")) else mimetypes.guess_type(fp)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except BrokenPipeError:
            pass
        except Exception as e:  # noqa: BLE001
            self.send_json({"error": "server error: %s" % type(e).__name__}, 500)

    def proplab_flag(self):
        ip = self.client_address[0]
        if not flag_rate_ok(ip):
            return self.send_json({"error": "rate limit: 30 requests per minute"}, 429)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self.send_json({"error": "bad Content-Length"}, 400)
        if n < 0 or n > FLAG_MAX_BODY:
            return self.send_json({"error": "payload too large (max 2 KB)"}, 413)
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return self.send_json({"error": "expected a JSON body"}, 400)
        if not isinstance(body, dict):
            return self.send_json({"error": "expected a JSON object"}, 400)
        uid, action, tech = body.get("post_uid"), body.get("action"), body.get("technique")
        if not isinstance(uid, str) or not FLAG_UID.match(uid):
            return self.send_json({"error": "invalid post_uid"}, 400)
        if not isinstance(action, str) or action not in FLAG_ACTIONS:
            return self.send_json({"error": "invalid action"}, 400)
        out = {"post_uid": uid, "action": action}
        if tech is not None:
            if not isinstance(tech, str) or tech not in flag_techniques():
                return self.send_json({"error": "invalid technique"}, 400)
            out["technique"] = tech
        req = urllib.request.Request(FLAG_URL, data=json.dumps(out).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=FLAG_TIMEOUT) as r:
                code, raw = r.status, r.read(65536)
        except urllib.error.HTTPError as e:
            code, raw = e.code, e.read(65536)
        except Exception:  # noqa: BLE001
            return self.send_json({"error": "flag service unavailable"}, 502)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            payload = {"error": "bad response from flag service"}
            code = 502
        if 200 <= code < 300:
            print("%s flag ip=%s action=%s post_uid=%s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), ip, action, uid), flush=True)
        self.send_json(payload, code)

    def analyse(self):
        """POST /api/analyse {text, lang?, force_propaganda?}: index match + engine + combined verdict (claimcheck)."""
        if not self.need_data():
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 100_000:
                return self.send_json({"error": "payload too large"}, 413)
            body = json.loads(self.rfile.read(max(n, 0)) or b"{}")
            if not isinstance(body, dict) or not isinstance(body.get("text", ""), str):
                raise ValueError("bad body")
        except Exception:  # noqa: BLE001
            return self.send_json({"error": "expected JSON body {\"text\": ...}"}, 400)
        text = body.get("text", "").strip()
        if len(text) < 3:
            return self.send_json({"error": "text too short"}, 400)
        lang = body.get("lang")
        lang = lang if isinstance(lang, str) and re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?", lang) else None
        try:
            out, code, hdr = claimcheck.analyse(text, lang, body.get("force_propaganda") is True,
                                                self.client_address[0], api_check)
        except Exception:  # noqa: BLE001
            return self.send_json({"error": "analysis failed"}, 500)
        self.send_json(out, code, hdr)

    def do_POST(self):
        if not self.gate():
            return
        if urlsplit(self.path).path.startswith("/api/rt/"):
            return self.evidence_post(unquote(urlsplit(self.path).path))
        if urlsplit(self.path).path == "/api/proplab/flag":
            return self.proplab_flag()
        if urlsplit(self.path).path == "/api/analyse":
            return self.analyse()
        if urlsplit(self.path).path != "/api/check":
            return self.send_json({"error": "unknown endpoint"}, 404)
        if not self.need_data():
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 100_000:
                return self.send_json({"error": "payload too large"}, 413)
            text = str(json.loads(self.rfile.read(n) or b"{}").get("text", "")).strip()
        except Exception:
            return self.send_json({"error": "expected JSON body {\"text\": ...}"}, 400)
        if len(text) < 3:
            return self.send_json({"error": "text too short"}, 400)
        self.send_json(api_check(text[:5000]))


def run():
    S.refresh()
    if not S.items:
        print("warning: build/items.json missing; run python3 Interface/build_index.py", file=sys.stderr)
    if bool(os.environ.get("FNP_AUTH_USER")) != bool(os.environ.get("FNP_AUTH_PASS")):
        print("warning: only one of FNP_AUTH_USER / FNP_AUTH_PASS is set; auth stays OFF", file=sys.stderr)
    httpd = ThreadingHTTPServer((HOST, PORT), H)
    syncing = maybe_start_sync()
    analysing = False
    try:
        analysing = maybe_start_analysis()
    except Exception as e:  # noqa: BLE001 - the scheduler is optional; never stop the server for it
        print("warning: scheduled analysis not started: %s" % type(e).__name__, file=sys.stderr)
    print("serving on http://%s:%d (auth: %s, realtime sync: %s, scheduled analysis: %s, realtime routes: %s)" % (
        HOST, PORT, "on" if AUTH else "off", "on" if syncing else "off", "on" if analysing else "off",
        "open" if AUTH or rt_open_allowed() else "closed (login required)"), flush=True)

    def stop(signum, frame):  # SIGTERM: leave serve_forever from another thread, then clean up below
        threading.Thread(target=httpd.shutdown, daemon=True).start()
    try:
        signal.signal(signal.SIGTERM, stop)
    except ValueError:  # not the main thread
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rt_scheduler.stop_background()
        realtime_sync.stop_background()
        httpd.server_close()


if __name__ == "__main__":
    run()
