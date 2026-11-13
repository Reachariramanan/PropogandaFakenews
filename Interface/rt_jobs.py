"""Analysis Workbench jobs (POST /api/rt/analyse, GET /api/rt/jobs/<id>, POST /api/rt/jobs/<id>/cancel).

A job is a list of realtime item ids run through the existing Claim Check pipeline (claimcheck.analyse: index match,
engine queue and lock, combine and gate), strictly one item at a time and one job at a time across the whole process,
so jobs never put more than one engine call in flight. Jobs live in memory only (the last JOBS_KEEP are kept).
Results of normal runs are cached in the SQLite `analyses` table (key: item id + engine version); a degraded run
(check_v1's own 55 s timeout) or a failed one is shown but never cached. Item text is never logged, only ids, seconds
and states.

Partial results (S2): a scheduled run that gets a degraded reply (check_v1 timeout / gemma_error) keeps what the reply
does carry (propaganda classes, techniques, spans) as a `partial` row in `analyses`: verdict absent, `partial: true`,
`partial_info: {retries, last_try_at, first_at, reason}`. It is never a cache hit (a later run goes to the engine) and
never a deep verdict. A reply with no extracted claim (`degraded_why: no_claims`) is a complete, deterministic result
and is stored and cached like any other. Workbench jobs do not store partials (store_partial is off for them).

Item states: queued | running | done | cached | degraded | skipped | error.
Job states:  queued | running | done | cancelled | error.
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import claimcheck  # noqa: E402
import rt_store  # noqa: E402

MAX_IDS = 25
MIN_TEXT_CHARS = 20          # below this the item is skipped: nothing worth checking
MAX_TEXT_CHARS = 8000        # what is sent to the engine (title + blank line + excerpt)
JOBS_KEEP = 20               # finished jobs kept in memory (oldest evicted first)
MAX_QUEUED_JOBS = 5          # jobs waiting behind the running one
BUSY_RETRIES = 2             # re-tries of one item when the engine answers busy
BUSY_WAIT_CAP = 45.0         # longest wait (s) between busy re-tries; the engine's Retry-After is honoured up to this
RESULT_MAX_BYTES = 400000    # same bound rt_api.item uses when reading stored results
_RE_ITEM = re.compile(r"^[A-Za-z0-9_.:\-]{1,200}$")
_RE_LANG = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?$")
ACTIVE = ("queued", "running")


def has_propaganda(body):
    """True when a Claim Check body carries the propaganda step (classes or techniques), i.e. a partial result is worth keeping."""
    p = ((body.get("step2") or {}).get("propaganda") if isinstance(body, dict) and isinstance(body.get("step2"), dict) else None)
    return isinstance(p, dict) and (bool(p.get("classes")) or bool(p.get("techniques")) or bool(p.get("technique_probs")))


def make_partial(body, sha, info):
    """A degraded Claim Check body -> the partial result stored in `analyses`: propaganda kept, every verdict removed
    (engine.verdict / fake_call, step1 label/plausibility/classes of the engine lead), `partial: true` + `partial_info`."""
    out = dict(body, text_sha1=sha, partial=True, partial_info=dict(info))
    eng = dict(out.get("engine") or {})
    eng["verdict"] = None
    eng["fake_call"] = None
    out["engine"] = eng
    s1 = dict(out.get("step1") or {})
    if s1.get("lead") != "index":
        s1.update(source="none", lead=None, label=None, plausibility=None, second_opinion=None, classes=[],
                  note="The deep verdict did not finish (%s), so there is no verdict. Only the propaganda signals are kept."
                       % info.get("reason", "unavailable"))
    elif isinstance(s1.get("second_opinion"), dict):
        s1["second_opinion"] = None
    out["step1"] = s1
    return out


class JobError(Exception):
    def __init__(self, code, msg, headers=None):
        super().__init__(msg)
        self.code, self.msg, self.headers = code, msg, headers or {}


def engine_version():
    """Cache key part: which engine contract the configured URL speaks."""
    url = claimcheck.ENGINE_URL or ""
    return "check_v1" if "check_v1" in url else "predict_v0" if "predict" in url else "engine"


def parse_ids(raw):
    """Validated, de-duplicated (order kept) list of item ids from the request body, or JobError(400)."""
    if not isinstance(raw, list) or not raw:
        raise JobError(400, "ids must be a non-empty list of item ids")
    if len(raw) > MAX_IDS * 4:
        raise JobError(400, "too many ids (max %d)" % MAX_IDS)
    out, seen = [], set()
    for v in raw:
        if not isinstance(v, str) or not _RE_ITEM.match(v):
            raise JobError(400, "ids must be item id strings")
        if v not in seen:
            seen.add(v)
            out.append(v)
    if len(out) > MAX_IDS:
        raise JobError(400, "too many ids (max %d per job)" % MAX_IDS)
    return out


def compose_text(title, excerpt):
    """title + blank line + excerpt (whichever exist), trimmed to what the engine takes."""
    t, e = (title or "").strip(), (excerpt or "").strip()
    text = (t + "\n\n" + e) if t and e else (t or e)
    return text[:MAX_TEXT_CHARS].strip()


def _iso(dt):
    return rt_store.utc_iso(dt)


class Job:
    def __init__(self, job_id, ids, force, created):
        self.id, self.force, self.created_at = job_id, force, created
        self.state = "queued"
        self.started_at = self.finished_at = None
        self.cancel = threading.Event()
        self.busy_retries = BUSY_RETRIES
        self.store_partial = False
        self.items = [dict(id=i, state="queued", started_at=None, secs=None, result=None, reason=None) for i in ids]


class Jobs:
    def __init__(self, db_path=None, index_fn=None, analyse_fn=None, clock=None, keep=JOBS_KEEP,
                 max_queued=MAX_QUEUED_JOBS):
        self.db_path = db_path or rt_store.DEFAULT_DB
        self.index_fn = index_fn
        self.analyse_fn = analyse_fn          # None -> claimcheck.analyse (looked up per call so tests can patch it)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.keep, self.max_queued = keep, max_queued
        self.cv = threading.Condition()
        self.jobs = {}                        # id -> Job, insertion ordered
        self.queue = deque()                  # queued Job objects
        self.running = None
        self._stop = False
        self._thread = None

    # ---- store access (short-lived connections; only the `analyses` table is ever written) ----------------------
    def _con(self):
        return rt_store.open_store(self.db_path, create=False)

    def _existing(self, ids):
        con = self._con()
        try:
            have = set()
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                have.update(r[0] for r in con.execute(
                    "SELECT id FROM items WHERE id IN (%s)" % ",".join("?" * len(chunk)), chunk))
            return have
        finally:
            con.close()

    def _load_item(self, item_id):
        con = self._con()
        try:
            r = con.execute("SELECT title, text_excerpt, lang FROM items WHERE id = ?", (item_id,)).fetchone()
            return dict(r) if r else None
        finally:
            con.close()

    def _cache_get(self, item_id, version, text_sha):
        con = self._con()
        try:
            r = con.execute("SELECT created_at, result_json FROM analyses WHERE item_id = ? AND engine_version = ?",
                            (item_id, version)).fetchone()
        finally:
            con.close()
        if r is None or len(r[1]) > RESULT_MAX_BYTES:
            return None
        try:
            res = json.loads(r[1])
        except ValueError:
            return None
        if not isinstance(res, dict) or res.get("text_sha1") != text_sha:  # the stored text changed since: stale
            return None
        if res.get("partial"):  # a partial result is not an answer: the next run goes to the engine
            return None
        return r[0], res

    def partial_info(self, item_id, version=None):
        """partial_info dict of the stored partial row of this item (retries, last_try_at, ...), or None."""
        con = self._con()
        try:
            for (rj,) in con.execute("SELECT result_json FROM analyses WHERE item_id = ?" +
                                     (" AND engine_version = ?" if version else ""),
                                     (item_id, version) if version else (item_id,)):
                if len(rj) > RESULT_MAX_BYTES:
                    continue
                try:
                    r = json.loads(rj)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("partial") and isinstance(r.get("partial_info"), dict):
                    return r["partial_info"]
            return None
        finally:
            con.close()

    def _cache_put(self, item_id, version, result):
        con = self._con()
        try:
            con.execute("INSERT OR REPLACE INTO analyses(item_id, created_at, engine_version, result_json) "
                        "VALUES (?,?,?,?)", (item_id, _iso(self.clock()), version,
                                             json.dumps(result, ensure_ascii=False, allow_nan=False)))
        finally:
            con.close()

    # ---- public API -----------------------------------------------------------------------------------------------
    def submit(self, ids, force=False):
        ids = parse_ids(ids)
        have = self._existing(ids)
        missing = [i for i in ids if i not in have]
        if missing:
            raise JobError(404, "unknown item ids: %s" % ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
        with self.cv:
            if len(self.queue) >= self.max_queued:
                raise JobError(429, "too many analysis jobs are waiting (max %d); try again shortly" % self.max_queued,
                               {"Retry-After": "30"})
            job = Job(os.urandom(8).hex(), ids, bool(force), _iso(self.clock()))
            self.jobs[job.id] = job
            self.queue.append(job)
            self._evict()
            self._ensure_thread()
            self.cv.notify_all()
            return self._view(job, results=False)

    def get(self, job_id, results=True):
        with self.cv:
            job = self.jobs.get(job_id)
            if job is None:
                raise JobError(404, "unknown job (jobs are kept for this server session only)")
            return self._view(job, results)

    def cancel(self, job_id):
        with self.cv:
            job = self.jobs.get(job_id)
            if job is None:
                raise JobError(404, "unknown job (jobs are kept for this server session only)")
            if job.state == "queued":
                self.queue.remove(job)
                self._finish(job, "cancelled")
            elif job.state == "running":
                job.cancel.set()  # the current engine call finishes, then the rest is skipped
            self.cv.notify_all()
            return self._view(job, results=False)

    def analyse_one(self, item_id, cancel=None, busy_retries=0, store_partial=True):
        """Run ONE item through the same path as a Workbench job, synchronously, without queueing a job (used by the
        scheduled precompute, rt_scheduler). Same cache, same engine lock (one engine call in flight), a degraded
        result is stored as a PARTIAL row (propaganda only, no verdict; see the module docstring) and is never a cache hit. A busy engine is NOT retried here (busy_retries=0): the caller decides
        how to back off. -> {state, reason, result, secs, ...} like a job item. Never raises."""
        job = Job("sched", [item_id], False, _iso(self.clock()))
        job.busy_retries = int(busy_retries)
        job.store_partial = bool(store_partial)
        if cancel is not None:
            job.cancel = cancel
        it = job.items[0]
        t0 = time.monotonic()
        try:
            out = self._run_item(job, it)
        except Exception as e:  # noqa: BLE001 - never log text
            out = dict(state="error", reason="analysis failed (%s)" % type(e).__name__, result=None)
        out["secs"] = round(time.monotonic() - t0, 2)
        return out

    def has_deep(self, item_id):
        """True when a usable stored result (an engine verdict) exists for this item under any engine version."""
        con = self._con()
        try:
            for (rj,) in con.execute("SELECT result_json FROM analyses WHERE item_id = ?", (item_id,)):
                if len(rj) <= RESULT_MAX_BYTES:
                    try:
                        r = json.loads(rj)
                    except ValueError:
                        continue
                    v = ((r.get("engine") or {}).get("verdict") if isinstance(r, dict) else None)
                    if isinstance(v, dict) and v.get("label"):
                        return True
            return False
        finally:
            con.close()

    def stored_result(self, item_id):
        """(created_at, result dict) of the newest usable stored result for the item (any engine version), or None."""
        con = self._con()
        try:
            rows = con.execute("SELECT created_at, result_json FROM analyses WHERE item_id = ? "
                               "ORDER BY created_at DESC", (item_id,)).fetchall()
        finally:
            con.close()
        for created, rj in rows:
            if len(rj) > RESULT_MAX_BYTES:
                continue
            try:
                r = json.loads(rj)
            except ValueError:
                continue
            v = ((r.get("engine") or {}).get("verdict") if isinstance(r, dict) else None)
            if isinstance(v, dict) and v.get("label"):
                return created, r
        return None

    def store_dup(self, item_id, source_id, source_result):
        """Record `source_result` (a stored deep result of a near-duplicate) for item_id, marked dup_of=source_id.
        Its text_sha1 is the SOURCE's, so a Workbench run on this item re-checks it instead of trusting the copy."""
        res = dict(source_result, dup_of=source_result.get("dup_of") or source_id)
        self._cache_put(item_id, engine_version(), res)

    def shutdown(self):
        with self.cv:
            self._stop = True
            if self.running:
                self.running.cancel.set()
            self.cv.notify_all()
        if self._thread:
            self._thread.join(10)

    # ---- internals ------------------------------------------------------------------------------------------------
    def _evict(self):
        """Keep at most `keep` jobs: drop the oldest finished ones (never a queued or running job)."""
        for jid in [j for j, job in self.jobs.items() if job.state not in ACTIVE]:
            if len(self.jobs) <= self.keep:
                break
            del self.jobs[jid]

    def _finish(self, job, state):
        for it in job.items:
            if it["state"] == "queued":
                it["state"], it["reason"] = "skipped", "cancelled"
        job.state, job.finished_at = state, _iso(self.clock())

    def _view(self, job, results):
        n_done = sum(1 for it in job.items if it["state"] not in ACTIVE)
        if job.state == "queued":
            pos = list(self.queue).index(job) + 1 + (1 if self.running else 0)
        elif job.state == "running":
            pos = 0
        else:
            pos = None
        items = []
        for it in job.items:
            d = dict(it)
            d["has_result"] = d["result"] is not None
            if not results:
                d["result"] = None  # the full body costs kilobytes per item; GET ...?results=0 polls without it
            items.append(d)
        return dict(job_id=job.id, state=job.state, total=len(job.items), done=n_done,
                    cancelling=bool(job.cancel.is_set() and job.state == "running"), queue_position=pos,
                    force=job.force, created_at=job.created_at, started_at=job.started_at,
                    finished_at=job.finished_at, server_time=_iso(self.clock()), engine_version=engine_version(),
                    items=items)

    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="rt-jobs", daemon=True)
            self._thread.start()

    def _loop(self):
        while True:
            with self.cv:
                while not self.queue and not self._stop:
                    self.cv.wait(1.0)
                if self._stop:
                    return
                job = self.queue.popleft()
                self.running = job
                job.state, job.started_at = "running", _iso(self.clock())
            try:
                self._run_job(job)
                final = "cancelled" if job.cancel.is_set() else "done"
            except Exception as e:  # noqa: BLE001 - a bug must not kill the worker; never log text
                print("%s rt-job job=%s crashed: %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), job.id, type(e).__name__),
                      file=sys.stderr, flush=True)
                final = "error"
            with self.cv:
                self._finish(job, final)
                self.running = None
                self._evict()
                self.cv.notify_all()

    def _run_job(self, job):
        for it in job.items:
            if job.cancel.is_set() or self._stop:
                break
            with self.cv:
                it["state"], it["started_at"] = "running", _iso(self.clock())
            t0 = time.monotonic()
            try:
                out = self._run_item(job, it)
            except Exception:  # noqa: BLE001
                out = dict(state="error", reason="analysis failed", result=None)
            secs = round(time.monotonic() - t0, 2)
            with self.cv:
                it.update(out)
                it["secs"] = secs
                it.pop("note", None)
                self.cv.notify_all()
            print("%s rt-job job=%s item=%s secs=%.1f state=%s" % (
                time.strftime("%Y-%m-%dT%H:%M:%S%z"), job.id, it["id"], secs, it["state"]), file=sys.stderr, flush=True)

    def _run_item(self, job, it):
        row = self._load_item(it["id"])
        if row is None:
            return dict(state="error", reason="the item is no longer in the realtime cache", result=None)
        text = compose_text(row["title"], row["text_excerpt"])
        if len(text) < MIN_TEXT_CHARS:
            return dict(state="skipped", reason="not enough text to check (under %d characters)" % MIN_TEXT_CHARS,
                        result=None)
        sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
        ver = engine_version()
        if not job.force:
            hit = self._cache_get(it["id"], ver, sha)
            if hit:
                return dict(state="cached", reason=None, result=claimcheck.with_blocks(hit[1]), cached_at=hit[0])
        lang = row["lang"] if isinstance(row["lang"], str) and _RE_LANG.match(row["lang"]) else None
        fn = self.analyse_fn or claimcheck.analyse
        index_fn = self.index_fn or (lambda t: {"state": "no_close_match", "score": None, "matches": [],
                                                "nearest_similarity": 0.0})
        if job.force:  # a forced re-run must reach the engine, not claimcheck's in-memory LRU in front of it
            claimcheck.CACHE.discard(claimcheck.Cache.key(text, lang, claimcheck.ENGINE_URL))
        tries = 0
        while True:
            body, code, headers = fn(text, lang, False, None, index_fn)  # ip None: no per-IP engine limit for jobs
            eng = body.get("engine") if isinstance(body, dict) else None
            if code != 200 or not isinstance(eng, dict):
                return dict(state="error", reason=(body or {}).get("error") or "analysis failed", result=None)
            if eng.get("state") == "busy" and tries < job.busy_retries and not job.cancel.is_set():
                wait = max(1.0, min(BUSY_WAIT_CAP, float(eng.get("retry_after_s") or headers.get("Retry-After") or 10)))
                with self.cv:
                    it["note"] = "engine busy, trying again in %d s" % round(wait)
                tries += 1
                if job.cancel.wait(wait):
                    return dict(state="error", reason="cancelled while waiting for the engine", result=None)
                continue
            break
        st = eng.get("state")
        if st == "ok":
            result = dict(body, text_sha1=sha)
            self._cache_put(it["id"], ver, result)
            return dict(state="done", reason=None, result=result)
        if st == "degraded" and eng.get("degraded_why") == "no_claims":
            # no checkable claim: complete and deterministic (same text, same outcome), so stored and cached like ok
            result = dict(body, text_sha1=sha)
            self._cache_put(it["id"], ver, result)
            return dict(state="done", reason=None, result=result, no_claims=True)
        if st == "degraded":
            out = dict(state="degraded", reason=eng.get("reason") or "the deep verdict timed out",
                       result=dict(body, text_sha1=sha))
            if job.store_partial and has_propaganda(body):
                prev = self.partial_info(it["id"], ver) or {}
                now = _iso(self.clock())
                retries = int(prev.get("retries", -1)) + 1 if prev else 0
                part = make_partial(body, sha, dict(retries=retries, last_try_at=now, first_at=prev.get("first_at") or now,
                                                    reason=eng.get("degraded_why") or "unavailable"))
                self._cache_put(it["id"], ver, part)
                out.update(result=part, partial=True, retries=retries)
            return out
        out = dict(state="error", reason=eng.get("reason") or "the engine could not be used", result=None)
        if st == "busy":
            out["retry_after_s"] = eng.get("retry_after_s")
            out["reason"] = "the engine is busy with other checks; try again shortly"
        out["engine_state"] = st
        return out
