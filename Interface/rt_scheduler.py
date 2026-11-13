"""Scheduled precompute of DEEP FNP analysis (Claim Check via check_v1) for the realtime items.

An in-process daemon thread, like the realtime sync thread: nothing persistent outside the server process (no systemd,
no cron). Off unless FNP_RT_ANALYSE=1. Every FNP_RT_ANALYSE_INTERVAL seconds (default 3 h) one cycle picks at most
FNP_RT_ANALYSE_BUDGET unanalysed items from the LOCAL store and runs them one at a time through the same path as a
Workbench job (rt_jobs.Jobs.analyse_one: claimcheck.analyse, the engine lock, the `analyses` cache). It reads only the
local SQLite store and calls the FNP engine; it never touches the Realtime Database.

Politeness (the engine is shared; 15-40 s per item):
  * strictly sequential: one engine call in flight from this thread;
  * a busy reply waits for the engine's Retry-After ONCE (capped at 60 s) and that item is tried once more; then the
    cycle moves on;
  * the cycle ENDS EARLY (reason recorded) when >= 70 % of the last 10 replies were busy / degraded / failed (and at
    least 6 replies were seen), when the engine is unreachable, and at the time cap FNP_RT_ANALYSE_MAX_SECS (5400 s);
  * adaptive pacing: when the median latency of the last 5 engine calls exceeds 40 s the REMAINING budget is halved
    (at most once per 5 calls);
  * a degraded reply (the engine's own timeout / gemma_error) is kept as a PARTIAL result (propaganda signals, no
    verdict; rt_jobs) and re-queued on later cycles: at most 3 retries, at least one cycle apart, after the fresh
    high-priority items. A reply with no extracted claim is a complete cacheable result, not a partial.

Selection (see select_candidates): text >= 40 characters, not a database duplicate, inside the sync window, no usable
stored result yet, not flagged `too_short` by the FNP export. Priority = +3 when the export found a fact-check
(nearest_factcheck.rerank present, i.e. cosine >= 0.72), else +2 when it has n_strong >= 1 and p_prop >= 0.6, + p_prop
(0-1) + recency (newer first, 0-1) + a small log(engagement) bonus; at least 30 % of the budget per kind (news / social)
when that many are available; with no export rows it is recency (and engagement) only. Near-duplicates (simhash within
3 bits) of an item that already has a deep result are not run again: the duplicate gets a copy of that result marked
`dup_of` (counted as cached_dupes).

Every cycle writes one row to `schedule_log`. Item text is never logged: only ids, states and counts.
"""
import math
import os
import statistics
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import realtime_sync  # noqa: E402
import rt_jobs  # noqa: E402
import rt_store  # noqa: E402
from rt_text import to_unsigned  # noqa: E402

DEFAULT_INTERVAL_S = 10800.0       # 3 hours
DEFAULT_BUDGET = 40                # items per cycle (S2: lowered from 100; 9 of 200 were analysed before)
DEFAULT_MAX_SECS = 5400.0          # cap per cycle (1.5 h)
DEFAULT_FIRST_DELAY_S = 600.0      # after server start
MIN_INTERVAL_S = 30.0
MAX_BUDGET = 500                   # hard ceiling on the env value: never a big bulk run
MIN_TEXT_CHARS = 40                # title + excerpt
SIMILARITY_BOOST = 3.0             # a nearest fact-check with a rerank (cosine >= 0.72) adds this
SIMILARITY_MIN = 0.72              # the FNP export writes `rerank` only from this cosine
PROP_BOOST = 2.0                   # n_strong >= 1 and p_prop >= PROP_MIN
PROP_MIN = 0.6
KIND_SHARE = 0.30                  # at least this share of the budget per kind when available
STOP_WINDOW = 10                   # the stop rule looks at the last 10 replies ...
STOP_MIN_REPLIES = 6               # ... once at least 6 have been seen ...
STOP_BAD_SHARE = 0.7               # ... and ends the cycle when this share was busy / degraded / failed
SLOW_WINDOW = 5                    # adaptive pacing: median latency of the last 5 calls ...
SLOW_MEDIAN_S = 40.0               # ... above this halves the remaining budget
BUSY_WAIT_CAP = 60.0
RETRY_MAX = 3                      # re-tries of a partial item for its verdict
RETRY_SHARE = 0.25                 # at most this share of the budget goes to partial re-tries
POOL_PER_KIND = 600                # newest unanalysed candidates read per kind
SIGNAL_PER_KIND = 300              # FNP-signal candidates read per kind
READY_RECHECK_S = 60.0
FRESH_HOURS = 24.0                 # the store counts as live when it holds an item newer than this
LOG_KEEP = 10


def _env_float(env, name, default, lo, hi):
    try:
        v = float(env.get(name))
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v)) if math.isfinite(v) else default


def enabled_from_env(env=None):
    return (os.environ if env is None else env).get("FNP_RT_ANALYSE") == "1"


def config_from_env(env=None):
    env = os.environ if env is None else env
    return dict(interval_s=_env_float(env, "FNP_RT_ANALYSE_INTERVAL", DEFAULT_INTERVAL_S, MIN_INTERVAL_S, 7 * 86400.0),
                budget=int(_env_float(env, "FNP_RT_ANALYSE_BUDGET", DEFAULT_BUDGET, 1, MAX_BUDGET)),
                max_secs=_env_float(env, "FNP_RT_ANALYSE_MAX_SECS", DEFAULT_MAX_SECS, 10, 86400.0),
                first_delay_s=_env_float(env, "FNP_RT_ANALYSE_FIRST_DELAY", DEFAULT_FIRST_DELAY_S, 0, 86400.0))


def _iso(dt):
    return rt_store.utc_iso(dt)


def priority(sim, p_prop, age_hours, engagement, n_strong=None, rerank=None):
    """Priority score of one candidate (see the module docstring). Pure, for tests."""
    pp = p_prop if isinstance(p_prop, (int, float)) and 0 <= p_prop <= 1 else None
    if rerank is not None or (sim is not None and sim >= SIMILARITY_MIN):
        s = SIMILARITY_BOOST
    elif pp is not None and pp >= PROP_MIN and (n_strong is None or n_strong >= 1):
        s = PROP_BOOST
    else:
        s = 0.0
    s += pp if pp is not None else 0.0
    s += 1.0 / (1.0 + max(0.0, age_hours) / 24.0)
    if isinstance(engagement, (int, float)) and engagement > 0:
        s += min(0.5, 0.1 * math.log10(1.0 + engagement))
    return s


def pick_balanced(scored, budget, share=KIND_SHARE):
    """scored: [(score, kind, id)] -> up to `budget` picks, best first. Each kind gets up to ceil(share*budget) of its best
    before the rest of the budget is filled by score, so neither news nor social starves."""
    ranked = sorted(scored, key=lambda t: (-t[0], t[2]))
    quota = math.ceil(share * budget)
    taken, per_kind = [], {}
    for t in ranked:
        if len(taken) < budget and per_kind.get(t[1], 0) < quota:
            taken.append(t)
            per_kind[t[1]] = per_kind.get(t[1], 0) + 1
    # top up what is left of the budget by overall score (a kind with fewer than `quota` leaves room)
    chosen = {t[2] for t in taken}
    for t in ranked:
        if len(taken) >= budget:
            break
        if t[2] not in chosen:
            taken.append(t)
            chosen.add(t[2])
    return sorted(taken, key=lambda t: (-t[0], t[2]))[:budget]


class Scheduler:
    def __init__(self, db_path=None, enrich=None, jobs=None, interval_s=DEFAULT_INTERVAL_S, budget=DEFAULT_BUDGET,
                 max_secs=DEFAULT_MAX_SECS, first_delay_s=DEFAULT_FIRST_DELAY_S, clock=None, monotonic=None,
                 sync_runs_fn=None, sleep_cap=BUSY_WAIT_CAP, wait_fn=None):
        self.db_path = db_path or rt_store.DEFAULT_DB
        self.enrich = enrich
        self.jobs = jobs or rt_jobs.Jobs(db_path=self.db_path)
        self.interval_s = max(1.0, float(interval_s))
        self.budget = max(1, min(MAX_BUDGET, int(budget)))
        self.max_secs = float(max_secs)
        self.first_delay_s = max(0.0, float(first_delay_s))
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.mono = monotonic or time.monotonic
        self.sync_runs_fn = sync_runs_fn or (lambda: realtime_sync.background_status().get("runs", 0) or 0)
        self.sleep_cap = sleep_cap
        self._wait_fn = wait_fn           # tests: callable(seconds) -> bool(stop requested); default waits on the stop event
        self._stop = threading.Event()
        self._cycle_lock = threading.Lock()
        self._thread = None
        self._state_lock = threading.Lock()
        self._running_now = False
        self._next_run_at = None
        self._waiting = None            # why the loop is not running a cycle yet
        self._last = None               # dict of the most recent cycle of this process

    # ---- store helpers ------------------------------------------------------------------------------------------
    def _con(self, readonly=True):
        con = rt_store.open_store(self.db_path, create=False)
        rt_store.ensure_deep_view(con)   # before query_only
        if readonly:
            con.execute("PRAGMA query_only=ON")
            if self.enrich is not None:
                self.enrich.attach(con)
        return con

    def ensure_table(self):
        con = rt_store.open_store(self.db_path, create=False)
        try:
            rt_store.ensure_schedule_log(con)
            # a row left open by a previous process (crash, kill) is closed as interrupted
            con.execute("UPDATE schedule_log SET ended_at = started_at, stop_reason = 'interrupted' "
                        "WHERE ended_at IS NULL")
        finally:
            con.close()

    # ---- selection ----------------------------------------------------------------------------------------------
    def _enr_cols(self, con):
        """(sim, p_prop, n_strong, rerank, too_short) SQL expressions for the export sidecar, or NULLs without one.
        Older sidecars (before the real export format) lack the fc_cos / fc_rerank / n_strong columns."""
        if self.enrich is None:
            return ("NULL", "NULL", "NULL", "NULL", "0"), ""
        have = {r[1] for r in con.execute("PRAGMA enr.table_info(enr)")}
        col = lambda c, e="NULL": ("e." + c) if c in have else e
        sim = "coalesce(%s, json_extract(e.extra, '$.nearest_factcheck.similarity'))" % col("fc_cos")
        return (sim, "e.p_prop", col("n_strong"), col("fc_rerank"), "coalesce(%s, 0)" % col("too_short", "0")), \
            "LEFT JOIN enr.enr e ON e.id = i.id"

    def select_candidates(self, now=None, budget=None):
        """[(id, kind, score, sim)] best first: up to `budget` (default: the cycle budget) unanalysed, eligible items
        (see the module docstring). Items with a stored partial result are not here (see select_retries)."""
        now = now or self.clock()
        budget = self.budget if budget is None else max(0, int(budget))
        if budget == 0:
            return []
        pol = realtime_sync.effective_policy()
        cutoff = {"news": _iso(now - timedelta(days=pol["news_days"])),
                  "social": _iso(now - timedelta(days=pol["social_days"]))}
        have_enr = self.enrich is not None
        con = self._con()
        try:
            (c_sim, c_pp, c_ns, c_rr, c_short), frm = self._enr_cols(con)
            sel = ("SELECT i.id, i.kind, i.published_at, i.engagement, %s AS sim, %s AS p_prop, %s AS n_strong, "
                   "%s AS rerank FROM items i %s LEFT JOIN deep d ON d.id = i.id WHERE d.id IS NULL "
                   "AND length(coalesce(i.title, '')) + length(coalesce(i.text_excerpt, '')) >= ? "
                   "AND (i.is_duplicate IS NULL OR i.is_duplicate = 0) AND i.kind = ? AND i.published_at >= ? "
                   "AND %s = 0 "
                   # complete results that carry no verdict (an old engine's no_claims reply) and partial rows' own
                   # retry gating are handled here: only partials are re-selected, by select_retries
                   "AND NOT EXISTS (SELECT 1 FROM analyses a WHERE a.item_id = i.id AND json_valid(a.result_json) AND "
                   "(json_extract(a.result_json, '$.partial') = 1 OR json_extract(a.result_json, '$.engine.degraded_why') = 'no_claims'))"
                   ) % (c_sim, c_pp, c_ns, c_rr, frm, c_short)
            rows = {}
            for kind in ("news", "social"):
                args = (MIN_TEXT_CHARS, kind, cutoff[kind])
                for r in con.execute(sel + " ORDER BY i.published_at DESC, i.id LIMIT %d" % POOL_PER_KIND, args):
                    rows[r["id"]] = r
                if have_enr:  # items the FNP export flagged as worth a deep look, even when not the newest
                    for r in con.execute(
                            sel + " AND (%s IS NOT NULL OR %s >= %s OR (%s >= %s AND coalesce(%s, 1) >= 1)) "
                            "ORDER BY (%s IS NOT NULL OR %s >= %s) DESC, coalesce(%s, 0) DESC, i.published_at DESC, i.id LIMIT %d"
                            % (c_rr, c_sim, SIMILARITY_MIN, c_pp, PROP_MIN, c_ns, c_rr, c_sim, SIMILARITY_MIN, c_pp,
                               SIGNAL_PER_KIND), args):
                        rows[r["id"]] = r
        finally:
            con.close()
        scored, sims = [], {}
        for r in rows.values():
            try:
                age_h = (now - rt_store.parse_iso(r["published_at"])).total_seconds() / 3600.0
            except (TypeError, ValueError):
                continue
            sim = r["sim"] if isinstance(r["sim"], (int, float)) else None
            sims[r["id"]] = sim
            scored.append((priority(sim, r["p_prop"], age_h, r["engagement"], r["n_strong"], r["rerank"]),
                           r["kind"], r["id"]))
        return [(i, k, s, sims.get(i)) for s, k, i in pick_balanced(scored, budget)]

    def select_retries(self, now=None, limit=None):
        """[(id, kind, 0.0, None)] partial items due another try for a verdict: fewer than RETRY_MAX retries, the last
        try at least one cycle (the interval) ago, still in the sync window, newest first. At most RETRY_SHARE of the
        budget (`limit` overrides)."""
        now = now or self.clock()
        limit = max(1, math.ceil(RETRY_SHARE * self.budget)) if limit is None else int(limit)
        pol = realtime_sync.effective_policy()
        cutoff = {"news": _iso(now - timedelta(days=pol["news_days"])),
                  "social": _iso(now - timedelta(days=pol["social_days"]))}
        ready_before = _iso(now - timedelta(seconds=0.9 * self.interval_s))   # tolerance for scheduling jitter
        con = self._con()
        try:
            rows = con.execute(
                "SELECT i.id, i.kind, i.published_at FROM partial p JOIN items i ON i.id = p.id "
                "WHERE p.retries < ? AND coalesce(p.last_try_at, p.partial_at) <= ? "
                "AND length(coalesce(i.title, '')) + length(coalesce(i.text_excerpt, '')) >= ? "
                "AND (i.is_duplicate IS NULL OR i.is_duplicate = 0) ORDER BY i.published_at DESC, i.id",
                (RETRY_MAX, ready_before, MIN_TEXT_CHARS)).fetchall()
        finally:
            con.close()
        out = [(r["id"], r["kind"], 0.0, None) for r in rows if (r["published_at"] or "") >= cutoff.get(r["kind"], "")]
        return out[:limit]

    # ---- one cycle ----------------------------------------------------------------------------------------------
    def run_cycle(self, next_run_in=None):
        """Run one cycle now. Returns the log row dict, or None when a cycle is already running (no overlap).
        Never raises. `next_run_in` seconds is recorded as next_run_at (the loop passes the interval)."""
        if not self._cycle_lock.acquire(blocking=False):
            return None
        started = self.clock()
        t0 = self.mono()
        rec = dict(started_at=_iso(started), ended_at=None, picked=0, analysed=0, cached_dupes=0, degraded=0, errors=0,
                   skipped=0, partial=0, retried=0, median_secs=None, stop_reason=None, next_run_at=None)
        rowid = None
        with self._state_lock:
            self._running_now, self._waiting = True, None
        try:
            try:
                rowid = self._log_start(rec)
            except Exception as e:  # noqa: BLE001
                self._err("log start", e)
            try:
                self._execute(rec, t0)
            except Exception as e:  # noqa: BLE001 - a bug in one cycle must not end the scheduler
                rec["errors"] += 1
                rec["stop_reason"] = "cycle error: %s" % type(e).__name__
                self._err("cycle", e)
            ended = self.clock()
            rec["ended_at"] = _iso(ended)
            if next_run_in is not None:
                rec["next_run_at"] = _iso(ended + timedelta(seconds=next_run_in))
            if rec["stop_reason"] is None:
                rec["stop_reason"] = "done"
            try:
                self._log_end(rowid, rec)
            except Exception as e:  # noqa: BLE001
                self._err("log end", e)
            with self._state_lock:
                self._last = dict(rec)
            print("%s rt-sched cycle picked=%d analysed=%d partial=%d retried=%d dupes=%d degraded=%d errors=%d skipped=%d "
                  "median_secs=%s stop=%s" % (
                      time.strftime("%Y-%m-%dT%H:%M:%S%z"), rec["picked"], rec["analysed"], rec["partial"], rec["retried"],
                      rec["cached_dupes"], rec["degraded"], rec["errors"], rec["skipped"], rec["median_secs"],
                      rec["stop_reason"]), file=sys.stderr, flush=True)
            return rec
        finally:
            with self._state_lock:
                self._running_now = False
            self._cycle_lock.release()

    def _wait(self, seconds):
        """Interruptible pause; True when the server is stopping."""
        if self._wait_fn is not None:
            return bool(self._wait_fn(seconds))
        return self._stop.wait(seconds)

    def _execute(self, rec, t0):
        try:
            retries = self.select_retries()
            cands = self.select_candidates(budget=self.budget - len(retries)) + retries   # fresh first, retries after
        except Exception as e:  # noqa: BLE001
            rec["errors"] += 1
            rec["stop_reason"] = "selection failed: %s" % type(e).__name__
            self._err("select", e)
            return
        retry_ids = {c[0] for c in retries}
        rec["picked"] = len(cands)
        if not cands:
            rec["stop_reason"] = "nothing to analyse"
            return
        window, lat, since_cut, halved = [], [], 0, False   # window: True = a good reply; lat: engine call seconds
        stop = None
        con = self._con(readonly=True)
        try:
            i = 0
            while i < len(cands):
                item_id = cands[i][0]
                i += 1
                if self._stop.is_set():
                    stop = "server stopping"
                    break
                if self.mono() - t0 >= self.max_secs:
                    stop = "time cap (%d s)" % int(self.max_secs)
                    break
                try:
                    if item_id not in retry_ids and self._try_duplicate(con, item_id):
                        rec["cached_dupes"] += 1
                        continue
                except Exception as e:  # noqa: BLE001 - a failed duplicate lookup just means: run it
                    self._err("dup", e)
                if item_id in retry_ids:
                    rec["retried"] += 1
                busy_waited = False
                while True:
                    t_call = self.mono()
                    out = self.jobs.analyse_one(item_id, cancel=self._stop)
                    st = out.get("state")
                    secs = max(0.0, self.mono() - t_call)
                    print("%s rt-sched item=%s secs=%.1f state=%s%s" % (
                        time.strftime("%Y-%m-%dT%H:%M:%S%z"), item_id, out.get("secs") or 0, st,
                        " retry" if item_id in retry_ids else ""), file=sys.stderr, flush=True)
                    if st == "cached":
                        rec["cached_dupes"] += 1
                        break                             # no engine call: not a reply for the stop rule
                    if st == "skipped":
                        rec["skipped"] += 1
                        break
                    lat.append(secs)
                    since_cut += 1
                    if st == "done":
                        rec["analysed"] += 1
                        window.append(True)
                        break
                    window.append(False)
                    if st == "degraded":
                        rec["degraded"] += 1
                        rec["partial"] += 1 if out.get("partial") else 0
                        break
                    rec["errors"] += 1
                    if out.get("engine_state") == "unavailable":
                        stop = "engine unreachable"
                        break
                    if out.get("engine_state") == "busy" and not busy_waited:
                        busy_waited = True                # honour Retry-After once, then this item gets one more try
                        try:
                            wait = float(out.get("retry_after_s"))
                        except (TypeError, ValueError):
                            wait = 10.0
                        if self._wait(min(self.sleep_cap, max(1.0, wait))):
                            stop = "server stopping"
                            break
                        continue
                    break
                if stop:
                    break
                recent = window[-STOP_WINDOW:]
                if len(window) >= STOP_MIN_REPLIES and recent.count(False) >= STOP_BAD_SHARE * len(recent) - 1e-9:
                    stop = "%d of the last %d replies were busy, degraded or failed" % (recent.count(False), len(recent))
                    break
                if since_cut >= SLOW_WINDOW and len(lat) >= SLOW_WINDOW:
                    med = statistics.median(lat[-SLOW_WINDOW:])
                    if med > SLOW_MEDIAN_S:
                        remaining = len(cands) - i
                        keep = remaining // 2
                        if remaining > 0:
                            del cands[i + keep:]
                            halved = True
                            since_cut = 0
        finally:
            con.close()
            if lat:
                rec["median_secs"] = round(statistics.median(lat), 1)
        if stop is None and halved:
            stop = "engine slow (median over %d s): half of the remaining items were left for the next run" % int(SLOW_MEDIAN_S)
        rec["stop_reason"] = stop or "done"

    def _try_duplicate(self, con, item_id):
        """If a near-duplicate (simhash within 3 bits) of this item already has a deep result, store a copy marked
        dup_of for this item and return True; otherwise False."""
        r = con.execute("SELECT simhash FROM items WHERE id = ?", (item_id,)).fetchone()
        if r is None or r[0] is None:
            return False
        for _d, _pub, other, _t, _s in rt_store.near_dup_scan(con, item_id, to_unsigned(r[0]), 3):
            got = self.jobs.stored_result(other)
            if got is not None:
                self.jobs.store_dup(item_id, got[1].get("dup_of") or other, got[1])
                return True
        return False

    # ---- schedule_log -------------------------------------------------------------------------------------------
    def _log_start(self, rec):
        con = rt_store.open_store(self.db_path, create=False)
        try:
            rt_store.ensure_schedule_log(con)
            cur = con.execute("INSERT INTO schedule_log(started_at, picked, analysed, cached_dupes, degraded, errors, "
                              "skipped) VALUES (?,0,0,0,0,0,0)", (rec["started_at"],))
            return cur.lastrowid
        finally:
            con.close()

    def _log_end(self, rowid, rec):
        con = rt_store.open_store(self.db_path, create=False)
        try:
            rt_store.ensure_schedule_log(con)
            vals = (rec["ended_at"], rec["picked"], rec["analysed"], rec["cached_dupes"], rec["degraded"],
                    rec["errors"], rec["skipped"], rec["stop_reason"], rec["next_run_at"], rec["partial"],
                    rec["retried"], rec["median_secs"])
            if rowid is None:
                con.execute("INSERT INTO schedule_log(ended_at, picked, analysed, cached_dupes, degraded, errors, "
                            "skipped, stop_reason, next_run_at, partial, retried, median_secs, started_at) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", vals + (rec["started_at"],))
            else:
                con.execute("UPDATE schedule_log SET ended_at=?, picked=?, analysed=?, cached_dupes=?, degraded=?, "
                            "errors=?, skipped=?, stop_reason=?, next_run_at=?, partial=?, retried=?, median_secs=? "
                            "WHERE rowid=?", vals + (rowid,))
        finally:
            con.close()

    @staticmethod
    def _err(where, e):
        print("%s rt-sched %s failed: %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), where, type(e).__name__),
              file=sys.stderr, flush=True)

    # ---- thread -------------------------------------------------------------------------------------------------
    def ready(self):
        """True once the realtime sync has completed a cycle in this process, or the store has data newer than 24 h."""
        try:
            if (self.sync_runs_fn() or 0) >= 1:
                return True
        except Exception:  # noqa: BLE001
            pass
        try:
            con = rt_store.open_store(self.db_path, create=False)
            try:
                newest = con.execute("SELECT max(published_at) FROM items").fetchone()[0]
            finally:
                con.close()
            return bool(newest) and (self.clock() - rt_store.parse_iso(newest)) < timedelta(hours=FRESH_HOURS)
        except Exception:  # noqa: BLE001
            return False

    def _set_next(self, seconds, waiting=None):
        with self._state_lock:
            self._next_run_at = _iso(self.clock() + timedelta(seconds=seconds))
            self._waiting = waiting

    def _loop(self):
        try:
            self.ensure_table()
        except Exception as e:  # noqa: BLE001
            self._err("init", e)
        self._set_next(self.first_delay_s, "first delay")
        if self._stop.wait(self.first_delay_s):
            return
        while not self._stop.is_set():
            if not self.ready():
                self._set_next(READY_RECHECK_S, "waiting for the first sync")
                if self._stop.wait(READY_RECHECK_S):
                    return
                continue
            with self._state_lock:
                self._next_run_at = None
            self.run_cycle(next_run_in=self.interval_s)
            self._set_next(self.interval_s, None)
            if self._stop.wait(self.interval_s):
                return

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._set_next(self.first_delay_s, "first delay")
            self._thread = threading.Thread(target=self._loop, name="fnp-rt-analyse", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout=5.0):
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and threading.current_thread() is not t:
            t.join(timeout)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def info(self):
        with self._state_lock:
            return dict(running=self.is_running(), running_now=self._running_now, next_run_at=self._next_run_at,
                        waiting=self._waiting, interval_s=self.interval_s, budget=self.budget, max_secs=self.max_secs,
                        first_delay_s=self.first_delay_s)


# ---- module level start/stop (like realtime_sync) and status ----------------------------------------------------

_scheduler = None
_lock = threading.Lock()


def start_background(db_path=None, enrich=None, jobs=None, env=None, **kw):
    """Start (once) the scheduler thread from the FNP_RT_ANALYSE_* environment; returns the Scheduler."""
    global _scheduler
    with _lock:
        if _scheduler is not None and _scheduler.is_running():
            return _scheduler
        cfg = config_from_env(env)
        cfg.update(kw)
        _scheduler = Scheduler(db_path=db_path, enrich=enrich, jobs=jobs, **cfg).start()
        return _scheduler


def stop_background(timeout=5.0):
    global _scheduler
    with _lock:
        sch, _scheduler = _scheduler, None
    if sch is not None:
        sch.stop(timeout)


def _rolling(rows):
    """Rolling success rate over the last cycles: replies that were a complete result / all engine replies
    (complete + degraded + failed; busy counts as failed). None when there were no replies."""
    ok = sum(r["analysed"] or 0 for r in rows)
    bad = sum((r["degraded"] or 0) + (r["errors"] or 0) for r in rows)
    return dict(cycles=len(rows), analysed=ok, degraded=sum(r["degraded"] or 0 for r in rows),
                errors=sum(r["errors"] or 0 for r in rows), replies=ok + bad,
                success_rate=(ok / (ok + bad)) if ok + bad else None)


def _row(r):
    d = dict(r)
    try:
        d["duration_s"] = int((rt_store.parse_iso(d["ended_at"]) - rt_store.parse_iso(d["started_at"])).total_seconds()) \
            if d.get("ended_at") else None
    except (TypeError, ValueError):
        d["duration_s"] = None
    return d


def schedule_status(db_path=None, env=None, log_rows=0):
    """{enabled, running_now, state, interval_s, budget, max_secs, first_delay_s, last_run, next_run_at,
    totals:{analysed_deep, cycles, scheduled_analysed}[, log:[last rows]]}. Never raises; reads only the local store."""
    env = os.environ if env is None else env
    sch = _scheduler
    cfg = config_from_env(env)
    info = sch.info() if sch is not None else {}
    running = bool(info.get("running"))
    enabled = running or enabled_from_env(env)
    out = dict(enabled=enabled, running_now=bool(info.get("running_now")),
               interval_s=info.get("interval_s", cfg["interval_s"]), budget=info.get("budget", cfg["budget"]),
               max_secs=info.get("max_secs", cfg["max_secs"]), first_delay_s=info.get("first_delay_s", cfg["first_delay_s"]),
               last_run=None, next_run_at=info.get("next_run_at") if enabled else None,
               totals=dict(analysed_deep=0, partial_items=0, cycles=0, scheduled_analysed=0), log=[],
               rolling=_rolling([]))
    out["state"] = ("off" if not enabled else "running" if out["running_now"] else
                    {"first delay": "waiting_first_delay", "waiting for the first sync": "waiting_for_sync"}.get(
                        info.get("waiting"), "idle"))
    path = db_path or rt_store.DEFAULT_DB
    try:
        if not os.path.exists(path):
            return out
        con = rt_store.open_store(path, create=False)
        try:
            con.execute("PRAGMA query_only=ON")
            has_log = con.execute("SELECT 1 FROM sqlite_master WHERE name = 'schedule_log'").fetchone()
            if has_log:
                have = {r[1] for r in con.execute("PRAGMA table_info(schedule_log)")}   # an old store may lack the S2 columns
                extra = ", ".join(c if c in have else "%s AS %s" % ("NULL" if c == "median_secs" else "0", c)
                                  for c in ("partial", "retried", "median_secs"))
                rows = [_row(r) for r in con.execute(
                    "SELECT started_at, ended_at, picked, analysed, cached_dupes, degraded, errors, skipped, stop_reason, "
                    "next_run_at, %s FROM schedule_log ORDER BY started_at DESC, rowid DESC LIMIT %d"
                    % (extra, max(1, int(log_rows or 1))))]
                if rows:
                    last = rows[0]
                    if last["ended_at"] is None and not out["running_now"]:
                        last = dict(last, stop_reason="interrupted")
                    out["last_run"] = last
                    if log_rows:
                        out["log"] = rows
                tot = con.execute("SELECT count(*), coalesce(sum(analysed), 0) FROM schedule_log "
                                  "WHERE ended_at IS NOT NULL").fetchone()
                out["totals"]["cycles"], out["totals"]["scheduled_analysed"] = tot[0], tot[1]
                out["rolling"] = _rolling([_row(r) for r in con.execute(
                    "SELECT analysed, degraded, errors FROM schedule_log WHERE ended_at IS NOT NULL AND "
                    "stop_reason IS NOT 'interrupted' ORDER BY started_at DESC, rowid DESC LIMIT %d" % LOG_KEEP)])
            out["totals"]["analysed_deep"] = _cached_deep_count(con, path, fresh=bool(log_rows))
            out["totals"]["partial_items"] = partial_count(con)
        finally:
            con.close()
    except Exception:  # noqa: BLE001 - status must never fail a request
        pass
    return out


_deep_cache = {}
DEEP_COUNT_TTL = 15.0


def _cached_deep_count(con, key, fresh=False):
    """deep_count with a short TTL (healthz polls it; it parses the stored results). `fresh` bypasses the cache."""
    now = time.monotonic()
    hit = _deep_cache.get(key)
    if hit and not fresh and now - hit[0] < DEEP_COUNT_TTL:
        return hit[1]
    n = deep_count(con)
    _deep_cache[key] = (now, n)
    return n


def deep_count(con):
    """Items with a usable deep result (an engine verdict) in `analyses`; any engine version, counted once."""
    return con.execute(
        "SELECT count(DISTINCT item_id) FROM analyses WHERE json_valid(result_json) AND "
        "json_extract(result_json, '$.engine.verdict.label') IS NOT NULL").fetchone()[0]


def partial_count(con):
    """Items whose only stored deep result is a partial one (propaganda signals, no verdict)."""
    return con.execute(
        "SELECT count(DISTINCT item_id) FROM analyses WHERE json_valid(result_json) AND "
        "json_extract(result_json, '$.partial') = 1 AND item_id NOT IN (SELECT item_id FROM analyses WHERE "
        "json_valid(result_json) AND json_extract(result_json, '$.engine.verdict.label') IS NOT NULL)").fetchone()[0]
