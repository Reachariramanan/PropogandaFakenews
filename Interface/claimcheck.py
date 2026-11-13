"""Claim Check back end: POST /api/analyse orchestration and GET /api/examples sampling (stdlib only, no fnp import).

Flow of `analyse` (see engine_adapter for the engine shapes):
  1a  index match   : the existing /api/check logic (`index_fn`, passed in by server.py), instant and local.
  1b  engine call   : FNP_ENGINE_URL (default http://127.0.0.1:8879/api/check_v1), one request at a time, at most
                      MAX_WAITING requests queued behind it. In-memory LRU cache (200 entries, 1 h) in front, a
                      per-IP limit of 10 engine calls per minute (HTTP 429 + Retry-After) before the queue.
  2   combine          : see `combine`. Propaganda is NEVER gated: both answers are always shown.

Failure behaviour (one consistent rule): the index result always exists, so the endpoint always answers 200 with it
when the engine cannot be used, and says why in `engine.state`:
    unavailable  engine not reachable            timeout  no answer within the timeout
    degraded     check_v1 returned degraded:true: fast result + propaganda techniques only. engine.degraded_why says
                 which: 'timeout' (its own 55 s limit; never cached), 'no_claims' (no checkable claim was extracted, a
                 deterministic outcome, cached) or 'unavailable'
    busy         queue full or waited too long   error    engine answered with an error / unreadable response
`busy` also carries a Retry-After header (seconds). Only a rate-limit hit is not a 200 (429), and only an unbuilt
index or a bad body is an error (503 / 400 / 413), exactly like /api/check. Messages are generic: no stack traces and
no claim text are ever returned or logged.

Response: {text_chars, engine_chars, truncated,
           index:  <api_check result>,
           engine: {state:'ok', engine:'predict_v0'|'check_v1', verdict:{label,probs,plausibly_fake,rationale,decider},
                    model_versions, secs, cached, queue_position, queue_wait_s}
                   | {state:'unavailable'|'timeout'|'busy'|'error', reason, queue_position?, queue_wait_s?},
           step1:  {source:'fact-check'|'model estimate'|'index only'|'none', lead:'index'|'engine'|None, label,
                    plausibility, note, second_opinion:{label,plausibility,agrees}|None,
                    classes:[{label, probability, secondary}]   FAKE, MISLEADING, LEGIT, INCONCLUSIVE, then the raw
                                                                PROPAGANDA (secondary:true); [] without an engine verdict
                    classes_note},
           step2:  {shown (true whenever the engine returned a propaganda block), gated_out (always false, kept for
                    old clients), forced (always false), available, gate_plausibility, reason,
                    propaganda:{is_propaganda_p, laya_is_propaganda_raw|None, techniques:[{technique,start,end,text,
                                confidence}], one_sided (bool|None), one_sided_score (0-1|None), text_words, n,
                                dropped, technique_probs|None,
                                signal:{level:'strong'|'weak'|'none', strong_n, weak_n, max_confidence, threshold,
                                        techniques[]}   span-only summary (Laya is not merged into it),
                                classes:[{technique, probability, basis:'span_max'|'engine'}]  all 14 merged buckets,
                                                                sorted by probability descending, explicit 0s,
                                basis, basis_note}|None,
                    laya:{available, state:'ok'|'not_run_no_claims'|'timeout'|'not_returned', label, confidence,
                          probs:[{label,probability,secondary}], propaganda_head|None, head_in_range, short_text,
                          text_words, compared:'agree'|'differs'|'not_compared', compared_note, independent:false}|None}}
           engine (check_v1) also carries degraded_why and n_claims.
Technique offsets index into the text the engine saw (the submitted text, stripped, first 8,000 characters).
"""
import hashlib
import math
import os
import random
import re
import sys
import threading
import time
from collections import OrderedDict, deque

import engine_adapter

ENGINE_URL = os.environ.get("FNP_ENGINE_URL", "http://127.0.0.1:8879/api/check_v1")
ENGINE_TIMEOUT = float(os.environ.get("FNP_ENGINE_TIMEOUT", "60"))
QUEUE_WAIT = float(os.environ.get("FNP_ENGINE_QUEUE_WAIT", "75"))  # longest a queued request waits for its turn
MAX_WAITING = 3
ENGINE_RATE = 10            # engine calls per client IP per minute
CACHE_MAX, CACHE_TTL = 200, 3600
MAX_ENGINE_CHARS = 8000
STRONG_LEAD = 0.5           # index similarity at which the fact-checker's verdict leads
FAKEISH = ("FAKE", "MISLEADING")


# ------------------------------------------------------------------ queue, cache, rate limit
class Gate:
    """One engine request at a time; up to `max_waiting` more wait in FIFO order."""

    def __init__(self, max_waiting=MAX_WAITING):
        self.cv = threading.Condition()
        self.running = False
        self.q = deque()
        self.max_waiting = max_waiting

    def enter(self, max_wait):
        """(queue_position, waited_s) once it is this caller's turn (position 0 = ran at once), or None when the
        queue is full or the wait exceeded `max_wait`."""
        t0 = time.monotonic()
        with self.cv:
            if not self.running and not self.q:
                self.running = True
                return 0, 0.0
            if len(self.q) >= self.max_waiting:
                return None
            me = object()
            self.q.append(me)
            pos = len(self.q)
            if not self.cv.wait_for(lambda: not self.running and self.q[0] is me, timeout=max_wait):
                self.q.remove(me)
                self.cv.notify_all()
                return None
            self.q.popleft()
            self.running = True
            return pos, time.monotonic() - t0

    def leave(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()

    def waiting(self):
        with self.cv:
            return len(self.q)


class Cache:
    """sha1(stripped text + lang + engine id) -> adapted engine result. LRU, TTL in seconds."""

    def __init__(self, max_entries=CACHE_MAX, ttl=CACHE_TTL):
        self.lock = threading.Lock()
        self.d = OrderedDict()
        self.max, self.ttl = max_entries, ttl

    @staticmethod
    def key(text, lang, engine_id):
        return hashlib.sha1("\x00".join((text.strip().replace("\r\n", "\n"), lang or "", engine_id)).encode()).hexdigest()

    def get(self, k):
        with self.lock:
            hit = self.d.get(k)
            if not hit:
                return None
            if time.monotonic() - hit[0] > self.ttl:
                del self.d[k]
                return None
            self.d.move_to_end(k)
            return hit[1]

    def discard(self, k):
        with self.lock:
            self.d.pop(k, None)

    def put(self, k, v):
        with self.lock:
            self.d[k] = (time.monotonic(), v)
            self.d.move_to_end(k)
            while len(self.d) > self.max:
                self.d.popitem(last=False)

    def __len__(self):
        return len(self.d)


class RateLimit:
    def __init__(self, per_min=ENGINE_RATE):
        self.lock = threading.Lock()
        self.hits = {}
        self.per_min = per_min

    def check(self, ip):
        """0 when allowed (and counted), else seconds until the oldest hit ages out."""
        now = time.monotonic()
        with self.lock:
            hits = [t for t in self.hits.get(ip, ()) if now - t < 60]
            if len(hits) >= self.per_min:
                self.hits[ip] = hits
                return int(60 - (now - hits[0])) + 1
            hits.append(now)
            self.hits[ip] = hits
            if len(self.hits) > 2000:
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] >= 60]:
                    del self.hits[k]
        return 0


GATE, CACHE, LIMIT = Gate(), Cache(), RateLimit()
_recent_secs = deque(maxlen=10)  # engine call durations, for the Retry-After estimate


def reset_state():
    """Fresh gate, cache and limiter (tests)."""
    global GATE, CACHE, LIMIT
    GATE, CACHE, LIMIT = Gate(), Cache(), RateLimit()
    _recent_secs.clear()


def _retry_after():
    avg = sum(_recent_secs) / len(_recent_secs) if _recent_secs else 10.0
    return max(5, min(60, int(math.ceil(avg * (GATE.waiting() + 1)))))


def _log(ip, secs, state, qpos):
    print("%s engine ip=%s secs=%.1f state=%s queue=%d" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), ip or "job", secs, state, qpos),
          file=sys.stderr, flush=True)


# ------------------------------------------------------------------ engine step
def run_engine(text, lang, ip):
    """-> (engine_block, headers, http_status_override). The block is {state:'ok', ...internal shape...} or a
    failure block. Cached results are answered without touching the gate or the rate limit."""
    url = ENGINE_URL
    key = Cache.key(text, lang, url)
    hit = CACHE.get(key)
    if hit is not None:
        return {**hit, "state": hit.get("state", "ok"), "cached": True, "queue_position": 0, "queue_wait_s": 0.0}, {}, None
    wait = LIMIT.check(ip) if ip is not None else 0  # ip None = a server-side job (already one at a time)
    if wait:
        return None, {"Retry-After": str(wait)}, 429
    entered = GATE.enter(QUEUE_WAIT)
    if entered is None:
        ra = _retry_after()
        _log(ip, 0.0, "busy", 0)
        return {"state": "busy", "reason": "the engine is busy with other checks; try again shortly",
                "retry_after_s": ra}, {"Retry-After": str(ra)}, None
    qpos, waited = entered
    t0 = time.monotonic()
    try:
        hit = CACHE.get(key)  # an identical request that was ahead of us may have filled it
        if hit is not None:
            return {**hit, "state": hit.get("state", "ok"), "cached": True, "queue_position": qpos,
                    "queue_wait_s": round(waited, 2)}, {}, None
        up_ra = None
        try:
            res = engine_adapter.normalise(engine_adapter.call_engine(url, text, lang, ENGINE_TIMEOUT), text)
            state = "ok"
        except engine_adapter.EngineError as e:
            res, state, reason, up_ra = None, e.state, e.reason, e.retry_after
        except engine_adapter.AdapterError as e:
            res, state, reason = None, "error", str(e)
        except Exception:  # noqa: BLE001 - never leak internals
            res, state, reason = None, "error", "the engine response could not be processed"
    finally:
        GATE.leave()
    dt = time.monotonic() - t0
    _log(ip, dt, state, qpos)
    if res is None:
        blk = {"state": state, "reason": reason, "queue_position": qpos, "queue_wait_s": round(waited, 2)}
        if state == "busy":  # the engine's own 503 + Retry-After
            ra = max(1, min(120, up_ra)) if up_ra else _retry_after()
            blk["retry_after_s"] = ra
            return blk, {"Retry-After": str(ra)}, None
        return blk, {}, None
    _recent_secs.append(dt)
    res["secs"] = res.get("secs") if res.get("secs") is not None else round(dt, 1)
    if res.get("degraded"):
        dwhy = res.get("degraded_why") or "unavailable"
        if (res.get("verdict") or {}).get("basis") == "none":
            res = {**res, "verdict": None}  # no factual call was made: nothing to present as a verdict
        if dwhy == "no_claims":
            reason = "no checkable factual claim was found, so there is no verdict; propaganda techniques are still shown"
        else:
            reason = ("the deep verdict timed out" if dwhy == "timeout" else "the deep verdict was unavailable") + \
                "; only the fast result and propaganda techniques are shown"
        blk = {**res, "state": "degraded", "reason": reason}
        if dwhy == "no_claims":  # deterministic (same text, same outcome), so cacheable; a timeout is not a result and is never cached
            CACHE.put(key, blk)
        return {**blk, "cached": False, "queue_position": qpos, "queue_wait_s": round(waited, 2)}, {}, None
    CACHE.put(key, res)
    return {**res, "state": "ok", "cached": False, "queue_position": qpos, "queue_wait_s": round(waited, 2)}, {}, None


# ------------------------------------------------------------------ combine + gate
def engine_plausibility(v):
    """P(FAKE) + P(MISLEADING) of an engine verdict; without probabilities 1/0 by label. A 'plausibly fake' flag
    lifts it to at least 0.5."""
    probs = v.get("probs") or {}
    if v.get("basis") == "none":  # check_v1 made no factual call: INCONCLUSIVE, no plausibility
        return 0.0
    fs = v.get("fake_score")
    if isinstance(fs, (int, float)) and 0 <= fs <= 1:
        p = float(fs)
    elif any(k in probs for k in FAKEISH):
        p = min(1.0, sum(probs.get(k, 0.0) for k in FAKEISH))
    else:
        p = 1.0 if v.get("label") in FAKEISH else 0.0
    return max(p, 0.5) if v.get("plausibly_fake") else p


def verdict_axis_label(v):
    """The fake-or-not label for step 1. The engine's top class can be PROPAGANDA, whose probability is a length
    artefact on short texts; propaganda belongs to step 2, so use the highest of FAKE/MISLEADING/LEGIT/INCONCLUSIVE."""
    lab = v.get("label")
    if lab != "PROPAGANDA":
        return lab
    probs = v.get("probs") or {}
    cands = [(float(probs.get(k) or 0.0), k) for k in ("FAKE", "MISLEADING", "LEGIT", "INCONCLUSIVE")]
    best = max(cands) if any(c[0] > 0 for c in cands) else (0.0, "INCONCLUSIVE")
    return best[1]


def combine(idx, eng, force=False):
    """-> (step1, step2). Strong index match (similarity >= 0.5): the fact-checker leads and the engine is a second
    opinion. Otherwise the engine leads ('model estimate'). Step 2 (propaganda) is never gated: it is shown whenever
    the engine returned a propaganda block, whatever the step 1 verdict. `force` is accepted for old callers and
    ignored (gated_out is always false, forced always false)."""
    st = eng.get("state") if isinstance(eng, dict) else None
    degraded = st == "degraded"
    dwhy = eng.get("degraded_why") if isinstance(eng, dict) and st in ("ok", "degraded") else None
    ok = (st == "ok") or (degraded and bool(eng.get("verdict")))  # degraded: only a fake_call verdict is usable
    matches = idx.get("matches") or []
    strong = bool(matches) and idx.get("score") is not None and (idx.get("nearest_similarity") or 0) >= STRONG_LEAD
    ev = eng["verdict"] if ok else None
    if strong:
        top = matches[0]
        step1 = {"source": "fact-check", "lead": "index", "label": top.get("verdict_norm"),
                 "plausibility": idx["score"],
                 "note": "Matches a fact-check by %s (similarity %.2f)." % (top.get("site") or "a fact-checker",
                                                                           idx.get("nearest_similarity") or 0),
                 "second_opinion": None}
        if ok:
            ep = engine_plausibility(ev)
            el = verdict_axis_label(ev)
            step1["second_opinion"] = {"label": el, "plausibility": round(ep, 3),
                                       "agrees": (el in FAKEISH) == (top.get("verdict_norm") in FAKEISH)}
    elif ok:
        step1 = {"source": "model estimate", "lead": "engine", "label": verdict_axis_label(ev),
                 "plausibility": round(engine_plausibility(ev), 3),
                 "note": ("no matching fact-check; no checkable factual claim was found, so the engine made no factual call"
                          if dwhy == "no_claims" else "no matching fact-check"), "second_opinion": None}
    else:
        sc = idx.get("score")
        step1 = {"source": "index only" if sc is not None else "none", "lead": None, "label": None,
                 "plausibility": sc,
                 "note": "The analysis engine is not available, so only the fact-check index match is shown."
                         if sc is not None else
                         "No matching fact-check and the analysis engine is not available.",
                 "second_opinion": None}
    if degraded and step1["lead"] != "index":
        if dwhy == "no_claims":
            step1["note"] = ("No checkable factual claim was found in this text, so there is no engine verdict" +
                             ("; only the fact-check index match is shown." if step1["plausibility"] is not None
                              else ". Step 2 below still reads the wording."))
        else:
            what = "timed out" if dwhy == "timeout" else "was unavailable"
            step1["note"] = ("The deep verdict %s, so this is the fast result only." % what if ok else
                             "The deep verdict %s, so only the fact-check index match is shown." % what)
    step1["classes"] = engine_adapter.verdict_classes(ev.get("probs"), eng.get("raw_propaganda_p")) if ok else []
    step1["classes_note"] = engine_adapter.VERDICT_NOTE
    prop = eng.get("propaganda") if st in ("ok", "degraded") else None
    laya = None
    if prop is not None:
        classes, basis, note = engine_adapter.propaganda_classes(prop)
        laya = engine_adapter.laya_view(eng.get("laya"), prop, eng.get("n_claims"), eng.get("degraded_why"), eng.get("laya_reason"))
        prop = {**prop, "classes": classes, "basis": basis, "basis_note": note,
                "signal": engine_adapter.propaganda_signal(prop)}
    step2 = {"shown": prop is not None, "gated_out": False, "forced": False, "available": prop is not None,
             "gate_plausibility": step1["plausibility"],
             "reason": (("shown because the deep verdict " + ("timed out" if dwhy == "timeout" else "was not available"))
                        if degraded and prop is not None else
                        "propaganda describes how the text is worded, whether or not it is true"
                        if prop is not None else "no engine result"),
             "propaganda": prop, "laya": laya}
    return step1, step2


def with_blocks(body):
    """A stored Claim Check body (e.g. a Workbench result cached before the always-both flow) -> a copy that has the
    current step1.classes and step2.propaganda.classes blocks and no gating. Bodies that already carry them, and
    anything that is not a Claim Check body, are returned unchanged."""
    if not isinstance(body, dict) or not isinstance(body.get("step1"), dict) or not isinstance(body.get("step2"), dict):
        return body
    s1, s2 = body["step1"], body["step2"]
    prop = s2.get("propaganda")
    if ("classes" in s1 and not s2.get("gated_out") and
            (not isinstance(prop, dict) or ("classes" in prop and "signal" in prop and "laya" in s2))):
        return body
    out = dict(body)
    if "classes" not in s1:
        eng = body.get("engine") if isinstance(body.get("engine"), dict) else {}
        v = eng.get("verdict") if isinstance(eng.get("verdict"), dict) else None
        out["step1"] = dict(s1, classes=engine_adapter.verdict_classes(v.get("probs"), eng.get("raw_propaganda_p"))
                            if v else [], classes_note=engine_adapter.VERDICT_NOTE)
    if isinstance(prop, dict):
        if "classes" not in prop:
            classes, basis, note = engine_adapter.propaganda_classes(prop)
            prop = {**prop, "classes": classes, "basis": basis, "basis_note": note}
        eng = body.get("engine") if isinstance(body.get("engine"), dict) else {}
        if "signal" not in prop:
            prop = {**prop, "signal": engine_adapter.propaganda_signal(prop)}
        lay = s2.get("laya") if "laya" in s2 else engine_adapter.laya_view(
            eng.get("laya") if isinstance(eng.get("laya"), dict) else None, prop, eng.get("n_claims"), eng.get("degraded_why"),
            eng.get("laya_reason"))
        out["step2"] = dict(s2, propaganda=prop, laya=lay, shown=True, gated_out=False, forced=False,
                            reason="propaganda describes how the text is worded, whether or not it is true")
    return out


def analyse(text, lang, force, ip, index_fn):
    """-> (body, http_status, headers). `text` is already stripped and length-checked by the caller."""
    engine_text = text[:MAX_ENGINE_CHARS].strip()
    idx = index_fn(text[:5000])
    eng, headers, override = run_engine(engine_text, lang, ip)
    if override == 429:
        return {"error": "rate limit: %d engine checks per minute" % ENGINE_RATE}, 429, headers
    eng_out = eng if eng.get("state") not in ("ok", "degraded") else {k: v for k, v in eng.items() if k != "propaganda"}
    step1, step2 = combine(idx, eng, force)
    return ({"text_chars": len(text), "engine_chars": len(engine_text), "truncated": len(text) > MAX_ENGINE_CHARS,
             "index": idx, "engine": eng_out, "step1": step1, "step2": step2}, 200, headers)


# ------------------------------------------------------------------ /api/examples
EXAMPLE_VERDICTS = ("FAKE", "MISLEADING", "LEGIT")
_RNG = random.SystemRandom()


def _example_ok(it):
    if it.get("kind") != "factcheck" or it.get("verdict_norm") not in EXAMPLE_VERDICTS:
        return False
    title, claim = (it.get("title") or "").strip(), (it.get("claim") or "").strip()
    return (25 <= len(title) and not title.isupper() and 25 <= len(claim) <= 500 and not claim.isupper()
            and bool(it.get("site")))


def build_pool(items):
    """{verdict: {site: [item, ...]}} of the items usable as example claims."""
    pool = {v: {} for v in EXAMPLE_VERDICTS}
    for it in items:
        if _example_ok(it):
            pool[it["verdict_norm"]].setdefault(it["site"], []).append(it)
    return {v: s for v, s in pool.items() if s}


def _lang_family(it):
    return str(it.get("lang") or "").split("-")[0].lower()


def sample(pool, n, rng=_RNG):
    """n items balanced across verdicts (random order, round robin), each from a different site while possible,
    preferring languages not yet picked."""
    verdicts = list(pool)
    rng.shuffle(verdicts)
    if not verdicts:
        return []
    seq = [verdicts[i % len(verdicts)] for i in range(n)]
    out, used_sites, used_langs, ids = [], set(), set(), set()
    for v in seq:
        sites = pool[v]
        fresh = [s for s in sites if s not in used_sites] or list(sites)
        site = rng.choice(fresh)
        cands = [it for it in (rng.choice(sites[site]) for _ in range(6)) if it["id"] not in ids]
        if not cands:
            continue
        pick = next((it for it in cands if _lang_family(it) not in used_langs), cands[0])
        out.append(pick)
        used_sites.add(site)
        used_langs.add(_lang_family(pick))
        ids.add(pick["id"])
    rng.shuffle(out)
    return out


class ExamplePool:
    """The filtered pool is built once per index reload (keyed on the loaded items list), not per request."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sig, self.pool = None, {}

    def get(self, items, mtime):
        sig = (id(items), len(items), mtime)
        with self.lock:
            if self.sig != sig:
                self.pool, self.sig = build_pool(items), sig
            return self.pool


EXAMPLES = ExamplePool()


def example_item(it):
    return {"id": it["id"], "title": it["title"].strip(), "claim": it["claim"].strip(),
            "verdict_norm": it["verdict_norm"], "site": it["site"], "lang": it.get("lang")}


def api_examples(items, mtime, q):
    try:
        n = int((q.get("n") or ["4"])[0])
    except ValueError:
        n = 4
    n = min(max(n, 1), 8)
    return {"items": [example_item(it) for it in sample(EXAMPLES.get(items, mtime), n)]}
