"""Triage tab API (/api/triage/*): a ranked "what to read first" queue from two sources. Read-only; server.py owns routing,
the gate and the rate limit (same gate and per-IP limit as /api/rt/*).

  source=fnp       the FNP triage export (reports/triage/v1/{meta.json,items.jsonl}, contract docs/TRIAGE_CONTRACT.md),
                   streamed into memory (about 7.6k rows), reloaded when meta.built_at / the files change, an empty state
                   when the directory is missing. Tiers are FNP's; this module only filters, sorts and explains them.
  source=realtime  a queue computed IN SQL over the local realtime store joined to the FNP batch sidecar (rt_enrich) and the
                   deep Claim Check results (`deep` view): never 630k rows in Python memory.

Realtime tier rules (a ranking aid, NOT validated; thresholds are assumptions from the S3 plan, listed below):
  HIGH    nearest fact-check has a rerank score (FNP only reranks matches with cosine >= 0.72), or at least two strong
          propaganda spans and p_prop >= 0.8.  If a deep verdict exists and its fake-or-misleading probability is below
          0.3, the item is demoted to MEDIUM (the deep check does not support it).
  MEDIUM  p_prop >= 0.4, or nearest fact-check cosine >= 0.6, or a deep fake-or-misleading probability >= 0.5.
  LOW     analysed (batch or deep) but below every rule above.
  not in the queue: items FNP has not scored ("unscored"), and items too short for the encoder ("too short"); both are
          counted separately, never ranked. Rows that only came from the older triage export (src='triage') are left out
          here: they are shown by the FNP source, so no item is counted twice.
Order inside a tier: freshness band first (under 6 h, 12 h, 1, 2, 3, 7, 14, 30, 60, 90 days, then older: fresher first), then deep fake-or-misleading probability, rerank, cosine, p_prop (signal strength), then engagement (reach),
then recency. The same rules exist once in rt_tier() (used for explanations) and once in TIER_SQL (the queue); a test keeps
the two identical on a grid of boundary values.

Every request value is whitelisted and bound as a SQL parameter; the only text interpolated into SQL is code constants.
"""
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rt_api  # noqa: E402
import rt_enrich  # noqa: E402
import rt_store  # noqa: E402
from rt_api import ApiError, _enum, _g, _int  # noqa: E402

TIERS = ("high", "medium", "low")
SOURCES = ("fnp", "realtime")
SORTS = ("priority", "recent")
PAGE_SIZE_MAX = rt_api.PAGE_SIZE_MAX
OFFSET_MAX = rt_api.OFFSET_MAX
SNIPPET = 260
MAX_SPANS = 12
STATS_TTL = 30.0
FACETS_TTL = 300.0
QUERY_SECONDS = 12.0
STATS_SECONDS = 40.0
TIER_RANK = {"high": 3, "medium": 2, "low": 1}

# ---- realtime tier thresholds (assumed, S3 plan; a ranking aid, not calibrated; keep in docs/review when FNP sees them) ----
# tightened from (1 strong span, 0.6) after it gave HIGH to 329 of 1,035 ranked items (32%); on the 1,509 FNP-scored items the new
# pair keeps 142 of the 502 with p_prop >= 0.6. ASSUMED, not validated (p_prop is an uncalibrated max strong-span probability).
N_STRONG_HIGH = 2      # strong propaganda spans needed (with P_PROP_HIGH) to make an item HIGH
P_PROP_HIGH = 0.8      # strong-span propaganda signal that makes an item HIGH (with N_STRONG_HIGH strong spans)
P_PROP_MEDIUM = 0.4    # propaganda signal that makes an item MEDIUM
COS_MEDIUM = 0.6       # nearest fact-check cosine similarity that makes an item MEDIUM
DEEP_MEDIUM = 0.5      # deep P(fake)+P(misleading) that makes an item MEDIUM
DEEP_DEMOTE = 0.3      # deep P(fake)+P(misleading) below this lowers a HIGH item to MEDIUM
# ---- FNP export tier thresholds (from docs/TRIAGE_CONTRACT.md; used only to explain the export's own tiers) ----
FNP_DEBUNK_RERANK, FNP_DEBUNK_COS, FNP_MED_FAKE, FNP_MED_RERANK, FNP_LOW_COS = 0.8, 0.72, 0.5, 0.5, 0.65
LIKELY_FAKE = ("FAKE", "MISLEADING")

CAVEATS = [
    "Tiers are a ranking aid to decide what to read first. They are not verdicts and no tier has been validated against hand-labelled social posts.",
    "No item has been validated by a person yet: every item here is unvalidated.",
    "A close fact-check match is often the same topic, not the same claim. Open the fact-check and compare before acting.",
    "The propaganda signal (p_prop) comes from an encoder trained on long news articles and is not validated on social media or short posts.",
]

RT_RULES = {
    "high": "The nearest known fact-check was close enough to be reranked (similarity of at least 0.72), or the item has at least 2 strong "
            "propaganda spans and a propaganda signal of 0.80 or more. If a deep verdict exists and says the fake-or-misleading probability "
            "is below 0.30, the item is lowered to medium.",
    "medium": "The propaganda signal is 0.40 or more, or the nearest fact-check similarity is 0.60 or more, or a deep verdict puts the "
              "fake-or-misleading probability at 0.50 or more.",
    "low": "Analysed by FNP (batch pass or deep verdict) but below every rule above.",
    "excluded": "Items FNP has not scored yet, and items too short for the encoder to rate, are not ranked. They are counted separately.",
    "order": "Inside a tier: deep fake-or-misleading probability, then rerank, similarity and propaganda signal (signal strength), then "
             "engagement (reach), then newest first.",
}

# the same rules as rt_tier(), as a SQL CASE over the aliases e (enr), d (deep). 3 = high, 2 = medium, 1 = low.
TIER_SQL = ("(CASE WHEN e.fc_rerank IS NOT NULL OR (coalesce(e.n_strong, 0) >= %d AND e.p_prop >= %r) "
            "THEN (CASE WHEN d.id IS NOT NULL AND d.d_pfm < %r THEN 2 ELSE 3 END) "
            "WHEN e.p_prop >= %r OR e.fc_cos >= %r OR d.d_pfm >= %r THEN 2 ELSE 1 END)"
            % (N_STRONG_HIGH, P_PROP_HIGH, DEEP_DEMOTE, P_PROP_MEDIUM, COS_MEDIUM, DEEP_MEDIUM))
TIER_NAME = {3: "high", 2: "medium", 1: "low"}
# candidates: every non-triage batch row and every deep verdict; scored = a deep verdict, or a batch row that is not too short
RT_FROM = ("(SELECT id FROM enr.enr WHERE src <> 'triage' UNION SELECT id FROM deep) c "
           "JOIN items i ON i.id = c.id "
           "LEFT JOIN enr.enr e ON e.id = c.id AND e.src <> 'triage' "
           "LEFT JOIN deep d ON d.id = c.id")
RT_SCORED = "(d.id IS NOT NULL OR (e.id IS NOT NULL AND coalesce(e.too_short, 0) = 0))"
# Freshness bands (days): the priority order puts fresher items first WITHIN a tier, then stronger signals. Without this, old posts with
# strong scores outrank today's items (the FNP export holds posts from months ago). The band edges are assumed, not tuned.
AGE_DAYS = (0.25, 0.5, 1, 2, 3, 7, 14, 30, 60, 90)
_SIGNAL_ORDER = ("coalesce(d.d_pfm, -1) DESC, coalesce(e.fc_rerank, -1) DESC, coalesce(e.fc_cos, -1) DESC, "
                 "coalesce(e.p_prop, -1) DESC, coalesce(i.engagement, 0) DESC, i.published_at DESC, i.id")


def _epoch(x):
    """The injectable clock returns a datetime (tests) or epoch seconds: normalise to epoch seconds."""
    return x.timestamp() if hasattr(x, "timestamp") else float(x)


def age_band(ts, now):
    """Freshness band of an epoch timestamp: 0 = under 1 day old ... len(AGE_DAYS) = older than the last edge; unknown time sorts last."""
    if ts is None:
        return len(AGE_DAYS) + 1
    age = (now - ts) / 86400.0
    for i, d in enumerate(AGE_DAYS):
        if age < d:
            return i
    return len(AGE_DAYS)


def order_priority(now=None):
    """SQL ORDER BY for the realtime queue: tier, freshness band (fresher first), then signal strength. Cutoffs are generated here
    (never from user input) as ISO strings that compare correctly with items.published_at."""
    now = time.time() if now is None else now
    cuts = [datetime.fromtimestamp(now - d * 86400, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") for d in AGE_DAYS]
    band = "CASE " + " ".join("WHEN i.published_at >= '%s' THEN %d" % (c, i) for i, c in enumerate(cuts)) + " ELSE %d END" % len(AGE_DAYS)
    return "t DESC, %s ASC, %s" % (band, _SIGNAL_ORDER)


ORDER_RECENT = "i.published_at DESC, i.id"

_RE_PLATFORM = re.compile(r"^[a-z0-9_]{1,40}$")
_RE_LANG = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?$|^und$")
_RE_ISO3 = re.compile(r"^[A-Za-z]{3}$")
_RE_ID = re.compile(r"^[A-Za-z0-9_.:\-]{1,200}$")
_RE_SHA1 = re.compile(r"^[0-9a-f]{40}$")


def _bad(c):
    raise ValueError("non-strict JSON constant %s" % c)


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _n2(x):
    return "n/a" if x is None else "%.2f" % x


def _parse_ts(s):
    if not isinstance(s, str) or not s:
        return None
    try:
        d = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def _oneline(s, n=SNIPPET):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


# ====================================================================================================== tier logic
def rt_tier(p_prop, n_strong, rerank, cosine, deep_pfm, has_deep):
    """Realtime tier rules -> (tier, rule id). Mirrors TIER_SQL exactly (see the module docstring)."""
    strong = (n_strong or 0) >= N_STRONG_HIGH and p_prop is not None and p_prop >= P_PROP_HIGH
    if rerank is not None or strong:
        if has_deep and deep_pfm is not None and deep_pfm < DEEP_DEMOTE:
            return "medium", "deep_demoted"
        return "high", "rerank_match" if rerank is not None else "strong_propaganda"
    if p_prop is not None and p_prop >= P_PROP_MEDIUM:
        return "medium", "p_prop"
    if cosine is not None and cosine >= COS_MEDIUM:
        return "medium", "cosine"
    if has_deep and deep_pfm is not None and deep_pfm >= DEEP_MEDIUM:
        return "medium", "deep"
    return "low", "below"


def explain_rt(tier, rule, s):
    """Plain-words reasons with the numbers for a realtime tier. `s` holds p_prop, n_strong, rerank, cosine, deep_pfm, has_deep."""
    pp, ns, rr, cs, dp = s.get("p_prop"), s.get("n_strong"), s.get("rerank"), s.get("cosine"), s.get("deep_pfm")
    out = []
    if rule == "rerank_match":
        out.append("FNP's nearest known fact-check is close enough to be reranked (similarity %s, rerank %s; FNP only reranks matches "
                   "with similarity of at least 0.72). A close match is often the same topic, not the same claim." % (_n2(cs), _n2(rr)))
    elif rule == "strong_propaganda":
        out.append("The text has %s strong propaganda span%s and a propaganda signal of %s: HIGH needs at least %d strong spans and a signal of %.2f or more."
                   % (ns, "" if ns == 1 else "s", _n2(pp), N_STRONG_HIGH, P_PROP_HIGH))
    elif rule == "deep_demoted":
        why = ("a rerank score of %s" % _n2(rr)) if rr is not None else ("%s strong propaganda span%s and a propaganda signal of %s"
                                                                      % (ns, "" if ns == 1 else "s", _n2(pp)))
        out.append("It met the HIGH rule (%s), but its deep verdict puts the fake-or-misleading probability at %s, below %.2f, so it was "
                   "lowered to MEDIUM." % (why, _n2(dp), DEEP_DEMOTE))
    elif rule == "p_prop":
        out.append("The propaganda signal is %s, at or above the %.2f line for MEDIUM, but there is no fact-check match or strong span "
                   "combination that would make it HIGH." % (_n2(pp), P_PROP_MEDIUM))
    elif rule == "cosine":
        out.append("The nearest known fact-check has similarity %s, at or above %.2f. It is not close enough to be reranked."
                   % (_n2(cs), COS_MEDIUM))
    elif rule == "deep":
        out.append("A deep verdict puts the fake-or-misleading probability at %s, at or above %.2f." % (_n2(dp), DEEP_MEDIUM))
    else:
        out.append("FNP analysed it, but no signal reaches a MEDIUM line: propaganda signal %s (needs %.2f), nearest fact-check "
                   "similarity %s (needs %.2f)%s." % (_n2(pp), P_PROP_MEDIUM, _n2(cs), COS_MEDIUM,
                                                       (", deep probability %s (needs %.2f)" % (_n2(dp), DEEP_MEDIUM)) if s.get("has_deep") else ""))
    if rule not in ("deep_demoted", "deep") and s.get("has_deep") and dp is not None and dp >= DEEP_MEDIUM:
        out.append("A deep verdict agrees: fake-or-misleading probability %s." % _n2(dp))
    elif rule in ("rerank_match", "strong_propaganda", "p_prop", "cosine") and s.get("has_deep") and dp is not None:
        out.append("A deep verdict exists: fake-or-misleading probability %s." % _n2(dp))
    return out


def fnp_why(it, rules):
    """Which of the export's own tier conditions hold for an FNP triage item, in words with numbers; falls back to the export's
    stated rule when none of the documented conditions can be recomputed."""
    tier = it["tier"]
    v = it.get("verdict") if isinstance(it.get("verdict"), dict) else None
    label = v.get("label") if v else None
    nf = it.get("nearest_factcheck") if isinstance(it.get("nearest_factcheck"), dict) else {}
    cos, rr, nl = _num(nf.get("cos", nf.get("cosine"))), _num(nf.get("rerank")), nf.get("label")
    fs = _num(it.get("fake_score"))
    fm = nl in LIKELY_FAKE
    out = []
    if tier == "high":
        if label in LIKELY_FAKE:
            out.append("FNP's own verdict on this post is %s (fake score %s%s)." % (label.title(), _n2(fs),
                       (", basis: %s" % v["basis"]) if isinstance(v.get("basis"), str) else ""))
        if fm and rr is not None and cos is not None and rr >= FNP_DEBUNK_RERANK and cos >= FNP_DEBUNK_COS:
            out.append("A debunk match: the nearest fact-check is rated %s, with similarity %s (at least %.2f) and rerank %s (at least %.2f)."
                       % (nl.title(), _n2(cos), FNP_DEBUNK_COS, _n2(rr), FNP_DEBUNK_RERANK))
    elif tier == "medium":
        if fs is not None and fs >= FNP_MED_FAKE:
            out.append("FNP's verdict gives a fake score of %s, at or above %.2f, but the verdict itself is not FAKE or MISLEADING%s."
                       % (_n2(fs), FNP_MED_FAKE, (" (it is %s)" % label.title()) if label else ""))
        if fm and rr is not None and rr >= FNP_MED_RERANK:
            out.append("The nearest fact-check is rated %s with rerank %s (at least %.2f), below the debunk-match line for HIGH."
                       % (nl.title(), _n2(rr), FNP_MED_RERANK))
    else:
        if cos is not None:
            out.append("It is on the list because its nearest fact-check has similarity %s (at least %.2f)%s." % (
                _n2(cos), FNP_LOW_COS, (", rated %s" % nl.title()) if isinstance(nl, str) and nl else ""))
        if label:
            out.append("FNP has a verdict for it (%s, fake score %s) that does not reach the MEDIUM or HIGH rule." % (label.title(), _n2(fs)))
    if not out:
        out.append("The export placed it in the %s tier. Its stated rule: %s" % (tier.upper(), (rules or {}).get(tier) or "not stated"))
    return out


# ============================================================================================================ FNP source
class FnpExport:
    """The FNP triage export, in memory. refresh() is a cheap stat check per request; a changed meta.json/items.jsonl reloads it
    (the new data is built aside and swapped in, so a request never sees a half-loaded list). Missing directory: empty state."""

    def __init__(self, directory=None, country_names=None, desk_url=None):
        self.dir = directory or rt_enrich.TRIAGE_DIR
        self.country_names = country_names
        self.desk_url = desk_url
        self._lock = threading.Lock()
        self._sig = None
        self.state = self._empty("not loaded yet")

    @staticmethod
    def _empty(reason, error=None):
        return dict(available=False, reason=reason, error=error, meta={}, items=[], by_id={}, tiers={}, built_at=None, invalid=0,
                    counts_match=None, facets={}, anchor=None, loaded_at=None, load_seconds=None, n_with_verdict=0, n_validated=0)

    def _sigs(self):
        out = []
        for name in ("meta.json", "items.jsonl"):
            try:
                st = os.stat(os.path.join(self.dir, name))
                out.append((st.st_mtime_ns, st.st_size, st.st_ino))
            except OSError:
                out.append(None)
        return tuple(out)

    def refresh(self):
        sig = self._sigs()
        if sig == self._sig:
            return self.state
        with self._lock:
            if sig == self._sig:
                return self.state
            if sig[0] is None or sig[1] is None:
                # a rebuild swaps directories: keep serving what was loaded; only an empty start shows the empty state
                if not self.state["available"]:
                    self.state = self._empty("no triage export found" if not os.path.isdir(self.dir)
                                             else "the export directory has no meta.json or items.jsonl yet")
                else:
                    self.state = dict(self.state, error="export files are missing right now; showing the last loaded copy")
                self._sig = sig
                return self.state
            try:
                self.state = self._load()
            except (OSError, ValueError) as e:
                keep = self.state if self.state["available"] else self._empty("the export could not be read")
                self.state = dict(keep, error="could not read the export (%s); %s" % (
                    type(e).__name__, "showing the last loaded copy" if keep["available"] else "nothing loaded"))
            self._sig = sig
            return self.state

    def _load(self):
        t0 = time.monotonic()
        with open(os.path.join(self.dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f, parse_constant=_bad)
        if not isinstance(meta, dict):
            raise ValueError("meta.json is not an object")
        items, by_id, invalid = [], {}, 0
        with open(os.path.join(self.dir, "items.jsonl"), "rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    it = self._check(json.loads(line.decode("utf-8"), parse_constant=_bad))
                except (ValueError, UnicodeDecodeError, RecursionError):
                    invalid += 1
                    continue
                if it["post_uid"] in by_id:
                    invalid += 1
                    continue
                by_id[it["post_uid"]] = it
                items.append(it)
        items.sort(key=lambda r: (-TIER_RANK[r["tier"]], -(r["_fake"] if r["_fake"] is not None else -1.0),
                                  -(r["_rerank"] if r["_rerank"] is not None else -1.0), -(r["_reach"] or 0)))
        tiers = {t: 0 for t in TIERS}
        for r in items:
            tiers[r["tier"]] += 1
        mt = meta.get("tiers") if isinstance(meta.get("tiers"), dict) else {}
        match = (all(int(_num(mt.get(t)) or 0) == tiers[t] for t in TIERS) and
                 (not isinstance(meta.get("n_items"), int) or meta["n_items"] == len(items)))
        stamps = [r["_ts"] for r in items if r["_ts"] is not None]
        return dict(available=True, reason=None, error=None, meta=meta, items=items, by_id=by_id, tiers=tiers,
                    built_at=meta.get("built_at") if isinstance(meta.get("built_at"), str) else None, invalid=invalid,
                    counts_match=bool(match), facets=self._facets(items), anchor=max(stamps) if stamps else None,
                    loaded_at=rt_store.utc_iso(datetime.now(timezone.utc)), load_seconds=round(time.monotonic() - t0, 3),
                    n_with_verdict=sum(1 for r in items if r.get("verdict")), n_validated=sum(1 for r in items if r.get("validated") is True))

    @staticmethod
    def _check(d):
        """One export line -> the item with private `_` fields for filtering, or ValueError."""
        if not isinstance(d, dict):
            raise ValueError("not an object")
        uid = d.get("post_uid")
        if not isinstance(uid, str) or not _RE_ID.match(uid):
            raise ValueError("bad post_uid")
        if d.get("tier") not in TIERS:
            raise ValueError("bad tier")
        text = d.get("text")
        if text is not None and not isinstance(text, str):
            raise ValueError("bad text")
        d = dict(d)
        d["text"] = text or ""
        for k in ("p_fake", "p_misleading", "fake_score", "p_prop"):
            d[k] = _num(d.get(k))
        d["reach"] = int(d["reach"]) if _num(d.get("reach")) is not None and d["reach"] >= 0 else None
        d["techniques"] = [t[:80] for t in d.get("techniques") or [] if isinstance(t, str)][:30] if isinstance(d.get("techniques"), list) else []
        d["countries"] = [c.upper() for c in d.get("countries") or [] if isinstance(c, str) and _RE_ISO3.match(c)][:30] \
            if isinstance(d.get("countries"), list) else []
        if not isinstance(d.get("verdict"), dict):
            d["verdict"] = None
        nf = d.get("nearest_factcheck")
        d["nearest_factcheck"] = nf if isinstance(nf, dict) else None
        d["validated"] = d.get("validated") is True
        d["_hay"] = (d["text"] + "\n" + (d.get("author") or "") + "\n" + (d.get("platform") or "")).lower()
        d["_ts"] = _parse_ts(d.get("posted_at"))
        d["_fake"] = d["fake_score"]
        d["_rerank"] = _num(nf.get("rerank")) if nf else None
        d["_reach"] = d["reach"]
        d["_countries"] = frozenset(d["countries"])
        d["_techs"] = frozenset(d["techniques"])
        return d

    @staticmethod
    def _facets(items):
        def top(counter, n):
            return [dict(value=k, count=v) for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]]
        plat, lang, ctry, tech = {}, {}, {}, {}
        for r in items:
            for c, k in ((plat, r.get("platform")), (lang, r.get("lang"))):
                if isinstance(k, str) and k:
                    c[k] = c.get(k, 0) + 1
            for k in r["_countries"]:
                ctry[k] = ctry.get(k, 0) + 1
            for k in r["_techs"]:
                tech[k] = tech.get(k, 0) + 1
        return dict(platforms=top(plat, 40), langs=top(lang, 40), countries=top(ctry, 60), techniques=top(tech, 20))

    # ---- views --------------------------------------------------------------------------------------------------
    def row(self, r):
        nf = r.get("nearest_factcheck")
        v = r.get("verdict")
        counts = {}
        for s in r.get("spans") or []:
            if isinstance(s, list) and len(s) >= 3 and isinstance(s[2], str):
                counts[s[2]] = counts.get(s[2], 0) + 1
        top = sorted(r["techniques"], key=lambda t: (-counts.get(t, 0), t))[:3]
        return dict(
            id=r["post_uid"], source="fnp", kind=r.get("source") if r.get("source") in ("social", "news") else None,
            platform=r.get("platform"), author=r.get("author"), snippet=_oneline(r["text"]), lang=r.get("lang"),
            posted_at=r.get("posted_at"), reach=r["reach"], tier=r["tier"], validated=r["validated"], p_prop=r["p_prop"],
            fake_score=r["fake_score"], p_fake=r["p_fake"], p_misleading=r["p_misleading"],
            verdict=v.get("label") if v and isinstance(v.get("label"), str) else None,
            nearest_factcheck=dict(label=nf.get("label"), cosine=_num(nf.get("cos", nf.get("cosine"))), rerank=_num(nf.get("rerank")),
                                   debunk_match=nf.get("debunk_match") is True) if nf else None,
            techniques=top, countries=r["countries"][:5])

    def list(self, flt, sort, page, size, now=None):
        st = self.refresh()
        if not st["available"]:
            return dict(available=False, reason=st["reason"], total=0, tiers={t: 0 for t in TIERS}, page=page, size=size, pages=0, items=[])
        t, platform, lang, country, tech, days, q = (flt.get(k) for k in ("tier", "platform", "lang", "country", "technique", "days", "q"))
        cutoff = (st["anchor"] - days * 86400) if days and st["anchor"] is not None else None
        words = [w for w in (q or "").lower().split()][:8]

        def ok(r, tier_too):
            if tier_too and t and r["tier"] != t:
                return False
            if platform and r.get("platform") != platform:
                return False
            if lang and (r.get("lang") or "").lower() != lang.lower():
                return False
            if country and country not in r["_countries"]:
                return False
            if tech and tech not in r["_techs"]:
                return False
            if cutoff is not None and (r["_ts"] is None or r["_ts"] < cutoff):
                return False
            return all(w in r["_hay"] for w in words)
        base = [r for r in st["items"] if ok(r, False)]
        tiers = {x: 0 for x in TIERS}
        for r in base:
            tiers[r["tier"]] += 1
        sel = [r for r in base if not t or r["tier"] == t]
        if sort == "recent":
            sel = sorted(sel, key=lambda r: (r["_ts"] is None, -(r["_ts"] or 0)))
        else:  # priority: tier, then freshness band (fresher first), then signal strength, reach, newest
            # the FNP export is a snapshot: measure freshness from its newest post, not from today
            now = st["anchor"] if st["anchor"] is not None else (time.time() if now is None else now)
            sel = sorted(sel, key=lambda r: (-TIER_RANK[r["tier"]], age_band(r["_ts"], now),
                                             -(r["_fake"] if r["_fake"] is not None else -1.0),
                                             -(r["_rerank"] if r["_rerank"] is not None else -1.0), -(r["_reach"] or 0),
                                             -(r["_ts"] or 0)))
        total = len(sel)
        off = (page - 1) * size
        return dict(available=True, total=total, tiers=tiers, page=page, size=size, pages=max(1, -(-total // size)) if total else 0,
                    items=[self.row(r) for r in sel[off:off + size]], built_at=st["built_at"],
                    window_anchor=rt_store.utc_iso(datetime.fromtimestamp(st["anchor"], timezone.utc)) if st["anchor"] is not None else None)

    def detail(self, uid):
        st = self.refresh()
        r = st["by_id"].get(uid) if st["available"] else None
        if r is None:
            raise ApiError(404, "not found")
        text = r["text"]
        spans = []
        for s in r.get("spans") or []:
            if (isinstance(s, list) and len(s) >= 3 and isinstance(s[0], int) and isinstance(s[1], int) and not isinstance(s[0], bool)
                    and not isinstance(s[1], bool) and 0 <= s[0] < s[1] <= len(text) and isinstance(s[2], str)):
                spans.append(dict(start=s[0], end=s[1], technique=s[2][:80], confidence=_num(s[3]) if len(s) > 3 else None,
                                  text=text[s[0]:s[1]], field="text"))
        dropped = len(r.get("spans") or []) - len(spans)
        shown = sorted(sorted(spans, key=lambda x: -(x["confidence"] or 0))[:MAX_SPANS], key=lambda x: x["start"])
        agg = {}
        for s in spans:
            n, m = agg.get(s["technique"], (0, None))
            agg[s["technique"]] = (n + 1, s["confidence"] if m is None or (s["confidence"] is not None and s["confidence"] > m) else m)
        techs = [dict(technique=t, count=agg.get(t, (1, None))[0], max_confidence=agg.get(t, (1, None))[1]) for t in r["techniques"]]
        techs.sort(key=lambda x: (-x["count"], x["technique"]))
        names = self.country_names
        nf = r.get("nearest_factcheck")
        card = None
        if nf and isinstance(nf.get("kb_id"), str):
            url = nf.get("url") if isinstance(nf.get("url"), str) and nf["url"].lower().startswith(("http://", "https://")) else None
            card = dict(id=nf["kb_id"], label=nf.get("label") if isinstance(nf.get("label"), str) else None,
                        claim=nf.get("claim") if isinstance(nf.get("claim"), str) else None, url=url,
                        similarity=_num(nf.get("cos", nf.get("cosine"))), rerank=_num(nf.get("rerank")),
                        debunk_match=nf.get("debunk_match") is True, desk_id=self.desk_url(url) if self.desk_url and url else None)
        v = r.get("verdict")
        verdict = dict(label=v.get("label"), basis=v.get("basis"), source=v.get("source"), rationale=v.get("rationale"),
                       at=v.get("at")) if v and isinstance(v.get("label"), str) else None
        item = dict(
            id=uid, source="fnp", kind=r.get("source"), platform=r.get("platform"), author=r.get("author"), url=r.get("url"),
            text=text, lang=r.get("lang"), posted_at=r.get("posted_at"), reach=r["reach"], tier=r["tier"], validated=r["validated"],
            p_prop=r["p_prop"], p_fake=r["p_fake"], p_misleading=r["p_misleading"], fake_score=r["fake_score"], verdict=verdict,
            techniques=techs, spans=shown, n_spans=len(spans), spans_dropped=dropped,
            countries=[dict(iso3=c, name=names.get(c) if names else None) for c in r["countries"]],
            nearest_factcheck=card, top_evidence_ids=[x for x in (r.get("top_evidence_ids") or []) if isinstance(x, str)][:10],
            model_version=r.get("model_version"), built_at=r.get("built_at"))
        rules = st["meta"].get("rules") if isinstance(st["meta"].get("rules"), dict) else {}
        triage = dict(tier=r["tier"], validated=r["validated"], rule_text=rules.get(r["tier"]), why=fnp_why(r, rules),
                      caveats=_str_list(st["meta"].get("caveats")))
        return dict(source="fnp", item=item, triage=triage)


def _str_list(v, n=8):
    return [x[:400] for x in v if isinstance(x, str)][:n] if isinstance(v, list) else []


# ======================================================================================================= realtime source
class Triage:
    def __init__(self, api, fnp_dir=None, clock=None):
        self.api = api
        self.clock = clock or api.clock
        self.fnp = FnpExport(fnp_dir, country_names=api.country_names, desk_url=api.desk_url)
        self._cache = {}
        self._lock = threading.Lock()

    # ---- helpers ------------------------------------------------------------------------------------------------
    def _cached(self, key, ttl, make):
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < ttl:
                return hit[1]
        val = make()
        with self._lock:
            self._cache[key] = (now, val)
            if len(self._cache) > 48:
                for k in sorted(self._cache, key=lambda k: self._cache[k][0])[:12]:
                    del self._cache[k]
        return val

    def _sig(self):
        enr = self.api.enrich.status()
        # never open a store that does not exist: opening one would create an empty file
        return (enr.get("loaded_at"), enr.get("records"), self.api._deep_sig() if os.path.exists(self.api.db_path) else None)

    def _source(self, q):
        s = _enum(q, "source", SOURCES, "fnp")
        return s

    # ---- /api/triage/meta ---------------------------------------------------------------------------------------
    def meta(self):
        st = self.fnp.refresh()
        fm = st["meta"]
        fnp = dict(
            available=st["available"], reason=st["reason"], error=st["error"], built_at=st["built_at"],
            model_version=fm.get("model_version") if isinstance(fm.get("model_version"), str) else None,
            data_through=fm.get("data_through") if isinstance(fm.get("data_through"), str) else None,
            n_items=len(st["items"]), tiers=dict(st["tiers"]) if st["available"] else {t: 0 for t in TIERS},
            tiers_in_meta=fm.get("tiers") if isinstance(fm.get("tiers"), dict) else None, counts_match=st["counts_match"],
            n_with_verdict=st["n_with_verdict"], validated=st["n_validated"], invalid_lines=st["invalid"],
            rules=fm.get("rules") if isinstance(fm.get("rules"), dict) else {}, caveats=_str_list(fm.get("caveats")),
            window_note="Time windows count back from the newest post in the export, not from today.",
            load_seconds=st["load_seconds"], next_run_at=fm.get("next_run_at") if isinstance(fm.get("next_run_at"), str) else None,
            refresh_note="Rebuilt by FNP after each live-scan or Propaganda Lab run; there is no fixed schedule.")
        try:
            rtm = self._cached(("meta", self._sig()), STATS_TTL, lambda: self.api._run(self._rt_meta, STATS_SECONDS))
        except ApiError as e:
            rtm = dict(available=False, reason=e.msg)
        return dict(generated_at=rt_store.utc_iso(self.clock()), caveats=CAVEATS, tier_order=list(TIERS),
                    sources=dict(fnp=fnp, realtime=rtm), validated=0)

    def _rt_meta(self, con):
        counts = {t: 0 for t in TIERS}
        for t, n in con.execute("SELECT %s AS t, count(*) FROM %s WHERE %s GROUP BY t" % (TIER_SQL, RT_FROM, RT_SCORED)):
            counts[TIER_NAME[t]] = n
        n_queue = sum(counts.values())
        n_short = con.execute("SELECT count(*) FROM (SELECT id FROM enr.enr WHERE src <> 'triage' AND coalesce(too_short, 0) = 1) c "
                              "JOIN items i ON i.id = c.id WHERE c.id NOT IN (SELECT id FROM deep)").fetchone()[0]
        n_items = con.execute("SELECT count(*) FROM items").fetchone()[0]
        n_deep = con.execute("SELECT count(*) FROM (SELECT id FROM deep) c JOIN items i ON i.id = c.id").fetchone()[0]
        bm = rt_api.batch_block(self.api.enrich.batch_meta(), n_short, 0)
        scored = n_queue + n_short
        return dict(
            available=True, built_at=bm.get("built_at"), next_run_at=bm.get("next_run_at"), model_version=bm.get("model_version"),
            tiers=counts, n_queue=n_queue, n_too_short=n_short, n_deep=n_deep, n_items=n_items, n_unscored=max(0, n_items - scored),
            share_scored=round(scored / n_items, 6) if n_items else None, validated=0, caveats=[bm["caveat"]] if bm.get("caveat") else [],
            rules=RT_RULES, thresholds=dict(n_strong_high=N_STRONG_HIGH, p_prop_high=P_PROP_HIGH, p_prop_medium=P_PROP_MEDIUM, cosine_medium=COS_MEDIUM,
                                            deep_medium=DEEP_MEDIUM, deep_demote=DEEP_DEMOTE, rerank_floor=rt_enrich.NF_RERANK_FLOOR),
            batch=dict(n_in_window=bm.get("n_in_window"), n_unscored_in_window=bm.get("n_unscored_in_window"),
                       n_scored_this_run=bm.get("n_scored_this_run"), n_failed_this_run=bm.get("n_failed_this_run"),
                       interval_hours=bm.get("interval_hours")),
            refresh_note="Recomputed from the local store each time the FNP batch export or a deep result changes.")

    # ---- /api/triage/facets -------------------------------------------------------------------------------------
    def facets(self, q):
        src = self._source(q)
        if src == "fnp":
            st = self.fnp.refresh()
            return dict(source=src, available=st["available"], **(st["facets"] or dict(platforms=[], langs=[], countries=[], techniques=[])),
                        generated_at=rt_store.utc_iso(self.clock()))
        return self._cached(("facets", self._sig()), FACETS_TTL, lambda: self.api._run(self._rt_facets, STATS_SECONDS))

    def _rt_facets(self, con):
        where = " WHERE " + RT_SCORED

        def grp(col):
            return [dict(value=r[0], count=r[1]) for r in con.execute(
                "SELECT %s, count(*) AS n FROM %s%s AND %s IS NOT NULL GROUP BY 1 ORDER BY n DESC, 1 LIMIT 40" % (col, RT_FROM, where, col))]
        ctry = []
        for r in con.execute("SELECT upper(json_extract(j.value, '$.iso3')) AS k, count(*) AS n FROM %s, "
                             "json_each(e.extra, '$.countries') j%s AND json_valid(e.extra) AND json_type(e.extra, '$.countries') = 'array' "
                             "AND k IS NOT NULL GROUP BY k ORDER BY n DESC, k LIMIT 60" % (RT_FROM, where)):
            ctry.append(dict(value=r[0], count=r[1], name=self.api.country_names.get(r[0])))
        tech = [dict(value=r[0], count=r[1]) for r in con.execute(
            "SELECT t.technique, count(DISTINCT c.id) AS n FROM %s JOIN enr.enr_tech t ON t.id = c.id%s GROUP BY 1 ORDER BY n DESC, 1 LIMIT 20"
            % (RT_FROM, where))]
        return dict(source="realtime", available=True, platforms=grp("i.platform"), langs=grp("i.lang"), countries=ctry, techniques=tech,
                    generated_at=rt_store.utc_iso(self.clock()))

    # ---- /api/triage/items --------------------------------------------------------------------------------------
    def parse(self, q):
        src = self._source(q)
        tier = _enum(q, "tier", TIERS)
        platform = _g(q, "platform").lower()
        if platform and not _RE_PLATFORM.match(platform):
            raise ApiError(400, "platform must be lowercase letters, digits or underscore")
        lang = _g(q, "lang")
        if lang and not _RE_LANG.match(lang):
            raise ApiError(400, "bad lang")
        country = _g(q, "country")
        if country and not _RE_ISO3.match(country):
            raise ApiError(400, "country must be a three-letter ISO code like IND")
        technique = _g(q, "technique")
        if technique and technique not in rt_enrich.TECHNIQUE_CLASSES:
            raise ApiError(400, "technique must be one of the 14 technique classes")
        days = _int(q, "days", 0, 0, 3650)
        text_q = _g(q, "q")
        if len(text_q) > 300:
            raise ApiError(400, "q too long (max 300 characters)")
        sort = _enum(q, "sort", SORTS, "priority")
        size = _int(q, "size", 25, 1, PAGE_SIZE_MAX)
        page = _int(q, "page", 1, 1, 100000)
        if (page - 1) * size > OFFSET_MAX:
            raise ApiError(400, "page too deep (limit %d results); narrow the filters" % OFFSET_MAX)
        return src, dict(tier=tier, platform=platform, lang=lang, country=country.upper(), technique=technique, days=days, q=text_q), sort, page, size

    def items(self, q):
        src, flt, sort, page, size = self.parse(q)
        if src == "fnp":
            out = self.fnp.list(flt, sort, page, size, now=_epoch(self.clock()))
        else:
            out = self._rt_items(flt, sort, page, size)
        out.update(source=src, sort=sort, generated_at=rt_store.utc_iso(self.clock()))
        return out

    def _rt_where(self, flt, with_tier):
        conds, args = [RT_SCORED], []
        if flt["platform"]:
            conds.append("i.platform = ?")
            args.append(flt["platform"])
        if flt["lang"]:
            conds.append("i.lang = ?")
            args.append(flt["lang"])
        if flt["days"]:
            conds.append("i.published_at >= ?")
            args.append(rt_store.utc_iso(self.clock() - timedelta(days=flt["days"])))
        if flt["country"]:
            conds.append("instr(e.extra, ?) > 0")
            args.append('"iso3":"%s"' % flt["country"])
        if flt["technique"]:
            conds.append("EXISTS (SELECT 1 FROM enr.enr_tech tt WHERE tt.id = c.id AND tt.technique = ?)")
            args.append(flt["technique"])
        if flt["q"]:
            m = rt_store.fts_query(flt["q"])
            if m is None:
                raise ApiError(400, "q has no searchable words")
            conds.append("i.rowid IN (SELECT rowid FROM items_fts WHERE items_fts MATCH ?)")
            args.append(m)
        if with_tier and flt["tier"]:
            conds.append("%s = ?" % TIER_SQL)
            args.append(TIER_RANK[flt["tier"]])
        return " WHERE " + " AND ".join(conds), args

    def _rt_items(self, flt, sort, page, size):
        order = order_priority(_epoch(self.clock())) if sort == "priority" else ORDER_RECENT
        base_where, base_args = self._rt_where(flt, False)
        page_where, page_args = self._rt_where(flt, True)
        off = (page - 1) * size

        def go(con):
            tiers = {t: 0 for t in TIERS}
            for t, n in con.execute("SELECT %s AS t, count(*) FROM %s%s GROUP BY t" % (TIER_SQL, RT_FROM, base_where), base_args):
                tiers[TIER_NAME[t]] = n
            total = tiers[flt["tier"]] if flt["tier"] else sum(tiers.values())
            ids = [r[0] for r in con.execute("SELECT i.id, %s AS t FROM %s%s ORDER BY %s LIMIT ? OFFSET ?"
                                              % (TIER_SQL, RT_FROM, page_where, order), page_args + [size, off])]
            rows = self._rt_rows(con, ids)
            return dict(available=True, total=total, tiers=tiers, page=page, size=size, pages=max(1, -(-total // size)) if total else 0,
                        items=rows)
        return self.api._run(go)

    def _rt_rows(self, con, ids):
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        got = {}
        for r in con.execute(
                "SELECT i.id, i.kind, i.platform, i.title, i.text_excerpt, i.source, i.lang, i.published_at, i.engagement, "
                "e.p_prop, e.n_strong, e.too_short, e.fc_id, e.fc_cos, e.fc_rerank, f.label AS fc_label, d.id AS d_id, d.d_label, d.d_pfm, "
                "(SELECT group_concat(technique, '|') FROM (SELECT technique FROM enr.enr_tech WHERE id = i.id ORDER BY cnt DESC, technique LIMIT 3)) AS techs "
                "FROM items i LEFT JOIN enr.enr e ON e.id = i.id AND e.src <> 'triage' LEFT JOIN enr.enr_fc f ON f.kb_id = e.fc_id "
                "LEFT JOIN deep d ON d.id = i.id WHERE i.id IN (%s)" % marks, ids):
            tier, rule = rt_tier(r["p_prop"], r["n_strong"], r["fc_rerank"], r["fc_cos"], r["d_pfm"], r["d_id"] is not None)
            title = _oneline(r["title"] or "", 160)
            eng = r["engagement"]
            got[r["id"]] = dict(
                id=r["id"], source="realtime", kind=r["kind"], platform=r["platform"], author=r["source"], title=title or None,
                snippet=_oneline(r["text_excerpt"] or ""), lang=r["lang"], posted_at=r["published_at"],
                reach=int(eng) if isinstance(eng, (int, float)) and not isinstance(eng, bool) and eng >= 0 else None,
                tier=tier, rule=rule, validated=False, p_prop=_num(r["p_prop"]), n_strong=r["n_strong"],
                nearest_factcheck=dict(label=r["fc_label"], cosine=_num(r["fc_cos"]), rerank=_num(r["fc_rerank"])) if r["fc_id"] else None,
                deep=dict(verdict=r["d_label"], p_fake_misleading=_num(r["d_pfm"])) if r["d_id"] else None,
                techniques=(r["techs"] or "").split("|") if r["techs"] else [])
        return [got[i] for i in ids if i in got]

    # ---- /api/triage/item/<id> ----------------------------------------------------------------------------------
    def item(self, item_id, q=None):
        if not _RE_ID.match(item_id or ""):
            raise ApiError(404, "not found")
        src = _enum(q or {}, "source", SOURCES, "fnp" if _RE_SHA1.match(item_id) else "realtime")
        if src == "fnp":
            return self.fnp.detail(item_id)
        it = self.api.item(item_id)  # 404 when unknown; the full Evidence Base case-file record
        inf = it.get("inferred") or {}
        b, deep = inf.get("batch") or {}, inf.get("deep")
        from_triage = bool(b) and inf.get("export_source") == "triage"   # the older triage export: shown by the FNP source, not queued here
        if from_triage:
            b = {}
        nf = b.get("nearest_factcheck") or {}
        sig = dict(p_prop=_num(b.get("p_prop")), n_strong=b.get("n_strong"), rerank=_num(nf.get("rerank")), cosine=_num(nf.get("similarity")),
                   deep_pfm=_num(deep.get("p_fake_misleading")) if deep else None, has_deep=bool(deep))
        scored = bool(deep) or (bool(b) and not b.get("too_short"))
        if scored:
            tier, rule = rt_tier(sig["p_prop"], sig["n_strong"], sig["rerank"], sig["cosine"], sig["deep_pfm"], sig["has_deep"])
            why = explain_rt(tier, rule, sig)
            rule_text = RT_RULES[tier]
        else:
            tier = rule = None
            why = ["FNP has not scored this item in its realtime batch yet, so it is not in the queue." if not b else
                   "This item is too short for the FNP encoder to rate, so it is not ranked. That says nothing about whether it is true."]
            if from_triage:
                why = ["Only the older FNP triage export has scored this item, so it is listed under the FNP triage export, not in this queue."]
            rule_text = RT_RULES["excluded"]
        return dict(source="realtime", item=it, triage=dict(tier=tier, rule=rule, rule_text=rule_text, why=why, signals=sig, validated=False,
                                                              in_queue=scored, caveats=[b["caveat"]] if isinstance(b.get("caveat"), str) else []))
