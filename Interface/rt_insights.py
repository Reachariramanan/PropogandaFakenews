"""Grounded insights for the Evidence Base (POST /api/rt/insights).

Everything shown is derived from the LOCAL realtime store only (read-only SQL, no outside knowledge):
  1. FACTS      counts, shares with their denominators, the 20 most recent rows in scope.
  2. TEMPLATE   deterministic sentences written from the facts alone; a sentence whose denominator is 0 is omitted
                and no sentence ever states a cause. Always available.
  3. PHRASED    optional Gemma phrasing through the FNP service (FNP_INSIGHT_URL, default
                http://127.0.0.1:8879/api/insight_v1) on request. The answer is validated AGAIN here: every cited
                id must be one of the rows we sent, every number in the text must be in the facts (percentages may be
                rounded), otherwise a refusal is returned and the unvalidated text is never shown. An unreachable
                service just means `phrased: null` and the template insights stay.
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rt_api  # noqa: E402
import rt_store  # noqa: E402
from rt_api import ApiError  # noqa: E402

INSIGHT_URL_DEFAULT = "http://127.0.0.1:8879/api/insight_v1"
PHRASE_TIMEOUT = 60.0
MAX_IDS = 25
SAMPLE_ROWS = 20
ROW_TEXT = 300
CACHE_TTL = 300.0
CACHE_MAX = 64
DOWN_TTL = 60.0              # after a failed phrasing call, don't retry for this long
TOP = 8
DAILY_MAX = 60
P_PROP_HIGH = 0.7            # descriptive cut-off for "high propaganda score" (a display bucket, not a decision rule)
QUERY_SECONDS = 40.0
MAX_ANSWER_BYTES = 256 * 1024
MAX_ANSWER_CHARS = 4000
UNAVAILABLE = "phrasing service not available"
DEFAULT_QUESTION = "Summarise the main patterns in these items."
LIKELY_FAKE = ("FAKE", "MISLEADING")
SCOPE_KEYS = ("kind", "platform", "source", "topic", "lang", "from", "to", "days", "ids", "q", "analysed")


# ------------------------------------------------------------------------------------------------ scope parsing
def parse_request(body):
    """Validated body -> (scope dict, question str|None, phrase bool). ApiError(400) on anything unexpected."""
    if not isinstance(body, dict):
        raise ApiError(400, "expected a JSON object")
    extra = set(body) - {"scope", "question", "phrase"}
    if extra:
        raise ApiError(400, "unknown field: %s" % sorted(extra)[0])
    raw = body.get("scope", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ApiError(400, "scope must be an object")
    bad = set(raw) - set(SCOPE_KEYS)
    if bad:
        raise ApiError(400, "unknown scope field: %s" % sorted(bad)[0])
    scope = {}

    def text(key, maxlen, rx=None, lower=False):
        v = raw.get(key)
        if v is None or v == "":
            return
        if not isinstance(v, str):
            raise ApiError(400, "scope.%s must be a string" % key)
        v = v.strip()
        if not v:
            return
        if len(v) > maxlen or any(ord(c) < 32 for c in v) or (rx and not rx.match(v)):
            raise ApiError(400, "bad scope.%s" % key)
        scope[key] = v.lower() if lower else v

    kind = raw.get("kind")
    if kind not in (None, "", "news", "social"):
        raise ApiError(400, "scope.kind must be news or social")
    if kind:
        scope["kind"] = kind
    text("platform", 40, rt_api._RE_PLATFORM, lower=True)
    text("source", 160)
    topic = raw.get("topic")
    if topic not in (None, ""):
        if not isinstance(topic, str) or not (topic.lower() == "unclassified" or rt_api._RE_TOPIC.match(topic)):
            raise ApiError(400, "bad scope.topic")
        scope["topic"] = topic
    text("lang", 12, rt_api._RE_LANG)
    for k in ("from", "to"):
        v = raw.get(k)
        if v not in (None, ""):
            if not isinstance(v, str):
                raise ApiError(400, "scope.%s must be a string" % k)
            rt_api._date_bound(v.strip(), k, k == "to")  # validates, raises ApiError(400)
            scope[k] = v.strip()
    days = raw.get("days")
    if days not in (None, "", 0):
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 365:
            raise ApiError(400, "scope.days must be an integer between 1 and 365")
        scope["days"] = days
    text("q", 300)
    if scope.get("q") and rt_store.fts_query(scope["q"]) is None:
        raise ApiError(400, "scope.q has no searchable words")
    an = raw.get("analysed")
    if an not in (None, ""):
        if an not in ("yes", "no"):
            raise ApiError(400, "scope.analysed must be yes or no")
        scope["analysed"] = an
    ids = raw.get("ids")
    if ids is not None:
        if not isinstance(ids, list) or not ids or len(ids) > MAX_IDS or any(
                not isinstance(i, str) or not rt_api._RE_ITEM.match(i) for i in ids):
            raise ApiError(400, "scope.ids must be a list of 1 to %d item ids" % MAX_IDS)
        scope["ids"] = list(dict.fromkeys(ids))
    q = body.get("question")
    if q is not None:
        if not isinstance(q, str) or len(q.strip()) > 300 or any(ord(c) < 32 for c in q):
            raise ApiError(400, "question must be a string of at most 300 characters")
        q = q.strip() or None
    ph = body.get("phrase", False)
    if not isinstance(ph, bool):
        raise ApiError(400, "phrase must be true or false")
    return scope, q, ph


def scope_hash(scope):
    return hashlib.sha1(json.dumps(scope, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def scope_sql(scope, clock):
    conds, args = [], []
    if scope.get("q"):
        conds.append("i.rowid IN (SELECT rowid FROM items_fts WHERE items_fts MATCH ?)")
        args.append(rt_store.fts_query(scope["q"]))
    for col, key in (("kind", "kind"), ("platform", "platform"), ("source", "source"), ("lang", "lang")):
        if scope.get(key):
            conds.append("i.%s = ?" % col)
            args.append(scope[key])
    if scope.get("topic"):
        if scope["topic"].lower() == "unclassified":
            conds.append("i.db_topic_id IS NULL")
        else:
            conds.append("i.db_topic_id = ?")
            args.append(scope["topic"])
    for key, upper in (("from", False), ("to", True)):
        if scope.get(key):
            op, val = rt_api._date_bound(scope[key], key, upper)
            conds.append("i.published_at %s ?" % op)
            args.append(val)
    if scope.get("days"):
        conds.append("i.published_at >= ?")
        args.append(rt_store.utc_iso(clock() - timedelta(days=scope["days"])))
    if scope.get("analysed"):
        conds.append("i.id %sIN (SELECT id FROM enr.enr)" % ("" if scope["analysed"] == "yes" else "NOT "))
    if scope.get("ids"):
        conds.append("i.id IN (%s)" % ",".join("?" * len(scope["ids"])))
        args.extend(scope["ids"])
    return (" WHERE " + " AND ".join(conds)) if conds else "", args


def scope_desc(scope):
    """A short human description of the scope (for the UI), built from the validated scope only."""
    parts = []
    if scope.get("ids"):
        parts.append("%d selected items" % len(scope["ids"]))
    for k, lab in (("kind", "kind"), ("platform", "platform"), ("source", "source"), ("topic", "topic"),
                   ("lang", "language")):
        if scope.get(k):
            parts.append("%s %s" % (lab, scope[k]))
    if scope.get("q"):
        parts.append('keyword "%s"' % scope["q"])
    if scope.get("days"):
        parts.append("last %d days" % scope["days"])
    if scope.get("from") or scope.get("to"):
        parts.append("dates %s to %s" % (scope.get("from") or "start", scope.get("to") or "now"))
    if scope.get("analysed"):
        parts.append("analysed: %s" % scope["analysed"])
    return ", ".join(parts) if parts else "all items"


# --------------------------------------------------------------------------------------------------------- facts
def _pct(n, d):
    return round(100.0 * n / d, 1) if d else None


def build_facts(con, scope, clock):
    """Read-only SQL on the local store (+ the attached enrichment sidecar) -> the FACTS dict. Every share carries
    its numerator and denominator; analysed-only figures use the analysed items as denominator."""
    where, args = scope_sql(scope, clock)
    one = lambda sql, a=args: con.execute(sql, a).fetchone()
    rows = lambda sql, a=args: con.execute(sql, a).fetchall()
    keys = {}

    def reg(key, label, value, of=None):
        keys[key] = dict(label=label, value=value, of=of, pct=_pct(value, of) if of else None)

    n = one("SELECT count(*) FROM items i" + where)[0]
    f = dict(scope=scope, n_items=n)
    reg("n_items", "Items in scope", n)
    kinds = {r[0]: r[1] for r in rows("SELECT i.kind, count(*) FROM items i%s GROUP BY i.kind" % where)}
    f["n_news"], f["n_social"] = kinds.get("news", 0), kinds.get("social", 0)
    f["pct_news"], f["pct_social"] = _pct(f["n_news"], n), _pct(f["n_social"], n)
    reg("n_news", "News items", f["n_news"], n)
    reg("n_social", "Social posts", f["n_social"], n)
    if n == 0:
        f.update(platforms=[], sources=[], n_sources_distinct=0, topics=[], n_classified=0, n_unclassified=0,
                 pct_unclassified=None, languages=[], daily=[], busiest_day=None, n_analysed=0, pct_analysed=None,
                 n_with_verdict=0, verdicts={}, n_likely_fake_misleading=0, pct_likely_fake_misleading=None,
                 n_with_p_prop=0, p_prop_buckets=[], n_p_prop_ge_07=0, techniques=[], workbench=None,
                 n_sample_rows=0, sample_method="no items in scope", rows=[], keys=keys)
        return f
    f["platforms"] = [dict(platform=r[0], count=r[1], pct=_pct(r[1], n)) for r in rows(
        "SELECT i.platform, count(*) c FROM items i%s GROUP BY i.platform ORDER BY c DESC, i.platform LIMIT %d"
        % (where, TOP))]
    f["sources"] = [dict(source=r[0], count=r[1], pct=_pct(r[1], n)) for r in rows(
        "SELECT i.source, count(*) c FROM items i%s%s GROUP BY i.source ORDER BY c DESC, i.source LIMIT %d"
        % (where, (" AND" if where else " WHERE") + " i.source IS NOT NULL AND i.source != ''", TOP))]
    f["n_sources_distinct"] = one("SELECT count(DISTINCT i.source) FROM items i" + where)[0]
    cw = where + (" AND" if where else " WHERE") + " i.db_topic_id IS NOT NULL"
    f["n_classified"] = one("SELECT count(*) FROM items i" + cw)[0]
    f["n_unclassified"] = n - f["n_classified"]
    f["pct_unclassified"] = _pct(f["n_unclassified"], n)
    f["topics"] = []
    for r in rows("SELECT i.db_topic_id, count(*) c, max(i.topic_name) FROM items i%s GROUP BY i.db_topic_id "
                  "ORDER BY c DESC, i.db_topic_id LIMIT %d" % (cw, TOP)):
        f["topics"].append(dict(topic_id=r[0], name=r[2] or r[0], count=r[1], pct=_pct(r[1], n)))
    f["languages"] = [dict(lang=r[0], count=r[1], pct=_pct(r[1], n)) for r in rows(
        "SELECT i.lang, count(*) c FROM items i%s GROUP BY i.lang ORDER BY c DESC, i.lang LIMIT %d" % (where, TOP))]
    days = [(r[0], r[1]) for r in rows("SELECT substr(i.published_at,1,10) d, count(*) FROM items i%s GROUP BY d "
                                       "ORDER BY d" % where)]
    f["daily"] = [dict(day=d, count=c) for d, c in days[-DAILY_MAX:]]
    best = max(days, key=lambda t: (t[1], t[0])) if days else None
    f["busiest_day"] = dict(day=best[0], count=best[1], pct=_pct(best[1], n)) if best else None
    # --- analysed items only: every figure below uses the analysed items as its denominator
    analysed = with_verdict = with_prop = high = 0
    verdicts, buckets = {}, [0] * 10
    for v, bk, c in con.execute(
            "SELECT e.verdict, CASE WHEN e.p_prop IS NULL THEN -1 ELSE min(cast(e.p_prop * 10 AS INTEGER), 9) END b, "
            "count(*) FROM enr.enr e CROSS JOIN items i ON i.id = e.id%s GROUP BY e.verdict, b" % where, args):
        analysed += c
        if v is not None:
            with_verdict += c
            verdicts[v] = verdicts.get(v, 0) + c
        if bk >= 0:
            with_prop += c
            buckets[bk] += c
    high = one("SELECT count(*) FROM enr.enr e CROSS JOIN items i ON i.id = e.id%s%s e.p_prop >= %s"
               % (where, " AND" if where else " WHERE", P_PROP_HIGH))[0]
    flagged = sum(verdicts.get(v, 0) for v in LIKELY_FAKE)
    f.update(n_analysed=analysed, pct_analysed=_pct(analysed, n), n_with_verdict=with_verdict, verdicts=verdicts,
             n_likely_fake_misleading=flagged, pct_likely_fake_misleading=_pct(flagged, with_verdict),
             n_with_p_prop=with_prop,
             p_prop_buckets=[dict(lo=i / 10, hi=(i + 1) / 10, count=c) for i, c in enumerate(buckets)],
             n_p_prop_ge_07=high, p_prop_high_cutoff=P_PROP_HIGH)
    f["techniques"] = [dict(technique=r[0], items=r[1], mentions=r[2], pct_of_analysed=_pct(r[1], analysed))
                       for r in rows("SELECT t.technique, count(DISTINCT t.id) c, sum(t.cnt) m FROM enr.enr_tech t "
                                     "CROSS JOIN items i ON i.id = t.id%s GROUP BY t.technique "
                                     "ORDER BY c DESC, t.technique LIMIT %d" % (where, TOP))]
    # --- workbench results already stored for the selected items
    f["workbench"] = None
    if scope.get("ids"):
        ph = ",".join("?" * len(scope["ids"]))
        latest = {}
        for iid, rj in con.execute("SELECT a.item_id, a.result_json FROM analyses a WHERE a.item_id IN (%s) "
                                   "ORDER BY a.created_at" % ph, scope["ids"]):
            latest[iid] = rj
        labels = {}
        for rj in latest.values():
            try:
                lab = ((json.loads(rj) or {}).get("step1") or {}).get("label") if len(rj) < 400000 else None
            except (ValueError, AttributeError):
                lab = None
            if isinstance(lab, str):
                labels[lab] = labels.get(lab, 0) + 1
        n_lab = sum(labels.values())
        wf = sum(labels.get(v, 0) for v in LIKELY_FAKE)
        f["workbench"] = dict(n_selected=n, n_stored=len(latest), n_with_label=n_lab, labels=labels,
                              n_likely_fake_misleading=wf, pct_likely_fake_misleading=_pct(wf, n_lab),
                              stored_ids=sorted(latest))
        reg("workbench_n_stored", "Selected items with a stored Claim Check result", len(latest), n)
        reg("workbench_n_flagged", "Stored results labelled FAKE or MISLEADING", wf, n_lab)
    # --- sample rows: the most recent items in scope (the selection itself when ids are given)
    if scope.get("ids"):
        order = {iid: k for k, iid in enumerate(scope["ids"])}
        sel = rows("SELECT i.id, i.title, i.text_excerpt, i.platform, i.source, i.published_at FROM items i%s" % where)
        sel.sort(key=lambda r: order.get(r[0], 1 << 30))
        sample, method = sel[:SAMPLE_ROWS], "the first %d selected items" % SAMPLE_ROWS
    else:
        sample = rows("SELECT i.id, i.title, i.text_excerpt, i.platform, i.source, i.published_at FROM items i%s "
                      "ORDER BY i.published_at DESC, i.id DESC LIMIT %d" % (where, SAMPLE_ROWS))
        method = "the %d most recent items in scope" % SAMPLE_ROWS
    f["rows"] = [dict(id=r[0], text=_row_text(r[1], r[2]), platform=r[3], source=r[4], published_at=r[5])
                 for r in sample]
    f["n_sample_rows"] = len(f["rows"])
    f["sample_method"] = method
    # cited keys
    if f["platforms"]:
        t = f["platforms"][0]
        reg("top_platform", "Most common platform: %s" % t["platform"], t["count"], n)
    if f["sources"]:
        t = f["sources"][0]
        reg("top_source", "Most frequent source: %s" % t["source"], t["count"], n)
    if f["topics"]:
        t = f["topics"][0]
        reg("top_topic", "Most common database topic: %s" % t["name"], t["count"], n)
    reg("n_unclassified", "Items without a database topic (Unclassified)", f["n_unclassified"], n)
    if f["languages"]:
        t = f["languages"][0]
        reg("top_language", "Most common language: %s" % t["lang"], t["count"], n)
    if f["busiest_day"]:
        reg("busiest_day", "Busiest day: %s" % f["busiest_day"]["day"], f["busiest_day"]["count"], n)
    reg("n_analysed", "Items with an FNP record (analysed)", analysed, n)
    reg("n_with_verdict", "Analysed items with a verdict", with_verdict, analysed)
    reg("n_likely_fake_misleading", "Verdicts FAKE or MISLEADING", flagged, with_verdict)
    reg("n_p_prop_ge_07", "Analysed items with a propaganda score of at least 0.7", high, with_prop)
    if f["techniques"]:
        t = f["techniques"][0]
        reg("top_technique", "Most common technique: %s" % t["technique"], t["items"], analysed)
    f["keys"] = keys
    return f


def _row_text(title, excerpt):
    t, e = (title or "").strip(), (excerpt or "").strip()
    s = (t + " - " + e) if t and e else (t or e)
    s = re.sub(r"\s+", " ", s)
    return s[:ROW_TEXT]


# ------------------------------------------------------------------------------------------------- template text
def _n(v):
    return "{:,}".format(v)


def _p(num, den):
    return "%d%%" % int(round(100.0 * num / den))


def template_insights(f):
    """Deterministic sentences from the facts only. Each is {text, cites}. Insights with a 0 denominator are left
    out, and no sentence states a cause."""
    out = []
    n = f["n_items"]
    if n <= 0:
        return out

    def add(text, cites):
        out.append(dict(text=text, cites=cites))

    nn, ns = f["n_news"], f["n_social"]
    if nn and ns:
        big, small, bn, sn = ("Social posts", "news articles", ns, nn) if ns >= nn else ("News articles", "social posts", nn, ns)
        add("%s are %s of the %s items in scope (%s of %s); %s are the other %s." % (
            big, _p(bn, n), _n(n), _n(bn), _n(n), small, _n(sn)), ["fact:n_social", "fact:n_news", "fact:n_items"])
    elif nn:
        add("All %s items in scope are news articles." % _n(n), ["fact:n_news", "fact:n_items"])
    elif ns:
        add("All %s items in scope are social posts." % _n(n), ["fact:n_social", "fact:n_items"])
    if f["platforms"]:
        t = f["platforms"][0]
        add("The most common platform is %s: %s of %s items (%s)." % (t["platform"], _n(t["count"]), _n(n),
                                                                     _p(t["count"], n)), ["fact:top_platform"])
    if f["sources"]:
        t = f["sources"][0]
        add("The most frequent source is %s: %s of %s items (%s)." % (t["source"], _n(t["count"]), _n(n),
                                                                     _p(t["count"], n)), ["fact:top_source"])
    un = "%s of %s items (%s) have no database topic (Unclassified)." % (
        _n(f["n_unclassified"]), _n(n), _p(f["n_unclassified"], n))
    if f["topics"]:
        t = f["topics"][0]
        add("The most common database topic is %s: %s of %s items (%s). %s" % (
            t["name"], _n(t["count"]), _n(n), _p(t["count"], n), un), ["fact:top_topic", "fact:n_unclassified"])
    else:
        add(un, ["fact:n_unclassified"])
    if f["languages"]:
        t = f["languages"][0]
        add("The most common language is %s: %s of %s items (%s)." % (t["lang"], _n(t["count"]), _n(n),
                                                                     _p(t["count"], n)), ["fact:top_language"])
    b = f["busiest_day"]
    if b and len(f["daily"]) > 1:
        add("The busiest day is %s with %s of %s items (%s)." % (b["day"], _n(b["count"]), _n(n), _p(b["count"], n)),
            ["fact:busiest_day"])
    a = f["n_analysed"]
    if a:
        add("FNP has analysed %s of %s items (%s); the rest have no FNP record yet." % (
            _n(a), _n(n), _p(a, n)), ["fact:n_analysed"])
        wv = f["n_with_verdict"]
        if wv:
            add("Of the %s analysed items, %s have a verdict. Of those %s, %s (%s) are FAKE or MISLEADING." % (
                _n(a), _n(wv), _n(wv), _n(f["n_likely_fake_misleading"]), _p(f["n_likely_fake_misleading"], wv)),
                ["fact:n_with_verdict", "fact:n_likely_fake_misleading"])
        else:
            add("None of the %s analysed items has a verdict yet." % _n(a), ["fact:n_with_verdict"])
        wp = f["n_with_p_prop"]
        if wp:
            add("Among the %s analysed items with a propaganda score, %s (%s) score 0.7 or higher." % (
                _n(wp), _n(f["n_p_prop_ge_07"]), _p(f["n_p_prop_ge_07"], wp)), ["fact:n_p_prop_ge_07"])
        if f["techniques"]:
            t = f["techniques"][0]
            add("The most common propaganda technique is %s: found in %s of %s analysed items (%s)." % (
                t["technique"], _n(t["items"]), _n(a), _p(t["items"], a)), ["fact:top_technique"])
    w = f.get("workbench")
    if w and w["n_stored"]:
        s = "%s of the %s selected items have a stored Claim Check result." % (_n(w["n_stored"]), _n(w["n_selected"]))
        cites = ["fact:workbench_n_stored"] + w["stored_ids"][:5]
        if w["n_with_label"]:
            s += " Of the %s with a label, %s (%s) are FAKE or MISLEADING." % (
                _n(w["n_with_label"]), _n(w["n_likely_fake_misleading"]),
                _p(w["n_likely_fake_misleading"], w["n_with_label"]))
            cites.append("fact:workbench_n_flagged")
        add(s, cites)
    return out


def scope_line(n, synced_at):
    when = ("%s UTC" % synced_at[:16].replace("T", " ")) if synced_at else "(no sync recorded)"
    return "Based on %s items from the realtime database, synced %s. No outside knowledge used." % (_n(n), when)


# ---------------------------------------------------------------------------------------- phrased-answer validation
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s?%)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def allowed_numbers(facts):
    """[(value, is_float)] of every number in the facts (rows excluded) plus the components of date strings."""
    out = []

    def walk(v, skip=False):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            out.append((float(v), isinstance(v, float) and v != int(v)))
        elif isinstance(v, str):
            if _DATE.match(v):
                for part in re.split(r"[-T:Z ]", v[:19]):
                    if part.isdigit():
                        out.append((float(int(part)), False))
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk({k: v for k, v in facts.items() if k not in ("rows", "sample_method", "stored_ids")})
    return out


def numbers_in(text):
    """[(value, decimals, is_percent, literal)] for each number written in the text (1,234 and 12.5% understood)."""
    res = []
    for m in _NUM.finditer(text):
        whole, dec, pct = m.group(1), m.group(2), m.group(3)
        try:
            v = float(whole.replace(",", "") + ("." + dec if dec else ""))
        except ValueError:
            continue
        res.append((v, len(dec) if dec else 0, bool(pct), m.group(0).strip()))
    return res


def validate_phrased(text, citations, facts):
    """-> (ok, reason). Cited ids must be among facts.rows; every number written must appear in the facts (a percentage,
    or any number compared with a non-integer fact, may differ by the rounding of the digits written)."""
    if not isinstance(text, str) or not text.strip():
        return False, "the answer had no text"
    if len(text) > MAX_ANSWER_CHARS:
        return False, "the answer was too long"
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        return False, "the answer's citations were not a list of row ids"
    ids = {r["id"] for r in facts.get("rows", [])}
    if not citations:
        return False, "the answer cited no rows"
    bad = [c for c in citations if c not in ids]
    if bad:
        return False, "the answer cited a row that is not in the data it was given"
    scrub = text
    for i in sorted(ids, key=len, reverse=True):
        scrub = scrub.replace(i, " ")
    allowed = allowed_numbers(facts)
    for v, dec, is_pct, lit in numbers_in(scrub):
        tol = 0.5 * (10 ** -dec) + 1e-9
        ok = False
        for a, is_float in allowed:
            if abs(a - v) < 1e-9 or ((is_pct or is_float) and abs(a - v) <= tol):
                ok = True
                break
        if not ok:
            return False, "the answer contains a number (%s) that is not in the data it was given" % lit
    return True, None


# ---------------------------------------------------------------------------------------------------- the service
class Insights:
    def __init__(self, api, clock=None, url=None, post=None):
        self.api = api
        self.clock = clock or api.clock
        self.url = url                      # None -> FNP_INSIGHT_URL / default, read per call
        self._post = post or self._http_post
        self._lock = threading.Lock()
        self._cache = {}
        self._down_until = 0.0
        self._answered = False              # the service answered at least once since the last failure

    # ---- cache ---------------------------------------------------------------------------------------------------
    def _cget(self, key):
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < CACHE_TTL:
                return hit[1]
            self._cache.pop(key, None)
        return None

    def _cput(self, key, val):
        with self._lock:
            self._cache[key] = (time.monotonic(), val)
            if len(self._cache) > CACHE_MAX:
                for k in sorted(self._cache, key=lambda k: self._cache[k][0])[:16]:
                    del self._cache[k]

    # ---- entry point ---------------------------------------------------------------------------------------------
    def run(self, body):
        scope, question, phrase = parse_request(body)
        sh = scope_hash(scope)
        synced = self.api._run(lambda con: con.execute(
            "SELECT max(ts) FROM sync_log WHERE error IS NULL").fetchone()[0])
        base = self._cget(("base", sh, synced))
        cached = base is not None
        if base is None:
            facts = self.api._run(lambda con: build_facts(con, scope, self.clock), QUERY_SECONDS)
            base = dict(scope=scope, scope_desc=scope_desc(scope), scope_line=scope_line(facts["n_items"], synced),
                        synced_at=synced, facts=facts, template_insights=template_insights(facts))
            self._cput(("base", sh, synced), base)
        out = dict(base, phrased=None, cached=cached, generated_at=rt_store.utc_iso(self.clock()))
        q = question or DEFAULT_QUESTION
        if phrase:
            if base["facts"]["n_items"] == 0:
                out["phrased"] = None
                out["phrasing"] = dict(available=None, reason="no items in scope to phrase")
                return out
            pk = ("phrased", sh, synced, hashlib.sha1(q.encode("utf-8")).hexdigest())
            hit = self._cget(pk)
            if hit is not None:
                out["phrased"], out["phrasing"] = hit
                out["cached"] = True
                return out
            phrased, state = self._phrase(q, base["facts"])
            if phrased is not None:
                self._cput(pk, (phrased, state))  # answers and refusals are cached; "not available" is not
            out["phrased"], out["phrasing"] = phrased, state
        else:
            out["phrasing"] = self._state()
        return out

    def _state(self):
        if time.monotonic() < self._down_until:
            return dict(available=False, reason=UNAVAILABLE)
        return dict(available=True if self._answered else None, reason=None)

    # ---- the phrasing call ---------------------------------------------------------------------------------------
    def _endpoint(self):
        return self.url or os.environ.get("FNP_INSIGHT_URL") or INSIGHT_URL_DEFAULT

    def _http_post(self, url, payload, timeout):
        """-> (status, parsed JSON | None). Raises on transport failure."""
        req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                                     method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status, raw = r.status, r.read(MAX_ANSWER_BYTES)
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read(MAX_ANSWER_BYTES)

        def bad(c):
            raise ValueError("non-strict JSON constant")
        try:
            return status, json.loads(raw.decode("utf-8"), parse_constant=bad)
        except Exception:  # noqa: BLE001
            return status, None

    def _phrase(self, question, facts):
        """-> (phrased | None, phrasing state). Never raises."""
        if time.monotonic() < self._down_until:
            return None, dict(available=False, reason=UNAVAILABLE)
        payload = dict(question=question, facts={**{k: v for k, v in facts.items() if k not in ("rows", "scope")},
                                                 "rows": [dict(id=r["id"], text=r["text"]) for r in facts["rows"]]})
        try:
            status, resp = self._post(self._endpoint(), payload, PHRASE_TIMEOUT)
        except Exception:  # noqa: BLE001 - unreachable, refused connection, timeout, bad URL
            return self._down()
        if status in (404, 405, 502, 503, 504) or status >= 500 or status in (401, 403, 429):
            return self._down()
        self._answered = True
        if status in (400, 422):
            return dict(refused=True, reason="the phrasing service declined this request"), dict(available=True,
                                                                                                reason=None)
        if status != 200 or not isinstance(resp, dict):
            return dict(refused=True, reason="the phrasing service sent an unreadable answer"), dict(available=True,
                                                                                                   reason=None)
        if resp.get("refused") or resp.get("error") or "text" not in resp:
            return dict(refused=True, reason="the phrasing service declined to answer from this data"), dict(
                available=True, reason=None)
        ok, why = validate_phrased(resp.get("text"), resp.get("citations"), facts)
        if not ok:
            return dict(refused=True, reason="Not shown: " + why + "."), dict(available=True, reason=None)
        return dict(text=resp["text"].strip(), citations=list(dict.fromkeys(resp["citations"])),
                    source="fnp-insight_v1"), dict(available=True, reason=None)

    def _down(self):
        self._down_until = time.monotonic() + DOWN_TTL
        self._answered = False
        return None, dict(available=False, reason=UNAVAILABLE)
