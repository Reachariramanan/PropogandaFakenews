"""FNP inference layer for the Evidence Base: joins the FNP exports onto our realtime items. Read-only on the exports.

Inputs (all optional; a missing directory is normal and simply means "not analysed"):
  reports/realtime/v1/{news,social}/part_NNNN.jsonl + meta.json   the REAL format FNP writes: Interface/docs/REALTIME_CONTRACT.md
  (the older proposal shape, with meta["files"] and `techniques` lists, is still read)
  reports/triage/v1/items.jsonl + meta.json                contract: docs/TRIAGE_CONTRACT.md (FNP tree)

The exports are streamed line by line (never loaded whole) into a compact SQLite *sidecar* file
(Interface/data/rt_enrich.sqlite, mode 600, rebuilt by temp file + os.replace) instead of a Python dict: millions of
records stay out of the process heap, and the API can JOIN, filter and sort on p_prop / p_fake in SQL. The sidecar
holds only the compact fields: p_prop, too_short, the 14 technique probabilities (one JSON array per item, not 14
rows), strong techniques, countries, the nearest fact-check (deduplicated in `enr_fc`), p_fake, p_misleading, verdict,
model_version and up to 12 spans as OFFSETS. Span text is resolved lazily from the local items table when one item is
requested (title + "\\n" + text_excerpt, bounds-checked). It is reloaded when meta.json `built_at` changes (one stat
per request; in a background thread when `background=True`). An item without a record is "not analysed"; nothing is
ever guessed.

Join keys: the realtime export joins on our item id (or source_table + source_id). The triage export carries
sha1 post_uids that are not our ids, so it is joined on exact URL equality at build time (items whose URL matches a
triage record); everything else in it is skipped. A realtime-export record wins over a triage record.
"""
import hashlib
import json
import math
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXPORT_DIR = os.environ.get("FNP_RT_EXPORT_DIR") or os.path.join(ROOT, "reports", "realtime", "v1")
TRIAGE_DIR = os.environ.get("FNP_RT_TRIAGE_DIR") or os.path.join(ROOT, "reports", "triage", "v1")
DEFAULT_SIDECAR = os.environ.get("FNP_RT_ENRICH_DB") or os.path.join(HERE, "data", "rt_enrich.sqlite")

VERDICTS = ("FAKE", "PROPAGANDA", "LEGIT", "MISLEADING", "INCONCLUSIVE")
MAX_SPANS = 12
MAX_COUNTRIES = 10
BATCH = 5000
SCHEMA_VERSION = "2"
# the 14 merged technique buckets of FNP's span encoder, in the order they are stored in `enr.tp`
TECHNIQUE_CLASSES = ("Loaded_Language", "Name_Calling,Labeling", "Repetition", "Exaggeration,Minimisation", "Doubt",
                     "Appeal_to_fear-prejudice", "Flag-Waving", "Causal_Oversimplification", "Slogans",
                     "Appeal_to_Authority", "Black-and-White_Fallacy", "Thought-terminating_Cliches",
                     "Whataboutism,Straw_Men,Red_Herring", "Bandwagon,Reductio_ad_hitlerum")
_TC_INDEX = {t: i for i, t in enumerate(TECHNIQUE_CLASSES)}
TECH_BASIS_NOTE = "encoder score (max token probability): uncalibrated, not validated on social posts"
NF_RERANK_FLOOR = 0.72   # FNP writes nearest_factcheck.rerank only when cosine >= 0.72 (meta contract)
PART_RETRY_SECONDS = 120  # a partial load is retried this long after it finished
REASON_KEYS = 20          # distinct reject reasons kept in the load report
# meta.json keys copied into the sidecar (the batch coverage block of /api/rt/stats is built from them)
META_KEYS = ("built_at", "n_items", "n_items_total", "n_in_window", "n_unscored_in_window", "last_item_published_at",
             "next_run_at", "interval_hours", "n_scored_this_run", "n_failed_this_run", "run_seconds", "model_version",
             "technique_classes", "caveat", "analysed_text", "max_items_per_run", "per_run_limit", "max_per_run")

SCHEMA = """
CREATE TABLE IF NOT EXISTS enr (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, p_prop REAL, p_fake REAL, p_misleading REAL, verdict TEXT,
  model_version TEXT, built_at TEXT, src TEXT NOT NULL, extra TEXT,
  too_short INTEGER, n_tokens INTEGER, n_spans INTEGER, n_strong INTEGER, density REAL, tp TEXT,
  fc_id TEXT, fc_cos REAL, fc_rerank REAL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS enr_fc (kb_id TEXT PRIMARY KEY, label TEXT, claim TEXT, url TEXT) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS enr_pfake ON enr(p_fake);
CREATE INDEX IF NOT EXISTS enr_pprop ON enr(p_prop);
CREATE TABLE IF NOT EXISTS enr_tech (id TEXT NOT NULL, technique TEXT NOT NULL, cnt INTEGER NOT NULL, max_conf REAL);
CREATE INDEX IF NOT EXISTS enr_tech_id ON enr_tech(id);
CREATE TABLE IF NOT EXISTS enr_meta (k TEXT PRIMARY KEY, v TEXT);
"""


def _bad(c):
    raise ValueError("non-strict JSON constant %s" % c)


def _num01(v):
    """None or a finite number in [0,1] -> float/None; anything else raises ValueError."""
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1:
        raise ValueError("number outside [0,1]")
    return float(v)


def _kind_of(item_id):
    return "news" if item_id.startswith("news:") else "social"


def _is_real(d):
    """The real FNP row carries technique_probs / scored_at / too_short; the older proposal shape carries `techniques`."""
    return any(k in d for k in ("technique_probs", "scored_at", "too_short", "n_tokens", "strong_techniques")) \
        and "techniques" not in d


def _nonneg_int(v, what):
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError("bad %s" % what)
    return v


def _short_str(v, n):
    return v[:n] if isinstance(v, str) else None


def _clean_countries(c):
    """[{iso3, weight, methods, terms}] -> a bounded list of the same; malformed entries are dropped."""
    out = []
    for x in c[:MAX_COUNTRIES * 2]:
        if not isinstance(x, dict):
            continue
        iso = x.get("iso3")
        if not (isinstance(iso, str) and len(iso) == 3 and iso.isalpha()):
            continue
        w = x.get("weight")
        w = float(w) if isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w) and 0 <= w <= 1 else None
        out.append({"iso3": iso.upper(), "weight": w,
                    "methods": [m[:20] for m in (x.get("methods") or []) if isinstance(m, str)][:6],
                    "terms": [t[:40] for t in (x.get("terms") or []) if isinstance(t, str)][:10]})
        if len(out) >= MAX_COUNTRIES:
            break
    return out


def validate_real(d, iid, report=None):
    """One row in the REAL FNP format -> (id, row dict) or ValueError. `report` (a dict) counts dropped spans/countries."""
    too_short = d.get("too_short", False)
    if too_short is None:
        too_short = False
    if not isinstance(too_short, bool):
        raise ValueError("bad too_short")
    p_prop = None if too_short else _num01(d.get("p_prop"))
    if too_short and d.get("p_prop") is not None:
        _num01(d.get("p_prop"))   # still must be a valid number; it is simply not used
    tp = d.get("technique_probs")
    tp_list = None
    if tp is not None:
        if not isinstance(tp, dict):
            raise ValueError("technique_probs is not an object")
        tp_list = [_num01(tp.get(t)) for t in TECHNIQUE_CLASSES]
        if tp and all(v is None for v in tp_list):
            raise ValueError("technique_probs has no known technique")
    st = d.get("strong_techniques")
    if st is not None and not (isinstance(st, list) and all(isinstance(x, str) for x in st)):
        raise ValueError("bad strong_techniques")
    strong = [x[:80] for x in (st or [])][:len(TECHNIQUE_CLASSES)]
    raw_spans = d.get("spans")
    if raw_spans is not None and not isinstance(raw_spans, list):
        raise ValueError("spans is not a list")
    spans = []
    for s in raw_spans or []:
        ok = (isinstance(s, (list, tuple)) and len(s) >= 4 and all(isinstance(s[i], int) and not isinstance(s[i], bool)
              for i in (0, 1)) and s[0] >= 0 and s[1] > s[0] and isinstance(s[2], str))
        pr = None
        if ok:
            try:
                pr = _num01(s[3])
            except ValueError:
                ok = False
        if not ok:
            if report is not None:
                report["spans_dropped"] = report.get("spans_dropped", 0) + 1
            continue
        spans.append((s[0], s[1], s[2][:80], pr))
    n_spans = _nonneg_int(d.get("n_spans"), "n_spans")
    n_strong = _nonneg_int(d.get("n_strong"), "n_strong")
    n_tokens = _nonneg_int(d.get("n_tokens"), "n_tokens")
    n_spans = len(spans) if n_spans is None else n_spans
    dens = d.get("density")
    if dens is not None and (isinstance(dens, bool) or not isinstance(dens, (int, float)) or not math.isfinite(dens) or dens < 0):
        raise ValueError("bad density")
    # keep the 12 most confident spans, in text order (offsets only; the text is sliced lazily at lookup)
    keep = sorted(sorted(spans, key=lambda x: -(x[3] or 0))[:MAX_SPANS])
    rspans = [[a, b, _TC_INDEX.get(t, t), pr] for a, b, t, pr in keep]
    cnt = {}
    for a, b, t, pr in spans:
        cnt[t] = cnt.get(t, 0) + 1
    techs = [] if too_short else [(t, cnt.get(t, 1), (tp.get(t) if isinstance(tp, dict) and isinstance(tp.get(t), (int, float))
                                                       and not isinstance(tp.get(t), bool) else None)) for t in strong]
    techs = [(t, c, m if m is None or 0 <= m <= 1 else None) for t, c, m in techs]
    c = d.get("countries")
    if c is not None and not isinstance(c, list):
        raise ValueError("countries is not a list")
    countries = _clean_countries(c or [])
    nf = d.get("nearest_factcheck")
    fc, fc_cos, fc_rerank = None, None, None
    if nf is not None:
        if not isinstance(nf, dict):
            raise ValueError("nearest_factcheck is not an object")
        if isinstance(nf.get("kb_id"), str) and nf["kb_id"]:
            fc_cos, fc_rerank = _num01(nf.get("cosine")), _num01(nf.get("rerank"))
            url = nf.get("url") if isinstance(nf.get("url"), str) and nf["url"].lower().startswith(("http://", "https://")) else None
            fc = (nf["kb_id"][:80], _short_str(nf.get("label"), 40), _short_str(nf.get("claim"), 400), url[:600] if url else None)
        elif report is not None:
            report["factchecks_dropped"] = report.get("factchecks_dropped", 0) + 1
    extra = {"fmt": "real", "countries": countries, "rspans": rspans, "strong": strong,
             "lang": d.get("lang") if isinstance(d.get("lang"), str) else None,
             "techniques": [{"technique": t, "count": c2, "max_confidence": m} for t, c2, m in techs]}
    built = d.get("scored_at") if isinstance(d.get("scored_at"), str) else ""
    mv = d.get("model_version") if isinstance(d.get("model_version"), str) else None
    return iid, dict(kind=_kind_of(iid), p_prop=p_prop, p_fake=None, p_misleading=None, verdict=None, model_version=mv,
                     built_at=built, src="realtime_export", extra=json.dumps(extra, separators=(",", ":")),
                     too_short=1 if too_short else 0, n_tokens=n_tokens, n_spans=n_spans, n_strong=n_strong,
                     density=float(dens) if dens is not None else None,
                     tp=json.dumps(tp_list, separators=(",", ":")) if tp_list is not None else None,
                     fc_id=fc[0] if fc else None, fc_cos=fc_cos, fc_rerank=fc_rerank, fc=fc)


def validate_record(d, report=None):
    """One realtime-export line -> (id, row dict) or raises ValueError (the line is dropped and counted)."""
    if not isinstance(d, dict):
        raise ValueError("not an object")
    iid = d.get("id")
    if not isinstance(iid, str) or not iid:
        st, sid = d.get("source_table"), d.get("source_id")
        if not (isinstance(st, str) and st and sid not in (None, "")):
            raise ValueError("no id")
        iid = ("news:%s:%s" % (st, sid)) if d.get("kind") == "news" else ("%s:%s" % (st, sid))
    if _is_real(d):
        return validate_real(d, iid, report)
    p_prop, p_fake, p_mis = _num01(d.get("p_prop")), _num01(d.get("p_fake")), _num01(d.get("p_misleading"))
    verdict = d.get("verdict")
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError("bad verdict")
    if (verdict is None) != (p_fake is None):
        raise ValueError("verdict must be null exactly when p_fake is null")
    techs = []
    for t in d.get("techniques") or []:
        if not isinstance(t, dict) or not isinstance(t.get("technique"), str):
            raise ValueError("bad technique")
        cnt = t.get("count", 1)
        if isinstance(cnt, bool) or not isinstance(cnt, int) or cnt < 1:
            raise ValueError("bad technique count")
        techs.append((t["technique"][:80], cnt, _num01(t.get("max_confidence"))))
    c = d.get("countries")
    countries = None
    if isinstance(c, dict):
        about = [x for x in (c.get("about") or []) if isinstance(x, str)][:30]
        origin = c.get("origin") if isinstance(c.get("origin"), str) else None
        countries = {"about": about, "origin": origin}
    nf = d.get("nearest_factcheck")
    nearest = None
    if isinstance(nf, dict) and isinstance(nf.get("id"), str):
        nearest = {"id": nf["id"], "similarity": _num01(nf.get("similarity"))}
    spans = []
    for s in (d.get("spans") or [])[:MAX_SPANS]:
        if isinstance(s, dict) and isinstance(s.get("text"), str) and isinstance(s.get("technique"), str):
            spans.append({"text": s["text"][:200], "technique": s["technique"][:80],
                          "confidence": _num01(s.get("confidence")),
                          "field": s.get("field") if s.get("field") in ("title", "text") else None})
    extra = {"countries": countries, "nearest_factcheck": nearest, "spans": spans, "lang": d.get("lang")
             if isinstance(d.get("lang"), str) else None,
             "techniques": [{"technique": t, "count": c, "max_confidence": m} for t, c, m in techs]}
    built = d.get("built_at") if isinstance(d.get("built_at"), str) else ""
    mv = d.get("model_version") if isinstance(d.get("model_version"), str) else None
    return iid, dict(kind=_kind_of(iid), p_prop=p_prop, p_fake=p_fake, p_misleading=p_mis, verdict=verdict,
                     model_version=mv, built_at=built, src="realtime_export", extra=json.dumps(extra))


def validate_triage(d):
    """One triage line -> (url, row dict) or ValueError. Triage verdict is {label, basis, ...} or null."""
    if not isinstance(d, dict) or not isinstance(d.get("url"), str) or not d["url"].strip():
        raise ValueError("no url")
    p_prop, p_fake, p_mis = _num01(d.get("p_prop")), _num01(d.get("p_fake")), _num01(d.get("p_misleading"))
    v = d.get("verdict")
    label = v.get("label") if isinstance(v, dict) else None
    label = label if label in VERDICTS else None
    stats = {}
    for s in d.get("spans") or []:
        if isinstance(s, (list, tuple)) and len(s) >= 3 and isinstance(s[2], str):
            n, m = stats.get(s[2], (0, None))
            conf = s[3] if len(s) > 3 and isinstance(s[3], (int, float)) and math.isfinite(s[3]) else None
            stats[s[2]] = (n + 1, conf if m is None or (conf is not None and conf > m) else m)
    techs = [(t[:80], stats.get(t, (1, None))[0], stats.get(t, (1, None))[1])
             for t in (d.get("techniques") or []) if isinstance(t, str)]
    about = [x for x in (d.get("countries") or []) if isinstance(x, str)][:30]
    nf = d.get("nearest_factcheck")
    nearest = None
    if isinstance(nf, dict) and isinstance(nf.get("kb_id"), str):
        cos = nf.get("cos")
        nearest = {"id": nf["kb_id"], "similarity": cos if isinstance(cos, (int, float)) and 0 <= cos <= 1 else None}
    extra = {"countries": {"about": about, "origin": None, "format": "iso3"}, "nearest_factcheck": nearest,
             "spans": [], "lang": d.get("lang") if isinstance(d.get("lang"), str) else None,
             "techniques": [{"technique": t, "count": c, "max_confidence": m} for t, c, m in techs],
             "triage_tier": d.get("tier") if isinstance(d.get("tier"), str) else None,
             "verdict_basis": v.get("basis") if isinstance(v, dict) and isinstance(v.get("basis"), str) else None}
    return d["url"].strip(), dict(
        p_prop=p_prop, p_fake=p_fake, p_misleading=p_mis, verdict=label, model_version=d.get("model_version")
        if isinstance(d.get("model_version"), str) else None,
        built_at=d.get("built_at") if isinstance(d.get("built_at"), str) else "", src="triage",
        extra=json.dumps(extra))


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f, parse_constant=_bad)


def _stat_sig(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size, st.st_ino, st.st_ctime_ns)  # an atomic swap changes the inode
    except OSError:
        return None


def _safe_member(base, rel):
    """Absolute path of `rel` under `base`, or None when it is absolute, escapes base, or is not a .jsonl file."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or ".." in rel.replace("\\", "/").split("/"):
        return None
    if not rel.endswith(".jsonl"):
        return None
    p = os.path.realpath(os.path.join(base, rel))
    return p if p.startswith(os.path.realpath(base) + os.sep) else None


_COLS = ("id", "kind", "p_prop", "p_fake", "p_misleading", "verdict", "model_version", "built_at", "src", "extra",
         "too_short", "n_tokens", "n_spans", "n_strong", "density", "tp", "fc_id", "fc_cos", "fc_rerank")
_UPSERT = ("INSERT INTO enr(%s) VALUES (%s) ON CONFLICT(id) DO UPDATE SET %s WHERE excluded.built_at >= enr.built_at"
           % (", ".join(_COLS), ",".join("?" * len(_COLS)), ", ".join("%s=excluded.%s" % (c, c) for c in _COLS[1:])))


def _insert(con, rows, seen_fc=None):
    """rows: [(id, row)] -> enr (a newer built_at wins; ties go to the later row); fact-check cards go to enr_fc once.
    enr_tech is derived afterwards."""
    con.executemany(_UPSERT, [(i,) + tuple(r.get(c) for c in _COLS[1:]) for i, r in rows])
    fcs = []
    for _i, r in rows:
        f = r.get("fc")
        if f and (seen_fc is None or f[0] not in seen_fc):
            if seen_fc is not None:
                seen_fc.add(f[0])
            fcs.append(f)
    if fcs:
        con.executemany("INSERT OR REPLACE INTO enr_fc(kb_id, label, claim, url) VALUES (?,?,?,?)", fcs)


class Enrich:
    def __init__(self, export_dir=None, triage_dir=None, sidecar=None, store_path=None, background=False):
        self.export_dir = export_dir or EXPORT_DIR
        self.triage_dir = triage_dir or TRIAGE_DIR
        self.sidecar = sidecar or DEFAULT_SIDECAR
        self.store_path = store_path
        self.background = background   # True: a changed meta.json reloads in a thread (requests keep the old sidecar)
        self._lock = threading.Lock()
        self._spawn = threading.Lock()
        self._thread = None
        self._migrated = False
        self._sig = None          # stat signatures of the two meta files at the last check
        self._loaded = None       # (realtime built_at, triage built_at) of the sidecar in use
        self._retry_at = None     # monotonic time at which a partial load is tried again
        self._cov_cache = None
        self._status = dict(present=False, built_at=None, triage_built_at=None, model_version=None, records=0,
                            invalid_lines=0, loaded_at=None, error=None, files=0, partial=False, loading=False,
                            load_report=None)

    # ---- cheap per-request check -------------------------------------------------------------------------------
    def refresh(self, force=False):
        """Reload when a meta.json built_at changed. Never raises; on any problem the previous sidecar stays."""
        sig = (_stat_sig(os.path.join(self.export_dir, "meta.json")),
               _stat_sig(os.path.join(self.triage_dir, "meta.json")), _stat_sig(self.sidecar) is not None)
        retry = self._retry_at is not None and time.monotonic() >= self._retry_at
        if not force and sig == self._sig and not retry:
            return False
        if self.background and not force:
            with self._spawn:
                if self._thread is not None and self._thread.is_alive():
                    return False
                self._status["loading"] = True
                if not os.path.exists(self.sidecar) and self._lock.acquire(blocking=False):
                    try:
                        self._create_empty()   # readers get an empty sidecar at once; the thread fills it
                    finally:
                        self._lock.release()
                self._thread = threading.Thread(target=self._refresh_locked, args=(sig, False), daemon=True,
                                                name="rt-enrich-load")
                self._thread.start()
            return False
        return self._refresh_locked(sig, force)

    def wait(self, timeout=60):
        """Block until a background reload (if any) has finished. For tests and one-off scripts."""
        t = self._thread
        if t is not None:
            t.join(timeout)

    def _refresh_locked(self, sig, force):
        if not self._lock.acquire(blocking=False):
            return False  # another thread is rebuilding; keep serving the old sidecar
        try:
            return self._reload(sig, force)
        except Exception as e:  # noqa: BLE001
            self._status["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
            self._sig = sig
            return False
        finally:
            self._status["loading"] = False
            self._lock.release()

    def _reload(self, sig, force):
        want_rt = self._read_meta(self.export_dir)
        want_tr = self._read_meta(self.triage_dir)
        key = (want_rt["built_at"] if want_rt else None, want_tr["built_at"] if want_tr else None)
        have = self._sidecar_meta()
        retrying = self._retry_at is not None and time.monotonic() >= self._retry_at
        fresh = bool(have) and have.get("schema") == SCHEMA_VERSION and ((have.get("rt") or None), (have.get("tr") or None)) == key
        if not force and not retrying and fresh and self._sidecar_ok():
            self._adopt(have)
            self._sig = (sig[0], sig[1], True)
            return False
        if key == (None, None):
            if have is None:
                self._create_empty()
            self._adopt(self._sidecar_meta())
            self._sig = (sig[0], sig[1], True)
            return False
        if not force and not retrying and self._loaded == key and (have or {}).get("schema") == SCHEMA_VERSION:
            self._sig = sig
            return False
        self._retry_at = None
        stats = self._build(want_rt, want_tr, key)
        self._status["error"] = stats.get("error")
        self._adopt(self._sidecar_meta())
        self._sig = (sig[0], sig[1], True)
        if stats.get("partial"):
            self._retry_at = time.monotonic() + PART_RETRY_SECONDS
        return stats.get("ok", False)

    @staticmethod
    def _read_meta(base):
        try:
            m = _read_json(os.path.join(base, "meta.json"))
        except (OSError, ValueError):
            return None
        if not isinstance(m, dict) or not isinstance(m.get("built_at"), str) or not m["built_at"]:
            return None
        return m

    def _sidecar_ok(self):
        return os.path.exists(self.sidecar)

    def _sidecar_meta(self):
        if not os.path.exists(self.sidecar):
            return None
        try:
            con = sqlite3.connect("file:%s?mode=ro" % self.sidecar, uri=True, timeout=10)
            try:
                d = dict(con.execute("SELECT k, v FROM enr_meta").fetchall())
                d["records"] = con.execute("SELECT count(*) FROM enr").fetchone()[0]
                return d
            finally:
                con.close()
        except sqlite3.Error:
            return None

    def _adopt(self, meta):
        if not meta:
            return
        rt, tr = meta.get("rt") or None, meta.get("tr") or None
        self._loaded = (rt, tr)
        try:
            report = json.loads(meta.get("report") or "null")
        except ValueError:
            report = None
        self._status.update(
            present=bool(rt or tr), built_at=rt, triage_built_at=tr, model_version=meta.get("model_version"),
            records=meta.get("records", 0), invalid_lines=int(meta.get("invalid_lines") or 0),
            files=int(meta.get("files") or 0), loaded_at=meta.get("loaded_at"),
            partial=meta.get("partial") == "1", load_report=report)

    def _create_empty(self):
        os.makedirs(os.path.dirname(self.sidecar) or ".", exist_ok=True)
        tmp = "%s.tmp%d" % (self.sidecar, os.getpid())
        self._unlink(tmp)
        con = sqlite3.connect(tmp)
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO enr_meta VALUES (?,?)", [("rt", ""), ("tr", ""), ("loaded_at", _now()),
                                                              ("schema", SCHEMA_VERSION)])
        con.commit()
        con.close()
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.sidecar)

    def _migrate(self):
        """An older sidecar (before the real-format columns) gets the new columns added in place, once per process, so
        queries never fail while the background rebuild runs. The rebuild then replaces it with a v2 file."""
        if self._migrated or not os.path.exists(self.sidecar):
            return
        self._migrated = True
        try:
            con = sqlite3.connect(self.sidecar, timeout=10)
            try:
                have = {r[1] for r in con.execute("PRAGMA table_info(enr)")}
                if have and "fc_id" not in have:
                    for c, t in (("too_short", "INTEGER"), ("n_tokens", "INTEGER"), ("n_spans", "INTEGER"),
                                 ("n_strong", "INTEGER"), ("density", "REAL"), ("tp", "TEXT"), ("fc_id", "TEXT"),
                                 ("fc_cos", "REAL"), ("fc_rerank", "REAL")):
                        if c not in have:
                            con.execute("ALTER TABLE enr ADD COLUMN %s %s" % (c, t))
                con.execute("CREATE TABLE IF NOT EXISTS enr_fc (kb_id TEXT PRIMARY KEY, label TEXT, claim TEXT, url TEXT)"
                            " WITHOUT ROWID")
                con.commit()
            finally:
                con.close()
        except sqlite3.Error:
            pass

    @staticmethod
    def _unlink(p):
        try:
            os.unlink(p)
        except OSError:
            pass

    # ---- build -------------------------------------------------------------------------------------------------
    def _build(self, meta_rt, meta_tr, key):
        os.makedirs(os.path.dirname(self.sidecar) or ".", exist_ok=True)
        tmp = "%s.tmp%d" % (self.sidecar, os.getpid())
        self._unlink(tmp)
        con = sqlite3.connect(tmp)
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.executescript(SCHEMA)
        t0 = time.monotonic()
        st = dict(invalid=0, files=0, ok=False, error=None, partial=False, report={})
        try:
            if meta_rt:
                self._load_realtime(con, meta_rt, st)
            if meta_tr and not st["error"]:
                self._load_triage(con, st)
            if st["error"]:
                raise ValueError(st["error"])
            con.execute(
                "INSERT INTO enr_tech(id, technique, cnt, max_conf) SELECT e.id, json_extract(j.value, '$.technique'), "
                "coalesce(json_extract(j.value, '$.count'), 1), json_extract(j.value, '$.max_confidence') "
                "FROM enr e, json_each(e.extra, '$.techniques') j")
            model = (meta_rt or meta_tr or {}).get("model_version")
            rep = dict(st["report"], seconds=round(time.monotonic() - t0, 2), loaded_at=_now(),
                       rows=con.execute("SELECT count(*) FROM enr").fetchone()[0],
                       too_short=con.execute("SELECT count(*) FROM enr WHERE too_short = 1").fetchone()[0])
            if meta_rt:
                rep["meta_n_items_total"] = meta_rt.get("n_items_total") if isinstance(meta_rt.get("n_items_total"), int) else None
            batch = {k: meta_rt[k] for k in META_KEYS if meta_rt and k in meta_rt}
            con.executemany("INSERT OR REPLACE INTO enr_meta VALUES (?,?)", [
                ("rt", key[0] or ""), ("tr", key[1] or ""), ("model_version", model if isinstance(model, str) else ""),
                ("invalid_lines", str(st["invalid"])), ("files", str(st["files"])), ("loaded_at", _now()),
                ("schema", SCHEMA_VERSION), ("partial", "1" if st["partial"] else "0"),
                ("report", json.dumps(rep)), ("fnp_meta", json.dumps(batch))])
            con.commit()
            con.close()
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.sidecar)
            st["ok"] = True
        except Exception as e:  # noqa: BLE001  keep the previous sidecar
            st["error"] = st["error"] or "%s: %s" % (type(e).__name__, str(e)[:200])
            try:
                con.close()
            except sqlite3.Error:
                pass
            self._unlink(tmp)
            if not os.path.exists(self.sidecar):
                self._create_empty()
        return st

    @staticmethod
    def _flush(con, rows, seen_fc=None):
        if rows:
            _insert(con, rows, seen_fc)

    @staticmethod
    def _list_parts(base):
        """[(relative name, absolute path)] of the *.jsonl part files under base/news and base/social, in name order.
        Hundreds of parts are fine (one scandir each); *.tmp files and anything that is not a regular file are skipped."""
        out = []
        for sub in ("news", "social"):
            try:
                with os.scandir(os.path.join(base, sub)) as it:
                    for e in it:
                        if e.name.endswith(".jsonl") and not e.name.startswith(".") and e.is_file(follow_symlinks=False):
                            out.append(("%s/%s" % (sub, e.name), e.path))
            except OSError:
                continue
        return sorted(out)

    def _load_realtime(self, con, meta, st):
        files = meta.get("files")
        if isinstance(files, list) and files:
            return self._load_listed(con, files, st)
        return self._load_parts(con, meta, st)

    def _load_listed(self, con, files, st):
        """The older contract: meta.json lists the files (and their sha256); a bad listing or a mismatch keeps the old build."""
        paths = []
        for f in files:
            rel = f.get("path") if isinstance(f, dict) else None
            p = _safe_member(self.export_dir, rel)
            if p is None or not os.path.isfile(p):
                st["error"] = "listed file missing or unsafe: %s" % str(rel)[:80]
                return
            paths.append((rel, p, f.get("sha256") if isinstance(f.get("sha256"), str) else None))
        rep = st["report"] = dict(mode="listed_files", rows_ok=0, rejected=0, reasons={}, files_listed=len(paths))
        seen_fc = set()
        for rel, p, sha in sorted(paths):
            h = hashlib.sha256()
            self._read_part(con, p, rep, st, seen_fc, h)
            if sha and h.hexdigest() != sha.lower():
                st["error"] = "sha256 mismatch for %s" % rel
                return
            st["files"] += 1
        rep["files_read"] = st["files"]

    def _read_part(self, con, path, rep, st, seen_fc, h=None):
        """Stream one part file into the sidecar. Raises OSError when the file is gone or unreadable (caller decides)."""
        rows = []
        with open(path, "rb") as fh:
            for line in fh:
                if h is not None:
                    h.update(line)
                line = line.strip()
                if not line:
                    continue
                try:
                    iid, r = validate_record(json.loads(line.decode("utf-8"), parse_constant=_bad), rep)
                except (ValueError, UnicodeDecodeError, RecursionError) as e:
                    st["invalid"] += 1
                    rep["rejected"] += 1
                    why = (type(e).__name__ if not isinstance(e, ValueError) else str(e))[:60]
                    reasons = rep["reasons"]
                    if why in reasons or len(reasons) < REASON_KEYS:
                        reasons[why] = reasons.get(why, 0) + 1
                    else:
                        reasons["(other)"] = reasons.get("(other)", 0) + 1
                    continue
                rep["rows_ok"] += 1
                rows.append((iid, r))
                if len(rows) >= BATCH:
                    self._flush(con, rows, seen_fc)
                    rows = []
        self._flush(con, rows, seen_fc)

    def _load_parts(self, con, meta, st):
        """The real FNP export: every part_*.jsonl under news/ and social/ (meta.json carries no file list).
        A part that vanishes mid-read (an atomic swap of the export) triggers ONE restart from a fresh listing; if
        parts are still missing the rows read so far are kept and the report says partial."""
        rep = st["report"] = dict(mode="export_parts", rows_ok=0, rejected=0, reasons={}, files_listed=0, files_read=0,
                                  parts_missing=[], restarts=0)
        seen_fc = set()
        for attempt in (1, 2):
            parts = self._list_parts(self.export_dir)
            rep.update(files_listed=len(parts), files_read=0, parts_missing=[], rows_ok=0, rejected=0, reasons={})
            st["invalid"] = 0
            if attempt == 2:
                con.execute("DELETE FROM enr")
                con.execute("DELETE FROM enr_fc")
                seen_fc.clear()
                rep["restarts"] = 1
            if not parts:
                st["error"] = "export has no part files"
                return
            missing = []
            for rel, p in parts:
                try:
                    self._read_part(con, p, rep, st, seen_fc)
                    rep["files_read"] += 1
                except OSError:
                    missing.append(rel)
            if not missing:
                break
            rep["parts_missing"] = missing[:10]
            if attempt == 2:
                st["partial"] = True
        st["files"] = rep["files_read"]
        rep["partial"] = st["partial"]
        # did the export change under us? (the next refresh reloads it; flag it so the report is honest)
        m2 = self._read_meta(self.export_dir)
        rep["meta_changed_during_load"] = bool(m2 and m2.get("built_at") != meta.get("built_at"))
        n_total = meta.get("n_items_total")
        rep["count_matches_meta"] = (rep["rows_ok"] + rep["rejected"] == n_total) if isinstance(n_total, int) else None

    def _load_triage(self, con, st):
        p = os.path.join(self.triage_dir, "items.jsonl")
        if not os.path.isfile(p) or not self.store_path or not os.path.exists(self.store_path):
            return
        by_url = {}
        with open(p, "rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    url, r = validate_triage(json.loads(line.decode("utf-8"), parse_constant=_bad))
                except (ValueError, UnicodeDecodeError, RecursionError):
                    st["invalid"] += 1
                    continue
                by_url[url] = r
        if not by_url:
            return
        store = sqlite3.connect("file:%s?mode=ro" % self.store_path, uri=True, timeout=30)
        try:
            rows = []
            for iid, url in store.execute("SELECT id, url FROM items WHERE url IS NOT NULL"):
                r = by_url.get(url.strip())
                if r is None or con.execute("SELECT 1 FROM enr WHERE id=?", (iid,)).fetchone():
                    continue  # no match, or the realtime export already covers this item
                r = dict(r, kind=_kind_of(iid))
                rows.append((iid, r))
                if len(rows) >= BATCH:
                    self._flush(con, rows)
                    rows = []
        finally:
            store.close()
        self._flush(con, rows)

    # ---- reading -----------------------------------------------------------------------------------------------
    def attach(self, con):
        """ATTACH the sidecar as schema `enr` on an (already query_only) connection. Creates an empty one if needed."""
        self.refresh()
        if not os.path.exists(self.sidecar):
            with self._lock:
                if not os.path.exists(self.sidecar):
                    self._create_empty()
        self._migrate()
        con.execute("ATTACH DATABASE ? AS enr", (self.sidecar,))

    def _ro(self):
        self.refresh()
        if not os.path.exists(self.sidecar):
            return None
        self._migrate()
        con = sqlite3.connect("file:%s?mode=ro" % self.sidecar, uri=True, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _col(row, k):
        try:
            return row[k]
        except (KeyError, IndexError):
            return None

    @staticmethod
    def technique_probs_block(tp_json):
        """The stored 14-float array -> {basis, basis_note, classes:[{technique, probability, basis}]} sorted by probability
        (descending, ties in taxonomy order). Missing buckets are left out, never filled with 0. None when absent."""
        try:
            arr = json.loads(tp_json) if tp_json else None
        except ValueError:
            arr = None
        if not isinstance(arr, list) or len(arr) != len(TECHNIQUE_CLASSES):
            return None
        cls = [dict(technique=t, probability=v, basis="encoder") for t, v in zip(TECHNIQUE_CLASSES, arr) if v is not None]
        cls.sort(key=lambda c: -c["probability"])   # stable: ties keep taxonomy order
        return dict(basis="encoder", basis_note=TECH_BASIS_NOTE, classes=cls)

    @staticmethod
    def resolve_spans(rspans, text, n_spans=None):
        """[[start, end, technique|index, prob]] over `title + "\\n" + text_excerpt` -> [{start, end, technique, prob, text,
        field}] with the text sliced from `text` (the item's title and excerpt as stored locally). Spans outside the
        string, empty after trimming, or straddling the title/excerpt boundary are dropped. `start`/`end` are relative
        to the field named in `field` ('title' or 'text'). Returns (spans, dropped)."""
        out, dropped = [], 0
        if text is None:
            return out, len(rspans or [])
        title, excerpt = text if isinstance(text, tuple) else ((text or "").partition("\n")[0], (text or "").partition("\n")[2])
        title, excerpt = title or "", excerpt or ""
        text = title + "\n" + excerpt
        full_len, tlen = len(text), len(title)
        for s in rspans or []:
            try:
                a, b, t, pr = s[0], s[1], s[2], s[3]
                tech = TECHNIQUE_CLASSES[t] if isinstance(t, int) and 0 <= t < len(TECHNIQUE_CLASSES) else str(t)
            except (IndexError, TypeError, ValueError):
                dropped += 1
                continue
            if not (isinstance(a, int) and isinstance(b, int)) or a < 0 or b <= a or b > full_len:
                dropped += 1
                continue
            if b <= tlen:
                field, off = "title", 0
            elif a > tlen:
                field, off = "text", tlen + 1
            else:
                dropped += 1   # straddles the title/excerpt boundary: not a phrase of either
                continue
            snippet = text[a:b]
            if not snippet.strip():
                dropped += 1
                continue
            out.append(dict(start=a - off, end=b - off, technique=tech, prob=pr, text=snippet[:200], field=field))
        return out, dropped

    @staticmethod
    def inferred_from_row(row, techniques=None, detail=False, **kw):
        """sidecar row (sqlite3.Row/dict with enr columns) -> inferred block, or None when row is None."""
        return _inferred(row, techniques, detail, **kw)

    def techniques_for(self, ids):
        """{item_id: [{technique, count, max_confidence}]} for the given ids."""
        ids = list(ids)
        con = self._ro()
        out = {}
        if con is None or not ids:
            return out
        try:
            for r in con.execute("SELECT id, technique, cnt, max_conf FROM enr_tech WHERE id IN (%s) "
                                 "ORDER BY cnt DESC, technique" % ",".join("?" * len(ids)), ids):
                out.setdefault(r["id"], []).append(dict(technique=r["technique"], count=r["cnt"],
                                                        max_confidence=r["max_conf"]))
        finally:
            con.close()
        return out

    def lookup(self, item_id, text=None, desk=None, country_names=None):
        """Full inferred block for one item id, or None when it has no record ('not analysed').
        `text` is the item's (title, text_excerpt) from the local store (or the joined string): span offsets are resolved against it
        (without it a real-format record returns no spans). `desk` maps a fact-check URL to a Desk item id."""
        con = self._ro()
        if con is None:
            return None
        try:
            r = con.execute("SELECT * FROM enr WHERE id=?", (item_id,)).fetchone()
            if r is None:
                return None
            techs = [dict(technique=t["technique"], count=t["cnt"], max_confidence=t["max_conf"]) for t in con.execute(
                "SELECT technique, cnt, max_conf FROM enr_tech WHERE id=? ORDER BY cnt DESC, technique", (item_id,))]
            fc = None
            if self._col(r, "fc_id"):
                fr = con.execute("SELECT label, claim, url FROM enr_fc WHERE kb_id=?", (r["fc_id"],)).fetchone()
                fc = dict(label=fr["label"], claim=fr["claim"], url=fr["url"]) if fr else dict(label=None, claim=None, url=None)
            return _inferred(r, techs, True, text, fc, desk, country_names)
        finally:
            con.close()

    def batch_meta(self):
        """The parsed meta.json fields of the export in use (copied into the sidecar at load), or None."""
        self.refresh()
        con = self._ro()
        if con is None:
            return None
        try:
            row = con.execute("SELECT v FROM enr_meta WHERE k='fnp_meta'").fetchone()
        except sqlite3.Error:
            return None
        finally:
            con.close()
        try:
            m = json.loads(row[0]) if row and row[0] else None
        except ValueError:
            return None
        return m if isinstance(m, dict) and m else None

    def coverage(self):
        """Analysed counts by kind with verdict counts, straight from the export (independent of the item cache).
        Cached per loaded sidecar: it is a full scan, and status() is called several times per stats request."""
        key = (self._status.get("loaded_at"), self._status.get("records"), self._status.get("built_at"))
        hit = self._cov_cache
        if hit is not None and hit[0] == key and key[0]:
            return hit[1]
        out = self._coverage()
        self._cov_cache = (key, out)
        return out

    def _coverage(self):
        con = self._ro()
        out = dict(analysed=0, by_kind={}, verdicts={}, with_verdict=0, with_p_prop=0, by_source={}, too_short=0)
        if con is None:
            return out
        try:
            for r in con.execute("SELECT kind, count(*), sum(verdict IS NOT NULL), sum(p_prop IS NOT NULL), "
                                 "sum(coalesce(too_short, 0)) FROM enr GROUP BY kind"):
                out["by_kind"][r[0]] = dict(analysed=r[1], with_verdict=r[2] or 0, with_p_prop=r[3] or 0, too_short=r[4] or 0)
                out["analysed"] += r[1]
                out["with_verdict"] += r[2] or 0
                out["with_p_prop"] += r[3] or 0
                out["too_short"] += r[4] or 0
            for r in con.execute("SELECT kind, verdict, count(*) FROM enr WHERE verdict IS NOT NULL "
                                 "GROUP BY kind, verdict"):
                out["by_kind"][r[0]].setdefault("verdicts", {})[r[1]] = r[2]
                out["verdicts"][r[1]] = out["verdicts"].get(r[1], 0) + r[2]
            for r in con.execute("SELECT src, count(*) FROM enr GROUP BY src"):
                out["by_source"][r[0]] = r[1]
        finally:
            con.close()
        return out

    def status(self):
        self.refresh()
        return dict(self._status, export_dir_present=os.path.isdir(self.export_dir),
                    triage_dir_present=os.path.isdir(self.triage_dir), coverage=self.coverage())


def _col(row, k):
    try:
        return row[k]
    except (KeyError, IndexError):
        return None


def _inferred(row, techniques=None, detail=False, text=None, fc=None, desk=None, country_names=None):
    """sidecar row (sqlite3.Row/dict with enr columns) -> inferred block, or None when row is None."""
    if row is None:
        return None
    out = dict(p_prop=row["p_prop"], p_fake=row["p_fake"], p_misleading=row["p_misleading"],
               verdict=row["verdict"], model_version=row["model_version"], built_at=row["built_at"] or None,
               source=row["src"], techniques=techniques or [], too_short=bool(_col(row, "too_short")))
    if not detail:
        out.update(n_spans=_col(row, "n_spans"), n_strong=_col(row, "n_strong"))
        return out
    try:
        ex = json.loads(row["extra"] or "{}")
    except ValueError:
        ex = {}
    real = ex.get("fmt") == "real"
    out.update(lang=ex.get("lang"), triage_tier=ex.get("triage_tier"), verdict_basis=ex.get("verdict_basis"))
    if not real:   # older contract / triage rows: spans carry their own text, nearest_factcheck is {id, similarity}
        out.update(countries=ex.get("countries"), nearest_factcheck=ex.get("nearest_factcheck"),
                   spans=ex.get("spans") or [])
        return out
    spans, dropped = Enrich.resolve_spans(ex.get("rspans"), text)
    out.update(
        n_tokens=_col(row, "n_tokens"), n_spans=_col(row, "n_spans"), n_strong=_col(row, "n_strong"),
        density=_col(row, "density"), strong_techniques=ex.get("strong") or [],
        technique_probs=Enrich.technique_probs_block(_col(row, "tp")),
        spans=spans, spans_dropped=dropped,
        spans_shown_of=_col(row, "n_spans"),
        countries=[dict(c, name=(country_names or {}).get(c.get("iso3"))) for c in ex.get("countries") or []])
    nf = None
    if _col(row, "fc_id"):
        url = (fc or {}).get("url")
        nf = dict(id=row["fc_id"], similarity=_col(row, "fc_cos"), rerank=_col(row, "fc_rerank"),
                  label=(fc or {}).get("label"), claim=(fc or {}).get("claim"), url=url,
                  desk_id=desk(url) if desk and url else None,
                  below_rerank_floor=(_col(row, "fc_cos") is not None and _col(row, "fc_cos") < NF_RERANK_FLOOR))
    out["nearest_factcheck"] = nf
    return out


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":  # quick manual check: python3 Interface/rt_enrich.py
    print(json.dumps(Enrich(store_path=os.path.join(HERE, "data", "realtime.sqlite")).status(), indent=2), file=sys.stdout)
