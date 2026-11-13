#!/usr/bin/env python3
"""FNP dashboard server (stdlib http.server + duckdb). Read-only over project data.

    cd UI && ../.venv/bin/python server.py [port] [--host H] [--port P]   # default 0.0.0.0:8878

Every request re-reads its sources from disk (the pipeline is still running). The only
caches are: units index (extended incrementally by byte offset), pytest result, and the
DuckDB snapshot used when data/fnp.duckdb is write-locked.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import duckdb

HERE = Path(__file__).resolve().parent
STARTED = time.time()
ROOT = HERE.parent
DATA = ROOT / "data"
DB = DATA / "fnp.duckdb"
SNAP = DATA / "cache" / "ui_snapshot.duckdb"
UNITS = DATA / "runs" / "units_v1.jsonl"
EXTERNAL = DATA / "runs" / "external_v1.jsonl"
REPORTS = ROOT / "reports"
N_UNITS_TARGET = 3709
LABELS = ["FAKE", "PROPAGANDA", "LEGIT", "MISLEADING", "INCONCLUSIVE"]
DOC_DIRS = ["docs", "docs/review", "methodologoes"]

# ---------------------------------------------------------------- helpers
def jread(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def tread(p: Path, default=None):
    try:
        return Path(p).read_text(errors="replace")
    except Exception:
        return default


def mtime(p: Path):
    try:
        return Path(p).stat().st_mtime
    except Exception:
        return None


def argmax(d):
    if not d:
        return None
    try:
        return max(d, key=lambda k: d[k])
    except Exception:
        return None


def count_lines(p: Path) -> int:
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


# ---------------------------------------------------------------- duckdb
_snap_lock = threading.Lock()


def db_connect():
    """Read-only handle on fnp.duckdb; on lock, a <=5 min old snapshot copy. Never writes project data."""
    if not DB.exists():
        raise FileNotFoundError("data/fnp.duckdb missing")
    try:
        return duckdb.connect(str(DB), read_only=True), "live"
    except duckdb.IOException:
        pass
    with _snap_lock:
        if not SNAP.exists() or time.time() - SNAP.stat().st_mtime > 300:
            SNAP.parent.mkdir(parents=True, exist_ok=True)
            tmp = SNAP.with_suffix(".tmp.duckdb")
            shutil.copyfile(DB, tmp)
            wal = DB.with_name(DB.name + ".wal")
            tmpwal = tmp.with_name(tmp.name + ".wal")
            if wal.exists():
                shutil.copyfile(wal, tmpwal)
            elif tmpwal.exists():
                tmpwal.unlink()
            # validate the copy before swapping (a torn copy must not replace a good snapshot)
            try:
                c = duckdb.connect(str(tmp), read_only=True)
                c.execute("show tables").fetchall()
                c.close()
                for suf in ("", ".wal"):
                    src = tmp.with_name(tmp.name + suf)
                    dst = SNAP.with_name(SNAP.name + suf)
                    if src.exists():
                        src.replace(dst)
                    elif dst.exists():
                        dst.unlink()
            except Exception:
                for suf in ("", ".wal"):
                    tmp.with_name(tmp.name + suf).unlink(missing_ok=True)
                if not SNAP.exists():
                    raise
    return duckdb.connect(str(SNAP), read_only=True), "snapshot"


def db_query(sql, params=()):
    con, mode = db_connect()
    try:
        cur = con.execute(sql, list(params))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()], mode
    finally:
        con.close()


# ---------------------------------------------------------------- units index
_idx_lock = threading.Lock()
_idx = {"rows": [], "pos": 0, "by_id": {}}


def _summ(r, off):
    c = r.get("claim") or {}
    f = r.get("features") or {}
    adj = r.get("gemma_adj") or {}
    sil = r.get("gemma_silver") or {}
    rule = r.get("rule") or {}
    return {
        "unit_id": r.get("unit_id"), "item_id": r.get("item_id"), "dataset": r.get("dataset"),
        "split": r.get("split"), "gold": r.get("label5"), "claim": c.get("claim", ""),
        "rule": argmax(rule), "adj": adj.get("label"), "adj_ok": adj.get("ok"),
        "raw": sil.get("label"),
        "p_true": f.get("p_true"), "p_true_lo": f.get("p_true_lo"), "p_true_hi": f.get("p_true_hi"),
        "ds_uncertainty": f.get("ds_uncertainty"), "n_eff": f.get("n_eff"),
        "kb_match": f.get("kb_match"), "p_prop": f.get("p_prop"),
        "n_evidence": len(r.get("evidence") or []), "_off": off,
    }


def units_index():
    """Incrementally extend the lightweight index with newly appended complete lines."""
    with _idx_lock:
        try:
            size = UNITS.stat().st_size
        except Exception:
            _idx.update(rows=[], pos=0, by_id={})
            return _idx["rows"]
        if size < _idx["pos"]:
            _idx.update(rows=[], pos=0, by_id={})
        if size > _idx["pos"]:
            with open(UNITS, "rb") as f:
                f.seek(_idx["pos"])
                off = _idx["pos"]
                for line in f:
                    if not line.endswith(b"\n"):
                        break  # partial write in progress
                    try:
                        r = json.loads(line)
                        s = _summ(r, off)
                        _idx["by_id"][s["unit_id"]] = len(_idx["rows"])
                        _idx["rows"].append(s)
                    except Exception:
                        pass
                    off += len(line)
                _idx["pos"] = off
        return _idx["rows"]


def unit_full(uid):
    units_index()
    i = _idx["by_id"].get(uid)
    if i is None:
        return None
    off = _idx["rows"][i]["_off"]
    with open(UNITS, "rb") as f:
        f.seek(off)
        return json.loads(f.readline())


def public(s):
    return {k: v for k, v in s.items() if k != "_off"}


# ---------------------------------------------------------------- endpoints
_tests = {"ts": 0, "res": None}
_tests_lock = threading.Lock()


def run_tests():
    with _tests_lock:
        if _tests["res"] and time.time() - _tests["ts"] < 300:
            return _tests["res"]
        t0 = time.time()
        try:
            p = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "tests", "-q", "-x", "--no-header",
                                "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, timeout=120)
            tail = (p.stdout or "").strip().splitlines()[-8:]
            res = {"ran": True, "ok": p.returncode == 0, "returncode": p.returncode,
                   "summary": tail[-1] if tail else "", "tail": tail}
        except subprocess.TimeoutExpired:
            res = {"ran": True, "ok": False, "summary": "timeout after 120 s", "tail": []}
        except Exception as e:
            res = {"ran": True, "ok": False, "summary": f"{type(e).__name__}: {e}", "tail": []}
        res["secs"] = round(time.time() - t0, 1)
        res["at"] = time.time()
        _tests.update(ts=time.time(), res=res)
        return res


def module_inventory():
    out = []
    for p in sorted((ROOT / "fnp").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        doc = ""
        try:
            import ast
            d = ast.get_docstring(ast.parse(p.read_text(errors="replace"))) or ""
            doc = d.strip().splitlines()[0] if d.strip() else ""
        except Exception:
            pass
        out.append({"path": str(p.relative_to(ROOT)), "doc": doc,
                    "lines": count_lines(p)})
    for p in sorted((ROOT / "scripts").glob("*.py")):
        doc = ""
        try:
            import ast
            d = ast.get_docstring(ast.parse(p.read_text(errors="replace"))) or ""
            doc = d.strip().splitlines()[0] if d.strip() else ""
        except Exception:
            pass
        out.append({"path": str(p.relative_to(ROOT)), "doc": doc, "lines": count_lines(p)})
    return out


def laya_cards():
    cards = []
    for p in sorted(glob.glob(str(DATA / "models" / "laya_fnp_v*"))):
        p = Path(p)
        card = jread(p / "training_card.json")
        rl = jread(p / "rl_agent_config.json")
        cards.append({"name": p.name, "card": card, "rl_agent": rl,
                      "has_weights": (p / "model.safetensors").exists(),
                      "mtime": mtime(p / "training_card.json")})
    return cards


def stage(id_, name, status, detail="", count=None, target=None, extra=None):
    return {"id": id_, "name": name, "status": status, "detail": detail, "count": count,
            "target": target, **(extra or {})}


def api_overview(q):
    stages = []
    dsinfo = {"mode": None}
    # datasets
    try:
        rows, mode = db_query("select 'gold_items_v1' t, count(*) n from gold_items_v1 union all "
                              "select 'ptc_articles_v1', count(*) from ptc_articles_v1 union all "
                              "select 'ptc_spans_v1', count(*) from ptc_spans_v1")
        tabs = {r["t"]: r["n"] for r in rows}
        try:
            tabs["gemma_cache"] = db_query("select count(*) n from gemma_cache")[0][0]["n"]
        except Exception:
            pass
        dsinfo = {"mode": mode, "tables": tabs}
        stages.append(stage("data", "Datasets (gold items + PTC)", "done" if tabs.get("gold_items_v1") else "pending",
                            f"{tabs.get('gold_items_v1', 0)} gold items, {tabs.get('ptc_articles_v1', 0)} PTC articles, "
                            f"{tabs.get('ptc_spans_v1', 0)} PTC spans", tabs.get("gold_items_v1")))
    except Exception as e:
        stages.append(stage("data", "Datasets (gold items + PTC)", "error", f"{type(e).__name__}: {e}"))
    # KB
    kbp = DATA / "models" / "kb_v1" / "meta.parquet"
    if kbp.exists():
        try:
            con = duckdb.connect()
            n = con.execute("select count(*) from read_parquet(?)", [str(kbp)]).fetchone()[0]
            stages.append(stage("kb", "Knowledge base (embeddings)", "done", f"{n} fact-check entries", n))
        except Exception as e:
            stages.append(stage("kb", "Knowledge base (embeddings)", "error", str(e)))
    else:
        stages.append(stage("kb", "Knowledge base (embeddings)", "pending", "meta.parquet missing"))
    # units
    n = len(units_index())
    st = "done" if n >= N_UNITS_TARGET else ("running" if n > 0 else "pending")
    if UNITS.exists() and st == "running" and time.time() - (mtime(UNITS) or 0) > 900:
        st = "stalled"
    stages.append(stage("units", "Units run (L1-L6 + Gemma)", st, f"{n} / {N_UNITS_TARGET} units",
                        n, N_UNITS_TARGET, {"updated": mtime(UNITS)}))
    # eval
    ev = REPORTS / "eval_v1.json"
    em = REPORTS / "EVAL_v1.md"
    stages.append(stage("eval", "Evaluation", "done" if ev.exists() or em.exists() else "pending",
                        "reports/EVAL_v1.md present" if em.exists() else "reports/EVAL_v1.md absent",
                        extra={"updated": mtime(em) or mtime(ev)}))
    # laya
    cards = laya_cards()
    lstat = "done" if any(c["card"] for c in cards) else "pending"
    stages.append(stage("laya", "Laya models", lstat,
                        ", ".join(f"{c['name']} (stage {((c['card'] or {}).get('stage'))})" for c in cards) or "none",
                        len(cards)))
    # external
    ne = count_lines(EXTERNAL) if EXTERNAL.exists() else 0
    nitems = sum(count_lines(p) for p in glob.glob(str(ROOT / "Interface" / "factchecks" / "*.jsonl")))
    stages.append(stage("external", "External scoring", "running" if ne else "pending",
                        f"{ne} external docs scored", ne, None, {"updated": mtime(EXTERNAL), "factcheck_items": nitems}))
    out = {"stages": stages, "db": dsinfo, "now": time.time(),
           "modules": module_inventory(),
           "tests": run_tests() if q.get("tests") == "1" else (_tests["res"] or {"ran": False}),
           "logs": [{"name": Path(p).name, "mtime": mtime(Path(p))} for p in sorted(glob.glob(str(ROOT / "logs" / "*.log")))]}
    return out


def api_datasets(q):
    out = {"counts": [], "factchecks": [], "ptc": {}}
    try:
        rows, mode = db_query("select dataset, split, label5, count(*) n from gold_items_v1 group by 1,2,3 order by 1,2,3")
        out["counts"] = rows
        out["mode"] = mode
        try:
            out["ptc"]["by_technique"] = db_query(
                "select technique, count(*) n from ptc_spans_v1 group by 1 order by 2 desc")[0]
            out["ptc"]["articles_by_split"] = db_query("select split, count(*) n from ptc_articles_v1 group by 1")[0]
        except Exception:
            pass
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    for p in sorted(glob.glob(str(ROOT / "Interface" / "factchecks" / "*.jsonl"))):
        site = Path(p).stem
        by = {}
        n = 0
        with open(p, errors="replace") as f:
            for line in f:
                n += 1
                try:
                    v = json.loads(line).get("verdict_norm") or "?"
                except Exception:
                    v = "?"
                by[v] = by.get(v, 0) + 1
        out["factchecks"].append({"site": site, "n": n, "by_verdict": by})
    return out


def api_units(q):
    rows = units_index()
    split, ds, lab, pred = q.get("split"), q.get("dataset"), q.get("label"), q.get("pred")
    text = (q.get("q") or "").lower()
    off = int(q.get("offset") or 0)
    lim = min(int(q.get("limit") or 50), 500)
    sel = []
    for r in rows:
        if split and r["split"] != split: continue
        if ds and r["dataset"] != ds: continue
        if lab and r["gold"] != lab: continue
        if pred and r["adj"] != pred: continue
        if text and text not in (r["claim"] or "").lower() and text not in (r["unit_id"] or "") and text not in (r["item_id"] or ""):
            continue
        sel.append(r)
    facets = {"dataset": sorted({r["dataset"] for r in rows if r["dataset"]}),
              "split": sorted({r["split"] for r in rows if r["split"]}), "labels": LABELS}
    return {"total": len(sel), "all": len(rows), "target": N_UNITS_TARGET, "offset": off, "limit": lim,
            "facets": facets, "rows": [public(r) for r in sel[off:off + lim]]}


def api_unit(uid):
    r = unit_full(uid)
    if r is None:
        return None
    r["rule_argmax"] = argmax(r.get("rule"))
    return r


def confusion(rows, key):
    m = {g: {p: 0 for p in LABELS} for g in LABELS}
    for r in rows:
        g, p = r["gold"], r[key]
        if g in m and p in m[g]:
            m[g][p] += 1
    return m


def prf(m):
    out = {}
    n = sum(sum(v.values()) for v in m.values())
    for c in LABELS:
        tp = m[c][c]
        fp = sum(m[g][c] for g in LABELS if g != c)
        fn = sum(m[c][p] for p in LABELS if p != c)
        out[c] = {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
                  "precision": tp / (tp + fp) if tp + fp else None,
                  "recall": tp / (tp + fn) if tp + fn else None,
                  "p_ci": wilson(tp, tp + fp), "r_ci": wilson(tp, tp + fn)}
    acc = sum(m[c][c] for c in LABELS)
    return {"per_class": out, "n": n, "accuracy": acc / n if n else None, "acc_ci": wilson(acc, n)}


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [max(0, c - h), min(1, c + h)]



def _finite(o):
    """Replace NaN/Infinity (invalid JSON for browsers) with None, recursively."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o

def derived_eval(rows):
    out = {"n": len(rows), "methods": {}, "by_dataset": {}, "by_kb_match": {}}
    for key, name in (("rule", "rule head"), ("adj", "Gemma adjudicator"), ("raw", "Gemma raw")):
        m = confusion(rows, key)
        out["methods"][key] = {"name": name, "confusion": m, **prf(m)}
    def acc(rs):
        d = {}
        for key in ("rule", "adj", "raw"):
            n = len(rs)
            k = sum(1 for r in rs if r["gold"] == r[key])
            d[key] = {"acc": k / n if n else None, "ci": wilson(k, n)}
        d["n"] = len(rs)
        return d
    for ds in sorted({r["dataset"] for r in rows}):
        out["by_dataset"][ds] = acc([r for r in rows if r["dataset"] == ds])
    out["by_kb_match"]["kb_match>0"] = acc([r for r in rows if (r["kb_match"] or 0) > 0])
    out["by_kb_match"]["kb_match=0"] = acc([r for r in rows if not (r["kb_match"] or 0) > 0])
    return out


def api_eval(q):
    rows = units_index()
    sp = q.get("split")
    if sp:
        rows = [r for r in rows if r["split"] == sp]
    return {"eval_json": jread(REPORTS / "eval_v1.json"), "markdown": tread(REPORTS / "EVAL_v1.md"),
            "derived": derived_eval(rows), "splits": sorted({r["split"] for r in units_index() if r["split"]}),
            "partial": len(units_index()) < N_UNITS_TARGET}


def _load_docs_verdicts():
    out = []
    seen = set()
    if EXTERNAL.exists():
        with open(EXTERNAL, errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                d = r.get("doc") or {}
                pr = d.get("probs") or {}
                out.append({"id": r.get("id"), "dataset": r.get("dataset"), "label": d.get("label"),
                            "plausibly_fake": bool(d.get("plausibly_fake")), "probs": pr,
                            "score": (pr.get("FAKE") or 0) + (pr.get("MISLEADING") or 0),
                            "decider": d.get("decider"), "rationale": d.get("rationale", ""),
                            "claims": r.get("claims") or [], "spans_summary": r.get("spans_summary"),
                            "src": "external_v1"})
                seen.add(r.get("id"))
    p = REPORTS / "verdict_v1_docs.jsonl"
    if p.exists():
        with open(p, errors="replace") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("doc_id") in seen:
                    continue
                pr = r.get("probs")
                if isinstance(pr, str):
                    try: pr = json.loads(pr)
                    except Exception: pr = {}
                pr = pr or {}
                out.append({"id": r.get("doc_id"), "dataset": r.get("dataset"), "label": r.get("label"),
                            "plausibly_fake": bool(r.get("plausibly_fake")), "probs": pr,
                            "score": (pr.get("FAKE") or 0) + (pr.get("MISLEADING") or 0),
                            "decider": r.get("decider"), "rationale": r.get("rationale", ""),
                            "claims": [], "spans_summary": None, "src": "verdict_v1_docs"})
    return out


def api_verdicts(q):
    rows = _load_docs_verdicts()
    ds, lab, text = q.get("dataset"), q.get("label"), (q.get("q") or "").lower()
    pf = q.get("plausibly_fake")
    facets = {"dataset": sorted({r["dataset"] for r in rows if r["dataset"]}),
              "labels": sorted({r["label"] for r in rows if r["label"]})}
    sel = [r for r in rows if (not ds or r["dataset"] == ds) and (not lab or r["label"] == lab)
           and (pf != "1" or r["plausibly_fake"]) and (not text or text in (r["id"] or "").lower()
                                                       or any(text in (c.get("claim") or "").lower() for c in r["claims"]))]
    sel.sort(key=lambda r: -r["score"])
    off, lim = int(q.get("offset") or 0), min(int(q.get("limit") or 50), 500)
    # attach item text for external docs (from Interface/factchecks) lazily for the page only
    page = sel[off:off + lim]
    return {"total": len(sel), "all": len(rows), "offset": off, "limit": lim, "facets": facets, "rows": page,
            "pending": not rows}


def doc_list():
    out = []
    for d in DOC_DIRS:
        for p in sorted((ROOT / d).glob("*.md")):
            out.append({"path": str(p.relative_to(ROOT)), "dir": d, "name": p.name,
                        "size": p.stat().st_size, "mtime": p.stat().st_mtime})
    return out


def api_docs(q):
    if q.get("path"):
        rel = q["path"]
        ok = {d["path"] for d in doc_list()} | {"reports/EVAL_v1.md"}
        if rel not in ok:
            return None
        return {"path": rel, "text": tread(ROOT / rel, "")}
    return {"docs": doc_list()}


def api_laya(q):
    bandit = []
    for pat in ("data/models/**/*bandit*.json", "data/runs/*bandit*", "reports/*bandit*", "reports/*rl*"):
        for p in glob.glob(str(ROOT / pat), recursive=True):
            pp = Path(p)
            if pp.is_file():
                bandit.append({"path": str(pp.relative_to(ROOT)), "data": jread(pp) if pp.suffix == ".json" else None,
                               "text": (tread(pp, "")[:3000] if pp.suffix in (".md", ".txt") else None)})
    return {"cards": laya_cards(), "bandit": bandit,
            "rl_modules": [str(p.relative_to(ROOT)) for p in sorted((ROOT / "fnp" / "rl").glob("*.py"))]}


def api_logs(q):
    n = min(int(q.get("lines") or 120), 1000)
    out = []
    for p in sorted(glob.glob(str(ROOT / "logs" / "*.log"))):
        p = Path(p)
        try:
            size = p.stat().st_size
            with open(p, "rb") as f:
                f.seek(max(0, size - 60000))
                data = f.read().decode("utf-8", "replace")
            lines = [l.rstrip("\r") for l in re.split(r"\n", data)][-n:]
            # collapse carriage-return progress bars
            lines = [l.split("\r")[-1] for l in lines]
        except Exception as e:
            lines = [f"<error {e}>"]
        out.append({"name": p.name, "size": size, "mtime": p.stat().st_mtime, "tail": lines})
    return {"logs": out}


_pred_lock = threading.Lock()
_deciders = {}


def api_predict(body):
    text = (body.get("text") or "").strip()
    if not text:
        return {"error": "empty text"}, 400
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    with _pred_lock:
        try:
            from fnp.pipeline import analyse_text, load_deciders
            if "d" not in _deciders:
                _deciders["d"] = load_deciders()
            t0 = time.time()
            res = analyse_text("ui:" + str(int(t0)), text, body.get("lang") or None, _deciders["d"])
            res["secs"] = round(time.time() - t0, 1)
            return _finite(json.loads(json.dumps(res, default=str))), 200
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}, 500


# ---------------------------------------------------------------- live gallery
LIVE = DATA / "runs" / "live_v2.jsonl"   # live_v1 is quarantined (data/runs/quarantine/README.md)
LIVE_STATE = DATA / "runs" / "live_state.json"
FLAGS = DATA / "feedback" / "flags_v1.jsonl"
FLAG_LABELS = ("FAKE", "PROPAGANDA", "LEGIT", "MISLEADING", "INCONCLUSIVE")
_flag_lock = threading.Lock()


def _latest_flags():
    out = {}
    for line in (FLAGS.read_text().splitlines() if FLAGS.exists() else []):
        try:
            r = json.loads(line)
            out[r["id"]] = r
        except Exception:
            pass
    return out


def _is_suspect(d):
    return d.get("label") in ("FAKE", "MISLEADING", "PROPAGANDA") or bool(d.get("plausibly_fake"))


def api_gallery(q):
    limit = min(int(q.get("limit", 60)), 300)
    show, src, lab = q.get("show", "flagged"), q.get("source", ""), q.get("label", "")
    flags = _latest_flags()
    rows, n_scored, n_total = [], 0, 0
    for line in (LIVE.read_text().splitlines() if LIVE.exists() else []):
        try:
            r = json.loads(line)
        except Exception:
            continue
        n_total += 1
        if r.get("status") != "scored":
            continue
        n_scored += 1
        d = r["doc"]
        fl = flags.get(r["id"])
        unfl = bool(fl and fl["action"] in ("unflag", "relabel") )
        if show == "flagged" and (not _is_suspect(d) or unfl):
            continue
        if show == "unflagged" and not unfl:
            continue
        if src and r["source"] != src:
            continue
        if lab and d.get("label") != lab:
            continue
        pr = d.get("probs", {})
        rows.append({"id": r["id"], "source": r["source"], "platform": r.get("platform"), "url": r.get("url"),
                     "text": r["text"], "posted_at": r.get("posted_at"), "scanned_at": r.get("scanned_at"),
                     "label": d.get("label"), "plausibly_fake": d.get("plausibly_fake"), "probs": pr,
                     "suspicion": (pr.get("FAKE", 0) + pr.get("MISLEADING", 0)), "kb_cos": r.get("kb_cos"),
                     "rationale": d.get("rationale"), "spans": r.get("spans_summary"),
                     "claims": [{"claim": (c["claim"].get("claim") if isinstance(c["claim"], dict) else c["claim"]), "label": c["verdict"].get("label"),
                                 "probs": c["verdict"].get("probs")} for c in r.get("claims", [])][:6],
                     "flag": fl})
    rows.sort(key=lambda x: x.get("posted_at") or "", reverse=True)
    st = jread(LIVE_STATE, {}) or {}
    return {"items": rows[:limit], "matched": len(rows), "scored": n_scored, "total_seen": n_total,
            "n_flags": len(flags), "state": st, "scanner_alive": (time.time() - (mtime(LIVE) or 0)) < 900}


# ---------------------------------------------------------------- engine: claim check v1
# Contract for the Evidence Desk "Claim Check" screen (docs/CHECK_V1_CONTRACT.md).
CHECK_MAX_CONCURRENT = 2
CHECK_TIMEOUT_S = 55
CHECK_MAX_CHARS = 20_000
LAYA_PATH = DATA / "models" / "laya_fnp_rl_v1"
_check_sem = threading.BoundedSemaphore(CHECK_MAX_CONCURRENT)
_check_pool = None
_check_models = {}
_check_models_lock = threading.Lock()


def _check_load():
    with _check_models_lock:
        if "d" not in _check_models:
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from fnp.pipeline import load_deciders
            from fnp.laya.predict import LayaDecider
            d = load_deciders()
            d["laya"] = LayaDecider.load(str(LAYA_PATH), device="cpu")   # CPU: GPU is shared with Gemma
            _check_models["d"] = d
        return _check_models["d"]


def _check_run(text, lang):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fnp.pipeline import analyse_text
    from fnp.llm.propaganda_enc import analyse_enc
    from fnp.triage import fake_call
    d = _check_load()
    t0 = time.time()
    res = analyse_text("check:" + str(int(t0 * 1000)), text, lang, d)
    cl = res["claims"]
    lay = d["laya"].predict([c["state_text"] for c in cl]) if cl else []
    laya_prop = max((x["noul"].get("is_propaganda") for x in lay if "is_propaganda" in x["noul"]), default=None)
    # Laya's verdict for the most fake-looking claim (reported alongside; the verdict itself is the
    # calibrated stacker until stacker_laya is refit on out-of-fold Laya probabilities -- P1).
    laya_doc = max(lay, key=lambda x: x["probs"].get("FAKE", 0) + x["probs"].get("MISLEADING", 0), default=None)
    doc = res["doc"]
    fc = fake_call([{"stacker": c["verdict"].get("probs"),
                     "gemma": c["gemma_adj"].get("probs") if c["gemma_adj"].get("ok") else None} for c in cl])
    gem_ok = bool(cl) and any(c["gemma_adj"].get("ok") for c in cl)
    enc = analyse_enc("check", text[:6000])
    tech = [{"technique": x["technique"], "start": x["start"], "end": x["end"], "text": x["text"],
             "confidence": round(x["confidence"], 4), "strong": x["strong"]} for x in enc["spans"]]
    reason = "no_claims" if not cl else (None if gem_ok else "gemma_error")
    return {
        "verdict": {"label": doc["label"], "probs": doc["probs"], "plausibly_fake": bool(doc.get("plausibly_fake")),
                    "prediction_set": doc.get("prediction_set", []),
                    "rationale": ((cl[0]["verdict"].get("rationale") if cl else "") or doc.get("rationale", ""))[:600],
                    "decider": "stacker_laya" if d.get("stacker_laya") is not None else "stacker"},
        "fake_call": fc,
        "claims": [{"claim": (c["claim"].get("claim") if isinstance(c["claim"], dict) else c["claim"]),
                    "label": c["verdict"]["label"], "probs": c["verdict"]["probs"],
                    "laya_probs": lay[i]["probs"] if i < len(lay) else None,
                    "evidence": [{"evidence_id": e.get("evidence_id"), "stance": e.get("stance"),
                                  "source": e.get("source"), "url": e.get("url"),
                                  "kb_claim": (e.get("kb_claim") or e.get("text") or "")[:300]}
                                 for e in c.get("evidence", [])[:5]]}
                   for i, c in enumerate(cl)],
        "laya": ({"label": laya_doc["label"], "probs": laya_doc["probs"], "confidence": laya_doc["confidence"]}
                 if laya_doc else None),
        "laya_reason": None if laya_doc else "no_claims",
        "propaganda": {"is_propaganda_p": None,       # no calibrated document-level propaganda probability yet
                       "laya_is_propaganda_raw": round(laya_prop, 4) if laya_prop is not None else None,
                       "techniques": tech, "technique_probs": enc["technique_probs"], "p_prop": enc["p_prop"],
                       "too_short": enc["too_short"], "n_tokens": enc["n_tokens"],
                       "one_sided": (res.get("spans_summary") or {}).get("one_sided")},
        "degraded": reason == "gemma_error", "degraded_reason": reason,
        "model_versions": {"laya": LAYA_PATH.name, "deciders": "deciders_v1", "span_encoder": "prop_span_v1",
                           "kb": os.environ.get("FNP_KB", "kb_v2"), "fake_call": "fake_call_v1"},
        "pipeline_s": round(time.time() - t0, 2)}


def api_check_v1(body):
    global _check_pool
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"error": "need non-empty text"}, 400, {}
    if len(text) > CHECK_MAX_CHARS:
        return {"error": f"text longer than {CHECK_MAX_CHARS} chars"}, 413, {}
    lang = body.get("lang") if isinstance(body.get("lang"), str) else None
    q0 = time.time()
    if not _check_sem.acquire(timeout=2):
        return {"error": "busy", "detail": f"{CHECK_MAX_CONCURRENT} checks already running"}, 503, {"Retry-After": "30"}
    wait = round(time.time() - q0, 2)
    with _check_models_lock:
        if _check_pool is None:
            from concurrent.futures import ThreadPoolExecutor
            _check_pool = ThreadPoolExecutor(CHECK_MAX_CONCURRENT)
    fut = _check_pool.submit(_check_run, text.strip(), lang)
    fut.add_done_callback(lambda f: _check_sem.release())     # slot frees only when the work really ends
    try:
        out = fut.result(timeout=CHECK_TIMEOUT_S)
        out.update(secs=round(time.time() - q0, 2), queue_wait_s=wait)
        return _finite(json.loads(json.dumps(out, default=str))), 200, {}
    except TimeoutError:
        try:
            from fnp.llm.propaganda_enc import analyse_enc
            enc = analyse_enc("check", text[:6000])
            tech = [{"technique": x["technique"], "start": x["start"], "end": x["end"], "text": x["text"],
                     "confidence": round(x["confidence"], 4), "strong": x["strong"]} for x in enc["spans"]]
            extra = {k: enc[k] for k in ("technique_probs", "p_prop", "too_short", "n_tokens")}
        except Exception:
            tech, extra = [], {}
        return {"verdict": None, "fake_call": None, "claims": [], "laya": None, "laya_reason": "timeout",
                "propaganda": {"is_propaganda_p": None, "laya_is_propaganda_raw": None, "techniques": tech,
                               "one_sided": None, **extra},
                "degraded": True, "degraded_reason": "timeout", "error": f"verdict timed out after {CHECK_TIMEOUT_S}s (Gemma busy)",
                "secs": round(time.time() - q0, 2), "queue_wait_s": wait}, 200, {}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300]}, 500, {}


PROP_FLAGS = DATA / "feedback" / "prop_flags_v1.jsonl"


def api_prop_flag(body):
    """Propaganda Lab feedback (append-only; scripts/32_prop_feedback_rows.py builds the gold set)."""
    i, act = body.get("post_uid"), body.get("action")
    if not i or not isinstance(i, str) or len(i) > 120 or act not in ("not_propaganda", "confirm", "clear"):
        return {"error": "need post_uid and action in not_propaganda|confirm|clear"}, 400
    tech = body.get("technique")
    rec = {"post_uid": i, "action": act, "technique": tech if isinstance(tech, str) else None,
           "note": str(body.get("note") or "")[:300], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with _flag_lock:
        PROP_FLAGS.parent.mkdir(parents=True, exist_ok=True)
        with PROP_FLAGS.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, "flag": rec}, 200


def api_flag(body):
    i, act = body.get("id"), body.get("action")
    if not i or act not in ("unflag", "confirm", "relabel", "clear"):
        return {"error": "need id and action in unflag|confirm|relabel|clear"}, 400
    lab = body.get("label")
    if act == "relabel" and lab not in FLAG_LABELS:
        return {"error": "relabel needs a valid label"}, 400
    rec = {"id": i, "action": act, "label": lab if act == "relabel" else None,
           "note": (body.get("note") or "")[:300], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with _flag_lock:
        FLAGS.parent.mkdir(parents=True, exist_ok=True)
        with FLAGS.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, "flag": rec}, 200


# ---------------------------------------------------------------- http
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, code=200, headers=None):
        b = json.dumps(_finite(json.loads(json.dumps(obj, default=str))), ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/healthz":                          # liveness for systemd/monitoring; no data
            return self._json({"ok": True, "service": "fnp-dashboard", "uptime_s": round(time.time() - STARTED)})
        if not u.path.startswith("/api/"):
            if u.path in ("", "/"):
                self.path = "/index.html"
            return super().do_GET()
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            p = u.path.rstrip("/")
            if p == "/api/overview": return self._json(api_overview(q))
            if p == "/api/datasets": return self._json(api_datasets(q))
            if p == "/api/units": return self._json(api_units(q))
            if p.startswith("/api/unit/"):
                r = api_unit(unquote(p[len("/api/unit/"):]))
                return self._json(r) if r else self._json({"error": "not found"}, 404)
            if p == "/api/eval": return self._json(api_eval(q))
            if p == "/api/verdicts": return self._json(api_verdicts(q))
            if p == "/api/docs":
                r = api_docs(q)
                return self._json(r) if r is not None else self._json({"error": "not found"}, 404)
            if p == "/api/gallery": return self._json(api_gallery(q))
            if p == "/api/laya": return self._json(api_laya(q))
            if p == "/api/logs": return self._json(api_logs(q))
            return self._json({"error": "unknown endpoint"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path.rstrip("/") not in ("/api/predict", "/api/flag", "/api/prop_flag", "/api/check_v1"):
            return self._json({"error": "unknown endpoint"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json({"error": "bad Content-Length"}, 400)
        if n > 256_000:
            return self._json({"error": "body too large"}, 413)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except Exception:
            return self._json({"error": "bad json"}, 400)
        if u.path.rstrip("/") == "/api/check_v1":
            res, code, hdr = api_check_v1(body)
            return self._json(res, code, hdr)
        fn = {"/api/flag": api_flag, "/api/prop_flag": api_prop_flag}.get(u.path.rstrip("/"), api_predict)
        res, code = fn(body)
        self._json(res, code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port_pos", nargs="?", type=int)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8878)
    a = ap.parse_args()
    port = a.port_pos or a.port
    srv = ThreadingHTTPServer((a.host, port), Handler)
    threading.Thread(target=_check_load, daemon=True, name="check-v1-warmup").start()   # avoid a ~35 s cold first check
    print(f"FNP dashboard: serving {HERE} + /api/* on http://{a.host}:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
