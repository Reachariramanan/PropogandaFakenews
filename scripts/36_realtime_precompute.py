"""Stage 36: incremental cheap-pass precompute for the Evidence Base (every 3 h via systemd timer).

Reads Interface/data/realtime.sqlite (read-only, table items; ids are used exactly as stored) and,
for every item not yet scored (newest first, failures retried), runs the cheap pass with NO Gemma:
  * propaganda span encoder (prop_span_v1): spans, 14-class technique_probs (max token prob per
    class), p_prop = max mean prob of strong spans (same definition as stage 35, uncalibrated);
  * countries the text is about (gazetteer matcher);
  * nearest kb_v2 fact-check (bge-m3 cosine, reranker score when cos >= 0.72).
Results append to data/realtime/v1/scored.jsonl (resume store). The export keeps only items still in
the SQLite window and is swapped atomically: reports/realtime/v1/{news,social}/part_NNNN.jsonl +
meta.json (built_at, n_items, last_item_published_at, next_run_at ...). Previous export -> v1.prev.
A flock prevents overlapping runs; --time-cap-min stops between parts.
  .venv/bin/python scripts/36_realtime_precompute.py [--max-items 6000] [--time-cap-min 100] [--dry-run]
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fnp.config import DATA_DIR, FNP_ROOT, REPORTS_DIR  # noqa: E402
from fnp.geo.countries import matcher  # noqa: E402
from fnp.retrieval.kb import embed_texts  # noqa: E402

SQLITE = FNP_ROOT / "Interface" / "data" / "realtime.sqlite"
STORE = DATA_DIR / "realtime" / "v1"
SCORED = STORE / "scored.jsonl"
OUT = REPORTS_DIR / "realtime" / "v1"
INTERVAL_H = 3
PART_SIZE = 500
ROWS_PER_FILE = 2000
RERANK_MIN_COS = 0.72
MODEL_VERSION = "prop_span_v1+kb_v2"
CAVEAT = ("p_prop and technique_probs come from an encoder trained on long news articles (PTC) and not "
          "validated on social media or short items; use as a ranking signal, not a calibrated probability. "
          "nearest_factcheck is the closest known fact-check by embedding similarity, not a verdict.")


def _stage30():
    spec = importlib.util.spec_from_file_location("s30", FNP_ROOT / "scripts" / "30_prop_lab_infer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def finite(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, 4)
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def read_items(done: set) -> tuple[list[dict], dict]:
    """New items (newest first) and {id: published_at} for everything currently in the window."""
    con = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT id, kind, platform, source, url, title, text_excerpt, lang, published_at "
                       "FROM items ORDER BY published_at DESC").fetchall()
    con.close()
    window = {r["id"]: (r["kind"], r["published_at"]) for r in rows}
    new = []
    for r in rows:
        if r["id"] in done:
            continue
        t = ((r["title"] or "").strip() + "\n" + (r["text_excerpt"] or "").strip()).strip()
        if len(t) < 5:
            continue
        new.append({**dict(r), "text": t})
    return new, window


def score_part(items, s30, enc, kbm, gm, dev) -> list[dict]:
    m, model, tok, thr = enc
    texts = [i["text"][:6000] for i in items]
    sig = s30.encode_spans(texts, m, model, tok, thr, dev)
    V = embed_texts([t[:1500] for t in texts], batch_size=32)
    cos, idx = kbm.top1(V)
    meta = kbm.kb.meta
    rr = np.full(len(items), np.nan, dtype=np.float32)
    cand = np.where(cos >= RERANK_MIN_COS)[0]
    if len(cand):
        rr[cand] = kbm.rerank([(texts[k][:1000], meta.claim.iat[idx[k]]) for k in cand])
    out = []
    for k, it in enumerate(items):
        s = sig[k]
        spans = json.loads(s["spans"])
        strong_min = {sp[2]: max(float(thr[m.CIDX[sp[2]]]), s30.STRONG_P) for sp in spans}
        strong = [sp for sp in spans if sp[3] >= strong_min[sp[2]]]
        out.append({
            "id": it["id"], "kind": it["kind"], "platform": it["platform"], "source": it["source"],
            "published_at": it["published_at"], "lang": it["lang"],
            "too_short": s["too_short"], "n_tokens": s["n_tokens"],
            "p_prop": None if s["too_short"] else max((sp[3] for sp in strong), default=0.0),
            "n_spans": s["n_spans"], "n_strong": s["n_strong"], "density": s["density"],
            "technique_probs": dict(zip(m.CLASSES14, json.loads(s["tech_max"]))),
            "strong_techniques": json.loads(s["strong_techniques"]),
            "spans": spans[:40],           # [start, end, technique, prob], offsets into title+"\n"+text_excerpt
            "countries": gm.match(texts[k]),
            "nearest_factcheck": {"kb_id": meta.kb_id.iat[idx[k]], "label": meta.label5.iat[idx[k]],
                                  "claim": meta.claim.iat[idx[k]][:300], "url": meta.url.iat[idx[k]],
                                  "cosine": float(cos[k]),
                                  "rerank": None if np.isnan(rr[k]) else float(rr[k])},
            "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_version": MODEL_VERSION})
    return out


def export(window: dict, run_started: float, n_new: int, n_failed: int) -> dict:
    keep = {}
    for line in SCORED.open():
        r = json.loads(line)
        if r["id"] in window:
            keep[r["id"]] = r                       # last write wins
    tmp, prev = OUT.with_name("v1.building"), OUT.with_name("v1.prev")
    shutil.rmtree(tmp, ignore_errors=True)
    counts, last_pub = {}, {}
    for kind in ("news", "social"):
        rows = sorted((r for r in keep.values() if r["kind"] == kind), key=lambda r: r["published_at"], reverse=True)
        d = tmp / kind
        d.mkdir(parents=True)
        for n, a in enumerate(range(0, max(len(rows), 1), ROWS_PER_FILE)):
            with (d / f"part_{n:04d}.jsonl").open("w") as f:
                for r in rows[a:a + ROWS_PER_FILE]:
                    f.write(json.dumps(finite(r), ensure_ascii=False, allow_nan=False) + "\n")
        counts[kind] = len(rows)
        last_pub[kind] = rows[0]["published_at"] if rows else None
    now = datetime.now(timezone.utc)
    meta = {"built_at": now.isoformat(timespec="seconds"), "n_items": counts, "n_items_total": sum(counts.values()),
            "n_in_window": len(window), "n_unscored_in_window": len(window) - len(keep),
            "last_item_published_at": last_pub, "next_run_at": (now + timedelta(hours=INTERVAL_H)).isoformat(timespec="seconds"),
            "interval_hours": INTERVAL_H, "n_scored_this_run": n_new, "n_failed_this_run": n_failed,
            "run_seconds": round(time.time() - run_started), "model_version": MODEL_VERSION,
            "technique_classes": list(json.loads(json.dumps(next(iter(keep.values()))["technique_probs"])).keys()) if keep else [],
            "analysed_text": "title + '\\n' + text_excerpt; span offsets index into this string",
            "caveat": CAVEAT}
    (tmp / "meta.json").write_text(json.dumps(finite(meta), ensure_ascii=False, allow_nan=False))
    if OUT.exists():
        shutil.rmtree(prev, ignore_errors=True)
        OUT.rename(prev)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp.rename(OUT)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-items", type=int, default=6000)
    ap.add_argument("--time-cap-min", type=float, default=100)
    ap.add_argument("--dry-run", action="store_true", help="score nothing, report what would run")
    a = ap.parse_args()
    STORE.mkdir(parents=True, exist_ok=True)
    lock = (STORE / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another run holds the lock; exiting", flush=True)
        return
    t0 = time.time()
    done = {json.loads(l)["id"] for l in SCORED.open()} if SCORED.exists() else set()
    items, window = read_items(done)
    todo = items[:a.max_items]
    print(f"window {len(window)} items, {len(done)} already scored, {len(items)} new, running {len(todo)}", flush=True)
    if a.dry_run:
        return
    n_new = n_failed = 0
    if todo:
        s30 = _stage30()
        dev = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**30 > 4 else "cpu"
        enc = s30.load_encoder(dev)
        kbm = s30.KBMatch(dev)
        gm = matcher()
        with SCORED.open("a") as f:
            for a0 in range(0, len(todo), PART_SIZE):
                if (time.time() - t0) / 60 > a.time_cap_min:
                    print("time cap reached; remaining items wait for the next run", flush=True)
                    break
                part = todo[a0:a0 + PART_SIZE]
                try:
                    for r in score_part(part, s30, enc, kbm, gm, dev):
                        f.write(json.dumps(finite(r), ensure_ascii=False) + "\n")
                    f.flush()
                    n_new += len(part)
                except Exception as e:      # failed part is not marked done, so it is retried next run
                    n_failed += len(part)
                    print(f"part failed ({type(e).__name__}: {str(e)[:150]}); retry next run", flush=True)
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                print(f"{a0 + len(part)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    if not SCORED.exists():
        print("nothing scored yet; no export")
        return
    meta = export(window, t0, n_new, n_failed)
    print(f"exported {meta['n_items_total']} items ({meta['n_unscored_in_window']} in window still unscored) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
