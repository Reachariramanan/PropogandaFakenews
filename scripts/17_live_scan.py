"""Stage 17: newest-first live scan -> data/runs/live_v2.jsonl (feeds the UI "Gallery" tab).

Per batch of the newest unseen items: cheap pass (bge-m3 cosine to the fact-check KB), then the
full pipeline (fnp.pipeline.analyse_text) on the most KB-similar share of the batch. The rest are
recorded as skipped so they are not re-embedded. Resumable by id. Read-only on all sources.
  .venv/bin/python scripts/17_live_scan.py [--source social|news|all] [--batch 200]
        [--top-frac 0.4] [--loop SECONDS] [--max-batches N]
Stop cleanly with: touch logs/live_scan.stop
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from fnp.config import DATA_DIR, FNP_ROOT, gemma_config  # noqa: E402
from fnp.live import sources  # noqa: E402
from fnp.pipeline import analyse_text, load_deciders  # noqa: E402
from fnp.retrieval.kb import KB, embed_texts  # noqa: E402

OUT = DATA_DIR / "runs" / "live_v2.jsonl"   # v1 = Oct-29..Nov-1 run, mostly failed Gemma calls
STATE = DATA_DIR / "runs" / "live_state.json"
STOP = FNP_ROOT / "logs" / "live_scan.stop"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seen_ids() -> set:
    """Ids already handled; `error` rows are not counted so they are retried (last row wins)."""
    st = {}
    for l in OUT.open() if OUT.exists() else []:
        r = json.loads(l)
        st[r["id"]] = r["status"]
    return {i for i, s in st.items() if s != "error"}


def write_state(**kw) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    s.update(kw, updated=now())
    STATE.write_text(json.dumps(s))


def next_items(source: str, n: int, seen: set) -> tuple[list, dict]:
    status, out = {}, []
    if source in ("social", "all"):
        out += list(islice((i for i in sources.social_local() if i["id"] not in seen), n))
        status["social"] = "local (EDA queue + parquet)"
    if source in ("news", "all"):
        ok, why = sources.pg_reachable("NEWS")
        status["news"] = "live postgres" if ok else f"UNREACHABLE: {why}"
        if ok:
            out += [i for i in sources.news_pg(limit=n * 3) if i["id"] not in seen][:n]
    return out, status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all", choices=["social", "news", "all"])
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--top-frac", type=float, default=0.4)
    ap.add_argument("--loop", type=int, default=0, help="seconds between polls; 0 = run once")
    ap.add_argument("--max-batches", type=int, default=None)
    a = ap.parse_args()

    kb = KB.load()
    deciders = load_deciders(None)
    conc = gemma_config()["concurrency"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = 0
    while not STOP.exists():
        seen = seen_ids()
        items, status = next_items(a.source, a.batch, seen)
        write_state(sources=status, pending_poll=now())
        if items:
            emb = embed_texts([i["text"][:1500] for i in items])
            cos = (emb @ kb.emb.T).max(axis=1)
            keep = set(np.argsort(-cos)[: max(1, int(len(items) * a.top_frac))].tolist())
            t0 = time.time()

            def run(k):
                it = items[k]
                res = analyse_text(it["id"], it["text"], it.get("lang"), deciders)
                return {"claims": [{"claim": c["claim"], "verdict": c["verdict"], "state_text": c["state_text"],
                                    "features": c["features"], "gemma_adj": c["gemma_adj"]}
                                   for c in res["claims"]],
                        "doc": res["doc"], "spans_summary": res["spans_summary"]}

            with OUT.open("a") as f, ThreadPoolExecutor(conc) as ex:
                futs = {k: ex.submit(run, k) for k in keep}
                for k, it in enumerate(items):
                    rec = {**it, "text": it["text"][:1500], "kb_cos": round(float(cos[k]), 3),
                           "scanned_at": now(), "status": "skipped"}
                    if k in futs:
                        try:
                            res = futs[k].result()
                            cl = res["claims"]
                            # Gemma down/overloaded -> no claims or every adjudication failed:
                            # record as error (retried on a later poll), never a fake INCONCLUSIVE.
                            failed = (not cl and len(it["text"]) >= 25) or (
                                bool(cl) and not any(c["gemma_adj"].get("ok") for c in cl))
                            rec.update(res, status="error" if failed else "scored",
                                       **({"error": "gemma calls failed"} if failed else {}))
                        except Exception as e:      # one bad item must not stop the scan
                            rec.update(status="error", error=f"{type(e).__name__}: {e}"[:200])
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    f.flush()
            write_state(last_batch=len(items), last_scored=len(keep), last_secs=round(time.time() - t0))
            print(f"{now()} batch {len(items)} scored {len(keep)} {time.time()-t0:.0f}s {status}", flush=True)
        else:
            print(f"{now()} nothing new {status}", flush=True)
        nb += 1
        if not a.loop or (a.max_batches and nb >= a.max_batches):
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
