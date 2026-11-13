"""Stage 16: score unlabelled external items and publish doc-level verdicts.

Inputs (written by the Interface/ scraping session, read-only here):
  Interface/data/db_claim_candidates.jsonl  social posts: id, platform, url, text, ...
  Interface/data/db_factchecks.jsonl        news items: id, claim, ... (verdict_raw is a
                                            keyword guess -- NOT used as gold)
Each item goes through fnp.pipeline.analyse_text (claims -> evidence -> maths -> deciders).
Results append to data/runs/external_v1.jsonl (resumable by id). --apply also upserts
claim + doc rows into data/fnp.duckdb:verdict_v1 (doc unit_id == the item id, which the
gallery joins on) and exports every doc row to reports/verdict_v1_docs.jsonl.
  .venv/bin/python scripts/16_score_external.py [--limit N] [--laya DIR] [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from fnp.config import BOOM_DIR, DATA_DIR, DUCKDB_PATH, REPORTS_DIR, gemma_config  # noqa: E402
from fnp.pipeline import analyse_text, load_deciders  # noqa: E402

SOURCES = {"db_claim_candidates": ("text", BOOM_DIR / "db_claim_candidates.jsonl"),
           "db_factchecks": ("claim", BOOM_DIR / "db_factchecks.jsonl")}
OUT = DATA_DIR / "runs" / "external_v1.jsonl"


def verdict_row(v: dict, unit: str, unit_id: str, doc_id: str, dataset: str) -> dict:
    return {"unit_id": unit_id, "unit": unit, "doc_id": doc_id, "label": v["label"],
            "plausibly_fake": bool(v.get("plausibly_fake")), "decider": v.get("decider"),
            "rationale": v.get("rationale", ""), "probs": json.dumps(v["probs"]),
            "secondary": json.dumps(v.get("secondary", [])),
            "prediction_set": json.dumps(v.get("prediction_set", [])),
            "features": json.dumps(v.get("features", {}), default=str),
            "evidence_ids": json.dumps(v.get("evidence_ids", [])),
            "p_fake": float(v["probs"].get("FAKE", 0.0)),
            "p_misleading": float(v["probs"].get("MISLEADING", 0.0)),
            "gold_label5": None, "dataset": dataset, "split": "external"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--laya", default=None)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    items = []
    for ds, (field, path) in SOURCES.items():
        for line in path.open():
            r = json.loads(line)
            if r.get(field):
                items.append({"id": r["id"], "dataset": ds, "text": r[field], "lang": r.get("lang")})
    done = {json.loads(l)["id"] for l in OUT.open()} if OUT.exists() else set()
    todo = [i for i in items if i["id"] not in done][: a.limit]
    print(f"{len(items)} items, {len(done)} done, running {len(todo)}", flush=True)

    deciders = load_deciders(a.laya)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def run(it):
        res = analyse_text(it["id"], it["text"], it.get("lang"), deciders)
        return {"id": it["id"], "dataset": it["dataset"], "doc": res["doc"],
                "claims": [{"claim": r["claim"], "verdict": r["verdict"]} for r in res["claims"]],
                "spans_summary": res["spans_summary"]}

    t0 = time.time()
    with OUT.open("a") as fout, ThreadPoolExecutor(gemma_config()["concurrency"]) as ex:
        for k, (it, fut) in enumerate(zip(todo, [ex.submit(run, it) for it in todo]), 1):
            try:
                r = fut.result()
            except Exception as e:      # one bad item must not stop the batch
                print(f"ERR {it['id']}: {type(e).__name__}: {e}", flush=True)
                continue
            fout.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            fout.flush()
            if k % 50 == 0:
                print(f"{k}/{len(todo)} {time.time()-t0:.0f}s", flush=True)

    if not a.apply:
        return
    rows = []
    for line in OUT.open():
        r = json.loads(line)
        rows.append(verdict_row(r["doc"], "doc", r["id"], r["id"], r["dataset"]))
        for c in r["claims"]:
            rows.append(verdict_row(c["verdict"], "claim", c["verdict"]["unit_id"], r["id"],
                                    r["dataset"]))
    out = pd.DataFrame(rows).drop_duplicates(["unit_id", "unit"], keep="last")
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("CREATE TABLE IF NOT EXISTS verdict_v1 AS SELECT * FROM out LIMIT 0")
    con.execute("DELETE FROM verdict_v1 WHERE (unit_id, unit) IN (SELECT unit_id, unit FROM out)")
    con.execute("INSERT INTO verdict_v1 BY NAME SELECT * FROM out")
    docs = con.execute("SELECT * FROM verdict_v1 WHERE unit='doc'").df()
    con.close()
    REPORTS_DIR.mkdir(exist_ok=True)
    docs.to_json(REPORTS_DIR / "verdict_v1_docs.jsonl", orient="records", lines=True,
                 force_ascii=False)
    print(f"verdict_v1 upserted {len(out)} rows; exported {len(docs)} doc rows")


if __name__ == "__main__":
    main()
