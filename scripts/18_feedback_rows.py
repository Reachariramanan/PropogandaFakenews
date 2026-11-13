"""Stage 18: turn gallery feedback (data/feedback/flags_v1.jsonl) into Laya training rows.

Only single-claim documents are used (the doc label then equals the claim label). Mapping:
  relabel -> the label you chose
  unflag  -> LEGIT, only when the model had flagged it FAKE/MISLEADING/PROPAGANDA
             (ASSUMPTION: "not suspicious" means LEGIT; relabel to INCONCLUSIVE if unsure)
  confirm -> the model's label, skipped when that label was INCONCLUSIVE
  clear   -> dropped
Rows are built with beta=1.0 (human gold only, no Gemma soft target). Written to
data/runs/laya_rows_feedback_v1.jsonl. Retrain by concatenating with laya_rows_train_v1.jsonl and
passing it to scripts/11_train_laya.py --rows ... --apply (dry-run by default here: use --apply).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnp.config import DATA_DIR  # noqa: E402
from fnp.laya.dataset import make_row, write_jsonl  # noqa: E402

FLAGS = DATA_DIR / "feedback" / "flags_v1.jsonl"
LIVE = DATA_DIR / "runs" / "live_v2.jsonl"   # live_v1 is quarantined
OUT = DATA_DIR / "runs" / "laya_rows_feedback_v1.jsonl"
FLAGGED = ("FAKE", "MISLEADING", "PROPAGANDA")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    flags = {}
    for l in FLAGS.open() if FLAGS.exists() else []:
        r = json.loads(l)
        flags[r["id"]] = r
    rows, skipped = [], Counter()
    for l in LIVE.open() if LIVE.exists() else []:
        r = json.loads(l)
        f = flags.get(r["id"])
        if not f or r.get("status") != "scored":
            continue
        if len(r["claims"]) != 1:
            skipped["multi-claim"] += 1
            continue
        pred, act = r["doc"]["label"], f["action"]
        gold = (f["label"] if act == "relabel" else
                "LEGIT" if act == "unflag" and pred in FLAGGED else
                pred if act == "confirm" and pred != "INCONCLUSIVE" else None)
        if gold is None:
            skipped[act] += 1
            continue
        c = r["claims"][0]
        row = make_row(c["state_text"], gold=gold, beta=1.0, row_id=r["id"], features=c["features"],
                       noul={"is_propaganda": gold == "PROPAGANDA"}, questions=("verdict", "is_propaganda"))
        if row:
            rows.append(row)
    print(f"{len(flags)} feedback items -> {len(rows)} rows; skipped {dict(skipped)}")
    if a.apply and rows:
        print(f"wrote {write_jsonl(rows, OUT)} rows -> {OUT}")


if __name__ == "__main__":
    main()
