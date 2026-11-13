"""Stage 10: build Laya training rows from the TRAIN split of data/runs/units_v1.jsonl.

Target = (1-beta)*gold + beta*Gemma-adjudicator probs (fnp.laya.dataset.verdict_target).
dev/test units are never written here: they are the held-out sets for 14_evaluate.
  .venv/bin/python scripts/10_laya_rows.py [--beta 0.3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnp.config import DATA_DIR  # noqa: E402
from fnp.laya.dataset import make_row, write_jsonl  # noqa: E402

UNITS = DATA_DIR / "runs" / "units_v1.jsonl"
OUT = DATA_DIR / "runs" / "laya_rows_train_v1.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.3)
    a = ap.parse_args()
    rows = []
    for line in UNITS.open():
        u = json.loads(line)
        if u["split"] != "train":
            continue
        gp = u["gemma_adj"]["probs"] if u["gemma_adj"].get("ok") else None
        r = make_row(u["state_text"], gold=u["label5"], gemma_probs=gp, beta=a.beta,
                     row_id=u["unit_id"], features=u["features"],
                     noul={"is_propaganda": u["label5"] == "PROPAGANDA"},
                     questions=("verdict", "is_propaganda"))
        if r:
            rows.append(r)
    n = write_jsonl(rows, OUT)
    print(f"wrote {n} rows -> {OUT}")


if __name__ == "__main__":
    main()
