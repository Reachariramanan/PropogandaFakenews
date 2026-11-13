"""Stage 32: Propaganda Lab feedback -> social gold rows for the span encoder.

data/feedback/prop_flags_v1.jsonl (written by POST /api/prop_flag; last action per post wins):
  not_propaganda -> negative document: every encoder span on the post is a false positive
                    (all-zero token labels), or only `technique` if one was given
  confirm        -> positive document: the stored strong encoder spans are kept as silver labels
  clear          -> dropped
Output: data/prop_lab/v1/feedback_gold_v1.jsonl  {post_uid, text, spans:[[s,e,technique]], negative_techniques}
Feed it to scripts/23_train_span_encoder.py as extra training docs (mix ratio is the trainer's
choice; keep a held-out slice of the feedback for the social validation the encoder lacks).
  .venv/bin/python scripts/32_prop_feedback_rows.py [--apply]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from fnp.config import DATA_DIR  # noqa: E402

FLAGS = DATA_DIR / "feedback" / "prop_flags_v1.jsonl"
PARTS = DATA_DIR / "prop_lab" / "v1" / "parts"
OUT = DATA_DIR / "prop_lab" / "v1" / "feedback_gold_v1.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    flags = {}
    for line in FLAGS.open() if FLAGS.exists() else []:
        r = json.loads(line)
        flags[r["post_uid"]] = r
    flags = {k: v for k, v in flags.items() if v["action"] != "clear"}
    if not flags:
        print("no feedback yet")
        return
    df = pd.concat([pd.read_parquet(f, columns=["post_uid", "text", "spans", "strong_techniques"])
                    for f in sorted(glob.glob(str(PARTS / "*.parquet")))])
    df = df[df.post_uid.isin(flags)].drop_duplicates("post_uid")
    rows, c = [], Counter()
    for r in df.itertuples():
        f = flags[r.post_uid]
        strong = set(json.loads(r.strong_techniques))
        spans = [s[:3] for s in json.loads(r.spans) if s[2] in strong]
        if f["action"] == "not_propaganda":
            neg = [f["technique"]] if f.get("technique") else sorted(strong)
            keep = [s for s in spans if s[2] not in neg]
            rows.append({"post_uid": r.post_uid, "text": r.text, "spans": keep, "negative_techniques": neg})
        else:
            rows.append({"post_uid": r.post_uid, "text": r.text, "spans": spans, "negative_techniques": []})
        c[f["action"]] += 1
    print(f"{len(flags)} flags -> {len(rows)} docs {dict(c)}")
    if a.apply:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
