"""Stage 33: remove near-duplicate and link-only items from the annotation pool (dry run unless --apply).

  .venv/bin/python scripts/33_annot_dedupe.py
  .venv/bin/python scripts/33_annot_dedupe.py --apply
Sets status='excluded' with a reason; never deletes, never touches items that already have an assignment.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fnp.annotate import cleanup  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    print("body_key backfilled for", cleanup.backfill_body_keys(), "items")
    p = cleanup.plan()
    print(json.dumps(p["report"], indent=1))
    if a.apply:
        print(json.dumps(cleanup.apply(p)))
    else:
        print("dry run: nothing excluded. Re-run with --apply.")
