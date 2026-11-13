"""Stage 30: export human annotations as a versioned training dataset (dry-run unless --apply).

  .venv/bin/python scripts/30_annot_export.py --kind claims --version claims_v1
  .venv/bin/python scripts/30_annot_export.py --kind propaganda --version spans_v1 --apply [--policy union]
Writes data/annotation/exports/<version>/ (canonical.jsonl, split files, agreement.json, label_map.json, manifest.json)
(class_weights.json, canonical_blind.jsonl and canonical_random.jsonl are the unbiased evaluation material)
and records the export + its dev/test split in annot.export_run / annot.export_split. Never overwrites a version.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fnp.annotate.export import build  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["claims", "propaganda"], required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--policy", choices=["agree", "union"], default="agree")
    ap.add_argument("--only", choices=["blind", "random"], help="restrict the export to blind items or to the unsteered random slice")
    ap.add_argument("--created-by")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build(a.kind, a.version, a.policy, a.apply, a.created_by, subset=a.only), indent=1, default=str))
    if not a.apply:
        print("dry run: nothing written. Re-run with --apply.")
