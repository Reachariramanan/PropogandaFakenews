"""Stage 29: run the annotation server (separate process from the dashboard; shares UI/assets).

  .venv/bin/python scripts/29_annot_server.py --port 8890
Prereqs: scripts/27 (database), `python -m fnp.annotate.migrate --apply`, an admin account
(`scripts/31_annot_users.py create NAME --role admin`), and items via `python -m fnp.annotate.ingest --apply`
(or `--fill-to 60000`) plus optional `python -m fnp.annotate.preannotate --apply` (steers class-balanced batches).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fnp.annotate.server import serve  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8890)
    a = ap.parse_args()
    serve(a.host, a.port)
