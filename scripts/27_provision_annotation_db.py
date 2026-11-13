"""Stage 27: create the dedicated annotation database + least-privilege role (one-off).

Uses the NEWS_DB_* account ONLY as the admin that issues CREATE ROLE / CREATE DATABASE on the
ShaktiDB news server. It never touches any existing database (shaktidb_news, Annotator_tool, ...).
The new role is NOSUPERUSER/NOCREATEDB/NOCREATEROLE and owns only `fnp_annotation`.
Credentials for the new role are appended to .env as FNP_ANNOT_DB_* (never printed).

  .venv/bin/python scripts/27_provision_annotation_db.py            # dry run: shows what it would do
  .venv/bin/python scripts/27_provision_annotation_db.py --apply
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fnp.config  # noqa: E402,F401  (loads .env)

DB_NAME, ROLE = "fnp_annotation", "fnp_annot"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    env_path = ROOT / ".env"
    have = {l.split("=", 1)[0] for l in env_path.read_text().splitlines() if "=" in l and not l.startswith("#")}
    if "FNP_ANNOT_DB_PASSWORD" in have:
        print("FNP_ANNOT_DB_* already in .env -> nothing to do (delete those lines to re-provision).")
        return
    host, port = os.environ["NEWS_DB_HOST"], int(os.environ["NEWS_DB_PORT"])
    print(f"admin server {host}:{port}; will create role {ROLE!r} and database {DB_NAME!r}")
    if not a.apply:
        print("dry run: no changes made. Re-run with --apply.")
        return
    pw = secrets.token_urlsafe(24)
    c = psycopg2.connect(host=host, port=port, dbname=os.environ["NEWS_DB_NAME"], user=os.environ["NEWS_DB_USER"],
                         password=os.environ["NEWS_DB_PASSWORD"], connect_timeout=8)
    c.autocommit = True
    cur = c.cursor()
    cur.execute("select 1 from pg_roles where rolname=%s", (ROLE,))
    if cur.fetchone():
        raise SystemExit(f"role {ROLE} already exists on the server; refusing to reuse it blindly")
    cur.execute("select 1 from pg_database where datname=%s", (DB_NAME,))
    if cur.fetchone():
        raise SystemExit(f"database {DB_NAME} already exists; refusing to touch it")
    cur.execute(sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}")
                .format(sql.Identifier(ROLE), sql.Literal(pw)))
    cur.execute(sql.SQL("CREATE DATABASE {} OWNER {} ENCODING 'UTF8'")
                .format(sql.Identifier(DB_NAME), sql.Identifier(ROLE)))
    cur.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(DB_NAME)))
    c.close()
    with env_path.open("a") as f:
        f.write(f"\n# FNP annotation database (read/write; created by scripts/27)\n"
                f"FNP_ANNOT_DB_HOST={host}\nFNP_ANNOT_DB_PORT={port}\nFNP_ANNOT_DB_NAME={DB_NAME}\n"
                f"FNP_ANNOT_DB_USER={ROLE}\nFNP_ANNOT_DB_PASSWORD={pw}\n")
    print("created; FNP_ANNOT_DB_* appended to .env")


if __name__ == "__main__":
    main()
