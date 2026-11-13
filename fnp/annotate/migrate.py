"""Apply numbered SQL migrations to the annotation DB, then seed taxonomies.

  .venv/bin/python -m fnp.annotate.migrate            # dry run: list pending
  .venv/bin/python -m fnp.annotate.migrate --apply
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import taxonomy
from .db import connect_annot

MIG = Path(__file__).parent / "migrations"


def sql_for(path: Path, schema: str = "annot") -> str:
    """Migration text bound to `schema` (tests apply the same files to a throwaway schema)."""
    t = path.read_text()
    if schema != "annot":
        t = t.replace("CREATE SCHEMA IF NOT EXISTS annot;", f"CREATE SCHEMA IF NOT EXISTS {schema};")
        t = t.replace("SET search_path = annot;", f"SET search_path = {schema};")
    return t


def apply_all(cur, schema: str = "annot") -> dict:
    """Apply every migration file (fresh schema) and seed the taxonomies. Used by tests."""
    for p in sorted(MIG.glob("*.sql")):
        cur.execute(sql_for(p, schema))
    cur.execute(f"SET search_path = {schema}, public")
    return taxonomy.seed(cur)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    c = connect_annot()
    cur = c.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS annot")
    cur.execute("CREATE TABLE IF NOT EXISTS annot.schema_migrations (version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
    c.commit()
    cur.execute("SELECT version FROM annot.schema_migrations")
    done = {r[0] for r in cur.fetchall()}
    pending = [p for p in sorted(MIG.glob("*.sql")) if p.stem not in done]
    print("applied:", sorted(done), "| pending:", [p.stem for p in pending])
    if not a.apply:
        print("dry run: nothing changed. Re-run with --apply.")
        return
    for p in pending:
        cur.execute(sql_for(p))
        cur.execute("INSERT INTO annot.schema_migrations (version) VALUES (%s)", (p.stem,))
        c.commit()
        print("applied", p.stem)
    cur.execute("SET search_path = annot, public")
    print("seeded:", taxonomy.seed(cur))
    c.commit()
    c.close()


if __name__ == "__main__":
    main()
