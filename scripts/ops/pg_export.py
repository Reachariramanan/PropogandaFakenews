"""Version-independent logical backup of the fnp_annotation Postgres (used by scripts/ops/backup.sh
because the server is PostgreSQL 18 and the host pg_dump is 16).

One REPEATABLE READ, read-only transaction -> every user table as CSV (COPY ... TO STDOUT) plus
schema.json (columns, types, nullability, primary keys) in a single .tar.gz.
Restore: create the schema with fnp/annotate/migrations/*.sql (scripts/27_provision_annotation_db.py),
then `COPY <table> FROM STDIN WITH CSV HEADER` per file, in foreign-key order (schema.json lists it).
  .venv/bin/python scripts/ops/pg_export.py OUT.tar.gz
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psycopg2  # noqa: E402

from fnp import config  # noqa: E402,F401  (loads .env)


def main(out: str) -> None:
    c = psycopg2.connect(host=os.environ["FNP_ANNOT_DB_HOST"], port=int(os.environ["FNP_ANNOT_DB_PORT"]),
                         dbname=os.environ["FNP_ANNOT_DB_NAME"], user=os.environ["FNP_ANNOT_DB_USER"],
                         password=os.environ["FNP_ANNOT_DB_PASSWORD"], connect_timeout=15)
    c.set_session(readonly=True, isolation_level="REPEATABLE READ")
    cur = c.cursor()
    cur.execute("""SELECT table_schema, table_name FROM information_schema.tables
                   WHERE table_type='BASE TABLE' AND table_schema NOT IN ('pg_catalog','information_schema')
                   ORDER BY 1, 2""")
    tables = cur.fetchall()
    schema = {"server_version": c.server_version, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "tables": []}
    tmp = out + ".part"
    with tarfile.open(tmp, "w:gz") as tar:
        for sch, t in tables:
            cur.execute("""SELECT column_name, data_type, is_nullable FROM information_schema.columns
                           WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position""", (sch, t))
            cols = [{"name": a, "type": b, "nullable": n == "YES"} for a, b, n in cur.fetchall()]
            cur.execute("""SELECT kcu.column_name FROM information_schema.table_constraints tc
                           JOIN information_schema.key_column_usage kcu USING (constraint_schema, constraint_name)
                           WHERE tc.table_schema=%s AND tc.table_name=%s AND tc.constraint_type='PRIMARY KEY'""", (sch, t))
            pk = [r[0] for r in cur.fetchall()]
            buf = io.BytesIO()
            cur.copy_expert(f'COPY "{sch}"."{t}" TO STDOUT WITH CSV HEADER', buf)
            data = buf.getvalue()
            info = tarfile.TarInfo(f"{sch}.{t}.csv")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            schema["tables"].append({"schema": sch, "table": t, "columns": cols, "primary_key": pk,
                                     "bytes": len(data), "rows": max(data.count(b"\n") - 1, 0)})
        js = json.dumps(schema, indent=1).encode()
        info = tarfile.TarInfo("schema.json")
        info.size = len(js)
        tar.addfile(info, io.BytesIO(js))
    c.rollback()
    c.close()
    os.replace(tmp, out)
    print(f"pg export ok: {len(tables)} tables, {sum(t['rows'] for t in schema['tables'])} rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
