#!/usr/bin/env bash
# Nightly FNP backup (run by systemd user timer fnp-backup.timer; safe to run by hand).
#   1. data/fnp.duckdb -> file copy, then verified by opening the copy read-only (a copy taken while a
#      writer is mid-checkpoint can be inconsistent; a copy that fails to open is discarded and logged).
#   2. fnp_annotation Postgres -> scripts/ops/pg_export.py (CSV + schema.json tar.gz; credentials from
#      .env via the environment, never on the command line).
# Keeps KEEP_DAYS days in $BACKUP_DIR (default ~/backups/fnp). Exit code != 0 if any part failed.
set -uo pipefail
umask 077                      # backups contain annotator password hashes and corpus text
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$HOME/backups/fnp}"
KEEP_DAYS="${KEEP_DAYS:-7}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"
fail=0

copy="$BACKUP_DIR/fnp_${TS}.duckdb"
cp "$ROOT/data/fnp.duckdb" "$copy.part" && mv "$copy.part" "$copy"
if "$ROOT/.venv/bin/python" -c "import duckdb,sys; c=duckdb.connect(sys.argv[1], read_only=True); \
n=c.execute('select count(*) from information_schema.tables').fetchone()[0]; print('duckdb tables', n)" "$copy"; then
  gzip -f "$copy"
else
  echo "duckdb copy unreadable, discarded: $copy" >&2; rm -f "$copy"; fail=1
fi

set -a; . "$ROOT/.env"; set +a
if [ -n "${FNP_ANNOT_DB_HOST:-}" ]; then
  # host pg_dump (16) refuses the PostgreSQL 18 server -> version-independent CSV + schema export
  if ! "$ROOT/.venv/bin/python" "$ROOT/scripts/ops/pg_export.py" "$BACKUP_DIR/fnp_annotation_${TS}.tar.gz"; then
    echo "annotation DB export failed" >&2; fail=1
  fi
fi

find "$BACKUP_DIR" -maxdepth 1 -type f \( -name 'fnp_*.duckdb.gz' -o -name 'fnp_annotation_*.tar.gz' \) \
     -mtime +"$KEEP_DAYS" -delete
exit $fail
