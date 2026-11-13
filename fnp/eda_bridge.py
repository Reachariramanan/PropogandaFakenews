"""Read-only bridge to the sibling EDA project.

* Statistical libraries (HMM bursts, event chains, stats_v2) are imported from
  EDA/scripts/lib on demand -- they are pure numpy/scipy and need no EDA credentials.
* Data is read from EDA parquet directly where possible: the EDA session holds DuckDB
  write locks while it runs, and DuckDB refuses even read-only opens from another
  process during that time. For DuckDB-only tables we fall back to a snapshot copy.
Nothing here ever writes to EDA paths.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import duckdb

from .config import CACHE_DIR, EDA_PARQUET_DIR, EDA_ROOT

_EDA_SCRIPTS = EDA_ROOT / "scripts"


def eda_lib(name: str):
    """Import `lib.<name>` from EDA (e.g. "hmm_v1", "event_chains", "stats_v2")."""
    if str(_EDA_SCRIPTS) not in sys.path:
        sys.path.append(str(_EDA_SCRIPTS))
    import importlib
    return importlib.import_module(f"lib.{name}")


def parquet_glob(table: str) -> str:
    p = EDA_PARQUET_DIR / table
    return str(p / "*.parquet") if p.is_dir() else str(EDA_PARQUET_DIR / f"{table}.parquet")


def read_parquet(table: str, sql_tail: str = "", con: Optional[duckdb.DuckDBPyConnection] = None):
    """`SELECT * FROM <table parquet> <sql_tail>` as a DataFrame, via an in-memory DuckDB."""
    own = con is None
    con = con or duckdb.connect()
    try:
        return con.execute(f"SELECT * FROM read_parquet('{parquet_glob(table)}', union_by_name=true) {sql_tail}").df()
    finally:
        if own:
            con.close()


def open_eda_db(db_path: Path, max_age_s: float = 6 * 3600) -> duckdb.DuckDBPyConnection:
    """Open an EDA DuckDB file read-only; if it is write-locked, open a snapshot copy.

    The snapshot lives in data/cache/eda_snapshots and is refreshed when older than
    `max_age_s`. A copy taken mid-write can be torn; DuckDB checks its own checksums
    and raises on open, in which case the caller sees the error rather than bad data.
    """
    try:
        return duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException:
        snap_dir = CACHE_DIR / "eda_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap = snap_dir / db_path.name
        if not snap.exists() or time.time() - snap.stat().st_mtime > max_age_s:
            shutil.copy2(db_path, snap)
            wal = db_path.with_suffix(db_path.suffix + ".wal")
            if wal.exists():
                shutil.copy2(wal, snap.with_suffix(snap.suffix + ".wal"))
        return duckdb.connect(str(snap), read_only=True)
