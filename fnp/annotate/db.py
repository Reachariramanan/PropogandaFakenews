"""Connections. Annotation DB = read/write (schema `annot`, role fnp_annot). Source DBs = read-only
sessions with a statement timeout; nothing in this package ever writes to them."""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

import fnp.config  # noqa: F401  (loads .env)

_pool: ThreadedConnectionPool | None = None
_lock = threading.Lock()


def annot_params() -> dict:
    try:
        return dict(host=os.environ["FNP_ANNOT_DB_HOST"], port=int(os.environ["FNP_ANNOT_DB_PORT"]),
                    dbname=os.environ["FNP_ANNOT_DB_NAME"], user=os.environ["FNP_ANNOT_DB_USER"],
                    password=os.environ["FNP_ANNOT_DB_PASSWORD"], connect_timeout=8,
                    options=f"-c search_path={os.environ.get('FNP_ANNOT_SCHEMA', 'annot')},public")
    except KeyError as e:
        raise RuntimeError(f"{e.args[0]} is not set; run scripts/27_provision_annotation_db.py --apply") from None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadedConnectionPool(1, 10, **annot_params())
        return _pool


@contextmanager
def tx(dict_rows: bool = True) -> Iterator[psycopg2.extensions.cursor]:
    """One transaction on the annotation DB: commit on success, rollback on error."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor if dict_rows else None)
        yield cur
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def connect_annot():
    """Stand-alone connection (scripts, migrations)."""
    return psycopg2.connect(**annot_params())


def connect_source(kind: str, timeout_ms: int = 120_000):
    """Read-only connection to the news ('news') or social ('social') database."""
    p = "NEWS" if kind == "news" else "SOCIAL"
    c = psycopg2.connect(host=os.environ[f"{p}_DB_HOST"], port=int(os.environ[f"{p}_DB_PORT"]),
                         dbname=os.environ[f"{p}_DB_NAME"], user=os.environ[f"{p}_DB_USER"],
                         password=os.environ[f"{p}_DB_PASSWORD"], connect_timeout=8,
                         options=f"-c default_transaction_read_only=on -c statement_timeout={timeout_ms}")
    c.set_session(readonly=True, autocommit=False)
    return c
