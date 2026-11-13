"""DuckDB-backed cache for Gemma calls: `gemma_cache` in `fnp.config.DUCKDB_PATH`.

One connection per `GemmaCache`, serialised by a `threading.Lock` (DuckDB connections are
not safe for concurrent use). Key = sha1(task | prompt_version | system | user), so bumping
a prompt version invalidates its entries. Failures are cached too (ok=false) so a
deterministic failure is visible/auditable, but `cached_call(retry_failed=True)` re-asks.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import warnings
from pathlib import Path
from typing import Callable, Iterable, Optional

import duckdb

from .. import config as _config
from .. import gemma as _gemma

DDL = """
CREATE TABLE IF NOT EXISTS gemma_cache (
    key VARCHAR PRIMARY KEY,
    task VARCHAR,
    prompt_version VARCHAR,
    input_sha VARCHAR,
    response_json VARCHAR,
    ok BOOLEAN,
    error VARCHAR,
    tokens INTEGER,
    latency_s DOUBLE,
    created_at TIMESTAMP DEFAULT current_timestamp
)
"""
_COLS = ("key", "task", "prompt_version", "input_sha", "response_json", "ok", "error",
         "tokens", "latency_s")


def make_key(task: str, version: str, system: str, user: str) -> str:
    return hashlib.sha1("|".join((task, version, system, user)).encode("utf-8")).hexdigest()


def input_sha(system: str, user: str) -> str:
    return hashlib.sha1(f"{system}\x1f{user}".encode("utf-8")).hexdigest()[:16]


class GemmaCache:
    def __init__(self, path: Optional[Path | str] = None):
        self.path = Path(path) if path else Path(_config.DUCKDB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        try:
            self._con = duckdb.connect(str(self.path))
        except duckdb.IOException:
            # Another process holds the main database; keep the cache usable beside it.
            alt = self.path.with_name(self.path.stem + "_gemma_cache.duckdb")
            warnings.warn(f"{self.path} is locked; using {alt} for gemma_cache")
            self.path = alt
            self._con = duckdb.connect(str(alt))
        self._con.execute(DDL)

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._con.execute(
                "SELECT response_json, ok, error, tokens, latency_s FROM gemma_cache WHERE key=?",
                [key]).fetchone()
        if row is None:
            return None
        resp, ok, err, tokens, lat = row
        return {"ok": bool(ok), "obj": json.loads(resp) if resp else None, "error": err,
                "tokens": int(tokens or 0), "latency_s": float(lat or 0.0)}

    def put_many(self, rows: Iterable[dict]) -> None:
        """Insert-or-replace several rows under one lock acquisition."""
        data = [[r["key"], r["task"], r["prompt_version"], r["input_sha"],
                 json.dumps(r["obj"], ensure_ascii=False) if r.get("obj") is not None else None,
                 bool(r["ok"]), r.get("error"), int(r.get("tokens") or 0),
                 float(r.get("latency_s") or 0.0)] for r in rows]
        if not data:
            return
        with self._lock:
            self._con.executemany(
                "INSERT OR REPLACE INTO gemma_cache (key, task, prompt_version, input_sha, "
                "response_json, ok, error, tokens, latency_s) VALUES (?,?,?,?,?,?,?,?,?)", data)

    def put(self, row: dict) -> None:
        self.put_many([row])

    def stats(self) -> list[tuple]:
        with self._lock:
            return self._con.execute(
                "SELECT task, prompt_version, count(*), sum(ok::INT), sum(tokens), avg(latency_s) "
                "FROM gemma_cache GROUP BY 1,2 ORDER BY 1,2").fetchall()

    def close(self) -> None:
        with self._lock:
            self._con.close()


_default: Optional[GemmaCache] = None
_default_lock = threading.Lock()


def get_cache() -> GemmaCache:
    """Process-wide cache, opened lazily at the *current* `config.DUCKDB_PATH`."""
    global _default
    with _default_lock:
        if _default is None:
            _default = GemmaCache()
        return _default


def reset_default_cache(path: Optional[Path | str] = None) -> Optional[GemmaCache]:
    """Close the default cache; with `path`, open a new default there (tests use tmp paths)."""
    global _default
    with _default_lock:
        if _default is not None:
            try:
                _default.close()
            except Exception:
                pass
        _default = GemmaCache(path) if path else None
        return _default


def cached_call(task: str, prompt_version: str, system: str, user: str,
                validate: Optional[Callable[[dict], Optional[str]]] = None,
                max_tokens: int = 1500, *, retry_failed: bool = True, retries: int = 1,
                cache: Optional[GemmaCache] = None) -> dict:
    """`gemma.chat_json` behind the cache. Returns chat_json's dict plus `cached`, `key`.

    A cached success is re-validated (cheap) so a tightened validator cannot serve stale
    objects. `gemma.chat_json` is looked up at call time so tests can monkeypatch it.
    """
    c = cache or get_cache()
    key = make_key(task, prompt_version, system, user)
    hit = c.get(key)
    if hit is not None and (hit["ok"] or not retry_failed):
        if not hit["ok"] or validate is None or validate(hit["obj"]) is None:
            return {**hit, "attempts": 0, "cached": True, "key": key}
    res = _gemma.chat_json(system, user, max_tokens=max_tokens, validate=validate, retries=retries)
    c.put({"key": key, "task": task, "prompt_version": prompt_version,
           "input_sha": input_sha(system, user), "obj": res.get("obj"), "ok": res["ok"],
           "error": res.get("error"), "tokens": res.get("tokens"),
           "latency_s": res.get("latency_s")})
    return {**res, "cached": False, "key": key}
