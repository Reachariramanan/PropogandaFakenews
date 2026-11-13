"""Embedding search over EDA social posts (read-only; EDA parquet is never written)."""
from __future__ import annotations

import glob
from typing import Optional

import numpy as np
import pandas as pd

from ..config import CACHE_DIR
from ..eda_bridge import parquet_glob

_MM = CACHE_DIR / "eda_post_emb_f32.npy"
_IDS = CACHE_DIR / "eda_post_emb_ids.parquet"
_state: dict = {}


def _build_cache(chunk_rows: int = 20000) -> None:
    """Stream the (DOUBLE[]) embedding parquet in record batches into a float32 .npy memmap."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(parquet_glob("post_embeddings")))
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(str(_MM) + ".tmp.npy", mode="w+", dtype=np.float32, shape=(n, 1024))
    ids, i = [], 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=chunk_rows, columns=["post_uid", "embedding"]):
            arr = b.column("embedding")
            m = np.asarray(arr.flatten().to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(b), -1)
            nrm = np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
            mm[i:i + len(b)] = m / nrm
            ids += b.column("post_uid").to_pylist()
            i += len(b)
    mm.flush()
    del mm
    import os
    os.replace(str(_MM) + ".tmp.npy", _MM)
    pd.DataFrame({"post_uid": ids}).to_parquet(_IDS)


def _load():
    if "emb" not in _state:
        if not (_MM.exists() and _IDS.exists()):
            _build_cache()
        _state["emb"] = np.load(_MM, mmap_mode="r")
        _state["ids"] = pd.read_parquet(_IDS).post_uid.to_numpy()
    return _state


def _facts(uids: list[str]) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect()
    try:
        con.register("u", pd.DataFrame({"post_uid": uids}))
        return con.execute(
            f"SELECT f.post_uid, f.text, f.author_uid, f.platform, f.posted_at FROM "
            f"read_parquet('{parquet_glob('post_facts')}', union_by_name=true) f JOIN u USING(post_uid)").df()
    finally:
        con.close()


def search(vec_or_text, k: int = 20, exclude_post_uids=(), chunk: int = 50000) -> list[dict]:
    st = _load()
    if isinstance(vec_or_text, str):
        from .kb import embed_texts
        q = embed_texts([vec_or_text])[0]
    else:
        q = np.asarray(vec_or_text, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    emb, ids = st["emb"], st["ids"]
    ex = set(exclude_post_uids)
    best_s, best_i = [], []
    for s in range(0, len(emb), chunk):
        sc = np.asarray(emb[s:s + chunk]) @ q
        kk = min(k + len(ex), len(sc))
        top = np.argpartition(-sc, kk - 1)[:kk]
        best_s.append(sc[top]); best_i.append(top + s)
    sc = np.concatenate(best_s); ix = np.concatenate(best_i)
    order = np.argsort(-sc)
    sel = [(float(sc[o]), int(ix[o])) for o in order if ids[ix[o]] not in ex][:k]
    uids = [str(ids[i]) for _, i in sel]
    f = _facts(uids).drop_duplicates("post_uid").set_index("post_uid")
    out = []
    for (s, i), u in zip(sel, uids):
        r = f.loc[u] if u in f.index else None
        out.append(dict(post_uid=u, text=None if r is None else r.text,
                        author_uid=None if r is None else r.author_uid,
                        platform=None if r is None else r.platform,
                        posted_at=None if r is None else str(r.posted_at), cosine=s))
    return out
