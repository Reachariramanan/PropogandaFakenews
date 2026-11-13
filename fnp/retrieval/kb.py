"""Debunked / fact-checked claim KB (train splits only: fact-check sites, X-Fact, LIAR)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import EMBEDDING_MODEL, MODELS_DIR, RERANKER_MODEL

# Newest built KB unless FNP_KB pins one ("kb_v1" = the KB used by EVAL_v1).
KB_DIR = MODELS_DIR / os.environ.get(
    "FNP_KB", "kb_v2" if (MODELS_DIR / "kb_v2" / "emb.npy").exists() else "kb_v1")
_models: dict = {}


def _device_ok(min_gb: float = 3.0) -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**30 >= min_gb
    except Exception:
        return False


def get_embedder():
    """Lazy singleton bge-m3 SentenceTransformer (GPU fp16 if >=3 GB free, else CPU)."""
    if "emb" not in _models:
        from sentence_transformers import SentenceTransformer
        if _device_ok():
            m = SentenceTransformer(EMBEDDING_MODEL, device="cuda", model_kwargs={"torch_dtype": "float16"})
        else:
            m = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        m.max_seq_length = 256
        _models["emb"] = m
    return _models["emb"]


def get_reranker():
    if "rr" not in _models:
        from sentence_transformers import CrossEncoder
        _models["rr"] = CrossEncoder(RERANKER_MODEL, device="cuda" if _device_ok() else "cpu",
                                     max_length=256)
    return _models["rr"]


def embed_texts(texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
    v = get_embedder().encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
    return v.astype(np.float32)


def build_kb_frame() -> pd.DataFrame:
    """kb rows from TRAIN splits only. Columns: kb_id, claim, label5, raw_label, source, url,
    lang, evidence (debunk passage, never used as the claim)."""
    from ..datasets import load_factchecks, load_liar, load_xfact
    parts = []
    fc = load_factchecks()
    for g in (fc, load_xfact(), load_liar()):
        g = g[(g.split == "train") & g.label5.notna() & g.claim.notna()]
        parts.append(g)
    g = pd.concat(parts, ignore_index=True)

    def ev(m):
        try:
            d = json.loads(m)
            return (d.get("description") or "") + ("\n" + d["body"] if d.get("body") else "")
        except Exception:
            return ""
    out = pd.DataFrame(dict(kb_id=g.item_id, claim=g.claim, label5=g.label5, raw_label=g.raw_label,
                            source=g.dataset, url=g.url, lang=g.lang,
                            evidence=g.meta_json.map(ev).str.slice(0, 2000)))
    return out.drop_duplicates("kb_id").reset_index(drop=True)


class KB:
    def __init__(self, emb: np.ndarray, meta: pd.DataFrame,
                 embed_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
                 rerank_fn: Optional[Callable[[list[tuple[str, str]]], Sequence[float]]] = None):
        assert len(emb) == len(meta)
        self.emb = np.asarray(emb, dtype=np.float32)
        self.meta = meta.reset_index(drop=True)
        self._embed = embed_fn or embed_texts
        self._rerank = rerank_fn
        self._ids = self.meta.kb_id.tolist()

    @classmethod
    def load(cls, path: Path = KB_DIR, **kw) -> "KB":
        return cls(np.load(Path(path) / "emb.npy"), pd.read_parquet(Path(path) / "meta.parquet"), **kw)

    @classmethod
    def default(cls) -> "KB":
        if "kb" not in _models:
            _models["kb"] = cls.load()
        return _models["kb"]

    def __len__(self):
        return len(self.meta)

    def _rr(self, pairs):
        if self._rerank is not None:
            return self._rerank(pairs)
        return get_reranker().predict(pairs, batch_size=16, show_progress_bar=False)

    def search(self, text_or_list, k: int = 10, exclude_ids=(), rerank: bool = True):
        single = isinstance(text_or_list, str)
        texts = [text_or_list] if single else list(text_or_list)
        q = np.asarray(self._embed(texts), dtype=np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        sims = q @ self.emb.T
        ex = set(exclude_ids)
        if ex:
            sims[:, [i for i, x in enumerate(self._ids) if x in ex]] = -np.inf
        res = []
        for qi, t in enumerate(texts):
            n = min(50 if rerank else k, sims.shape[1])
            top = np.argpartition(-sims[qi], n - 1)[:n]
            top = top[np.argsort(-sims[qi][top])]
            top = [i for i in top if np.isfinite(sims[qi][i])]
            rows = []
            for i in top:
                m = self.meta.iloc[i]
                rows.append(dict(kb_id=m.kb_id, claim=m.claim, label5=m.label5, raw_label=m.raw_label,
                                 source=m.source, url=m.url, cosine=float(sims[qi][i]),
                                 rerank_score=None))
            if rerank and rows:
                sc = self._rr([(t, r["claim"]) for r in rows])
                for r, s in zip(rows, sc):
                    r["rerank_score"] = float(s)
                rows.sort(key=lambda r: -r["rerank_score"])
            res.append(rows[:k])
        return res[0] if single else res
