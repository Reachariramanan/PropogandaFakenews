"""Convert FNP rows into the JSONL that `laya.train` reads, and split deterministically.

Laya row format (read by `laya.train.read_jsonl` / `items_from_rows`)::

    {"state": <str>,
     "questions": {qid: {"type": "choice"|"noul"|"score", "instructions": str, "criteria": ...}},
     "gold": {qid: {"probabilities": {<choice label> | "false"/"true": p, ...}}}}

`gold.probabilities` are normalised by the package, so SOFT targets are supported natively
(the alternative `expected` form is one-hot only). A question without a gold entry is simply
not trained on for that row, which is how rows with only a verdict (no noul labels) work.
Extra keys (`id`, `src`, `meta`) are ignored by the package and used by our RL stage.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Optional

from fnp.schema import LABELS

from .state_builder import QUESTIONS

BETA_DEFAULT = 0.3   # assumed (docs/review/02_ASSUMPTIONS_REGISTER.md): weight of Gemma probs vs gold
SPLIT_FRACTIONS = {"train": 0.8, "calib": 0.1, "eval": 0.1}


def verdict_target(gold: Optional[str], gemma_probs: Optional[dict], beta: float = BETA_DEFAULT) -> Optional[dict]:
    """target = (1-beta)*onehot(gold) + beta*gemma ; gold-only -> one-hot ; gemma-only -> gemma."""
    g = None
    if gold is not None:
        if gold not in LABELS:
            raise ValueError(f"unknown gold label {gold!r}")
        g = [1.0 if l == gold else 0.0 for l in LABELS]
    m = None
    if gemma_probs:
        v = [max(0.0, float(gemma_probs.get(l, 0.0))) for l in LABELS]
        s = sum(v)
        if s > 0 and all(math.isfinite(x) for x in v):
            m = [x / s for x in v]
    if g is not None and m is not None:
        t = [(1 - beta) * a + beta * b for a, b in zip(g, m)]
    else:
        t = g if g is not None else m
    return None if t is None else {l: round(x, 6) for l, x in zip(LABELS, t)}


def noul_target(v) -> Optional[dict]:
    if v is None:
        return None
    p = float(v)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"noul target must be in [0,1], got {v!r}")
    return {"false": round(1 - p, 6), "true": round(p, 6)}


def make_row(state: str, *, gold: Optional[str] = None, gemma_probs: Optional[dict] = None,
             noul: Optional[dict] = None, beta: float = BETA_DEFAULT, row_id: Optional[str] = None,
             questions: Optional[Iterable[str]] = None, meta: Optional[dict] = None,
             features: Optional[dict] = None) -> Optional[dict]:
    """One Laya row, or None if the row carries no label at all.

    Extra keys for stage B (RLCD): `reward_target` (one-hot gold where available, else the Gemma
    consensus distribution -- the soft training target is NOT used for reward) and `strong`
    (evidence is strong: n_eff >= 2 and ds_uncertainty < 0.3, so INCONCLUSIVE is penalised).
    """
    qids = list(questions) if questions is not None else list(QUESTIONS)
    goldq = {}
    vt = verdict_target(gold, gemma_probs, beta)
    if vt is not None and "verdict" in qids:
        goldq["verdict"] = {"probabilities": vt}
    for qid, val in (noul or {}).items():
        nt = noul_target(val)
        if nt is not None and qid in qids:
            goldq[qid] = {"probabilities": nt}
    if not goldq:
        return None
    rt = verdict_target(gold, None) if gold else verdict_target(None, gemma_probs)
    f = features or {}
    try:
        strong = float(f.get("n_eff", 0)) >= 2 and float(f.get("ds_uncertainty", 1)) < 0.3
    except (TypeError, ValueError):
        strong = False
    row = {"id": row_id or hashlib.sha1(state.encode()).hexdigest()[:16],
           "state": state,
           "questions": {q: QUESTIONS[q] for q in qids},
           "gold": goldq,
           "reward_target": rt, "strong": bool(strong),
           "src": ("gold+gemma" if gold and gemma_probs else "gold" if gold else "gemma")}
    if meta:
        row["meta"] = meta
    return row


def split_of(row_id: str, fractions: dict = SPLIT_FRACTIONS, salt: str = "fnp-laya-v1") -> str:
    """Deterministic split from a hash of the id: stable when rows are added or reordered."""
    h = int(hashlib.sha256(f"{salt}:{row_id}".encode()).hexdigest()[:8], 16) / 0x100000000
    acc = 0.0
    for name, frac in fractions.items():
        acc += frac
        if h < acc:
            return name
    return list(fractions)[-1]


def split_rows(rows: Iterable[dict], fractions: dict = SPLIT_FRACTIONS) -> dict[str, list[dict]]:
    out = {k: [] for k in fractions}
    for r in rows:
        out[split_of(r["id"], fractions)].append(r)
    return out


def write_jsonl(rows: Iterable[dict], path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def write_splits(rows: Iterable[dict], out_dir, fractions: dict = SPLIT_FRACTIONS) -> dict[str, int]:
    out_dir = Path(out_dir)
    parts = split_rows(rows, fractions)
    return {k: write_jsonl(v, out_dir / f"{k}.jsonl") for k, v in parts.items()}


def read_jsonl(path) -> list[dict]:
    from laya.train import read_jsonl as _r
    return _r(str(path))
