"""SemEval-2020 Task 11 style scoring for propaganda spans (Da San Martino et al.).

Span identification (SI): predicted and gold spans are first merged per document when they
overlap, then with C(s,t,h) = |s ∩ t| / h
    P = Σ_{s,t same doc} C(s,t,|s|) / |S|        R = Σ C(s,t,|t|) / |T|
Technique-aware score ("TC-span"): same formula, but a (s,t) pair only counts when the
(merged-class) technique agrees. Gold overlapping spans of one technique are merged.
Per-technique P/R/F1 use the same rule restricted to one technique.

Spans are (start, end, technique) with half-open character offsets.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from ..schema import PROPAGANDA_TECHNIQUES

# 18 -> 14 SemEval-2020 merge. Mirrors fnp.schema.PTC_MERGE but is explicit so the scorer
# never silently drops a technique name.
MERGE14 = {
    "Whataboutism": "Whataboutism,Straw_Men,Red_Herring",
    "Straw_Men": "Whataboutism,Straw_Men,Red_Herring",
    "Red_Herring": "Whataboutism,Straw_Men,Red_Herring",
    "Bandwagon": "Bandwagon,Reductio_ad_hitlerum",
    "Reductio_ad_hitlerum": "Bandwagon,Reductio_ad_hitlerum",
}
# PTC gold only uses 14 classes (the corpus has no Obfuscation spans), so that class is not scored.
CLASSES14 = tuple(c for c in dict.fromkeys(MERGE14.get(t, t) for t in PROPAGANDA_TECHNIQUES)
                  if not c.startswith("Obfuscation"))


def to14(t: str) -> str:
    return MERGE14.get(t, t)


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(spans):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def _inter(a, b) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _pr(pred: list, gold: list) -> tuple[float, float, int, int]:
    """Sum C over pairs; returns (precision numerator, recall numerator, |pred|, |gold|)."""
    pn = rn = 0.0
    for s in pred:
        ls = s[1] - s[0]
        pn += sum(_inter(s, t) / ls for t in gold)
    for t in gold:
        lt = t[1] - t[0]
        rn += sum(_inter(s, t) / lt for s in pred)
    return pn, rn, len(pred), len(gold)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def score(gold: dict[str, Iterable[tuple]], pred: dict[str, Iterable[tuple]],
          docs: Optional[Iterable[str]] = None) -> dict:
    """gold/pred: doc_id -> iterable of (start, end, technique). Scores only `docs`
    (default: every gold doc). A doc missing from pred counts as predicting nothing."""
    docs = list(docs if docs is not None else gold.keys())
    si = [0.0, 0.0, 0, 0]
    tc = [0.0, 0.0, 0, 0]
    per = {c: [0.0, 0.0, 0, 0] for c in CLASSES14}
    for d in docs:
        g = [(int(s), int(e), to14(t)) for s, e, t in gold.get(d, [])]
        p = [(int(s), int(e), to14(t)) for s, e, t in pred.get(d, [])]
        gm = _merge([(s, e) for s, e, _ in g])
        pm = _merge([(s, e) for s, e, _ in p])
        a = _pr(pm, gm)
        for i in range(4):
            si[i] += a[i]
        for c in CLASSES14:
            gc = _merge([(s, e) for s, e, t in g if t == c])
            pc = _merge([(s, e) for s, e, t in p if t == c])
            b = _pr(pc, gc)
            for i in range(4):
                per[c][i] += b[i]
                tc[i] += b[i]

    def fin(x):
        pr = x[0] / x[2] if x[2] else 0.0
        rc = x[1] / x[3] if x[3] else 0.0
        return {"precision": pr, "recall": rc, "f1": _f1(pr, rc), "n_pred": x[2], "n_gold": x[3]}

    return {"n_docs": len(docs), "SI": fin(si), "TC_span": fin(tc),
            "per_technique": {c: fin(v) for c, v in per.items()}}
