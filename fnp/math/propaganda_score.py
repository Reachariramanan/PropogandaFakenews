"""Propaganda scores from tagged spans + span-level F1 of Da San Martino et al. (2019).

P_prop = 1 - prod_i (1 - c_i * pi_t(i))       (noisy-OR of spans, c = span confidence,
                                               pi_t = tagger precision for technique t)
density = 100 * n_spans / n_tokens
entropy = Shannon entropy (bits) of the technique distribution
one_sidedness = mean of available {stance asymmetry |S-C|/(S+C), emotive-technique share}

Span F1 (Da San Martino et al. 2019, sec. 4): for predicted spans S and gold spans T,
  C(s, t, h) = |s ∩ t| / h * delta(l(s), l(t))
  P(S, T) = 1/|S| sum_{s in S} sum_{t in T} C(s, t, |s|)
  R(S, T) = 1/|T| sum_{s in S} sum_{t in T} C(s, t, |t|)
delta = 1 when techniques match (or always, for the technique-agnostic variant).
Spans only intersect within the same doc_id. Empty S gives P = 0; empty T gives R = 0.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Sequence

import numpy as np

from fnp.math.constants import M
from fnp.schema import PTC_MERGE

EMOTIVE_TECHNIQUES = frozenset({
    "Loaded_Language", "Name_Calling,Labeling", "Exaggeration,Minimisation",
    "Appeal_to_fear-prejudice", "Flag-Waving", "Slogans", "Black-and-White_Fallacy",
    "Reductio_ad_hitlerum", "Thought-terminating_Cliches",
})


def p_prop(spans: Sequence[dict], precision: Optional[dict] = None) -> float:
    prec = precision or {}
    dflt = M("technique_precision_default")
    q = 1.0
    for s in spans:
        c = float(s.get("confidence", 1.0) if s.get("confidence") is not None else 1.0)
        q *= 1.0 - min(max(c * prec.get(s.get("technique"), dflt), 0.0), 1.0)
    return 1.0 - q


def density_per_100(spans: Sequence[dict], n_tokens: Optional[int]) -> float:
    if not n_tokens:
        return float("nan")
    return 100.0 * len(spans) / n_tokens


def technique_entropy(spans: Sequence[dict]) -> float:
    cnt = Counter(s.get("technique") for s in spans)
    n = sum(cnt.values())
    if n == 0:
        return 0.0
    return float(-sum(v / n * math.log2(v / n) for v in cnt.values()))


def one_sidedness(support: float = float("nan"), contradiction: float = float("nan"),
                  spans: Optional[Sequence[dict]] = None) -> float:
    parts = []
    tot = (support or 0.0) + (contradiction or 0.0)
    if np.isfinite(tot) and tot > 0:
        parts.append(abs(support - contradiction) / tot)
    if spans:
        parts.append(sum(s.get("technique") in EMOTIVE_TECHNIQUES for s in spans) / len(spans))
    return float(np.mean(parts)) if parts else float("nan")


def _overlap(a: dict, b: dict) -> int:
    if a.get("doc_id") != b.get("doc_id"):
        return 0
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def span_prf(pred: Sequence[dict], gold: Sequence[dict], match_technique: bool = True,
             merge_ptc: bool = False) -> dict:
    def lab(s):
        t = s.get("technique")
        return PTC_MERGE.get(t, t) if merge_ptc else t

    P_sum = R_sum = 0.0
    for s in pred:
        for t in gold:
            if match_technique and lab(s) != lab(t):
                continue
            ov = _overlap(s, t)
            if ov:
                P_sum += ov / max(s["end"] - s["start"], 1)
                R_sum += ov / max(t["end"] - t["start"], 1)
    P = P_sum / len(pred) if pred else 0.0
    R = R_sum / len(gold) if gold else 0.0
    F = 2 * P * R / (P + R) if P + R > 0 else 0.0
    return {"precision": P, "recall": R, "f1": F}
