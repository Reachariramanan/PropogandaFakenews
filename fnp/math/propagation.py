"""Map an EDA-derived propagation dict onto feature keys (no EDA import here).

Missing keys / None / non-numeric -> NaN. Share-type keys (author_diversity, dup_rate,
coordination) are clipped to [0,1]; burst_intensity and platform_spread are kept raw (>= 0).
`propagation_risk` = mean of the available {dup_rate, coordination, 1-author_diversity,
1-exp(-burst_intensity)}; NaN if none are available.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from fnp.math._util import fnum

PROP_KEYS = ("burst_intensity", "platform_spread", "author_diversity", "dup_rate", "coordination")
_UNIT = {"author_diversity", "dup_rate", "coordination"}


def propagation_features(prop: Optional[dict]) -> dict:
    prop = prop or {}
    out = {}
    for k in PROP_KEYS:
        v = fnum(prop.get(k))
        if np.isfinite(v):
            v = float(np.clip(v, 0.0, 1.0)) if k in _UNIT else max(v, 0.0)
        out[k] = v
    return out


def propagation_risk(feats: dict) -> float:
    parts = [feats.get("dup_rate"), feats.get("coordination")]
    ad, bi = feats.get("author_diversity"), feats.get("burst_intensity")
    parts.append(None if ad is None else 1.0 - ad)
    parts.append(None if bi is None else 1.0 - np.exp(-bi))
    vals = [float(p) for p in parts if p is not None and np.isfinite(p)]
    return float(np.mean(vals)) if vals else float("nan")
