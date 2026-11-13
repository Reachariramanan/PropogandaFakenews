"""Timeline inconsistency as a noisy-OR (a one-child slice of the event-level BN, Method 7).

P(inconsistent) = 1 - (1 - leak) * prod_i (1 - w_i * p_i)
p_i = detector probability that flag i is true; w_i = P(flag alone implies inconsistency).
"""
from __future__ import annotations

from typing import Sequence

from fnp.math.constants import M

# assumed: causal strength of each timeline flag (see 02_ASSUMPTIONS_REGISTER.md)
FLAG_STRENGTH = {
    "media_older_than_event": 0.90,
    "date_order_impossible": 0.95,
    "future_reference": 0.80,
    "location_mismatch": 0.70,
    "speaker_not_present": 0.60,
}
DEFAULT_FLAG_STRENGTH = 0.5   # assumed: unknown flag names


def timeline_inconsistency(flags: Sequence[dict], leak: float | None = None) -> float:
    """flags: [{"name": str, "p": float, "w": optional float}]."""
    leak = M("timeline_leak") if leak is None else leak
    q = 1.0 - leak
    for f in flags:
        w = f.get("w", FLAG_STRENGTH.get(f.get("name"), DEFAULT_FLAG_STRENGTH))
        q *= 1.0 - min(max(w * float(f.get("p", 1.0)), 0.0), 1.0)
    return 1.0 - q
