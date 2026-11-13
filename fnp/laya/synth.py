"""Synthetic-but-plausible verdict records for smoke tests and plumbing checks (NOT for any claim
about accuracy: labels are generated from the same features the state shows)."""
from __future__ import annotations

import random

from fnp.schema import LABELS, TYPE_RELIABILITY

from .dataset import make_row
from .state_builder import build_state

_TOPICS = ["vaccine side effects", "election results", "flood relief funds", "border incident",
           "stock market crash", "school closures", "water contamination", "new tax rules"]
_STANCE = {"FAKE": "refute", "LEGIT": "support", "MISLEADING": "context_mismatch",
           "PROPAGANDA": "neutral", "INCONCLUSIVE": "neutral"}


def synth_records(n: int, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        label = LABELS[i % len(LABELS)] if rnd.random() < 0.8 else rnd.choice(LABELS)
        topic = rnd.choice(_TOPICS)
        strong = label in ("FAKE", "LEGIT", "MISLEADING") and rnd.random() < 0.8
        n_ev = rnd.randint(2, 6) if strong else rnd.randint(0, 2)
        pt = {"FAKE": 0.1, "LEGIT": 0.9, "MISLEADING": 0.45, "PROPAGANDA": 0.5, "INCONCLUSIVE": 0.5}[label]
        pt = min(0.99, max(0.01, pt + rnd.gauss(0, 0.08)))
        feats = {"p_true": pt, "p_true_lo": max(0, pt - 0.2), "p_true_hi": min(1, pt + 0.2),
                 "support": pt * n_ev, "contradiction": (1 - pt) * n_ev, "n_evidence": n_ev,
                 "n_eff": max(0.0, n_ev - rnd.random()), "ds_uncertainty": 0.15 if strong else 0.7 + 0.2 * rnd.random(),
                 "p_prop": 0.85 if label == "PROPAGANDA" else rnd.random() * 0.3,
                 "one_sidedness": 0.8 if label == "PROPAGANDA" else rnd.random() * 0.4,
                 "kb_match_score": rnd.random() * (0.9 if label == "FAKE" else 0.3),
                 "context_mismatch": 0.8 if label == "MISLEADING" else rnd.random() * 0.2}
        ev = [{"text": f"Report on {topic}: officials and fact-checkers comment (item {j}).",
               "source": rnd.choice(["PTI", "BOOM", "Reuters", "blog"]), "stance": _STANCE[label],
               "evidence_type": rnd.choice(["FACT_CHECK", "MEDIA_REPORT", "OFFICIAL_RECORD"]),
               "reliability": TYPE_RELIABILITY["FACT_CHECK"] - 0.1 * rnd.random(), "independence": 0.9,
               "relevance": 0.8, "stance_conf": 0.8} for j in range(n_ev)]
        claim = {"claim": f"Claim {i} about {topic}: " + ("!!! They are hiding the truth, share now!" if label == "PROPAGANDA"
                                                          else "an event is reported to have happened."),
                 "claim_type": "factual", "lang": "en"}
        spans = {"n_spans": 3, "techniques": ["Loaded_Language", "Appeal_to_fear-prejudice"]} if label == "PROPAGANDA" else None
        state = build_state(claim, feats, ev, spans, None)
        probs = {l: (0.6 if l == label else 0.1) for l in LABELS}
        rows.append(make_row(state, gold=label, gemma_probs=probs, row_id=f"synth{seed}-{i}",
                             noul={"is_propaganda": float(label == "PROPAGANDA"),
                                   "context_mismatch": float(label == "MISLEADING"),
                                   "check_worthy": 1.0 if label != "INCONCLUSIVE" else 0.5},
                             features=feats))
    return rows
