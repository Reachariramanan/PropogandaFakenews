"""Per-claim feature vector (exactly schema.FEATURE_NAMES) and document roll-up."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from fnp.config import C
from fnp.math._util import fnum, sigmoid
from fnp.math.centrality import claim_centrality
from fnp.math.constants import M
from fnp.math.dempster_shafer import ds_fuse
from fnp.math.event_bn import timeline_inconsistency
from fnp.math.independence import assign_independence, cluster_n_eff
from fnp.math.infogain import kl_bernoulli_bits
from fnp.math.meta import meta_pool
from fnp.math.propaganda_score import density_per_100, one_sidedness, p_prop
from fnp.math.propagation import propagation_features
from fnp.math.reliability import assign_reliability
from fnp.math.woe import DIRECTION, strengths, woe_posterior
from fnp.schema import FEATURE_NAMES, LABELS, Verdict

NAN = float("nan")
_FALSE_KB = {"false", "fake", "misleading", "debunked", "FAKE", "MISLEADING"}


def _kb(kb_hits: Optional[Sequence[dict]]) -> tuple[float, float]:
    if kb_hits is None:
        return NAN, NAN
    best = 0.0
    for h in kb_hits:
        lab = h.get("label") or h.get("verdict")
        if lab is not None and str(lab) not in _FALSE_KB:
            continue                          # match to a claim rated true is not a debunk
        best = max(best, fnum(h.get("score", h.get("similarity", h.get("retrieval_score"))), 0.0))
    return float(best >= M("kb_match_min")), best


def prepare_evidence(evidence: Sequence[dict], embeddings: Optional[np.ndarray] = None,
                     track_records: Optional[dict] = None) -> list[dict]:
    """Reliability (if missing) + independence (always recomputed within the claim)."""
    return assign_independence(assign_reliability(evidence, track_records), embeddings)


def claim_features(claim: dict, evidence: Sequence[dict], spans: Optional[Sequence[dict]] = None,
                   propagation: Optional[dict] = None, kb_hits: Optional[Sequence[dict]] = None,
                   graph: Optional[dict] = None, embeddings: Optional[np.ndarray] = None,
                   n_tokens: Optional[int] = None, timeline_flags: Optional[Sequence[dict]] = None,
                   track_records: Optional[dict] = None) -> dict[str, float]:
    """Feature dict with exactly FEATURE_NAMES keys (NaN = unknown).

    graph: optional {"centrality": {claim_id: x}, "bp_marginal": {claim_id: x}} computed
    over the story/doc (see story_graph). Without it centrality is computed on this claim's
    own evidence and bp_marginal = p_true (an isolated node's marginal is its unary).
    spans=None means the tagger did not run (NaN); [] means it found nothing (0).
    """
    ev = prepare_evidence(evidence, embeddings, track_records)
    cid = claim.get("claim_id", "c")
    w = woe_posterior(ev)
    mp = meta_pool(ev)
    # Interval is centred on the WoE logit (the point estimate) with the DerSimonian-Laird
    # random-effects half-width; otherwise p_true could fall outside its own interval.
    if mp["k"] == 0:
        lo, hi = 0.0, 1.0
    else:
        hw = 0.5 * (mp["re_hi"] - mp["re_lo"])
        lo, hi = float(sigmoid(w["logit"] - hw)), float(sigmoid(w["logit"] + hw))
    ds = ds_fuse(ev)
    s, d, cm = strengths(ev)
    informative = (d != 0) | cm
    labels = np.array([e["origin_cluster"] for e in ev], dtype=int)
    unit_w = np.array([fnum(e.get("reliability"), 0) * fnum(e.get("relevance"), 1.0)
                       * e["independence"] for e in ev]) if ev else np.zeros(0)
    n_eff = cluster_n_eff(labels[informative], unit_w[informative]) if ev else 0.0
    rel = np.array([fnum(e.get("reliability"), NAN) for e in ev])
    ind = np.array([e["independence"] for e in ev])
    ok = informative & np.isfinite(rel) if ev else np.zeros(0, bool)
    src_rel = float(np.average(rel[ok], weights=ind[ok])) if ok.any() else NAN

    graph = graph or {}
    cen = graph.get("centrality", {}).get(cid)
    if cen is None:
        cen = claim_centrality({cid: ev})[cid] if ev else NAN
    bpm = graph.get("bp_marginal", {}).get(cid, w["p_true"])
    kb_match, kb_score = _kb(kb_hits)
    if spans is None:
        pp = dens = ntech = NAN
        one = one_sidedness(w["support"], w["contradiction"], None) if ev else NAN
    else:
        pp = p_prop(spans)
        dens = density_per_100(spans, n_tokens)
        ntech = float(len({sp.get("technique") for sp in spans}))
        one = one_sidedness(w["support"], w["contradiction"], spans)
    prop = propagation_features(propagation)

    f = {
        "p_true": w["p_true"], "p_true_lo": lo, "p_true_hi": hi,
        "support": w["support"], "contradiction": w["contradiction"],
        "context_mismatch": w["context_mismatch"],
        "ds_bel_true": ds["bel_T"], "ds_bel_false": ds["bel_F"],
        "ds_uncertainty": ds["uncertainty"], "ds_conflict": ds["conflict"],
        "n_evidence": float(len(ev)), "n_eff": n_eff, "i2": mp["I2"] if mp["k"] else NAN,
        "info_gain_bits": kl_bernoulli_bits(w["p_true"], w["prior"]), "centrality": cen,
        "kb_match": kb_match, "kb_match_score": kb_score,
        "p_prop": pp, "prop_density": dens, "prop_n_techniques": ntech, "one_sidedness": one,
        "check_worthy": fnum(claim.get("check_worthy")), "bp_marginal": float(bpm),
        "timeline_inconsistency": timeline_inconsistency(timeline_flags) if timeline_flags is not None else NAN,
        **prop,
        "source_reliability": src_rel,
    }
    assert tuple(f) == FEATURE_NAMES, "feature order drifted from schema.FEATURE_NAMES"
    return {k: float(v) for k, v in f.items()}


def story_graph(claim_p_true: dict[str, float], edges: Sequence[tuple],
                evidence_by_claim: Optional[dict] = None) -> dict:
    """Doc/story-level graph features for claim_features(graph=...).
    edges: (claim_id_a, claim_id_b, sign, strength)."""
    from fnp.math.bp import loopy_bp
    ids = list(claim_p_true)
    ix = {c: i for i, c in enumerate(ids)}
    e = [(ix[a], ix[b], sg, st) for a, b, sg, st in edges if a in ix and b in ix]
    bp = loopy_bp([claim_p_true[c] for c in ids], e)
    out = {"bp_marginal": dict(zip(ids, map(float, bp["marginals"]))), "bp_converged": bp["converged"]}
    if evidence_by_claim:
        out["centrality"] = claim_centrality(evidence_by_claim)
    return out


def doc_rollup(claim_verdicts: Sequence[Verdict], doc_id: str = "",
               doc_p_prop: Optional[float] = None) -> Verdict:
    """Document verdict from claim verdicts.

    Considered claims: those with features.check_worthy >= M('check_worthy_min'); all
    claims if none pass. Order of rules:
      1. FAKE        if any considered claim is FAKE with P(FAKE) >= M('doc_fake_min');
      2. MISLEADING  if any considered claim is FAKE or MISLEADING (a mix of false/out-of-
                     context with other claims, or a weak FAKE);
      3. PROPAGANDA  if any claim is PROPAGANDA or doc_p_prop >= C('prop_threshold');
      4. LEGIT       if every considered claim is LEGIT;
      5. INCONCLUSIVE otherwise (incl. no claims).
    Doc probs: noisy-OR over claims for FAKE/MISLEADING/PROPAGANDA, product for LEGIT,
    mean for INCONCLUSIVE, then normalised (descriptive, not calibrated).
    Secondary = other labels present among claims; plausibly_fake = any claim's flag.
    """
    vs = list(claim_verdicts)
    cw = [v for v in vs if fnum(v.get("features", {}).get("check_worthy"), 1.0) >= M("check_worthy_min")]
    cons = cw or vs
    labs = [v.get("label") for v in cons]
    pf = lambda v, k: fnum(v.get("probs", {}).get(k), 0.0)
    if not cons:
        label = "INCONCLUSIVE"
    elif any(l == "FAKE" and pf(v, "FAKE") >= M("doc_fake_min") for v, l in zip(cons, labs)):
        label = "FAKE"
    elif any(l in ("FAKE", "MISLEADING") for l in labs):
        label = "MISLEADING"
    elif "PROPAGANDA" in [v.get("label") for v in vs] or (doc_p_prop is not None and doc_p_prop >= C("prop_threshold")):
        label = "PROPAGANDA"
    elif all(l == "LEGIT" for l in labs):
        label = "LEGIT"
    else:
        label = "INCONCLUSIVE"
    probs = {}
    for k in LABELS:
        x = np.array([pf(v, k) for v in cons]) if cons else np.zeros(1)
        if k == "LEGIT":
            probs[k] = float(np.prod(x)) if cons else 0.0
        elif k == "INCONCLUSIVE":
            probs[k] = float(x.mean()) if cons else 1.0
        else:
            probs[k] = float(1 - np.prod(1 - x))
    z = sum(probs.values()) or 1.0
    probs = {k: v / z for k, v in probs.items()}
    sec = sorted({v.get("label") for v in vs if v.get("label") not in (label, None, "INCONCLUSIVE")})
    return Verdict(unit_id=doc_id, unit="doc", label=label, probs=probs,
                   plausibly_fake=any(v.get("plausibly_fake") for v in vs), secondary=sec,
                   prediction_set=[label], features={}, decider="rollup",
                   rationale=f"{len(cons)} considered claim(s): {labs}",
                   evidence_ids=[i for v in vs for i in v.get("evidence_ids", [])])
