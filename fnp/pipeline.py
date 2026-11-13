"""Per-claim and per-document analysis shared by scripts/05_run_units.py and 15_predict.py.

analyse_claim: KB retrieval -> Gemma stance per hit -> maths features -> rule head ->
               Gemma adjudicator on the structured state. Propaganda spans are passed in
               (tagged once per document) so every claim of a document shares them.
analyse_text:  claim extraction (long text) or the text itself (short claim) -> spans ->
               analyse_claim per claim -> final verdicts -> doc rollup.
"""
from __future__ import annotations

import math
import pickle
from typing import Optional, Sequence

from fnp.config import MODELS_DIR
from fnp.laya.state_builder import build_state
from fnp.llm import adjudicate, propaganda, stance
from fnp.llm.claims import extract_claims
from fnp.math.features import claim_features, doc_rollup
from fnp.math.verdict import final_verdict, rule_head
from fnp.retrieval.kb import KB
from fnp.schema import TYPE_RELIABILITY
from fnp.text_utils import sha16

K_EVIDENCE = 5
SHORT_CLAIM_WORDS = 60     # below this a text is treated as a single claim (no extraction)


def kb_evidence_text(hit: dict) -> str:
    return (f"Fact-check by {hit['source']}: the claim \"{hit['claim']}\" was rated "
            f"'{hit['raw_label']}'.")


def _clean(feats: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else round(float(v), 3))
            for k, v in feats.items()}


def analyse_claim(claim: dict, spans: Sequence[dict], spans_summary: dict, n_tokens: int,
                  hits: Optional[list] = None, exclude_ids: Sequence[str] = ()) -> dict:
    if hits is None:
        hits = KB.default().search(claim["claim"], k=K_EVIDENCE, exclude_ids=tuple(exclude_ids))
    evidence, kb_hits = [], []
    for h in hits:
        j = stance.judge(claim, kb_evidence_text(h), {"source": h["source"], "url": h.get("url")})
        if not j.get("ok"):
            continue
        et = j.get("evidence_type") or "FACT_CHECK"
        evidence.append({
            "evidence_id": f"kb:{h['kb_id']}", "claim_id": claim["claim_id"],
            "text": h["claim"], "source": h["source"], "url": h.get("url"),
            "origin_key": h["kb_id"], "evidence_type": et, "stance": j["stance"],
            "stance_conf": j["stance_conf"], "relevance": j["relevance"],
            "reliability": TYPE_RELIABILITY.get(et, 0.3), "retrieval": "kb",
            "retrieval_score": h.get("rerank_score") or h.get("cosine")})
        if j["stance"] in ("refute", "context_mismatch"):
            kb_hits.append({"label": h["label5"],
                            "score": min(h.get("rerank_score") or 0.0, j["stance_conf"])})
    feats = claim_features(claim, evidence, spans=list(spans), kb_hits=kb_hits, n_tokens=n_tokens)
    feats["check_worthy"] = claim.get("check_worthy", float("nan"))
    rule = rule_head(feats)
    clean_f = _clean(feats)
    state = {"claim": {k: claim.get(k) for k in ("claim", "speaker", "date", "claim_type", "lang")},
             "features": clean_f, "rule_head": {k: round(v, 3) for k, v in rule.items()},
             "evidence": [{k: e[k] for k in ("evidence_id", "source", "evidence_type", "stance",
                                               "stance_conf", "reliability")}
                          | {"kb_claim": e["text"][:300]} for e in evidence],
             "propaganda": spans_summary}
    adj = adjudicate.decide(state)
    return {"claim": claim, "evidence": evidence, "features": clean_f, "features_raw": feats,
            "rule": rule,
            "gemma_adj": {"ok": adj.get("ok"), "probs": adj.get("probs"), "label": adj.get("label"),
                          "rationale": adj.get("rationale")},
            "state_text": build_state(claim, clean_f, evidence, spans_summary, None)}


def tag_document(doc_id: str, text: str) -> tuple[list, dict, int]:
    spans = propaganda.tag_spans(doc_id, text)
    summary = {"n": len(spans), "techniques": sorted({s["technique"] for s in spans}),
               "one_sided": getattr(spans, "one_sided", None)}
    return list(spans), summary, max(len(text.split()), 1)


def load_deciders(laya_path: Optional[str] = None) -> dict:
    p = MODELS_DIR / "deciders_v1.pkl"
    d = pickle.load(p.open("rb")) if p.exists() else {}
    if laya_path:
        from fnp.laya.predict import LayaDecider
        d["laya"] = LayaDecider.load(laya_path)
    return d


def analyse_text(doc_id: str, text: str, lang: Optional[str] = None,
                 deciders: Optional[dict] = None, exclude_ids: Sequence[str] = ()) -> dict:
    deciders = deciders or {}
    if len(text.split()) <= SHORT_CLAIM_WORDS:
        claims = [{"claim_id": sha16(doc_id, text), "doc_id": doc_id, "claim": text.strip(),
                   "lang": lang, "claim_type": "factual", "check_worthy": 1.0}]
    else:
        claims = [dict(c) for c in extract_claims(doc_id, text, lang)]
    spans, summary, ntok = tag_document(doc_id, text)
    laya = deciders.get("laya")
    use_laya = laya is not None and deciders.get("stacker_laya") is not None
    st = deciders.get("stacker_laya" if use_laya else "stacker")
    cf = deciders.get("conformal_laya" if use_laya else "conformal")
    out = []
    for c in claims:
        r = analyse_claim(c, spans, summary, ntok, exclude_ids=exclude_ids)
        lp = laya.predict([r["state_text"]])[0]["probs"] if laya is not None else None
        gp = r["gemma_adj"]["probs"] if r["gemma_adj"]["ok"] else None
        v = final_verdict(c["claim_id"], r["features_raw"], gp, lp if use_laya else None,
                          stacker=st, conformal=cf,
                          rationale=(r["gemma_adj"].get("rationale") or "")[:500],
                          evidence_ids=[e["evidence_id"] for e in r["evidence"]])
        v["doc_id"] = doc_id
        r["verdict"] = v
        out.append(r)
    doc = doc_rollup([r["verdict"] for r in out], doc_id=doc_id,
                     doc_p_prop=out[0]["features"].get("p_prop") if out else None)
    doc["doc_id"] = doc_id
    return {"doc": doc, "claims": out, "spans": spans, "spans_summary": summary}
