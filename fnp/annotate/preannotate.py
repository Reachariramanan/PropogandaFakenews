"""Model pre-annotations -> annot.suggestion (annotators correct them; blind items never show them).

Readers of raw text here are the existing L1 claim extractor (Gemma, verbatim-quote validated) and the
fine-tuned span encoder (propaganda techniques). Both write only to `suggestion`, never to human-label tables.

  .venv/bin/python -m fnp.annotate.preannotate --limit 100            # dry run: counts only
  .venv/bin/python -m fnp.annotate.preannotate --limit 100 --apply
  options: --no-gemma / --no-encoder, --workers N (Gemma concurrency, capped at 12)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ..config import FNP_ROOT
from .db import tx

ENC_VERSION = "enc_v1"
CLAIM_MODEL_PREFIX = "gemma_"


def map_claim_type(c: dict) -> str:
    """Existing L1 schema -> form-of-claim taxonomy. Conservative: unknown forms stay UNCLEAR, not guessed."""
    qr = (c.get("quoted_or_reported") or "").lower()
    ct = (c.get("claim_type") or "").lower()
    if ct == "prediction":
        return "PREDICTION_OR_FORECAST"
    if qr in ("quoted", "reported") or ct == "quote":
        return "REPORTED_OR_REQUOTED_STATEMENT"
    if ct in ("factual", "statistical", "media_attribution") and qr == "asserted":
        return "DIRECT_STATEMENT"
    if ct == "opinion" or ct == "other":
        return "OTHER"
    return "UNCLEAR"


def claim_suggestions(text: str, claims: list[dict]) -> list[dict]:
    """Gemma claims -> span suggestions. Only claims whose evidence_quote is found verbatim get a span."""
    out = []
    for c in claims:
        q = c.get("evidence_quote") or ""
        i = text.find(q) if q.strip() else -1
        if i < 0:
            continue
        out.append(dict(start=i, end=i + len(q), text=q, label=map_claim_type(c),
                        confidence=float(c.get("check_worthy") or 0.0),
                        payload={"normalised_claim": c.get("claim"), "speaker": c.get("speaker"),
                                 "check_worthy_score": c.get("check_worthy"), "l1_claim_type": c.get("claim_type"),
                                 "quoted_or_reported": c.get("quoted_or_reported")}))
    return out


def pending(layer_model: str, limit: int, include_blind: bool = True) -> list[dict]:
    """Items this model has not seen yet, pool first (served-soon items matter most for class-balanced batching).
    Hash order, not id order: a partial run (Gemma is slow) then covers the pool evenly instead of only the oldest rows.
    Blind items are included: their predictions only steer selection and are never shown to annotators."""
    with tx() as cur:
        cur.execute("""SELECT i.item_id, i.text, i.lang FROM item i
                       WHERE i.status IN ('queued','in_progress') AND (%s OR NOT i.is_blind)
                         AND NOT EXISTS (SELECT 1 FROM suggestion s WHERE s.item_id=i.item_id AND s.model_version=%s)
                         AND NOT EXISTS (SELECT 1 FROM suggestion_done d WHERE d.item_id=i.item_id AND d.model_version=%s)
                         AND i.lang_group IN ('en','hi')
                       ORDER BY (i.status = 'queued') DESC, md5(i.item_id::text) LIMIT %s""", (include_blind, layer_model, layer_model, limit))
        return [dict(r) for r in cur.fetchall()]


def _store(item_id: int, layer: str, model: str, sugg: list[dict]) -> int:
    with tx() as cur:
        for s in sugg:
            cur.execute("INSERT INTO suggestion (item_id, layer, model_version, start_char, end_char, span_text, label, "
                        "confidence, payload) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (item_id, layer, model, s["start"], s["end"], s["text"], s.get("label"), s.get("confidence"),
                         json.dumps(s.get("payload") or {})))
        # remember that the model ran on this item even when it found nothing, so it is not re-run
        cur.execute("INSERT INTO suggestion_done (item_id, model_version) VALUES (%s,%s) ON CONFLICT DO NOTHING", (item_id, model))
    return len(sugg)


def run(limit: int, apply: bool, use_encoder: bool = True, use_gemma: bool = True, workers: int = 8,
        include_blind: bool = True) -> dict:
    res = {"encoder": {"items": 0, "spans": 0}, "gemma": {"items": 0, "spans": 0}}
    if use_encoder:
        todo = pending(ENC_VERSION, limit, include_blind)
        res["encoder"]["items"] = len(todo)
        if apply and todo:
            from ..llm.propaganda_enc import tag_spans_enc
            for it in todo:
                sp = tag_spans_enc(f"item:{it['item_id']}", it["text"])
                res["encoder"]["spans"] += _store(it["item_id"], "propaganda", ENC_VERSION, [
                    dict(start=s["start"], end=s["end"], text=s["text"], label=s["technique"], confidence=s["confidence"])
                    for s in sp])
    if use_gemma:
        from ..llm import prompts as P
        model = CLAIM_MODEL_PREFIX + str(P.CLAIMS_VERSION)
        todo = pending(model, limit, include_blind)
        res["gemma"]["items"] = len(todo)
        if apply and todo:
            from .. import gemma
            from ..llm.cache import reset_default_cache
            from ..llm.claims import extract_claims
            reset_default_cache(FNP_ROOT / "data/cache/annot_preannot.duckdb")
            print("gemma host:", gemma.endpoint_host(), "| items:", len(todo), flush=True)

            def one(it):
                lang = it["lang"] if it["lang"] not in (None, "unknown") else None
                r = extract_claims(f"item:{it['item_id']}", it["text"], lang)
                return (it, None) if not r.ok else (it, [dict(c) for c in r])

            for it, claims in gemma.run_batch(todo, one, min(workers, 12)):
                if claims is None:      # transient failure: leave pending so a later run retries
                    continue
                res["gemma"]["spans"] += _store(it["item_id"], "claim", model, claim_suggestions(it["text"], claims))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-gemma", action="store_true")
    ap.add_argument("--no-encoder", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-blind", action="store_true",
                    help="do not pre-annotate blind items (default: do, because the predictions steer class-balanced batch "
                         "selection; blind items still never SHOW suggestions to annotators)")
    a = ap.parse_args()
    print(json.dumps(run(a.limit, a.apply, not a.no_encoder, not a.no_gemma, a.workers, not a.skip_blind)))
    if not a.apply:
        print("dry run: nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
