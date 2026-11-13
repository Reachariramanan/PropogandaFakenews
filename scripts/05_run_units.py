"""Stage 05: run the full per-unit pipeline on a stratified sample of gold items.

unit = one claim (claim datasets) or one article reduced to its most check-worthy claim
(PTC). For every unit:
  KB retrieval (top-k, own id excluded) -> Gemma stance per hit -> propaganda spans
  (run on EVERY unit, so missing-span indicators cannot leak the dataset) ->
  fnp.math.claim_features -> rule head -> Gemma adjudicator (structured state only) ->
  Gemma silver label on raw claim (the "Gemma-only" ablation baseline).

Writes data/runs/units_v1.jsonl (append, resumable by unit_id). Dry run = 10 units, printed.
  .venv/bin/python scripts/05_run_units.py                # pilot, no write
  .venv/bin/python scripts/05_run_units.py --apply        # full sample
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402

from fnp.config import DATA_DIR, DUCKDB_PATH, gemma_config  # noqa: E402
from fnp.laya.state_builder import build_state  # noqa: E402
from fnp.llm import adjudicate, propaganda, silver, stance  # noqa: E402
from fnp.llm.claims import extract_claims  # noqa: E402
from fnp.math.features import claim_features  # noqa: E402
from fnp.math.verdict import rule_head  # noqa: E402
from fnp.retrieval.kb import KB  # noqa: E402
from fnp.schema import TYPE_RELIABILITY  # noqa: E402
from fnp.text_utils import sha16  # noqa: E402

OUT = DATA_DIR / "runs" / "units_v1.jsonl"
K_EVIDENCE = 5
# Per-dataset sample sizes per split (stratified by label5). None = all rows.
SAMPLE = {
    "liar": {"train": 600, "dev": 300, "test": 400},
    "xfact": {"train": 600, "dev": 300, "test": 400},
    "boomlive": {"train": None, "dev": None, "test": None},
    "fakenewsnet": {"all": 300},
    "ptc": {"train": None},
}


def hsplit(item_id: str) -> str:
    """Deterministic 60/20/20 split for datasets without official splits."""
    h = int(hashlib.sha1(item_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 60 else ("dev" if h < 80 else "test")


def select_items(con) -> list[dict]:
    rows = []
    for ds, splits in SAMPLE.items():
        for sp, n in splits.items():
            df = con.execute(
                "SELECT * FROM gold_items_v1 WHERE dataset=? AND split=? AND label5 IS NOT NULL",
                [ds, sp]).df()
            df["_h"] = df["item_id"].map(lambda x: hashlib.sha1(x.encode()).hexdigest())
            df = df.sort_values("_h")
            if n is not None:   # stratified: equal share per label, capped by availability
                labs = df["label5"].unique()
                per = math.ceil(n / max(len(labs), 1))
                df = df.groupby("label5", group_keys=False).head(per).head(n)
            for r in df.to_dict("records"):
                r["eval_split"] = hsplit(r["item_id"]) if sp in ("all",) or ds == "ptc" else sp
                rows.append(r)
    return rows


def kb_evidence(hit: dict) -> tuple[str, dict]:
    text = (f"Fact-check by {hit['source']}: the claim \"{hit['claim']}\" was rated "
            f"'{hit['raw_label']}'.")
    return text, {"source": hit["source"], "url": hit.get("url")}


def unit_claim(item: dict) -> tuple[dict, str]:
    """-> (claim dict, text that propaganda tagging runs on)."""
    if item["dataset"] == "ptc":
        text = item["text"] or ""
        cl = extract_claims(item["item_id"], text, item.get("lang"))
        if cl:
            best = max(cl, key=lambda c: c.get("check_worthy", 0))
            return dict(best), text
        first = text.strip().split("\n")[0][:300]
        return {"claim_id": sha16(item["item_id"], first), "doc_id": item["item_id"],
                "claim": first, "claim_type": "other", "check_worthy": 0.0}, text
    c = item["claim"] or item["text"]
    return {"claim_id": sha16(item["item_id"], c), "doc_id": item["item_id"], "claim": c,
            "speaker": item.get("speaker"), "date": item.get("date"), "lang": item.get("lang"),
            "claim_type": "factual", "check_worthy": 1.0}, c


def run_unit(item: dict, hits: list[dict]) -> dict:
    t0 = time.time()
    claim, ptext = unit_claim(item)
    if item["dataset"] == "ptc":       # retrieval for PTC uses the extracted claim
        hits = KB.default().search(claim["claim"], k=K_EVIDENCE, exclude_ids=(item["item_id"],))
    evidence, kb_hits = [], []
    judged = [stance.judge(claim, *kb_evidence(h)) for h in hits]
    for h, j in zip(hits, judged):
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
            kb_hits.append({"label": h["label5"], "score": min(h.get("rerank_score") or 0.0,
                                                                j["stance_conf"])})
    spans = propaganda.tag_spans(item["item_id"], ptext)
    n_tok = max(len(ptext.split()), 1)
    feats = claim_features(claim, evidence, spans=list(spans), kb_hits=kb_hits, n_tokens=n_tok)
    feats["check_worthy"] = claim.get("check_worthy", float("nan"))
    rule = rule_head(feats)
    spans_summary = {"n": len(spans), "techniques": sorted({s["technique"] for s in spans}),
                     "one_sided": getattr(spans, "one_sided", None)}
    clean_f = {k: (None if isinstance(v, float) and math.isnan(v) else round(float(v), 3))
               for k, v in feats.items()}
    state = {"claim": {k: claim.get(k) for k in ("claim", "speaker", "date", "claim_type", "lang")},
             "features": clean_f, "rule_head": {k: round(v, 3) for k, v in rule.items()},
             "evidence": [{k: e[k] for k in ("evidence_id", "source", "evidence_type", "stance",
                                               "stance_conf", "reliability")} | {"kb_claim": e["text"][:300]}
                          for e in evidence],
             "propaganda": spans_summary}
    adj = adjudicate.decide(state)
    sil = silver.label({"claim": claim["claim"], "speaker": claim.get("speaker"),
                        "date": claim.get("date")})
    return {
        "unit_id": claim["claim_id"], "item_id": item["item_id"], "dataset": item["dataset"],
        "split": item["eval_split"], "label5": item["label5"], "claim": claim,
        "evidence": evidence, "spans": list(spans), "spans_summary": spans_summary,
        "features": clean_f, "rule": rule,
        "gemma_adj": {"ok": adj.get("ok"), "probs": adj.get("probs"), "label": adj.get("label"),
                      "rationale": adj.get("rationale")},
        "gemma_silver": {"ok": sil.get("ok"), "probs": sil.get("probs"), "label": sil.get("label")},
        "state_text": build_state(claim, clean_f, evidence, spans_summary, None),
        "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    items = select_items(con)
    con.close()
    done = set()
    if OUT.exists():
        done = {json.loads(l)["item_id"] for l in OUT.open()}
    items = [i for i in items if i["item_id"] not in done]
    if not a.apply:
        items = items[:: max(1, len(items) // 10)][:10]
    elif a.limit:
        items = items[: a.limit]
    print(f"{len(items)} units to run ({len(done)} already done); gemma concurrency "
          f"{gemma_config()['concurrency']}", flush=True)

    kb = KB.default()
    claim_items = [i for i in items if i["dataset"] != "ptc"]
    hits = {}
    for s in range(0, len(claim_items), 64):    # batched retrieval on the main thread
        chunk = claim_items[s:s + 64]
        for it in chunk:
            hits[it["item_id"]] = kb.search(it["claim"] or it["text"], k=K_EVIDENCE,
                                            exclude_ids=(it["item_id"],))
    print("retrieval done", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fout = OUT.open("a") if a.apply else None
    n_ok = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=gemma_config()["concurrency"]) as ex:
        futs = [ex.submit(run_unit, it, hits.get(it["item_id"], [])) for it in items]
        for k, f in enumerate(futs, 1):
            try:
                r = f.result()
            except Exception as e:    # one bad unit must not kill a long run
                print(f"ERR {items[k-1]['item_id']}: {type(e).__name__}: {e}", flush=True)
                continue
            n_ok += 1
            if fout:
                fout.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                fout.flush()
            else:
                print(json.dumps({x: r[x] for x in ("item_id", "label5", "rule", "secs")}
                                 | {"adj": r["gemma_adj"]["label"], "sil": r["gemma_silver"]["label"],
                                    "n_ev": len(r["evidence"]), "kb_match": r["features"].get("kb_match")},
                                 default=str), flush=True)
            if k % 100 == 0:
                print(f"{k}/{len(items)} done, {time.time()-t0:.0f}s", flush=True)
    print(f"finished {n_ok}/{len(items)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
