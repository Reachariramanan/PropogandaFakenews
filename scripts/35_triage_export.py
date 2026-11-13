"""Stage 35: triage export for the Evidence Desk "Triage" tab -> reports/triage/v1/ (contract:
docs/TRIAGE_CONTRACT.md). Strict JSON, atomic swap (v1.building -> v1, previous kept as v1.prev).

Candidates (social posts; news joins when the DB is reachable):
  * every post with an FNP verdict: data/runs/live_v2.jsonl (status=scored) and the Propaganda Lab
    verdict sample (data/prop_lab/v1/gemma_verdict.jsonl);
  * every post whose nearest kb_v2 fact-check has cosine >= CAND_MIN_COS (stage-30 parts).
Per item: fake/misleading call via fnp.triage.fake_call (PROPAGANDA mass removed), encoder propaganda
signal, nearest fact-check, reach, tier. Tiers (ranking aid, NOT validated -- no social gold yet):
  high   = FNP verdict FAKE/MISLEADING, or debunk match (fact-check FAKE/MISLEADING, rerank >= 0.8, cos >= 0.72)
  medium = verdict plausibly fake, or fact-check FAKE/MISLEADING with rerank >= 0.5
  low    = other candidates
  .venv/bin/python scripts/35_triage_export.py
"""
from __future__ import annotations

import glob
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fnp.config import DATA_DIR, REPORTS_DIR  # noqa: E402
from fnp.eda_bridge import read_parquet  # noqa: E402
from fnp.triage import fake_call  # noqa: E402

PARTS = DATA_DIR / "prop_lab" / "v1" / "parts"
LIVE = DATA_DIR / "runs" / "live_v2.jsonl"
GV = DATA_DIR / "prop_lab" / "v1" / "gemma_verdict.jsonl"
OUT = REPORTS_DIR / "triage" / "v1"
CAND_MIN_COS = 0.65
MODEL_VERSION = "prop_span_v1+kb_v2+stacker_v1(deciders_v1)+fake_call_v1"


def finite(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.generic):
        return finite(o.item())
    if isinstance(o, dict):
        return {str(k): finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def r4(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 4)


def load_verdicts() -> dict:
    """post_uid -> {label, probs, fake_score, basis, rationale, evidence_ids, source}; live scan wins."""
    out = {}
    for line in GV.open() if GV.exists() else []:
        r = json.loads(line)
        if r.get("ok"):
            out[r["post_uid"]] = {k: r.get(k) for k in ("label", "probs", "fake_score", "basis", "rationale")} \
                | {"evidence_ids": [], "verdict_source": "prop_lab_sample", "at": r.get("at")}
    for line in LIVE.open() if LIVE.exists() else []:
        r = json.loads(line)
        if r.get("status") != "scored":
            continue
        cl = r.get("claims") or []
        fc = fake_call([{"stacker": c["verdict"].get("probs"),
                         "gemma": c["gemma_adj"].get("probs") if c.get("gemma_adj", {}).get("ok") else None}
                        for c in cl])
        uid = r["id"].split(":", 1)[1] if r["id"].startswith("social:") else r["id"]
        out[uid] = {**fc, "rationale": (cl[0]["verdict"].get("rationale") if cl else "") or "",
                    "evidence_ids": [e for c in cl for e in c["verdict"].get("evidence_ids", [])][:10],
                    "verdict_source": "live_scan", "at": r.get("scanned_at")}
    return out


def main() -> None:
    t0 = time.time()
    cols = ["post_uid", "source", "platform", "url", "author_name", "lang", "posted_at", "text", "too_short",
            "n_strong", "spans", "strong_techniques", "countries", "kb_cos", "kb_rerank", "kb_id", "kb_label",
            "kb_claim", "kb_url"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(PARTS / "*.parquet")))])
    df = df.drop_duplicates("post_uid")
    ver = load_verdicts()
    df = df[(df.kb_cos >= CAND_MIN_COS) | df.post_uid.isin(ver)].copy()
    eng = read_parquet("post_facts")[["post_uid", "likes", "comments", "shares", "views"]]
    df = df.merge(eng, on="post_uid", how="left")
    fm = df.kb_label.isin(["FAKE", "MISLEADING"])
    rr = df.kb_rerank.fillna(0)
    debunk = fm & (rr >= 0.8) & (df.kb_cos >= 0.72)
    items = []
    for r, deb, fmm in zip(df.itertuples(), debunk, fm):
        v = ver.get(r.post_uid)
        strong = set(json.loads(r.strong_techniques))
        sp = [s for s in json.loads(r.spans) if s[2] in strong]
        p_prop = max((s[3] for s in sp), default=0.0) if not r.too_short else None
        if (v and v["label"] in ("FAKE", "MISLEADING")) or deb:
            tier = "high"
        elif (v and v.get("fake_score", 0) >= 0.5) or (fmm and (r.kb_rerank or 0) >= 0.5):
            tier = "medium"
        else:
            tier = "low"
        reach_parts = [x for x in (r.likes, r.comments, r.shares, r.views) if x is not None and not pd.isna(x)]
        items.append({
            "post_uid": r.post_uid, "source": r.source, "platform": r.platform,
            "url": r.url if isinstance(r.url, str) and r.url.startswith("http") else None,
            "author": r.author_name if isinstance(r.author_name, str) else None,
            "text": r.text[:1200], "lang": r.lang if isinstance(r.lang, str) and r.lang else "und",
            "posted_at": pd.to_datetime(r.posted_at, utc=True, errors="coerce").isoformat()
            if pd.notna(pd.to_datetime(r.posted_at, utc=True, errors="coerce")) else None,
            "reach": int(sum(reach_parts)) if reach_parts else None,
            "p_fake": r4(v["probs"]["FAKE"]) if v else None,
            "p_misleading": r4(v["probs"]["MISLEADING"]) if v else None,
            "fake_score": r4(v["fake_score"]) if v else None,
            "verdict": ({"label": v["label"], "basis": v["basis"], "source": v["verdict_source"],
                         "rationale": (v.get("rationale") or "")[:400], "at": v.get("at")} if v else None),
            "p_prop": r4(p_prop), "techniques": sorted(strong),
            "spans": [s for s in sp if s[0] < 1200][:12],
            "countries": [c["iso3"] for c in json.loads(r.countries)],
            "tier": tier, "validated": False,
            "top_evidence_ids": (v["evidence_ids"] if v and v["evidence_ids"] else [r.kb_id]),
            "nearest_factcheck": {"kb_id": r.kb_id, "claim": r.kb_claim, "label": r.kb_label, "url": r.kb_url,
                                  "cos": r4(r.kb_cos), "rerank": r4(r.kb_rerank), "debunk_match": bool(deb)},
            "model_version": MODEL_VERSION})
    order = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda x: (order[x["tier"]], -(x["fake_score"] or 0), -(x["nearest_factcheck"]["rerank"] or 0),
                              -(x["reach"] or 0)))
    built = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for it in items:
        it["built_at"] = built
    tmp = OUT.with_name(OUT.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with (tmp / "items.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(finite(it), ensure_ascii=False, allow_nan=False) + "\n")
    tiers = pd.Series([i["tier"] for i in items]).value_counts().to_dict()
    meta = {"version": "triage_v1", "built_at": built, "model_version": MODEL_VERSION, "n_items": len(items),
            "tiers": tiers, "n_with_verdict": sum(i["verdict"] is not None for i in items),
            "data_through": max((i["posted_at"] for i in items if i["posted_at"]), default=None),
            "validated": False,
            "rules": {"candidates": f"FNP verdict exists, or nearest kb_v2 fact-check cosine >= {CAND_MIN_COS}",
                      "high": "verdict FAKE/MISLEADING, or debunk match (fact-check FAKE/MISLEADING, rerank >= 0.8, cos >= 0.72)",
                      "medium": "verdict fake_score >= 0.5, or fact-check FAKE/MISLEADING with rerank >= 0.5",
                      "low": "other candidates",
                      "p_fake/p_misleading": "fnp.triage.fake_call: stacker (or Gemma adjudicator) probabilities with "
                                             "PROPAGANDA removed and renormalised; null when no verdict exists",
                      "p_prop": "max mean-probability of the post's strong encoder spans (uncalibrated signal); "
                                "null for posts < 12 tokens",
                      "reach": "likes + comments + shares + views where present (platform-dependent; null if none)"},
            "caveats": ["Tiers are a ranking aid and are NOT validated: no hand-labelled social gold set exists yet.",
                        "Fact-check matches are often same-topic rather than same-claim; check the evidence link.",
                        "p_prop is a propaganda-language signal from a model not validated on social media."]}
    (tmp / "meta.json").write_text(json.dumps(finite(meta), ensure_ascii=False, allow_nan=False, indent=1))
    if OUT.exists():
        shutil.rmtree(OUT.with_name(OUT.name + ".prev"), ignore_errors=True)
        OUT.rename(OUT.with_name(OUT.name + ".prev"))
    tmp.rename(OUT)
    print(f"triage: {len(items)} items {tiers} in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
