#!/usr/bin/env python
"""Live smoke test of every Gemma task (NOT pytest). ~20 calls: 3 BOOM claims (the `claim`
field only -- the body is the debunk and leaks the label) + one Hindi sentence.

Usage: .venv/bin/python scripts/02_llm_smoke.py [--fresh]   (--fresh bypasses the cache)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnp import config, gemma  # noqa: E402
from fnp.llm import adjudicate, claims, propaganda, redteam, silver, stance, timeline  # noqa: E402
from fnp.llm import cache as C  # noqa: E402
from fnp.llm import prompts as P  # noqa: E402
from fnp.schema import FEATURE_NAMES  # noqa: E402

HINDI = ("सोशल मीडिया पर वायरल वीडियो में दावा किया गया है कि पिछले हफ्ते दिल्ली में बाढ़ के कारण "
         "सरकार ने 500 लोगों को मार डाला। ये भ्रष्ट सरकार देश को बर्बाद कर रही है, हर कोई जानता है!")

STATS: dict[str, list] = defaultdict(list)


def rec(task, meta, ok):
    STATS[task].append((ok, meta.get("tokens") or 0, meta.get("latency_s") or 0, bool(meta.get("cached"))))


def show(title, obj):
    print(f"\n--- {title}")
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str)[:1800])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    print("endpoint:", gemma.endpoint_host(), "| prompt versions:", P.PROMPT_VERSIONS)
    if a.fresh:
        C.reset_default_cache(config.DATA_DIR / "smoke_nocache.duckdb")
        (config.DATA_DIR / "smoke_nocache.duckdb").unlink(missing_ok=True)
        C.reset_default_cache(config.DATA_DIR / "smoke_nocache.duckdb")
    arts = json.load(open(config.BOOM_DIR / "articles_full.json"))
    texts = [(f"boom{i}", arts[i]["claim"]) for i in (0, 1, 3)] + [("hindi", HINDI)]

    all_claims = []
    for doc_id, text in texts:
        cl = claims.extract_claims(doc_id, text)
        rec("claims", cl.meta, cl.ok)
        show(f"claims[{doc_id}] ok={cl.ok} err={cl.error} dropped={cl.dropped} meta={cl.meta}", list(cl))
        all_claims += list(cl)

        sp = propaganda.tag_spans(doc_id, text)
        rec("propaganda", sp.meta, sp.ok)
        show(f"spans[{doc_id}] ok={sp.ok} err={sp.error} one_sided={sp.one_sided} framing={sp.framing!r} "
             f"dropped={sp.dropped} meta={sp.meta}", list(sp))

        sv = silver.label(text)
        rec("silver", sv["meta"], sv["ok"])
        show(f"silver[{doc_id}]", sv)

    # Stance: hand-written evidence passages (no BOOM body text, to avoid label leakage).
    by_doc = {c["doc_id"]: c for c in all_claims}
    evid = [
        ("boom0", "BOOM FactCheck: The clip is from a 2019 rally in Etawah; Akhilesh Yadav wore a saffron "
                  "turban at a local event then. It is not from the 2027 election campaign.", {"source": "BOOM", "published": "2025-08-01"}),
        ("boom1", "Ministry of External Affairs press briefing: there has been no face-off between Indian and "
                  "Chinese troops in Arunachal Pradesh this year.", {"source": "MEA", "published": "2025-03-04"}),
        ("boom3", "Local weather report: heavy rain is expected over Kerala next week.", {"source": "IMD"}),
        ("hindi", "दिल्ली पुलिस ने बयान जारी कर कहा कि बाढ़ में किसी की मौत नहीं हुई और वायरल वीडियो 2023 का है।", {"source": "Delhi Police"}),
    ]
    results = []
    for doc_id, ev, meta in evid:
        c = by_doc.get(doc_id)
        if not c:
            print(f"(no claim extracted for {doc_id}, skipping stance)")
            continue
        r = stance.judge(c, ev, meta)
        rec("stance", r["meta"], r["ok"])
        show(f"stance[{doc_id}] claim={c['claim']!r}", r)
        results.append((c, r))

    # Timeline: extracted claims plus one hand-made chronologically impossible claim.
    tl_claims = all_claims[:3] + [
        {"claim_id": "x1", "claim": "A video filmed in 2019 shows the flood that hit the city in July 2024", "date": "2019"},
        {"claim_id": "x2", "claim": "The flood hit the city in July 2024", "date": "2024-07"}]
    t = timeline.check_ex(tl_claims)
    rec("timeline", t["meta"], t["ok"])
    show(f"timeline ok={t['ok']} err={t['error']}", t["flags"])

    # Adjudicate + redteam on a synthetic structured dossier.
    feats = {k: 0.0 for k in FEATURE_NAMES}
    feats.update(p_true=0.15, p_true_lo=0.05, p_true_hi=0.4, contradiction=1.6, context_mismatch=0.8,
                 n_evidence=3, n_eff=1.7, ds_uncertainty=0.3, p_prop=0.2, check_worthy=0.8)
    state = {"claims": [{"claim_id": c["claim_id"], "claim": c["claim"], "claim_type": c["claim_type"],
                         "check_worthy": c["check_worthy"]} for c in all_claims[:1]],
             "features": feats,
             "evidence": [{"evidence_id": "e1", "stance": "context_mismatch", "evidence_type": "FACT_CHECK",
                           "reliability": 0.9, "independence": 1.0, "relevance": 0.9,
                           "summary": "Fact-check: the clip is from 2019, not recent."},
                          {"evidence_id": "e2", "stance": "refute", "evidence_type": "MEDIA_REPORT",
                           "reliability": 0.6, "independence": 0.4, "relevance": 0.7,
                           "summary": "News site repeating the fact-check conclusion."}],
             "spans_summary": {"n_spans": 0, "top_techniques": [], "one_sided": 0.1},
             "propagation": {"burst_intensity": 0.7, "platform_spread": 2, "dup_rate": 0.5}}
    d = adjudicate.decide(state)
    rec("adjudicate", d["meta"], d["ok"])
    show("adjudicate", d)
    rt = redteam.review(state, d)
    rec("redteam", rt["meta"], rt["ok"])
    show("redteam", rt)

    print("\n=== summary (task: calls, ok, avg tokens, avg latency s, cached)")
    for task, rows in STATS.items():
        n = len(rows)
        print(f"{task:11s} n={n} ok={sum(r[0] for r in rows)} avg_tokens={sum(r[1] for r in rows)/n:.0f} "
              f"avg_latency={sum(r[2] for r in rows)/n:.1f} cached={sum(r[3] for r in rows)}")


if __name__ == "__main__":
    main()
