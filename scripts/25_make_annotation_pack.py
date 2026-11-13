"""Stage 25: build the social-media annotation pack (propaganda technique spans + claim spans).

Sample 300 posts from the EDA store (7 platforms; en/hi/other): 210 stratified-random + 90 enriched
for rare techniques (picked by the fine-tuned encoder on a candidate pool). Every item is
pre-annotated (encoder spans, Gemma claims) so annotators correct instead of start from scratch.
20% of items (every 5th) are flagged `double` for double annotation; 10% flagged `blind` get NO
pre-annotation (to measure anchoring bias).

  .venv/bin/python scripts/25_make_annotation_pack.py
Writes data/annotation/social_v1/{items.jsonl, label_studio_config.xml, label_studio_import.json, GUIDELINES.md}
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fnp import gemma  # noqa: E402
from fnp.eval.ptc_scorer import CLASSES14  # noqa: E402
from fnp.llm import prompts as P  # noqa: E402
from fnp.llm.cache import reset_default_cache  # noqa: E402
from fnp.llm.claims import extract_claims  # noqa: E402
from fnp.llm.propaganda_enc import tag_spans_enc  # noqa: E402

OUT = ROOT / "data/annotation/social_v1"
SEED = 11
PLATFORMS = ["instagram", "telegram", "snapchat", "youtube_video", "reddit", "facebook", "linkedin"]
RARE = {"Appeal_to_Authority", "Causal_Oversimplification", "Whataboutism,Straw_Men,Red_Herring",
        "Thought-terminating_Cliches", "Bandwagon,Reductio_ad_hitlerum", "Slogans", "Black-and-White_Fallacy"}


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def main() -> None:
    rng = random.Random(SEED)
    con = duckdb.connect(str(ROOT.parent / "EDA/data/analytics.duckdb"), read_only=True)
    rows = con.execute("""
        select f.post_uid, f.platform, coalesce(s.lang_resolved, f.lang, 'unknown') lang,
               cast(f.posted_at as varchar) posted_at, f.url, f.text
        from post_facts f left join post_scores s using (post_uid)
        where length(f.text) between 80 and 1500 using sample 40000 rows (reservoir, 11)""").fetchall()
    seen, pool = set(), defaultdict(list)
    for uid, plat, lang, ts, url, text in rows:
        k = norm(text)
        if plat not in PLATFORMS or k in seen or len(k) < 60:
            continue
        seen.add(k)
        g = "en" if lang == "en" else "hi" if lang == "hi" else "other"
        pool[(plat, g)].append(dict(post_uid=uid, platform=plat, lang=lang, lang_group=g,
                                    posted_at=ts, url=url, text=text))
    # --- 210 stratified random: 30 per platform, split over en/hi/other ~ 50/25/25
    chosen, used = [], set()
    for plat in PLATFORMS:
        want = {"en": 15, "hi": 8, "other": 7}
        got = []
        for g, n in want.items():
            c = pool[(plat, g)]
            rng.shuffle(c)
            got += c[:n]
        rest = [x for g in want for x in pool[(plat, g)] if x not in got]
        rng.shuffle(rest)
        got += rest[:max(0, 30 - len(got))]
        for x in got[:30]:
            x["stratum"] = "random"
            chosen.append(x); used.add(x["post_uid"])
    # --- 90 enriched: score a candidate pool with the encoder, take posts with rare techniques
    cand = [x for v in pool.values() for x in v if x["post_uid"] not in used]
    rng.shuffle(cand)
    cand = cand[:1500]
    print("scoring", len(cand), "candidates with the encoder", flush=True)
    for x in cand:
        x["_sp"] = tag_spans_enc(x["post_uid"], x["text"])
        x["_rare"] = sum(s["technique"] in RARE for s in x["_sp"])
    cand.sort(key=lambda x: -x["_rare"])
    per_plat, enriched = Counter(), []
    for x in cand:
        if x["_rare"] and per_plat[x["platform"]] < 14 and len(enriched) < 90:
            per_plat[x["platform"]] += 1
            x["stratum"] = "enriched_rare"
            enriched.append(x)
    chosen += enriched
    rng.shuffle(chosen)
    # --- pre-annotate + flags
    reset_default_cache(ROOT / "data/cache/annot_pack.duckdb")
    print("gemma host:", gemma.endpoint_host(), "| items:", len(chosen), flush=True)

    def claims(x):
        c = extract_claims(x["post_uid"], x["text"], x["lang"] if x["lang"] != "unknown" else None)
        return [{"claim": d["claim"], "evidence_quote": d.get("evidence_quote"), "claim_type": d["claim_type"],
                 "check_worthy": d["check_worthy"], "speaker": d.get("speaker")} for d in c] if c.ok else None
    cl = gemma.run_batch(chosen, claims, 8)
    items = []
    for i, (x, c) in enumerate(zip(chosen, cl)):
        blind = (i % 10 == 3)
        sp = x.pop("_sp", None) or tag_spans_enc(x["post_uid"], x["text"])
        x.pop("_rare", None)
        items.append({"item_id": f"social_v1:{i:04d}", **x, "double": (i % 5 == 0), "blind": blind,
                      "pre_spans": [] if blind else [{k: s[k] for k in ("start", "end", "text", "technique", "confidence")} for s in sp],
                      "pre_claims": [] if blind else (c or [])})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # --- Label Studio config + import (predictions pre-filled)
    labels = "\n".join(f'    <Label value="{c}"/>' for c in CLASSES14)
    (OUT / "label_studio_config.xml").write_text(f"""<View>
  <Text name="text" value="$text"/>
  <Labels name="technique" toName="text" choice="multiple">
{labels}
  </Labels>
  <Labels name="claim" toName="text">
    <Label value="CLAIM_CHECKWORTHY"/>
    <Label value="CLAIM_NOT_CHECKWORTHY"/>
  </Labels>
  <TextArea name="normalised_claim" toName="text" rows="2" editable="true" perRegion="true"
            placeholder="Self-contained English restatement of the selected claim"/>
</View>
""")
    ls = []
    for it in items:
        res = []
        for j, s in enumerate(it["pre_spans"]):
            res.append({"id": f"s{j}", "from_name": "technique", "to_name": "text", "type": "labels",
                        "value": {"start": s["start"], "end": s["end"], "text": s["text"], "labels": [s["technique"]]}})
        for j, c in enumerate(it["pre_claims"]):
            q = c.get("evidence_quote") or ""
            p = it["text"].find(q) if q else -1
            if p >= 0:
                res.append({"id": f"c{j}", "from_name": "claim", "to_name": "text", "type": "labels",
                            "value": {"start": p, "end": p + len(q), "text": q, "labels": ["CLAIM_CHECKWORTHY" if c["check_worthy"] >= 0.5 else "CLAIM_NOT_CHECKWORTHY"]}})
        ls.append({"data": {"text": it["text"], "item_id": it["item_id"], "platform": it["platform"], "lang": it["lang"],
                            "double": it["double"], "blind": it["blind"]},
                   "predictions": [{"model_version": "enc_v1+gemma_claims_v1", "result": res}] if res else []})
    (OUT / "label_studio_import.json").write_text(json.dumps(ls, ensure_ascii=False))
    defs = "\n".join(f"- **{k.replace(',', ' / ')}**: {v}" for k, v in P.TECHNIQUE_DEFS.items())
    (OUT / "GUIDELINES.md").write_text(f"""# Annotation guidelines, social_v1 (300 posts)

You label two things per post. Pre-annotations (grey) come from a model: **correct them, do not trust them**.
{sum(i['blind'] for i in items)} items are *blind* (no pre-annotation) and {sum(i['double'] for i in items)} are *double* (a second annotator labels the same items independently). Do not discuss these.

## 1. Propaganda techniques (span labels)
Select the shortest span carrying the technique; spans of different techniques may overlap. Use the 14 classes
(Whataboutism / Straw Men / Red Herring are one class; Bandwagon / Reductio ad Hitlerum are one class).
Neutral reporting, plain facts and ordinary emotion are **not** propaganda. Short neutral titles usually have no span.
{defs}

## 2. Claims
- Select the verbatim span containing each factual, check-worthy claim (events, numbers, attributed quotes, "photo/video shows X", official actions, science/health assertions).
- Not claims: opinions, insults, slogans, rhetorical questions, hopes, vague generalities.
- Mark CLAIM_CHECKWORTHY if a fact-checker could verify it and it matters to the public; otherwise CLAIM_NOT_CHECKWORTHY.
- Fill *normalised_claim*: one self-contained English sentence (resolve pronouns, add who/where/when, keep names and numbers exact).
- Hindi/other languages: keep the span in the original script; normalised claim in English.

## Process
Import `label_studio_import.json` into Label Studio with `label_studio_config.xml`, export as JSON (with offsets) to `data/annotation/social_v1/annotator_<name>.json`.
Sampling: 210 stratified-random (30 per platform: instagram, telegram, snapchat, youtube, reddit, facebook, linkedin) + 90 enriched for rare techniques.
Report agreement per technique (span-level) before merging; merge or drop classes with poor agreement.
""")
    c = Counter((i["platform"], i["stratum"]) for i in items)
    print(json.dumps({"n": len(items), "by_stratum": Counter(i["stratum"] for i in items),
                      "by_group": Counter(i["lang_group"] for i in items), "with_pre_spans": sum(bool(i["pre_spans"]) for i in items),
                      "with_pre_claims": sum(bool(i["pre_claims"]) for i in items), "double": sum(i["double"] for i in items),
                      "blind": sum(i["blind"] for i in items)}, indent=1))


if __name__ == "__main__":
    main()
