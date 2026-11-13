# Triage export contract (FNP → Evidence Desk "Triage" tab)

FNP writes `FakeNewsPropoganda/reports/triage/v1/` with `scripts/35_triage_export.py`. The Evidence Desk reads it read-only.

**Build and format**
- Builds are written to `v1.building/` and then renamed to `v1/`. The previous build is kept as `v1.prev/`.
- Reload when `meta.json` `built_at` changes.
- Strict JSON throughout: no NaN or Infinity, and missing values are `null`.

## `meta.json`
```json
{ "version": "triage_v1", "built_at": "...Z", "model_version": "prop_span_v1+kb_v2+stacker_v1(deciders_v1)+fake_call_v1",
  "n_items": 7584, "tiers": {"high": 345, "medium": 79, "low": 7160}, "n_with_verdict": 229,
  "data_through": "2026-10-27T...", "validated": false,
  "rules": {"candidates": "...", "high": "...", "medium": "...", "low": "...", "p_fake/p_misleading": "...",
            "p_prop": "...", "reach": "..."},
  "caveats": ["...show in the UI..."] }
```

## `items.jsonl` (one item per line, already sorted: tier, then fake_score, rerank, reach)
```json
{ "post_uid": "...", "source": "social|news", "platform": "telegram", "url": "https://..."|null,
  "author": "..."|null, "text": "first 1200 chars", "lang": "en|und|...", "posted_at": "ISO-8601"|null,
  "reach": 1234|null,                        // likes+comments+shares+views where present (platform-dependent)
  "p_fake": 0.62|null, "p_misleading": 0.21|null, "fake_score": 0.83|null,   // null = no FNP verdict yet
  "verdict": {"label": "FAKE|MISLEADING|LEGIT|INCONCLUSIVE", "basis": "stacker|gemma|none",
              "source": "live_scan|prop_lab_sample", "rationale": "...", "at": "..."}|null,
  "p_prop": 0.81|null,                       // max strong-span probability (uncalibrated signal); null if < 12 tokens
  "techniques": ["Loaded_Language"], "spans": [[start, end, "Loaded_Language", 0.81]],   // offsets into text
  "countries": ["IND"],
  "tier": "high|medium|low", "validated": false,
  "top_evidence_ids": ["kb_id", ...],
  "nearest_factcheck": {"kb_id": "...", "claim": "...", "label": "FAKE", "url": "...", "cos": 0.74,
                        "rerank": 0.93, "debunk_match": true},
  "model_version": "...", "built_at": "...Z" }
```

**Tiers.** Tiers are a ranking aid and are not validated, because no hand-labelled social gold set exists yet. Show an "unvalidated" chip.
- **high:** the FNP verdict is FAKE/MISLEADING, or there is a debunk match (the fact-check is FAKE/MISLEADING with rerank ≥ 0.8 and cosine ≥ 0.72).
- **medium:** fake_score ≥ 0.5, or the fact-check is FAKE/MISLEADING with rerank ≥ 0.5.
- **low:** every other candidate (fact-check cosine ≥ 0.65).

**Verdict probabilities.** `p_fake`/`p_misleading` exclude PROPAGANDA (`fnp/triage.py`): the stacker's PROPAGANDA probability is a length artefact.

**Freshness.** The export is rebuilt after each live-scan or Propaganda Lab run. Once FNP reads the realtime DB directly (<DB_HOST>, pending the user's confirmation in the FNP session), the export becomes incremental, newest first.
