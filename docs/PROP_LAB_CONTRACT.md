# Propaganda Lab data contract (FNP → Evidence Desk)

FNP writes everything under `FakeNewsPropoganda/reports/prop_lab/v1/` (built by `scripts/31_prop_lab_aggregate.py`). Evidence Desk (`Interface/`, session fakenewspropoganda-7c) reads these files read-only and never runs inference.

**Build and freshness**
- A build is written to `v1.building/` and then renamed to `v1/`. The previous build is kept as `v1.prev/`, so readers never see a half-written build.
- Reload when `meta.json`'s mtime or `built_at` changes.

**Format rules**
- All files are strict JSON: no NaN or Infinity, and missing numbers are `null`.
- Country keys are ISO 3166-1 alpha-3 codes.
- `"UNL"` is the **unlocated** bucket. It is always present and must never be dropped.
- **Windows** are `7d`, `30d` and `all`. They are relative to `meta.data_through`, the newest post, not to the wall clock.
- **Sources** are `social`, `news` and `all`. News is empty until shaktidb_news is reachable.
- **Modes** are `about` (countries the text mentions) and `origin` (publisher or channel country).

## Files

### `meta.json`
```json
{ "version": "prop_lab_v1", "built_at": "2026-11-05T09:00:00Z", "data_through": "2026-10-27T08:53:48+00:00",
  "n_posts": 274000, "n_articles": 0, "n_located_about": 0, "n_located_origin": 0, "n_gemma_verified": 0,
  "n_gemma_span_checked": 0,
  "windows": ["7d","30d","all"], "sources": ["social","news","all"], "modes": ["about","origin"],
  "techniques": ["Loaded_Language", "...14 PTC classes"], "min_n": 30, "unlocated_key": "UNL",
  "names": {"IND": "India", "UNL": "Unlocated", "...": "..."},
  "global": { "<mode>": { "<window>": { "<source>": {
      "n": 0, "n_signal": 0, "rate": 0.0, "ci": [0,0], "debunk_n": 0, "debunk_rate": 0.0,
      "eb_prior": [a,b], "technique_share": {"Loaded_Language": 0.0, "...": 0.0}, "located": 0 }}}},
  "method": { "engine": "...", "signal_rule": "...", "debunk_rule": "...", "about": "...", "origin": "...",
              "lift": "...", "narratives": "...", "windows": "..." },
  "caveats": ["...show these in the UI..."] }
```

### `world_paths.json`
`{ "viewBox": "x y w h", "projection": "Equal Earth", "source": "...", "countries": { "IND": {"name": "India", "d": "M..Z"} } }`
- These are SVG paths that are already projected, so render them inside `<svg viewBox=...>` with no map library.
- Boundaries follow the **India worldview** of Natural Earth (public domain).
- Antarctica is omitted.

### `countries.json`
`{ "countries": { "<mode>": { "<window>": { "<source>": { "<ISO3|UNL>": CountryStats } } } } }`

```json
CountryStats = {
  "n": 1357,                 // posts mentioning / from the country (a multi-country post counts in each)
  "n_weighted": 802.5,       // sum of 1/k weights (sums to the global n across countries + UNL)
  "n_signal": 854, "rate": 0.629, "ci": [0.603, 0.655],          // Wilson 95%
  "lift": 1.21,              // HEADLINE: length-adjusted observed/expected signal (1 = as expected)
  "expected_signal": 705.3, "lift_raw": 1.98,                    // raw EB-shrunk rate / global rate
  "insufficient": false,     // n < 30: hatch on the map, grey in tables
  "top_technique": "Loaded_Language", "top_technique_ratio": 1.1,
  "debunk_n": 7, "debunk_rate": 0.005, "debunk_ci": [0.002, 0.011],   // fake/misleading layer
  "verified_n": 12, "verified_fake": 2, "verified_misleading": 3, "verified_plausibly_fake": 6,
                             // FNP verdicts on posts with a candidate fact-check match (<= 15 per country)
  "agreement_n": 5, "agreement_post": 0.8, "agreement_span": 0.45 }
                             // Gemma span check on 5 signal posts per country (n >= 30): share of those posts
                             // Gemma also flags, and mean share of encoder spans Gemma overlaps. null = not checked
```

### `heatmap.json`
`{ "techniques": [...14], "views": { "<mode>": { "<window>": { "<source>": { "countries": ["IND", ...], "share": [[14 floats], ...], "ratio": [[14 floats|null], ...] } } } } }`
- Countries are limited to n ≥ 30, sorted by n.
- `share` is the fraction of the country's signal posts that use the technique.
- `ratio` is share divided by the global share. Use a diverging colour around 1.

### `country/<ISO3>.json` (every country with n ≥ 5 in either mode, plus `UNL`)
```json
{ "iso3": "IND", "name": "India",
  "modes": { "<mode>": { "<window>": {
     "kpis": { "social": CountryStats|null, "news": CountryStats|null, "all": CountryStats|null },
     "techniques": [ {"technique": "...", "n": 0, "share": 0.0, "global_share": 0.0, "ratio": 0.0} ],
     "platforms": {"reddit": 0}, "languages": {"en": 0},
     "narratives": [ {"cluster": 3, "name": "Gemma-written title (falls back to terms)", "summary": "one sentence or ''",
                      "terms": "c-TF-IDF terms", "cohesion": 0.71, "n": 0, "n_unique": 0, "templated": false,
                      "n_signal": 0, "rate": 0.0,   // templated = < 30% unique texts (show a "templated" chip)
                      "top_technique": "...", "exemplars": [Post, Post]} ],
                      // per-country clusters (stage 34); cohesion = mean cosine to centroid (0-1, higher = tighter)
     "narratives_coverage": 0.79,          // share of the country's posts in a listed narrative
     "entities": [ {"name": "Narendra Modi", "type": "PERSON", "n": 0} ],
     "fake_layer": { "debunk_n": 0, "n": 0,
        "matches": [ {"post_uid": "...", "text": "...", "kb_claim": "...", "kb_label": "FAKE", "kb_url": "...", "rerank": 0.99} ],
        "verified": {"n": 12, "FAKE": 2, "MISLEADING": 3, "LEGIT": 5, "INCONCLUSIVE": 2},   // label keys only when > 0
        "verified_posts": [Post] },
     "trend": [ {"week": "2026-10-19", "n": 0, "n_signal": 0, "rate": 0.0} ]     // only in window "all", last 26 weeks
  } } } }
```
- `kpis.*` can be `null` when the country has no posts for that source.
- `narratives` are per-country k-means clusters of bge-m3 post vectors. `entities` come from EDA (posts up to 29 Jul 2026), so recent windows may have none. Show "no narrative/entity data" when a list is empty.

### `posts/<mode>/<ISO3>.jsonl` (one Post per line, up to 150, best first)
```json
Post = { "post_uid": "...", "source": "social", "platform": "telegram", "url": "https://..."|null,
  "author": "@rtnews"|null, "posted_at": "2026-10-23T21:32:57+00:00", "lang": "en",   // "und" when unknown
  "text": "first 900 chars",
  "spans": [[start, end, "Loaded_Language", 0.82]],     // char offsets into `text`; strong spans only
  "techniques": ["Loaded_Language"], "signal": true, "density": 4.2, "n_strong": 2,
  "about": ["ISR","IRN"], "origin": ["RUS"],
  "debunk": {"kb_claim": "...", "kb_label": "FAKE", "kb_url": "...", "rerank": 0.99, "cos": 0.8} | null,
  "verified": {"label": "MISLEADING", "probs": {"FAKE":0.1,"MISLEADING":0.6,"LEGIT":0.1,"INCONCLUSIVE":0.2},
               "fake_score": 0.7, "basis": "stacker|gemma|none", "plausibly_fake": true, "rationale": "..."} | null,
  "gemma_check": {"gemma_spans": [[s,e,"Doubt"]],
                  "agreement": {"enc_spans": 2, "gemma_spans": 3, "overlap": 0.5, "same_technique": 0.5,
                                "gemma_flags": true, "encoder_flags": true}} | null }
```
- The UI filters posts client-side by window (`posted_at` ≥ data_through − 7/30 days) and by technique.
- Span offsets index into `text`. Highlight with `text.slice(s, e)` and escape everything.
- A span may end past 900 characters (the text is truncated), so clamp it.

## Gemma checks (stage 33)
- **Verdicts** cover only posts with a candidate fact-check match, so `verified_*` is evidence-grounded but small. Show it as "N checked against fact-checks", not as a rate over all posts.
- **The span check** (`agreement_*`) tells the reader how far to trust the encoder for that country. Suggested panel line: "Gemma agrees on X of N sampled posts".

## Feedback (optional "not propaganda" button)
`POST http://<SITE_HOST>:8879/api/prop_flag {"post_uid": "...", "action": "not_propaganda|confirm|clear", "technique": optional}`
- FNP appends the request to `data/feedback/prop_flags_v1.jsonl`, and `scripts/32_prop_feedback_rows.py` turns it into a social gold set for the encoder.
- Proxy it from server.py, or skip it in v1.

## Presentation notes (agreed design)
- **KPI strip:** posts, articles, countries, global signal rate, debunk rate, built_at, and the chip "span encoder v1 – signal, not verdict".
- **Controls:**
  - View: Map | Heatmap
  - Attribution: About | Origin. With Origin, show "origin known for N%".
  - Layer: Propaganda signal | Fake/misleading
  - Metric: lift (default) | rate | volume | technique
  - Window and Source
- **Map:**
  - A sequential ramp from the paper colour to `--accent` for rates.
  - A diverging ramp around 1 for lift.
  - `--fake` for the fake layer.
  - Hatching where `insufficient` is true, and a grey fill where there is no data.
- **Country panel:** tabs Overview | Narratives | Posts | Fake/misleading.
- **Leaderboard:** sortable, with `UNL` as the last row. It doubles as the keyboard-accessible alternative to the map.
- **Caveats:** always show `meta.caveats` in a collapsible "How to read this".
