# FNP — current architecture and status

*Snapshot 2026-10-28, updated after the external scoring run. The approved plan is `docs/ARCHITECTURE.md`, the maths is `docs/METHODS.md`, and open problems are in `docs/review/`.*

## What it is

FNP is a hybrid LLM + mathematical-models system. Every claim, post or article gets one of **FAKE / PROPAGANDA / LEGIT / MISLEADING / INCONCLUSIVE**, plus a **"plausibly fake"** tier, defined as P(FAKE)+P(MISLEADING) ∈ [0.5, 0.8).

It turns the design notes in `methodologoes/` (10 probability frameworks, plus the propaganda-span and forensic-evidence notes) into code. Each model's assumptions are stated explicitly.

- **Gemma** (`gemma-4-26b-a4b-it`, sglang `localhost:11650`, shared with the EDA session) reads text and adjudicates.
- **Laya** (`convaiinnovations/laya`, multilingual 322M) is trained as a cheap, calibrated decider. The RL part is RLCD, i.e. REINFORCE on proper scoring rules.
- The work is **claim-centric**: articles are reduced to claims, claims become evidence units, evidence is de-duplicated and then fused into a verdict.

## Pipeline (layers → code)

| Layer | What | Code | Status |
|---|---|---|---|
| L0 Ingest | Gold datasets + BOOM/fact-check sites + PTC propaganda spans; EDA social posts read-only | `fnp/datasets.py`, `scripts/00_fetch_datasets.py`, `fnp/eda_bridge.py` | ✅ |
| KB | train-only fact-checked claims: kb_v2 32,225 (with the new sites, now the default); kb_v1 29,632 (used for EVAL_v1). bge-m3 + bge-reranker-v2-m3 | `fnp/retrieval/kb.py`, `scripts/01_build_kb.py` | ✅ |
| L1 Claims | Gemma claim extraction with verbatim quote check | `fnp/llm/claims.py` | ✅ |
| L2 Propaganda | Gemma span + technique tagging (18 PTC techniques), P_prop, span-F1 | `fnp/llm/propaganda.py`, `fnp/math/propaganda_score.py` | ✅ |
| L3 Evidence | KB / corpus / web retrieval → Gemma stance (support / refute / context_mismatch / neutral) + evidence type | `fnp/retrieval/*`, `fnp/llm/stance.py` | ✅ (web search blocked) |
| L4 Independence | origin clustering, 1/\|cluster\|, Kish n_eff, AccuCopy | `fnp/math/independence.py` | ✅ |
| L5 Reliability | Beta prior from evidence-type tiers + track record; Dawid–Skene EM, TruthFinder | `fnp/math/reliability.py`, `truth_discovery.py` | ✅ |
| L6 Fusion | log-odds WoE, Dempster–Shafer (Yager), DerSimonian–Laird CI, information gain (bits), loopy BP, PageRank, CP-tensor coordination, noisy-OR timeline BN, EDA HMM/chain propagation hooks | `fnp/math/*` → `features.claim_features` (30 features, `schema.FEATURE_NAMES`) | ✅ |
| L7 Verdict | rule head → Gemma adjudicator (structured state only) → multinomial stacker + temperature → split-conformal abstention → INCONCLUSIVE; doc rollup | `fnp/math/verdict.py`, `conformal.py`, `calibration.py`, `fnp/llm/adjudicate.py`, `redteam.py` | ✅ stacker + conformal fit (`deciders_v1.pkl`) |
| Laya | state builder, SL distillation (gold + 0.3·Gemma soft targets), custom RLCD (group baseline, INCONCLUSIVE penalty, KL anchor), temperature + abstention fit | `fnp/laya/*`, `scripts/10,11,12` | ✅ trained: `laya_fnp_sl_v1`, `laya_fnp_rl_v1` (passes the adoption gate) |
| RL routing | contextual Thompson / LinUCB bandit over {laya, gemma, human}; VOI stopping for evidence acquisition | `fnp/rl/*` | ✅ code; not yet trained on real logs |
| Shared runner | per-claim / per-doc analysis used by the batch runner and the CLI | `fnp/pipeline.py` | ✅ |

## Scripts (numbered = run order; dry-run unless `--apply`)

| Script | Purpose |
|---|---|
| `00_fetch_datasets` | Download and normalise gold data into `data/fnp.duckdb` |
| `01_build_kb` | Build the KB |
| `02_llm_smoke` | Live Gemma check |
| `05_run_units` | Full per-unit pipeline on a stratified gold sample → `data/runs/units_v1.jsonl` |
| `10_laya_rows` | Laya training rows from the train units |
| `11_train_laya` / `12_rl_laya` | Laya stage A (SL) / stage B (RLCD) |
| `14_evaluate` | Stacker (train) + conformal (dev); ablation on test; writes `verdict_v1`, `reports/EVAL_v1.md`, `data/models/deciders_v1.pkl` |
| `15_predict` | `--text`/`--file`/`--url`: verdict JSON with evidence trail |
| `16_score_external` | (✅ ran on 712 items) Score `Interface/data/db_claim_candidates.jsonl` + `db_factchecks.jsonl`; upsert doc rows into `verdict_v1`; export `reports/verdict_v1_docs.jsonl` for the gallery |

## Data

**Gold data (`gold_items_v1`)**, label5 counts:

| Dataset | Rows | FAKE | MISLEADING | LEGIT | INCONCLUSIVE | Notes |
|---|---|---|---|---|---|---|
| BOOM | 486 | 455 | 31 | — | — | `claim` field only; the body leaks the label |
| LIAR | 12.8k | ✓ | ✓ | ✓ | — | 6 labels mapped to 3 |
| X-Fact | 25.4k | ✓ | ✓ | ✓ | ✓ | multilingual, includes Hindi |
| FakeNewsNet-PolitiFact | 1,056 | ✓ | — | ✓ | — | |
| PTC | 371 articles | | | | | 6,129 propaganda spans |

**Ingested since v1.** New fact-check sites from the Interface session in `Interface/factchecks/`, now in `gold_items_v1` and kb_v2:
- vishvasnews 2,098, quint_webqoof 1,069, snopes 601, fullfact 600 and newsmeter 5.
- fullfact verdicts are free text.

**Missing:** PTC dev/test labels, IFND, QProp and live web search. See `docs/review/01_DATA_GAPS.md`.

**Run sample:** 3,709 units. LIAR and X-Fact give 600/300/400 per split, stratified; all of BOOM; 300 FakeNewsNet; all of PTC. Each unit gets top-5 KB evidence with Gemma stance, propaganda tagging (run on every unit so a missing-span indicator cannot leak which dataset it came from), the maths features, the rule head, the Gemma adjudicator and a Gemma raw-text baseline.

## Status (2026-10-28)

**Done**
- Package, contracts (`fnp/schema.py`) and the constants register (`fnp/config.py`, `fnp/math/constants.py`).
- 64 offline tests passing.
- Datasets and KB built.
- Gemma layer: 7 tasks, cached in DuckDB, 0 validation failures in the smoke run.
- The full maths library.
- Laya SL + RLCD pipeline: about 3.4 GB GPU peak, runs on the ~7 GB that is free.
- Routing bandit and VOI.
- End-to-end CLI.
- Dashboard.

**Done since the first snapshot**
- `05_run_units`: 3,709 / 3,709 units, 0 errors.
- `14_evaluate --apply`: produced `verdict_v1` (7,418 rows), `reports/EVAL_v1.md` and `data/models/deciders_v1.pkl`.
- Laya SL (`laya_fnp_sl_v1`) and RLCD (`laya_fnp_rl_v1`) trained on the real units. Both pass the adoption gate.
- New fact-check sites ingested into `gold_items_v1`; `kb_v2` built (32,225 claims). Retrieval now uses v2; `FNP_KB=kb_v1` reproduces EVAL_v1.

**Results (test, 945 decidable units)**

| decider | macro-F1 | ECE |
|---|---|---|
| rule | 0.382 | 0.46 |
| Gemma raw | 0.254 | 0.33 |
| Gemma adjudicator | 0.379 | 0.35 |
| stacker | 0.554 | 0.055 |
| Laya SL | 0.608 | 0.082 |
| Laya RL | 0.607 | 0.098 |
| stacker+Laya, conformal | 0.726 at 26% coverage | — |

PROPAGANDA = 1.0 for every decider is a length/format artefact; see `docs/review/05_EVAL_FINDINGS.md`.

**External scoring done:** `16_score_external --apply` scored 712 / 712 items with 0 errors, upserted 2,560 rows into `verdict_v1`, and exported 4,421 doc rows to `reports/verdict_v1_docs.jsonl`.
- Social posts (628): 454 PROPAGANDA, 148 INCONCLUSIVE, 18 FAKE, 6 LEGIT, 2 MISLEADING.
- Fact-check items (84): mostly INCONCLUSIVE.
- 233 items are flagged plausibly fake.
- The PROPAGANDA count on posts is over-labelling (P_prop saturates); see `docs/review/05` problem 7.

**Next**
1. Stance v2 with a `same_claim` check: KB refute evidence is only weakly diagnostic (72% FAKE vs a 58% base rate).
2. Out-of-fold Laya probabilities for the stacker.
3. Fit the routing bandit on dev outcomes.
4. Conformal α decision (suggest 0.2).
5. Propaganda data on short posts.

**Early observations (10-unit pilot, not an evaluation)**
- KB claim-matching drives FAKE well (FAKE ≈ 0.96 on matched debunks).
- LEGIT tends to fall to INCONCLUSIVE, because the KB is debunk-heavy and rarely holds supporting evidence.
- Gemma raw-text silver labels over-predict PROPAGANDA on short "video shows X" claims, so they are treated as noisy.

## Live scan + gallery (added 2026-10-29)

- `fnp/live/sources.py`: read-only newest-first sources.
  - Social: EDA's incoming queue (4,146 posts newer than the extract), then the EDA `post_facts` parquet (270k posts).
  - News: `shaktidb_news.scraped_articles_trusted` over Postgres. Columns are discovered at runtime and this path is **untested**.
- `scripts/17_live_scan.py`: loops newest-first over batches of 200.
  - Cheap pass: bge-m3 cosine to the KB. Full pipeline on the top 40% of each batch. The rest are logged as skipped.
  - Writes `data/runs/live_v2.jsonl` (resumable) and `live_state.json`. Stop with `touch logs/live_scan.stop`.
  - **Bug fixed 2026-11-05:** `live_v1` (29 Oct–1 Nov, 104,698 "scored" rows) is about 99.8% failed Gemma calls (Gemma was overloaded or restarting) that were recorded as INCONCLUSIVE. `live_v1` is kept untouched and **must not be used**. The scanner now marks such rows `status=error` and retries them, and writes to `live_v2`.
  - Stopped on 2026-11-05 while the Propaganda Lab's Gemma sample runs (shared concurrency).
- **Live gallery** tab in the dashboard: newest-first suspects, refreshing every 10 s, with Unflag / Confirm / Relabel / undo buttons. Feedback is appended to `data/feedback/flags_v1.jsonl`.
- `scripts/18_feedback_rows.py`: feedback becomes Laya rows (`laya_rows_feedback_v1.jsonl`) for retraining via `11_train_laya --rows`. Unflag is mapped to LEGIT (assumption, stated in the script).
- **Network:** <OLD_DB_HOST> (ports 13366 and 13355) times out from this machine, and EDA's watcher fails the same way, so the live Postgres sources are unreachable. Credentials are in the gitignored `.env`. The scan picks up news automatically on the next poll once the route is back.
- Throughput is about 10 s per fully scored item, so roughly 150–250 scored items per hour on the shared Gemma.

## Propaganda Lab (added 2026-11-05; module for Evidence Desk :8766)

A world map of the propaganda signal by country, precomputed and stored. The plan is `~/.claude/plans/…snappy-tarjan.md`, and the data contract for the Evidence Desk tab (built by session fakenewspropoganda-7c) is `docs/PROP_LAB_CONTRACT.md`.

| Piece | Code | Notes |
|---|---|---|
| Country "about" | `fnp/geo/countries.py`, `data/geo/gazetteer_v1.json` | pycountry + geonamescache (capitals, cities ≥ 1M), Gemma multilingual aliases and demonyms (12 languages), overrides, flag emoji, ambiguity rules and region blockers. Matching uses Aho-Corasick at about 0.1 ms per post. |
| Country "origin" | `fnp/geo/origin.py`, `data/geo/channels_v1.json` | A Gemma-drafted channel registry (423 broadcast accounts across telegram, youtube, instagram and others), reviewed in `docs/review/07_OUTLET_COUNTRY.csv`. News uses outlet/ccTLD. Language is never used. |
| Map geometry | `fnp/geo/worldmap.py` → `data/geo/world_paths_v1.json` | Natural Earth 10m India worldview, simplified and projected to Equal Earth SVG paths (258 KB, 242 countries). |
| Inference | `scripts/30_prop_lab_infer.py` | Span encoder `prop_span_v1` on GPU (bf16, about 150–400 posts/s, 5.9 GB peak). Also country matching, and the kb_v2 nearest fact-check (EDA's stored bge-m3 vectors plus the reranker). Output: `data/prop_lab/v1/parts/*.parquet`. |
| Gemma sample | `scripts/33_prop_lab_gemma_sample.py` | 20 posts per country (2/3 strongest, 1/3 random). Gemma propaganda_v2 spans → encoder agreement. The FNP verdict with PROPAGANDA excluded (the stacker's PROPAGANDA probability is a length artefact) gives the fake/misleading call. |
| Aggregates | `scripts/31_prop_lab_aggregate.py` → `reports/prop_lab/v1/` | Computed by mode × window × source. Wilson CIs, length-adjusted lift (headline), EB-shrunk raw lift, heatmap, narratives (EDA topic v4, or the nearest topic centroid at cosine ≥ 0.55), entities, trend, fake layer, top 150 posts per country. Strict JSON with atomic swap. |
| Feedback | `POST :8879/api/prop_flag` → `data/feedback/prop_flags_v1.jsonl`; `scripts/32_prop_feedback_rows.py` | The "not propaganda" / confirm buttons become the social gold set for retraining the encoder. |

Rules:
- **signal:** ≥ 12 tokens and at least one strong span (mean prob ≥ max(class thr, 0.5)).
- **debunk:** the nearest fact-check is FAKE/MISLEADING with rerank ≥ 0.8 and cosine ≥ 0.72.
- **windows:** relative to the newest post in the data.
- **insufficient:** n < 30.

Caveats:
- This is a *signal*, not a verdict, because the encoder is not validated on social media.
- Long posts both mention more countries and carry more spans, so the headline lift is length-adjusted.
- EDA topics and entities stop at 29 Jul 2026, so narratives for newer posts come from centroid assignment.
- News: 0 until shaktidb_news is reachable. Its `news_engine.article_geo` then supplies the country for articles.

## Realtime cheap-pass precompute (added 2026-11-12)

- `scripts/36_realtime_precompute.py`: reads `Interface/data/realtime.sqlite` read-only and scores every unscored item newest-first. No Gemma.
  - Per item: span encoder `prop_span_v1` (spans, 14-class `technique_probs`, `p_prop`), countries, and the nearest kb_v2 fact-check.
  - Resume store `data/realtime/v1/scored.jsonl`. Failed parts are retried on the next run. A flock stops overlapping runs, and a time cap stops between parts.
  - Export: `reports/realtime/v1/{news,social}/part_NNNN.jsonl` plus `meta.json`, swapped atomically (previous export kept as `v1.prev`). Only items still in the SQLite window are exported.
- `fnp-realtime-precompute.timer` (user systemd): starts 3 h after the previous run finishes, up to 40,000 items per run, 100-minute cap. Log `logs/realtime_precompute.log`.
- Throughput about 9 items/s. The first window was 628k items, so the backlog takes about 13 runs. `meta.json` reports `n_unscored_in_window`.
- `p_prop` and `technique_probs` come from an encoder trained on long PTC articles, not validated on social media or short items. The export says so.
- The Interface session's deep verdict batch is separate. FNP runs no deep verdicts on these items.

## Dashboard

`UI/server.py` serves http://<SITE_HOST>:8879/ (log `logs/ui_server.log`). Port 8878 was taken. Tabs:
- Overview (live stage status)
- Methods
- Data
- Runs explorer (per-unit evidence trail)
- Evaluation
- Verdicts
- Laya & RL
- Review
- Try it (live predict)

It reads the run files fresh on each request. If `fnp.duckdb` is locked, it reads a snapshot copy.

## Integration with other sessions

- **EDA ("soft-topic-hmm-events"):** read-only. FNP reuses its parquet data (posts, bge-m3 embeddings, topics) and its HMM / event-chain / stats libraries via `fnp/eda_bridge.py`. The Gemma client is vendored as `fnp/gemma.py`. FNP keeps Gemma concurrency at 12 so EDA is not starved.
- **Interface session (fact-check crawl + Evidence Desk):**
  - Site: `Interface/server.py` serves http://<SITE_HOST>:8766/ (stdlib server). It runs as the systemd user service `fnp-interface` (`systemctl --user status|restart fnp-interface`, logs with `journalctl --user -u fnp-interface`). Settings are read from `~/.config/fnp/interface.env` (mode 600; `FNP_HOST`, `FNP_PORT`, `FNP_RT_ALLOW_OPEN`, `FNP_RT_SYNC`, `FNP_RT_ANALYSE`, optional budget, engine URL and Basic login; real environment variables win). Screens: Overview, Desk (the **Fake News Database**: the fact-check index), **Evidence Base** (the **Realtime Database**: Overview, Browse, Workbench, Insights), **Triage**, Claim Check, Propaganda Lab, Sources. Four themes (System, Light, Dark, Command Center, the default) and two typefaces. Tests in `Interface/tests`: 552 pass as of 2026-11-12 (node-based front-end helper tests skip when node is missing). Rebuild the index with `python3 Interface/build_index.py` (about 40 s; it also repairs language tags from the script of the text, 368 changed on 2026-11-12); the server reloads `Interface/build/` on its own, but code changes need a service restart.
  - Gallery filters: source, country, source type (IFCN signatory, independent, government, OSINT), channel (web, instagram, telegram, whatsapp, reddit and others), verdict, language, date, minimum score.
  - Data in (index: 37,888 items = 36,657 fact-checks + 1,231 unrated posts, de-duplicated by URL):
    - `Interface/factchecks/*.jsonl`: 4,859 rows from six sites (vishvasnews, quint_webqoof, snopes, fullfact, boomlive, newsmeter), scraped by `Interface/tools/fc_scrape.py` from each article's ClaimReview markup.
    - `Interface/data/factchecks_3y.jsonl`: 32,458 rows, 2023-10-30 to 2026-10-30, 11 sites (adds Alt News, Factly, dfrac, PIB Fact Check, FactCheck.org). Where a site has no ClaimReview markup, the verdict is inferred from categories, tags or title (`verdict_basis`).
    - `Interface/data/factcheckers.jsonl`: 6,732 posts from fact-checkers' own Instagram, Telegram and WhatsApp accounts (PIB, BOOM, Factly, Alt News, Newschecker, India Today, EUvsDisinfo, Bellingcat). Verdicts are LLM-extracted (`gemma`). The 2,610 rows flagged `is_fact_check: False` are skipped; rows without a verdict are listed as unrated and kept out of the search index. `TRUE` is mapped to LEGIT and `pibfactcheck` is merged into `pib_factcheck`.
    - `Interface/data/db_factchecks.jsonl` (84 rows, PolitiFact and FactCheck.org from the news DB, keyword-guessed verdicts) and `db_claim_candidates.jsonl` (628 unrated social posts, from keyword search in the DBs; they talk about misinformation, which is not the same as being misinformation).
    - The Interface index already combines all of the files above (42,151 rows read, 37,888 unique items after URL de-duplication: `factchecks/*` 4,859 rows, `factchecks_3y` 29,550 new of 32,458, `factcheckers` 2,824 new of 4,122, the DB exports 655 new of 712). Checked 2026-11-02 against a snapshot of `fnp.duckdb`: the engine's `gold_items_v1` (44,562 items) holds all 4,859 rows of `Interface/factchecks/*.jsonl` and only 2,908 of `factchecks_3y.jsonl`; it holds none of `factcheckers.jsonl` or `db_factchecks.jsonl`. So altnews, factly, dfrac, pib_factcheck and factcheck_org, and about 29,500 newer rows overall, are not yet in the engine's gold data or in `kb_v2`. PolitiFact's own site returned 403, so its only rows are the 67 from the DB.
  - Scoring: `plaus` = P(fake or misleading). Fact-checker label scores are FAKE 0.95, MISLEADING 0.70, INCONCLUSIVE 0.50, LEGIT 0.05. Where the verdict is inferred (`verdict_basis` is not `claimreview`), the score is pulled 40% toward 0.5 (a constant I chose, not tuned). Where `verdict_v1` has a doc row (`unit_id == <item id>`), the site blends in the engine's P(FAKE)+P(MISLEADING); items with only an engine score use it alone. Only BOOM and the 712 DB items have engine scores, read from `reports/verdict_v1_docs.jsonl` with `fnp.duckdb` as a fallback.
  - Index match (the instant part of Claim Check): TF-IDF over title + claim, top 5 neighbours weighted by similarity squared, shrunk toward 0.5 when the best match is weak, and "no close match" below 0.25. It matches words only.
  - Claim Check (`POST /api/analyse`, `Interface/claimcheck.py`, `engine_adapter.py`): runs the index match, then calls FNP's `POST /api/check_v1` (`FNP_ENGINE_URL`, 55 s hard limit, one request at a time with a short queue, results cached). Step 1 (fake or not) uses `fake_call`; the engine's raw PROPAGANDA class is shown only as a secondary tag. Propaganda is never gated. Step 2 shows a probability for each of the 14 technique buckets: the engine's `technique_probs` when `check_v1` sends all 14 (with `p_prop`), otherwise the highest span confidence; both are labelled not calibrated. Step 2 also shows Laya's own verdict and its raw propaganda head next to the engine's; this comes from a dataset shortcut and has not been validated. A timed-out reply is shown as degraded and never cached as normal.
  - Evidence Base (Realtime Database; `realtime_sync.py`, `rt_store.py`, `rt_enrich.py`, `rt_api.py`, `rt_jobs.py`, `rt_insights.py`): a read-only sync (read-only sessions, SELECT only) copies the last 7 days of news and 30 days of social posts from <DB_HOST> (news :13355, social :13366) into `Interface/data/realtime.sqlite` (at most 100,000 rows per social table; credentials in `~/.config/fnp/realtime.env`, mode 600). The sync thread runs only when `FNP_RT_SYNC=1` (on in the settings file). `/api/rt/*` is refused unless login is configured or `FNP_RT_ALLOW_OPEN=1`. Database fields are badged "From the database"; FNP fields "Inferred by FNP". The Overview has a Source freshness card (newest item per source table, flags for feeds with no new rows at the source and for tables at the row cap). The Workbench runs Claim Check on up to 25 selected items, one at a time; the Insights tab builds statements only from counts in the local store (optional Gemma phrasing through `/api/insight_v1`, validated again).
  - FNP realtime export integration: `rt_enrich.py` reads `reports/realtime/v1/` (`meta.json` plus part files; read-only) into a sidecar. FNP's cheap pass (p_prop, 14 technique probabilities, spans, countries, nearest fact-check) is exported every 3 hours and still covers a small part of the window (meta of 2026-11-12: 41,000 scored of 627,962 in the window). The banner and the coverage card take their numbers from `meta.json`, and the caveat text is shown verbatim: the scores are uncalibrated and not validated on social posts.
  - Deep scheduler (`rt_scheduler.py`, on when `FNP_RT_ANALYSE=1`): every 3 hours it runs Claim Check on up to 40 new items, one at a time. Priority: items with a fact-check match or strong propaganda signals first, then newest. A degraded reply is kept as a partial result (propaganda only, no verdict) and retried up to 3 times. A cycle stops when 70% or more of the last 10 replies are busy, degraded or failed, when the engine is unreachable, or at the time cap; it halves the rest of the budget when the engine is slow (median over 40 s). Counts per cycle are in `schedule_log` and the Overview Schedule card.
  - Triage tab (`rt_triage.py`): two sources, the FNP triage export (`reports/triage/v1`) and a queue ranked from the Realtime Database. Tiers high, medium and low are rules that were assumed, not fitted, and no item has been validated by a person, so a tier is a lead and not a verdict.
  - Detector accuracy, measured 2026-11-01 on a 4,500-claim leave-one-out sample of publisher-stated verdicts (index of 35,064 items, before the social posts were added): against any other item it answered 48% of claims with AUC 0.90 (90% accuracy vs 72% majority baseline); against other publishers only it answered 23% with AUC 0.59 (no better than the baseline). On the 588 DB social posts it gave "no close match" for 540. It works as a lookup for already-checked claims, not as a detector for new text. Not re-measured since the new data was added.
  - PROPAGANDA on social posts is shown as a secondary tag with a caveat, not as the verdict.
  - Known gaps: `verdict_norm` is a keyword mapping of the site's verdict text. Full Fact verdicts are free text, so many map to INCONCLUSIVE. Vishvas News and Quint are almost all FAKE or MISLEADING (about 93% of all fact-checks are FAKE or MISLEADING). Many fact-checker social posts repeat articles already in the web data, so the same claim can appear twice.
  - FNP never writes to `Interface/`.

## Known issues (details in `docs/review/`)

- **Evidence relevance:** KB fact-checks about a *different* but similar item can count as refute or context_mismatch, e.g. a Malda video used against a Sialkot claim. That inflates n_eff. The relevance discount needs checking against the eval data.
- **Base-rate shift:** the PROPAGANDA gold set is only PTC news articles (long text) against short claims elsewhere, so format confounds the label. There is also no article-level LEGIT set.
- **Uncalibrated maths (from the maths review):**
  - The rule-head margins are uncalibrated.
  - P_prop saturates as spans accumulate (noisy-OR).
  - one_sidedness measures evidence consensus.
  - BP double-counts evidence that linked claims share.
  - The p_true CI is now centred on the WoE logit with the DerSimonian–Laird half-width (fixed).
- **PROPAGANDA over-labelled on social posts:** 454 of 628 posts, because P_prop saturates and the doc rollup applies its rule. See `docs/review/05` problem 7.
- **Laya:** the model card reports that its "noul" answers can follow option labels (#156) and that its escalate head carries no signal (#185). Temperature fits on tiny calibration sets hit the clamp. Free GPU memory fluctuates between 2.9 and 6.5 GB.
- **Web search:** DuckDuckGo returns a bot challenge, so web evidence is off. A search API key would enable it.
