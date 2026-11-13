# Realtime export contract v1: what FNP actually writes, and how the Interface reads it

Status: documentation of the REAL format (first export 2026-11-12), written by the Interface session from the files FNP
produced. The FNP tree is not modified by this document. Earlier versions of this file were a proposal; the proposal
shape (section 7) is still accepted by the reader but FNP does not write it.
Purpose: FNP writes its per-item "cheap pass" results for the customer's realtime news and social items; the Evidence
Desk (`Interface/rt_enrich.py`) joins them to its local cache and shows them badged "Inferred by FNP".
Items without a record show "Not analysed yet". The Interface never guesses a missing field.

## 1. What the Interface holds (so you know what you join to)

`Interface/data/realtime.sqlite`, table `items`, filled read-only from the customer databases (news :13355, social
:13366 on <DB_HOST>) for a bounded window (default last 30 days, newest first, capped per source). The cache keeps a
title, a text excerpt of at most 1,500 characters (HTML and control characters stripped) and the database's own fields.
FNP reads the full text from the databases itself; the cache is not an input for FNP.

### Item id scheme (the join key)

    id = "<source_table>:<source_id>"          social, e.g. "reddit_posts:9824575"
    id = "news:<source_table>:<article_id>"    news,   e.g. "news:scraped_articles:19557256"

Write `id` exactly like this, and also write `source_table` and `source_id` separately (the Interface joins on either).
`source_id` is always a string in the export (ids are bigint for most tables, varchar for some).

| kind | source_table | source_id is | platform label |
|---|---|---|---|
| news | `scraped_articles`, `scraped_articles_trusted` | `news_engine.article_fact.article_id` | news |
| social | `instagram_posts`, `twitter_posts`, `telegram_posts`, `reddit_posts`, `linkedin_posts`, `snapchat_posts`, `whatsapp_posts`, `pinterest_posts`, `baidu_tieba_posts`, `discord_posts`, `facebook_posts`, `facebook_profile_posts`, `mastodon_posts` | the table's `id` column | table name without `_posts` |
| social | `x_posts` | `tweet_id` | x |
| social | `tiktok_posts`, `truthsocial_posts` | `post_id` | tiktok / truthsocial |

(`mastodon_posts.id` is a varchar snowflake; the others are bigint.) Tables not listed are not synced. `youtube_comments`
and the seen/helper tables are excluded for now. If FNP analyses other tables, say so and the map will be extended.

## 2. Files FNP writes

    reports/realtime/v1/meta.json
    reports/realtime/v1/news/part_NNNN.jsonl
    reports/realtime/v1/social/part_NNNN.jsonl
    reports/realtime/v1.prev/...            the previous export (the reader ignores it)

* JSONL, UTF-8, one JSON object per line, **strict JSON** (no `NaN`, `Infinity`; write `null`). 2,000 rows per part, newest
  first. Part names are `part_0000.jsonl`, `part_0001.jsonl`, ... Hundreds of parts are expected: a timer (every 3 h)
  grows the export until it covers the whole window (about 628k items, roughly 13 runs).
* **Atomic swap**: the whole export directory is replaced; `v1.prev` keeps the previous one. `meta.json` is the
  last file written. The reader lists `news/*.jsonl` and `social/*.jsonl` itself (meta carries no file list, no sha256),
  ignores `*.tmp`, and tolerates a part disappearing mid-read: it restarts once from a fresh listing, and if parts are
  still missing it serves what it read and reports `partial` (retry after 2 minutes).
* **Reload trigger**: the reader compares `meta.json` `built_at` (a string) with the one it loaded; one `stat` per request.
  A full re-read follows (rows move between parts when new items arrive).
* One id appears once; if it appears twice the row with the later `scored_at` wins.

## 3. Per-row record (real format)

| field | type | notes |
|---|---|---|
| `id` | string | exactly `items.id` (section 1). The only field a row must have |
| `kind` | `"news"` / `"social"` | |
| `platform`, `source` | string | informational |
| `published_at` | string | UTC, informational |
| `lang` | string | FNP's own detection (`und` when unknown) |
| `too_short` | bool | true when the text is too short to score; then `p_prop` is `null`. Reader shows "Too short to score" and counts these separately |
| `n_tokens` | int | |
| `p_prop` | number in [0,1] or null | encoder score, **uncalibrated**; null when `too_short` |
| `n_spans`, `n_strong` | int | all spans found / spans above the "strong" threshold |
| `density` | number | |
| `technique_probs` | object | all 14 buckets (`technique_classes` in meta) -> maximum token probability of that class, in [0,1]; uncalibrated. Missing buckets are shown as missing, never as 0 |
| `strong_techniques` | list of string | the buckets FNP calls strong; drives the technique mix |
| `spans` | `[[start, end, technique, prob], ...]` | OFFSETS into `analysed_text` (below); end exclusive; Python str (code point) indexing. Can be many (38 seen); the reader keeps the 12 most confident. Offsets follow the encoder's word pieces, so a span can start or end inside a word |
| `countries` | `[{iso3, weight, methods, terms}]` | ISO 3166 alpha-3; `methods` e.g. `name`, `capital`; `terms` the matched words |
| `nearest_factcheck` | `{kb_id, label, claim, url, cosine, rerank}` | closest known fact-check by embedding similarity: a lookup, not a verdict. `rerank` is `null` when `cosine` < 0.72. In the first export every `cosine` was below 0.72, so the UI marks such matches as weak |
| `scored_at` | string | ISO time with offset; used as the row's `built_at` |
| `model_version` | string | e.g. `prop_span_v1+kb_v2` |

`p_fake`, `p_misleading` and `verdict` are **not** in the real export (the realtime verdicts come from deep Claim Check
runs, not from the batch). `analysed_text` in meta defines the offset base: `title + "\n" + text_excerpt` (the
Interface's cached `items.title` and `items.text_excerpt`, which are the same database fields FNP read). The reader
slices span text from its own copy lazily when one item is opened, drops spans that fall outside the string or straddle
the title/excerpt boundary, and reports how many it dropped. No article text is written by FNP except the fact-check
`claim` of `nearest_factcheck`.

## 4. `meta.json` (real)

    built_at, n_items{news,social}, n_items_total, n_in_window, n_unscored_in_window,
    last_item_published_at{news,social}, next_run_at, interval_hours, n_scored_this_run, n_failed_this_run,
    run_seconds, model_version, technique_classes (14), caveat, analysed_text

* `n_items_total` is the number of rows across all parts; the reader checks `rows loaded + rows rejected == n_items_total`
  and reports `count_matches_meta`.
* `n_in_window` / `n_unscored_in_window`: the window FNP is working through and how much of it is still unscored. The UI
  banner is "FNP batch: `n_items_total` of `n_in_window` scored, last export `built_at`, next run `next_run_at`".
* `n_failed_this_run > 0` shows a calm note. `next_run_at` may be absent (the UI then omits the next-run part).
* `caveat` is shown verbatim (escaped) in the Overview "How to read this" panel.
* Optional, **not written today**: `max_items_per_run` (also accepted as `per_run_limit` / `max_per_run`). With it the
  Interface estimates the runs left as `ceil(n_unscored_in_window / min(n_scored_this_run, limit))`; without it the
  estimate is `null` and the UI says so. FNP can add it to get a backlog estimate.

## 5. Reader behaviour (Interface)

1. Per request, one `stat` of `meta.json`. A changed `built_at` starts a background rebuild of the sidecar
   (`Interface/data/rt_enrich.sqlite`, schema version 2); requests keep the previous sidecar until the new one is swapped in.
2. Rows are streamed (bounded memory, flush every 5,000). Validation per row: `id` present; `p_prop`, probabilities and
   `cosine`/`rerank` numbers in [0,1]; `technique_probs` an object with at least one known class; `spans` a list;
   `countries` a list; `nearest_factcheck` an object or null; `too_short` a bool. Malformed span or country entries are
   dropped (counted); a bad row is rejected, counted by reason, and never stops the load.
3. Load report (in `/api/rt/status` and `coverage.batch.load`): `rows_ok`, `rejected`, `reasons`, `files_listed`,
   `files_read`, `parts_missing`, `restarts`, `partial`, `meta_changed_during_load`, `count_matches_meta`, `seconds`,
   `too_short`, `spans_dropped`.
4. A missing directory or meta: "Not analysed yet". An unscored item is unscored, never "clean".
5. The triage export (`reports/triage/v1/items.jsonl`, FNP `docs/TRIAGE_CONTRACT.md`) is joined by exact URL as before.
6. The nearest fact-check URL is normalised (`build_index.norm_url`: host without `www.`, path without trailing slash,
   lower case) and looked up in the Fake News Database index; a hit gives `desk_id` and the UI links `#desk/<id>`,
   otherwise the external URL.

## 6. What the Interface shows (so FNP knows the meaning it will be given)

* `p_prop`: "propaganda signal", labelled uncalibrated; technique bars: the 14 values sorted high to low with the note
  "encoder score (max token probability): uncalibrated, not validated on social posts".
* `too_short`: "Too short to score", no signal, no bars, not in the signal chart or technique mix.
* `nearest_factcheck`: a card with similarity (and rerank when present), the fact-check's own label, claim and a link;
  below 0.72 it is labelled a weak match.

## 7. Older proposal shape (still read, not written by FNP)

`meta.json` with `files:[{path, rows, sha256}]` and rows with `techniques:[{technique, count, max_confidence}]`, `spans`
as dicts with verbatim `text`, `countries:{about, origin}`, `nearest_factcheck:{id, similarity}`, and `p_fake` /
`p_misleading` / `verdict` (verdict null exactly when `p_fake` is null). With `files` present the reader verifies the
listed files and sha256 and keeps the previous build on any mismatch. The Interface's own tests use this shape for
regression.
