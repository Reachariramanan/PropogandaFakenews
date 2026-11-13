# Evidence Desk: fake-news database, Claim Check and Propaganda Lab

A stdlib-only Python server (`server.py`) plus a static single-page UI (`web/`). It serves a searchable database of fact-checks, Claim Check (a two-step test of a pasted claim), source statistics, and a read-only Propaganda Lab view of the FNP aggregate build. scikit-learn is used for TF-IDF when installed (pure-Python fallback otherwise). The only outbound call is the Propaganda Lab flag proxy to a local service (see below).

## Ports and where it runs

- **Live site: `http://<SITE_HOST>:8766/`**, run as the systemd user service `fnp-interface` (see "Running as a service"; the manual recipes below are for development).
- Port **8765 is a different application** on this machine. `server.py` still defaults to `127.0.0.1:8765`, so always set `FNP_PORT=8766` (and `FNP_HOST`) explicitly or the start will fail with "address in use".
- `:8879` is the flag service the proxy talks to, `:8890` is another service. This server never touches either except for the flag proxy to `:8879`.

## Run manually (development)

This does not use the service. If the service is running, stop it first or use another `FNP_PORT`. When started as `python3 Interface/server.py`, the settings file (next section) is read too; variables you set on the command line win over it.

From the repository root (`/home/hariramanan/NTRO/FakeNewsPropoganda`):

    python3 Interface/build_index.py     # merge data, score, build TF-IDF index -> Interface/build/
    FNP_HOST=<SITE_HOST> FNP_PORT=8766 python3 Interface/server.py

Re-run `build_index.py` whenever new data lands. The running server reloads `build/` when the files change; no restart needed for data.

### Restart recipe for a hand-started server (by pid, never `pkill -f`)

Only needed for a manual start; the service restarts with `systemctl --user restart fnp-interface`.

    PID=$(ss -ltnp | grep ':8766 ' | grep -o 'pid=[0-9]*' | cut -d= -f2)
    ps -o pid,cmd -p "$PID"                       # check it is python3 Interface/server.py
    kill "$PID"                                   # SIGTERM, this one pid only
    while ss -ltn | grep -q ':8766 '; do sleep 0.5; done
    cd /home/hariramanan/NTRO/FakeNewsPropoganda && (FNP_HOST=<SITE_HOST> FNP_PORT=8766 setsid nohup python3 Interface/server.py > /tmp/fnp_interface_server.log 2>&1 &)
    sleep 5; curl -s http://<SITE_HOST>:8766/healthz

The settings file already supplies `FNP_HOST`, `FNP_PORT`, `FNP_RT_ALLOW_OPEN`, `FNP_RT_SYNC` and `FNP_RT_ANALYSE`; add or override variables on the command line as needed (scheduled deep analysis: see its section) and `FNP_AUTH_USER=... FNP_AUTH_PASS=...` to that environment to turn on login (next section). The log of a hand-started server is `/tmp/fnp_interface_server.log`; the service logs to journald (access log, flag audit lines, failed-login lines; never credentials).

## Running as a service

The live server runs as the systemd **user** service `fnp-interface` (`~/.config/systemd/user/fnp-interface.service`): `python3 Interface/server.py` from the repo root, `Restart=on-failure` (3 s), logs to journald, enabled at login/boot (`WantedBy=default.target`). User lingering is on (`loginctl show-user $USER | grep Linger` says `Linger=yes`), so it starts at boot and survives logout. No sudo is involved.

    systemctl --user status fnp-interface
    systemctl --user restart fnp-interface        # also: start, stop
    journalctl --user -u fnp-interface -n 50 --no-pager     # add -f to follow
    systemctl --user daemon-reload                # after editing the unit file

Stop is a clean SIGTERM: the HTTP server shuts down and the sync and scheduler threads stop before exit. `Restart=on-failure` restarts after a crash or kill -9, not after a deliberate `stop` or a clean exit on SIGTERM.

### Settings file: `~/.config/fnp/interface.env` (mode 600)

`KEY=VALUE` per line, `#` comments, blank lines, optional surrounding quotes, no shell expansion. Only keys starting with `FNP_` are accepted; others are ignored (the startup line prints a count). Keys in use: `FNP_HOST`, `FNP_PORT`, `FNP_RT_ALLOW_OPEN`, `FNP_RT_SYNC`, `FNP_RT_ANALYSE`; commented examples for `FNP_RT_ANALYSE_BUDGET`, `FNP_ENGINE_URL`, `FNP_AUTH_USER`, `FNP_AUTH_PASS`. Any variable in the table below can go there.

- **Precedence:** a real environment variable always wins over the file. systemd loads the file as `EnvironmentFile` and `Interface/settings.py` loads it again at the very start of `server.py` (before any module reads its settings) for variables not already set, so a hand start behaves the same as the service.
- Only key names and counts are ever logged, never values. Restart the service after editing the file.
- Database credentials are not here: they stay in `~/.config/fnp/realtime.env`, which `realtime_sync` loads itself.
- **Turn login on:** uncomment and fill `FNP_AUTH_USER` and `FNP_AUTH_PASS` (both, non-empty) in the file, then `systemctl --user restart fnp-interface`. Check `curl -s http://<SITE_HOST>:8766/api/rt/gate` shows `"auth": true`. The file must stay mode 600.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `FNP_HOST` | `127.0.0.1` | Bind address. Use `<SITE_HOST>` for the LAN site. |
| `FNP_PORT` | `8765` | Bind port. Use `8766` (8765 belongs to another app). |
| `FNP_FLAG_URL` | `http://127.0.0.1:8879/api/prop_flag` | Upstream the flag proxy forwards to. |
| `FNP_PROPLAB_DIR` | `<repo>/reports/prop_lab/v1` | Propaganda Lab aggregate build (read-only). |
| `FNP_AUTH_USER`, `FNP_AUTH_PASS` | unset | Both set and non-empty turns HTTP Basic auth on. Otherwise off. |
| `FNP_RT_ALLOW_OPEN` | unset | `1` lets `/api/rt/*` answer without login. Default: refused (403) unless login is configured. |
| `FNP_RT_SYNC`, `FNP_RT_INTERVAL` | unset, `300` | `FNP_RT_SYNC=1` starts the realtime sync thread inside this process; interval in seconds. |
| `FNP_RT_ANALYSE` | unset | `1` starts the scheduled deep analysis thread inside this process (see Scheduled deep analysis). Default off. |
| `FNP_RT_ANALYSE_INTERVAL` | `10800` | Seconds between cycles (3 h; minimum 30). |
| `FNP_RT_ANALYSE_BUDGET` | `40` | Items per cycle (hard ceiling 500). Lowered from 100: the engine takes 15 to 40 s per item. Adaptive pacing halves what is left when the engine is slow. |
| `FNP_RT_ANALYSE_MAX_SECS` | `5400` | Time cap per cycle. |
| `FNP_RT_ANALYSE_FIRST_DELAY` | `600` | Seconds after server start before the first cycle (it also waits for the first sync, see below). |
| `FNP_RT_NEWS_DAYS/LIMIT`, `FNP_RT_SOCIAL_DAYS/LIMIT` | 7 / 400000, 30 / 100000 | Sync window and per-source caps (see Realtime data layer). |

## Optional login (HTTP Basic), off by default

Set both `FNP_AUTH_USER` and `FNP_AUTH_PASS` before starting. If only one is set, auth stays off and a warning is printed. When on:

- every route (static files, `/api/*`, every POST) needs `Authorization: Basic ...`, except `GET /healthz`;
- no or wrong credentials give `401` with `WWW-Authenticate: Basic realm="Evidence Desk"`; credentials are compared with `hmac.compare_digest`;
- throttle: 10 failed logins from one client IP within 5 minutes gives `429` (with `Retry-After`) for that IP, even for the right password, until the oldest failure ages out. A request with no `Authorization` header (a browser's first try) is not counted as a failure;
- credentials are never logged. The usernames must not contain `:`.

Basic auth over plain HTTP sends the credentials base64-encoded, not encrypted: it keeps casual visitors out of the LAN site, it is not protection against someone sniffing the network. The browser caches the login for the origin; `web/` uses only same-origin relative or root-relative URLs, so cached credentials apply to every fetch. The password is visible to the same Unix user through `/proc/<pid>/environ`.

## `GET /healthz` (never needs auth)

    {"ok": true, "build_built_at": "<build/items.json meta.built>", "n_items": 37888,
     "proplab_built_at": "<reports/prop_lab/v1/meta.json built_at> or null", "uptime_s": 123.4,
     "realtime": {"enabled": false, "running": false, "last_sync": "<ISO> or null", "lag_minutes": 37.4, "rows": 17917, "errors": 0,
                  "analysis": {"enabled": false, "last_run": null, "next_run_at": null, "analysed_deep": 0}}}

Returns `503 {"ok": false, ...}` when `build/items.json` is missing or unreadable. The large file is parsed at most once per rebuild (or the already loaded copy is reused), so polling is cheap.

## Data layout

Inputs (read by `build_index.py`):

- `factchecks/*.jsonl`: per-site ClaimReview scrapes (`tools/fc_scrape.py`). One JSON per line with `id, site, url, claim, verdict_raw, verdict_norm (FAKE|MISLEADING|LEGIT|INCONCLUSIVE), rating_value, published, lang, title, body, description`.
- `data/factchecks_3y.jsonl`: merged three-year collection (`tools/fc_collect.py` into `data/fc_shards/`, then `tools/fc_merge.py`). Same schema.
- `data/factcheckers.jsonl`: fact-checkers' own Instagram/Telegram/WhatsApp accounts. `platform` becomes `channel`; rows with `is_fact_check: false` are skipped; `TRUE` maps to `LEGIT`; `pibfactcheck` is renamed `pib_factcheck`.
- `data/db_factchecks.jsonl`: same schema as `factchecks/` (verdict may be missing, treated as INCONCLUSIVE).
- `data/db_claim_candidates.jsonl`: `id, platform, url, text, published, lang, engagement`; no verdict. Indexed as kind `candidate`, verdict `UNRATED`.
- Engine scores (optional, read-only): `../reports/verdict_v1_docs.jsonl` first, else `../data/fnp.duckdb` table `verdict_v1` (copied to a temp file if locked). Missing files or `duckdb` module are skipped and the UI shows "engine scores pending". Doc-level rows (`unit='doc'`) are joined first; if only claim-level rows exist for a doc, the most suspicious one is used.
- Rows are de-duplicated by normalised URL (host without `www`, no query string or trailing slash); the row with the longer body wins.

Generated: `build/items.json` (items + meta, about 190 MB) and `build/index.pkl` (TF-IDF over `title + claim`, unigrams and bigrams, sublinear tf, L2-normalised). Written through `.tmp` files and `os.replace`, so the server never sees a half-written build.

Other: `web/` is the UI (`index.html`, `app.js`, `claimcheck.js`, `proplab.js`, `evidence.js`, `styles.css`, `fonts/`), `tools/` the collectors, `tests/` the test suite. `data/articles*`, `data/thumbs`, `data/gallery.html` and the scrapers in `data/` are gallery material, not read by the server. `build/`, `data/`, `factchecks/` and `__pycache__/` are git-ignored.

## API

- `GET /api/items?q=&site=&verdict=&lang=&country=&tier=&channel=&kind=&from=&to=&min_plaus=&max_plaus=&sort=plaus|date&page=&page_size=` (comma lists for site, verdict, lang, country, tier, channel; `page_size` capped at 100).
- `GET /api/stats`, `GET /api/item/<id>`.
- `POST /api/check {"text": ...}`: the instant index match behind Claim Check step 1 (body up to 100 KB, text capped at 5000 characters).
- `GET /healthz`, see above.

### Propaganda Lab routes (read-only)

Served from `FNP_PROPLAB_DIR`, with `ETag`/`304`; an atomic `v1.building` to `v1` swap is picked up on the next request. `503 {"code": "not_built"}` when the directory is absent.

- `GET /api/proplab/meta | world | countries | heatmap`
- `GET /api/proplab/country/<ISO3>`: ISO3 must match `^[A-Z]{3}$`, otherwise `400`; unknown code `404`.
- `GET /api/proplab/posts?mode=about|origin&iso=<ISO3>&offset=&limit=`: streams `posts/<mode>/<ISO3>.jsonl`, `limit` capped at 50.

### Flag proxy

`POST /api/proplab/flag {"post_uid": ..., "action": "not_propaganda"|"confirm"|"clear", "technique": optional}`. The server validates and forwards to `FNP_FLAG_URL` (3 s timeout):

- body must be a JSON object of at most 2 KB (`413` over); `post_uid` matches `^[A-Za-z0-9:_.-]{1,128}$`; `technique`, if given, must be one of the techniques in `heatmap.json`; anything else is `400`;
- 30 requests per client IP per minute (`429` over);
- upstream error codes and bodies are passed through; upstream unreachable is `502 flag service unavailable`; a non-JSON upstream reply is `502`;
- each accepted flag prints one audit line (`flag ip=... action=... post_uid=...`) to the log.

## Scores

`plaus` in [0,1] is the estimated probability that a claim is fake or misleading.

- Label score: FAKE 0.95, MISLEADING 0.70, INCONCLUSIVE 0.50, LEGIT 0.05. If `rating_value` is numeric in 1..5 (1 = false) and agrees in direction with the verdict, it nudges the score: `0.85*label + 0.15*(5-r)/4`. Verdicts inferred from tags or titles (`verdict_basis` other than `claimreview`) are shrunk toward 0.5: `0.5 + 0.6*(label-0.5)`.
- Engine score: `P(FAKE)+P(MISLEADING)` from `verdict_v1.probs`.
- `score_source`: `fact-checker-label` (label only), `engine` (engine only), `blend` (mean of both), `none` (unrated post without an engine row, so no score).

Dynamic metrics per item: claim length, capitals and exclamation ratios, sensationalism (0..1 heuristic), age and recency (`exp(-days/180)`), `cross_checker_agreement` (distinct sites among near-duplicates with cosine >= 0.5, plus the item's own site for fact-checks), and the per-site fake/misleading rate.

## Claim Check: index match (`POST /api/check`)

The text is compared by TF-IDF cosine to all fact-checks. Similarity < 0.25: state `no_close_match`, no score, only linguistic signals. Otherwise the score is the average plaus of matches with cosine >= 0.25 weighted by cosine squared, shrunk toward 0.5 by `min(1, best/0.6)`. States: weak (< 0.40), moderate, strong (>= 0.60). The score measures resemblance to previously rated claims, not a ruling on the text. Because the database is dominated by debunks, a close match mostly means "this claim has been debunked before".

## Claim Check screen (`#check`, formerly Detector; `#detector` still redirects)

`web/claimcheck.js` drives the screen. On Check it fires `POST /api/check` (instant index match, shown at once as step 1) and `POST /api/analyse` (engine, 20-35 s, shown with an elapsed timer and a Cancel button that aborts the browser request only). Step 1 "Is it fake?" shows the gauge, verdict, nearest fact-checks and the engine's second opinion or model estimate; step 2 "What kind of propaganda?" is always shown when the engine returned a propaganda block (no gating): a probability bar for each of the 14 technique buckets (`step2.propaganda.classes`) with a basis note, a span-only "Propaganda signal" line on top (STRONG if any span is at least 0.50, WEAK if spans exist but all are lower, NONE if none; Laya is not merged into it; the one-sided chip sits beside it), then the highlighted spans (weak spans, below 0.50, are marked with a dotted underline and "(weak)"), the bucket summary and the secondary "Fine-tuned Laya check" block: Laya's own 5-class verdict with its PROPAGANDA probability ("Laya's own class, not validated"), the raw propaganda yes/no head (0 to 1, a raw output, not a verdict; FNP measured it at 1.0 on news-article states and 0.0 on others, so it appears to separate article-style training data rather than propaganda wording), and an agree / differs chip against the span tagger. An empty result says "No strong propaganda spans found" (never "detected"): on short texts (the engine's `too_short` flag, under 12 tokens, else under 60 words as a proxy) the tagger reports little, so that is not evidence the wording is neutral. When the engine sends `propaganda.technique_probs` the 14 bars use it (basis `engine`, the max token probability) and its per-span `strong` flag and `p_prop` drive the signal line; otherwise the 0.50 span-confidence rule and the strongest span per technique are the fallback. `step2.laya` follows the engine's `laya_reason` (no_claims | timeout); `engine.degraded_why` follows `degraded_reason` (a no_claims reply is now `degraded: false`; the old degraded shape is still understood). `one_sided` is shown as Gemma's framing judgement, unvalidated. Laya's input already contains the tagger's spans, so it is not an independent opinion. A degraded reply with no extracted claim says "No checkable claim found" (no Retry button); step 1 also draws verdict class bars from `step1.classes`. A "How we decided" panel lists models, timings, cache flag and caveats. "Your checks" keeps each full response in memory for the tab, so clicking a past check restores it without a request. Examples come from `GET /api/examples?n=4` with a Refresh button. Technique colours are the `--k1..--k8` tokens; every highlight also carries the technique name as text.

## Claim Check API (`POST /api/analyse`, `GET /api/examples`)

`/api/analyse` {text, lang?, force_propaganda?} = index match (same logic as `/api/check`) + one engine call (`FNP_ENGINE_URL`, default `http://127.0.0.1:8879/api/check_v1` (contract: `docs/CHECK_V1_CONTRACT.md`; the older `/api/predict` shape is still understood by key detection). Step 1 uses `fake_call`, with `laya` reported separately; a `degraded` reply (55 s limit) is shown as `engine.state` = `degraded` (fast result + techniques, never cached); a 503 busy uses the engine's `Retry-After`; `FNP_ENGINE_TIMEOUT` 60 s; `FNP_ENGINE_QUEUE_WAIT` 75 s) + combined verdict. Propaganda is never gated: `step2.shown` is true whenever the engine returned a propaganda block; `gated_out` (always false), `forced` (always false) and the `force_propaganda` request field are kept only for old clients. `step1.classes` = [{label, probability, secondary}] (FAKE, MISLEADING, LEGIT, INCONCLUSIVE from `fake_call.probs`, then the raw PROPAGANDA with `secondary: true`); `step2.propaganda` also carries `signal` {level strong|weak|none, strong_n, weak_n, max_confidence, threshold, techniques[]}, `laya_is_propaganda_raw` (0-1 or null; bools and out-of-range values become null), `one_sided_score` (0-1 or null; `one_sided` stays a bool = score >= 0.5) and `text_words`; `step2.laya` = {available, state ok|not_run_no_claims|timeout|not_returned, label, confidence, probs[], propaganda_head, head_in_range, short_text, text_words, compared agree|differs|not_compared, compared_note, independent: false}; `engine.degraded_why` (no_claims | timeout | unavailable) and `engine.n_claims` explain a degraded reply (a no_claims reply is deterministic and is cached, a timeout never is). Assumed constants (in `engine_adapter.py`): `SPAN_STRONG` 0.50 (FNP's own STRONG_P from `scripts/30_prop_lab_infer.py`, not calibrated), `LAYA_YES` 0.50 (assumed), `SHORT_WORDS` 60 (the claim-extraction cutoff; its use as the "short text" line is assumed). `step2.propaganda.classes` = [{technique, probability, basis}] for all 14 merged buckets, sorted descending, explicit 0s: basis `engine` when the engine sends `propaganda.technique_probs` (all 14 keys, 0-1), otherwise `span_max` (highest span confidence for the bucket); `basis_note` says it is not a calibrated probability. Cached Workbench results and `/api/rt/item` analysis rows get the same blocks. One engine request at a time, 3 may queue, 10 engine calls/min per IP (429), in-memory cache (200 entries, 1 h). If the engine cannot be used the answer is still 200 with the index result and `engine.state` = unavailable | timeout | busy (+ Retry-After) | error. Details and response shape: docstrings of `claimcheck.py` and `engine_adapter.py`. `/api/examples?n=4` (1..8) returns random, verdict-balanced real fact-check claims.

## Themes

Settings > Theme offers System, Light, Dark and **Command Center** (`data-theme="command"`, saved in `localStorage` key `fnp-theme`). Command Center is a dark navy control-room look: glowing blue panel borders, uppercase tracked titles, line icons on the Overview cards, a verdict-mix donut, a legend card on the 30-day chart, a faint dotted world map behind the Overview heading (built in the browser from `/api/proplab/world`), and a top bar with a live UTC clock and a quick theme button. Its tokens and rules sit at the end of `web/styles.css`; the other themes are unchanged and the Typeface setting works with all of them.

## Realtime data layer (Evidence Base, phase E1)

`realtime_sync.py` copies the realtime news and social posts into a local SQLite cache (`data/realtime.sqlite`, mode 600, WAL, FTS5, git-ignored) so the UI never queries the Realtime Database live. Helpers: `rt_store.py` (schema, search, near-duplicate index, status) and `rt_text.py` (text hygiene, simhash). The cache holds only titles, a <=1,500-character excerpt and the database's own fields, never raw HTML or full bodies.

    python3 Interface/realtime_sync.py --once --limit 1000 --days 3     # bounded dry run (limit = rows per source)
    python3 Interface/realtime_sync.py --once --sources reddit_posts,news
    python3 Interface/realtime_sync.py --status                         # cache status, no database access

- **Credentials**: `FNP_RT_HOST`, `FNP_RT_NEWS_PORT/DB/USER/PASSWORD`, `FNP_RT_SOCIAL_PORT/DB/USER/PASSWORD` from the environment, else `~/.config/fnp/realtime.env` (directory 700, file 600, outside the repo). Passwords never reach logs, errors, stdout or SQLite.
- **The two accounts are PostgreSQL superusers**, so read-only is enforced by this code: every session is opened with `default_transaction_read_only=on`, a 30 s statement timeout and an idle timeout, and verified; every statement must be a single `SELECT`/`WITH` or it is refused; no temp tables, one connection per database, strictly sequential. Please ask the database administrator for real read-only roles (`GRANT SELECT` only) and switch the env file to them.
- **Scope** (sync policy): news the last 7 days (`FNP_RT_NEWS_DAYS`), at most 400,000 rows per news source (`FNP_RT_NEWS_LIMIT`); social the last 30 days (`FNP_RT_SOCIAL_DAYS`), at most 100,000 rows per table (`FNP_RT_SOCIAL_LIMIT`). CLI `--days` / `--limit` override both kinds. Newest first, batches of 2,000 by keyset pagination, incremental by watermark afterwards (2 h overlap, `FNP_RT_OVERLAP_HOURS`). The background pass prunes items older than their window (unless they have a stored analysis). A source that errors is logged and skipped; the run continues. `youtube_comments` and the seen/helper tables are excluded.
- **Topic refresh**: the database fills topic/category some time after ingest (about 12% of the newest news has one at ingest), so every pass re-reads `db_topic_id`, category, subtopic and their names for stored news whose topic is still NULL (newest first, primary-key SELECTs, at most `FNP_RT_REFRESH_BUDGET` = 5,000 per source per cycle).
- **Background**: `server.py` starts the sync thread inside the server process only when `FNP_RT_SYNC=1` (interval `FNP_RT_INTERVAL`, default 300 s, minimum 30 s) and stops it on shutdown (SIGTERM/Ctrl-C); there is deliberately no systemd/cron entry.
- **FNP inference** (propaganda techniques, nearest fact-check, verdicts) arrives separately through the export described in `docs/REALTIME_CONTRACT.md` (a proposal to the FNP session).
- Do not `VACUUM` the SQLite file (the FTS index is keyed on `items.rowid`); `rt_store.rebuild_fts()` repairs it if you do.

## Evidence Base screens (phase E3): `web/evidence.js`

New nav item **Evidence Base** (`#evidence`, sub-tabs `#evidence/overview` (default) and `#evidence/browse`, item `#evidence/item/<id>`, `#evidence/workbench`, `#evidence/insights`). It reads only `/api/rt/gate`, `/api/rt/status`, `/api/rt/stats`, `/api/rt/items` and `/api/rt/item/<id>`; nothing is written to the server; the Workbench and Insights tabs add the routes listed in the next sections.

- **Two sources, labelled.** Evidence Base pages carry a `Realtime Database` banner (sync age, newest item age, row counts, "Analysed X% of N items"; a "Sync is behind" flag when the last sync is over 30 minutes old). The Desk list, Overview and Sources pages carry a small `Fake News Database` chip.
- **Gate.** On entry the page calls `/api/rt/gate`; when it is closed (`open:false, auth:false`, or any `403 login_required`) a calm full-panel explanation is shown, with the technical hint under "For administrators". `429` and network errors show a retry panel. When login is on, the browser's own Basic prompt does the work.
- **Overview.** Filters (kind, platform, window 1/3/7/30 days) re-query `/api/rt/stats`. KPI strip, volume per day (stacked news/social), top platforms and sources, topics with `Unclassified` shown apart, language mix, propaganda technique mix and p_prop distribution (analysed items only; empty state until FNP has analysed any), analysis coverage, schedule, **Source freshness** and sync. Every card states its denominator; "likely fake or misleading" shows `n/a` with an explanation when no item has a verdict. Bars drill into Browse.
- **Source freshness** (from `/api/rt/status` `source_freshness`, whole store, ignores the scope filters). One sortable row per source table: platform (table name beneath), newest item, a bar for how recent it is within 30 days, rows and share of the store. A row says "No new rows at the source since <date>" only when the sync is healthy (last successful sync within 30 minutes and no error in the latest sync) and the newest item is older than 48 hours, so the database itself has nothing newer; it is never shown while the sync is unhealthy. A row says "Capped at N rows: older posts in the window are not stored" when its row count reached the per-table cap (`FNP_RT_SOCIAL_LIMIT`, default 100,000; the news cap for news tables).
- **Browse.** Desk-style list | case file: full-text search, kind chips, platform/analysed/sort selects, More (source with a type-ahead, topic, language, dates), infinite scroll (`10,000+` when the total is capped; the API serves at most 10,000 results deep), `j`/`k`/Enter/Escape, push navigation below 960px. At 560 px and below the kind chips and the three selects sit behind a **Filters** button so the search box stays visible and the header is about 65 px instead of about 250 px. The **source type-ahead** calls `/api/rt/sources` (the 20 largest sources when empty, otherwise names containing the typed text; arrow keys, Enter and click). The case file badges every field **From the database** or **Inferred by FNP**, shows "Not analysed yet" with an Analyse button that adds the item to the Workbench selection and opens it, shows a probability bar for each of the 14 techniques (highest span confidence, not a calibrated probability) next to the techniques list for analysed items, highlights FNP technique spans (located by their text), and compares with near-duplicates and the nearest Fake News Database fact-checks.
- **Untrusted text.** Titles, text, source names, technique names and span text are database or pipeline data: they only reach the page through `Labels.esc`.
- Platform display names live in `Labels.platform` (`reddit` / `reddit_posts` -> Reddit, `twitter` and `x` -> X (Twitter), `news` -> News, ...). `twitter_posts` and `x_posts` are two different source tables that read the same; `Labels.platformTable` gives the table name, which is shown in a tooltip ("Source table: x_posts") and as secondary text on the Source freshness card.
- **Tour and Lab layout.** The tour card is placed by `FNPPure.placeCard` in the viewport corner where it overlaps the highlighted element least (the Workbench step highlights the Find column; on phones it docks at the top or bottom). In the Propaganda Lab with a country open, the map keeps at least 560 px from 1440 px up (narrower list and country columns); between 1280 and 1439 px the country file covers the map and Close returns to it.
- **Pure helpers.** `web/pure.js` holds the DOM-free helpers (escaping, number and time formatting, query-string building, the Workbench selection, the freshness sort and the tour card placement). It is loaded before the other scripts and exported with `module.exports` so node can test it (see Tests).

## Workbench (`#evidence/workbench`): `web/workbench.js`

Three areas. **Find items** searches `/api/rt/items` (keyword, kind, platform, topic, language, analysed state, sort) with an Add button per row and "Add all shown (max 25)". **Selection and runs** holds the selection (at most 25, remove or clear; saved under a name in `localStorage` key `fnp-wb-selections`, inside try/catch, with an in-memory fallback), "Run Claim Check on selection" (`POST /api/rt/analyse`), per-item states with elapsed timers, an overall progress bar, Cancel, and a results table (verdict, fake-or-not score, number of techniques, top technique; sortable by score or techniques). The page polls `GET /api/rt/jobs/<id>?results=0` every 2 s while the tab is visible and stops when the job is finished or the tab is hidden; full results are fetched once per finished item. **Result** shows the item with `ClaimCheck.renderResult` (step 1 verdict-class bars, step 2 with all 14 technique bars and highlighted spans, always both) and a badge: *Run by FNP now*, *cached result* or *Inferred by FNP (batch)*, plus a "From the database" fields block.

Tools: "Compare with the Fake News Database" (nearest matches with similarity, linking to `#desk/<id>`, near-duplicate count; a compact table for the whole selection, one `GET /api/rt/item/<id>` per item), "Insights for selection", and "Copy results as JSON" (`navigator.clipboard.writeText` in the click handler; if that is blocked, as it is on plain http, a visible select-all textarea; no file is downloaded).

Honest limits: the engine takes about 15 to 40 s per item and the server runs items one at a time (one job at a time, up to 5 queued jobs, 25 items per job, 6 new jobs per minute per IP). A stored (cached) result comes back instantly; a *degraded* result (the engine's 55 s timeout) is shown with a warning and never stored; busy, unavailable and skipped items (under 20 characters of text) say why and can be retried. Item text is untrusted and always escaped.

## Insights (`#evidence/insights`)

`POST /api/rt/insights` for a scope: the Browse filters in use, the Overview scope or the Workbench selection. The page shows the server's `scope_line`, the template insights (sentences written from counts only, each with a denominator) with citation chips (a fact chip opens a popover with the number and its denominator; a row chip opens the item's case file), a "Numbers used" table and the sample rows. "Ask Gemma to phrase this" (optional question, 300 characters) sends `phrase: true`; the server validates the phrasing (every cited row must be in the data, every number in the facts) and the page shows either the phrased text with its citations or the refusal reason in a calm notice while keeping the template insights. When the phrasing service is unreachable the button is disabled with the reason. No chart libraries and no outside knowledge are used.

Environment: `FNP_INSIGHT_URL` (default `http://127.0.0.1:8879/api/insight_v1`) is the phrasing service; without it the template insights still work. `FNP_ENGINE_URL` selects the Claim Check engine used by Workbench runs. The `/api/rt/*` gate (`FNP_AUTH_USER` and `FNP_AUTH_PASS`, or a deliberate `FNP_RT_ALLOW_OPEN=1`) applies to all of the above.

## Evidence Base API (phase E2): `/api/rt/*`

All GET, strict JSON (`allow_nan=False`), `Cache-Control: no-store`, errors as `{"error": ...}`. They read only the local SQLite cache and the FNP enrichment sidecar; they never connect to the Realtime Database. Every connection is `query_only`, every value is validated against a whitelist (bad input is a `400`), all SQL is parameterised, full-text input is reduced to quoted words (operators are just words) and each request has a statement time budget (`503 query too slow`).

**Gate.** Every `/api/rt/*` route except `/api/rt/gate` is refused with `403 {"error":"login_required","hint":"Set FNP_AUTH_USER and FNP_AUTH_PASS to use the Evidence Base"}` unless HTTP Basic login is configured (then the site-wide `401` login applies first) or `FNP_RT_ALLOW_OPEN=1` is set deliberately (any other value, including `true`, does not count). `GET /api/rt/gate` returns `{"open": bool, "auth": bool}` and no data. Per-IP rate limit 120 requests/min on `/api/rt/*` (`429` + `Retry-After`).

- `GET /api/rt/schedule`: scheduled deep analysis status and the last 10 cycles (see Scheduled deep analysis).
- `GET /api/rt/sources?q=&kind=&platform=&limit=20`: distinct sources whose name contains `q` (case-insensitive, `%` and `_` literal), with item counts, most items first. `q` at most 80 characters, `limit` 1..50, `kind` news|social, `platform` lowercase code; bad input is a `400`; parameterised SQL; same gate and rate limit; cached 30 s.
- `GET /api/rt/status`: store status (rows per source, newest item, last sync, lag, recent errors), `source_freshness` (`rows[{source_table, kind, platform, rows, newest_published_at, age_hours, share, stale, capped, cap}]`, `sync_healthy`, `stale_after_hours`), the sync policy in force, background-thread info and enrichment status with coverage.
- `GET /api/rt/stats?kind=news|social&platform=&days=0..365`: `total`, `by_kind`, `by_platform`, `by_source` (distinct + top 15), `by_topic` (top 15 with names + the `Unclassified` bucket + `classified`), `by_language` (distinct + top 15), `by_day` (default last 30 days), `coverage` (`items_in_scope`, `analysed`, `pct`, per kind, `with_verdict`, `verdicts`), `likely_fake_misleading` (`count`, `denominator` = items WITH a verdict, `share`, `pct`; null share when the denominator is 0), `p_prop_distribution` (10 buckets over analysed items with p_prop, with `n`), `technique_mix` (over analysed items, with the denominator), `freshness` (`newest_item_at`, `lag_minutes`, `synced_at`), `enrichment`. Cached 30 s.
- `GET /api/rt/items?q=&kind=&platform=&source=&topic=&lang=&from=&to=&analysed=yes|no&sort=recent|p_fake|p_prop|relevance&page=&size=` (size 1..50, default 25; at most 10,000 results deep). `q` is full-text over title and excerpt (best match first by default; its `total` stops at 10,000 with `total_capped: true`). `topic` is a database topic id or `unclassified`. `from`/`to` take `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SSZ`. Sorting by `p_fake`/`p_prop` puts items without a value last. Each item carries the database fields (`id, kind, platform, source_table, source_id, url, title, snippet, source, lang, published_at, topic{id,name}, category, subtopic_id, cluster_id, is_duplicate, engagement, synced_at`) and, separately, `inferred` (null when FNP has not analysed it): `p_prop, p_fake, p_misleading, verdict, model_version, built_at, source, techniques[]`.
- `GET /api/rt/item/<id>`: the full row (`text_excerpt`), the full `inferred` block (adds `countries`, `nearest_factcheck`, `spans`), `near_duplicates` (`count`, up to 5 `sample`; simhash within 3 bits, found by an indexless SQL band scan so no large in-memory index is held), `nearest_factchecks` (the Fake News Database TF-IDF matches with similarity, same thresholds as Claim Check step 1), `analyses` (stored Workbench results, if any). `404` for unknown or malformed ids.
- `GET /healthz` gains `realtime: {enabled, running, last_sync, lag_minutes, rows, errors, analysis:{enabled, last_run, next_run_at, analysed_deep}}`; computed from the local store only and never fails the probe.

**Enrichment (`rt_enrich.py`).** Reads the REAL FNP export, `reports/realtime/v1/{news,social}/part_NNNN.jsonl` + `meta.json` (format in `docs/REALTIME_CONTRACT.md`; `FNP_RT_EXPORT_DIR` overrides the directory; `v1.prev` and `*.tmp` are never read), and `reports/triage/v1/items.jsonl` (`FNP_RT_TRIAGE_DIR`), read-only. A missing directory is normal and means "not analysed". The parts are discovered by directory listing (hundreds are fine) and streamed line by line into a compact sidecar `data/rt_enrich.sqlite` (`FNP_RT_ENRICH_DB`, mode 600, about 0.8 KB per item): `p_prop`, `too_short`, the 14 technique probabilities as ONE JSON array per item, strong techniques, countries, span OFFSETS (at most the 12 most confident), the nearest fact-check (card text deduplicated in `enr_fc`). Span text is sliced lazily from the local `items` table (`title + "\n" + text_excerpt`) only when one item is opened; spans outside the string or straddling the title/excerpt boundary are dropped and counted. The sidecar is rebuilt (temp file + `os.replace`, in a background thread under the API) when `meta.json` `built_at` changes (one stat per request); a 630k-row export loads in about 50 s with a few MB of heap. Rows that fail validation are counted and listed by reason in the load report (`/api/rt/stats` -> `coverage.batch.load`, `/api/rt/status` -> `enrichment.load_report`), never fatal; `rows + rejected` should equal `meta.n_items_total`. A part that disappears mid-read (the export is swapped atomically) restarts the load once from a fresh listing; if parts are still missing the rows read so far are served and the report says `partial` (retried after 2 minutes). The older proposal shape (`meta.files` with sha256, `techniques` lists, span dicts with text) and the triage export are still read. Triage ids are sha1 hashes, not our ids, so triage records are joined on exact URL equality when the sidecar is built; a realtime-export record wins over a triage one.

**FNP batch block of `/api/rt/stats`.** `coverage.batch` carries FNP's own meta numbers next to the in-cache join: `scored` (meta `n_items_total`), `n_in_window`, `n_unscored_in_window`, `pct_of_window`, `built_at`, `next_run_at`, `interval_hours`, `n_scored_this_run`, `n_failed_this_run`, `caveat` (verbatim), `model_version`, `too_short`, and `estimated_runs` (null unless meta gives `max_items_per_run`; then `ceil(unscored / min(n_scored_this_run, limit))`). `inferred.batch` of `/api/rt/item/<id>` adds `technique_probs{basis:'encoder', basis_note, classes[14] sorted desc}`, `strong_techniques`, `spans[{start,end,technique,prob,text,field}]`, `countries[{iso3,name,weight,methods,terms}]` and `nearest_factcheck{similarity, rerank, label, claim, url, desk_id}`; `desk_id` is the Fake News Database item whose normalised URL equals the fact-check URL (map built once per index load in `server.py`, `desk_url_map`), so the UI opens `#desk/<id>`.

## Triage tab (`#triage`): `rt_triage.py`, `web/triage.js`

A ranked "what to read first" list with a source switch: **FNP triage export** (`reports/triage/v1/{meta.json,items.jsonl}`, contract `docs/TRIAGE_CONTRACT.md`, read-only; `FNP_RT_TRIAGE_DIR` overrides the directory) and **Realtime Database queue** (stored items ranked by FNP's batch signals and deep verdicts). Deep link `#triage/<fnp|realtime>/<id>`. The KPI strip, the banner "No item has been validated by a person yet" (shown while the validated count is 0) and "How the tiers are decided" come from `/api/triage/meta`; tiers are a ranking aid, never verdicts.

**API** (all GET, strict JSON, `no-store`, same gate and per-IP rate limit as `/api/rt/*`: `403 login_required` unless login is configured or `FNP_RT_ALLOW_OPEN=1`; errors `{"error": ...}`; every parameter is whitelisted and bound as a SQL parameter):
- `GET /api/triage/meta`: both sources' meta (`built_at`, tier counts, `validated`, rules, caveats, FNP `next_run_at`, queue and unscored counts). For `fnp`, `counts_match` says whether the loaded tier counts equal the export's own `meta.json`.
- `GET /api/triage/items?source=fnp|realtime&tier=high|medium|low&platform=&lang=&country=ISO3&technique=<one of the 14>&days=&q=&sort=priority|recent&page=&size=` (size 1..50, at most 10,000 results deep). The response carries `total` for the filters and `tiers` (counts under the other filters, ignoring `tier`, for the chips). For `fnp`, `days` counts back from the newest post in the export.
- `GET /api/triage/item/<id>[?source=]`: the case file record plus a `triage` block (tier, the rule that applied, plain-words `why` with the numbers). A 40-character hex id defaults to `fnp`, anything else to `realtime` (the same record as `/api/rt/item/<id>`, so the Evidence Base case file is reused). Unknown ids are `404`.
- `GET /api/triage/facets?source=`: platform, language, country and technique options with counts (cached 5 minutes).

**FNP source.** Streamed into memory (about 7.6k rows, 0.3 s), re-read when `meta.json` or `items.jsonl` changes (a rebuild swaps directories: the last good copy keeps being served and the problem is stated); a missing directory is an empty state, not an error. Bad lines are counted, strict JSON is enforced.

**Realtime queue tier rules** (assumptions from the S3 plan, not validated; constants at the top of `rt_triage.py`): **HIGH** = the nearest fact-check has a rerank score (FNP reranks only at cosine >= 0.72), or at least 2 strong propaganda spans and `p_prop` >= 0.8 (`N_STRONG_HIGH`, `P_PROP_HIGH`; tightened from 1 span and 0.6, which gave HIGH to 32% of the queue; assumed, not validated, `p_prop` is an uncalibrated maximum span probability); a deep verdict with fake-or-misleading probability < 0.3 lowers it to MEDIUM. **MEDIUM** = `p_prop` >= 0.4, or fact-check cosine >= 0.6, or deep probability >= 0.5. **LOW** = analysed but below those. Not in the queue: unscored items and items too short for the encoder (both counted), and rows that only came from the older triage export (they are shown by the FNP source). Order inside a tier: freshness band first (under 6 h, 12 h, 1, 2, 3, 7, 14, 30, 60 and 90 days, then older; fresher first, band edges assumed), then deep probability, rerank, cosine, `p_prop`, engagement, newest. Both sources use it, so an old post with a strong score no longer outranks today's items; `sort=recent` is plain newest first. For the FNP export (a fixed snapshot) the bands are measured from its newest post, for the realtime queue from now. The rules live once in Python (`rt_tier`, used for the explanations) and once as a SQL `CASE` (`TIER_SQL`, the queue; nothing is loaded into Python memory); a test keeps them identical on a boundary grid.

**UI.** Same viewport-high two-pane layout as Evidence Base > Browse (list | case file), `j`/`k`/Enter/Esc, tier chips with counts, platform, window, sort and More (language, country, technique). The realtime case file is the Evidence Base case file with a "Why this was flagged" section first; the FNP one adds the verdict, the 14 technique bars (span-confidence basis, uncalibrated), highlighted spans, countries and the nearest fact-check (a Desk link when its URL is in the Fake News Database). "Analyse in the Workbench" only adds the item to the Workbench selection: this tab never calls the engine.

## Scheduled deep analysis (`rt_scheduler.py`)

A precompute of the deep FNP analysis (Claim Check via `check_v1`) on the realtime items, shown in the Evidence Base. It is an in-process daemon thread like the sync thread: **off unless `FNP_RT_ANALYSE=1`**, nothing outside the server process (no systemd, no cron), it reads only the local SQLite store and calls the FNP engine, never the Realtime Database. To enable it add `FNP_RT_ANALYSE=1` to the start command (see the restart recipe); stop it by restarting without the flag.

- **When.** First cycle `FNP_RT_ANALYSE_FIRST_DELAY` (600 s) after start, and never before the realtime sync has completed a cycle in this process or the store holds an item newer than 24 h; then every `FNP_RT_ANALYSE_INTERVAL` (3 h). One cycle at a time (lock); an exception in one item or one cycle is caught and logged by type only.
- **What it picks** (at most `FNP_RT_ANALYSE_BUDGET`, incremental, never a bulk run): items with at least 40 characters of title + excerpt, not flagged `is_duplicate`, inside the sync window, with no usable stored result (any engine version). Items the export flags `too_short` are skipped. Priority = +3 when the FNP export found a fact-check (`nearest_factcheck.rerank` present, cosine >= 0.72), else +2 when it has `n_strong >= 1` and `p_prop >= 0.6`, + the batch `p_prop`, + recency (newer first, 0 to 1), + a small `log10(engagement)` bonus (max 0.5); with no export rows for the candidates it is recency and engagement only. Items that already have a partial result are not "fresh": they are re-tried (see below). At least 30 % of the budget goes to each kind (news, social) when available. A near-duplicate (simhash within 3 bits) of an item that already has a deep result is not run again: it gets a copy marked `dup_of` (counted as `cached_dupes`; a Workbench run on it still re-checks).
- **How it runs.** Strictly sequential through the Workbench path (`rt_jobs.Jobs.analyse_one`: Claim Check, the engine lock, the `analyses` cache), so at most one engine call is in flight from it (the engine allows 2 in total; Gemma is shared). **Stop rule:** a busy reply waits for the engine's Retry-After once (capped at 60 s) and that item is tried once more, then the cycle moves on; the cycle ends when >= 70 % of the last 10 replies were busy, degraded or failed (at least 6 replies seen), when the engine is unreachable, or at the time cap `FNP_RT_ANALYSE_MAX_SECS`. **Adaptive pacing:** when the median latency of the last 5 calls exceeds 40 s the remaining budget is halved (at most once per 5 calls). The stop reason is recorded in plain words.
- **Partial results.** A degraded reply (`check_v1` `degraded: true`, i.e. timeout or `gemma_error`) is kept as a **partial** row in `analyses`: the propaganda classes, techniques and spans, no verdict (`engine.verdict` and `step1.label` are null), `partial: true`, `partial_info: {retries, last_try_at, first_at, reason}`. It is shown as "Analysed (partial: propaganda only)", counted apart (`coverage.deep.partial_n`, `coverage.partial_n`), never in the likely fake/misleading share, and never a cache hit (a Workbench run on it goes to the engine). Later cycles re-queue partial items for a verdict, after the fresh items: at most 3 retries, at least one interval (3 h) apart, at most 25 % of the budget, newest first. A reply with no extracted claim (`degraded_why: no_claims`, `fake_call.basis: none`) is a complete deterministic result (INCONCLUSIVE), stored and cached, not partial. Workbench jobs never store partials.
- **Log.** One row per cycle in the SQLite table `schedule_log` (started_at, ended_at, picked, analysed, cached_dupes, degraded, errors, skipped, stop_reason, next_run_at, partial, retried, median_secs; older stores get the last three by a guarded `ALTER TABLE ADD COLUMN`, old rows are never rewritten). Counts and item ids only: item text is never logged. A row left open by a killed process is closed as `interrupted` at the next start.
- **API.** `GET /api/rt/schedule` (same gate and rate limit as the other `/api/rt/*` routes): `{enabled, running_now, state (off|waiting_first_delay|waiting_for_sync|idle|running), interval_s, budget, max_secs, first_delay_s, last_run, next_run_at, totals:{analysed_deep, partial_items, cycles, scheduled_analysed}, rolling:{cycles, analysed, degraded, errors, replies, success_rate} (last 10 cycles), log:[last 10 cycles]}`. `/healthz` `realtime.analysis` = `{enabled, last_run, next_run_at, analysed_deep, partial_items, rolling}`; it never fails the probe. There is deliberately **no "run now" button or endpoint**: there is no login on this site and nobody should be able to load the shared engine from the page.
- **Two kinds of analysis, never double counted.** *Batch cheap pass* = the FNP export (`rt_enrich`, fast, shallow, partial until the FNP timer has scored the whole window: see `coverage.batch.n_unscored_in_window`); *deep verdict* = a stored Claim Check result (scheduled or Workbench). `GET /api/rt/stats` reports `coverage.batch` and `coverage.deep` as separate numbers with their denominators (`coverage.analysed` = either, counted once; `both`, `deep_only`). `analysed=yes|no` on `/api/rt/items` means any source, partial included. `inferred` carries `source` (`deep` when a deep result exists, else `batch`, else `partial`), `sources` (every source present), `partial{state, created_at, retries, last_try_at, reason, max_retries, note}` (only when there is no deep verdict), `batch{...}` and `deep{verdict, p_fake, p_misleading, p_fake_misleading, created_at, engine_version, dup_of}`; the top-level `verdict`/`p_fake`/`p_misleading` are the deep ones when both exist, `p_prop` is always the batch one. The likely fake/misleading share is over items that have a verdict from either source, the deep one winning (deep: fake_call P(FAKE)+P(MISLEADING) >= 0.5; batch: a FAKE or MISLEADING verdict), with the denominator and a per-source split. The technique mix uses the batch technique list where an item has a batch record, otherwise the deep result's technique classes; the p_prop distribution is batch only (a deep result has no document-level p_prop).
- **UI.** The banner says `Deep analysis: last run <ago> (N analysed, P partial, D degraded, E errors), next run in <time>` and `FNP batch: last export <ago>` (a calm "scheduled analysis is off" otherwise); Overview shows both coverage numbers and a Schedule card (per cycle: analysed / partial / degraded / errors, median seconds, stop reason in plain words, rolling engine success rate, next run); Browse rows say Analysed (deep), Analysed (batch), Analysed (partial: propaganda only) or Not analysed yet; the case file labels each inferred block ("Inferred by FNP: deep verdict, run <time>" / "batch cheap pass").
- **Limits.** The engine takes 15 to 40 s per item, so a 40-item cycle is about 10 to 25 minutes; the schedule state is in memory (a restart restarts the first-delay countdown, the log persists); with the default budget at most about 320 items a day get a deep verdict, newest and most suspicious first.

## Tests

Offline, no network, no real flag is ever sent (the proxy is pointed at a closed port or an in-process stub on an ephemeral port; a guard refuses `:8879`). From the repository root:

    python3 -m unittest discover -s Interface/tests          # works with the stdlib alone (552 tests)
    .venv/bin/python -m pytest Interface/tests -q -p no:cacheprovider    # same tests under pytest

Covers `fnlib` (including `clean_json`), the `build_index` helpers and a full `main()` on a tiny fixture, and the real request handler on `127.0.0.1:<ephemeral>` with a temp `build/` and Propaganda Lab directory: `/healthz`, `/api/*`, static files and path traversal, ISO3 validation, flag validation and `502` when upstream is down, and auth off/on including the throttle and no-credentials-in-logs. `tests/js/*.test.js` test the pure front-end helpers (`web/pure.js`) and `labels.js` under node with no DOM; `tests/test_js_helpers.py` runs them (and `node --check` on every `web/*.js`) and skips cleanly when `node` is not installed.

## Known limits

- `/api/check` is lexical TF-IDF only (no semantic embeddings): paraphrases in other wording or languages may not match.
- No login unless `FNP_AUTH_USER` and `FNP_AUTH_PASS` are set, and then only HTTP Basic over plain HTTP (no TLS, no sessions, one shared account).
- Supervision: the live server is the systemd user service `fnp-interface` (see Running as a service); a hand-started server has no supervisor.
- Single `ThreadingHTTPServer` process with no socket timeouts; rate limits and the login throttle are in memory and reset on restart; the client IP is the socket peer (no `X-Forwarded-For`).
- The whole build (about 190 MB of JSON) is held in memory.
- `db_factchecks` verdicts are keyword-inferred and noisy. Date filters use the publication-date prefix.
- Engine rows whose `probs` is missing or unparsable currently score 0.0 instead of being skipped (`build_index.load_engine`).
- Language tags: sources sometimes tag Indic-script text as `en` or leave it empty, so `build_index.script_lang` repairs a tag that is `en*`, `und` or empty when more than 50% of the letters of title + claim are in one script (Devanagari -> hi, Bengali -> bn, Gujarati -> gu, Tamil -> ta, Telugu -> te, Kannada -> kn, Arabic script -> ur). Devanagari is also used by Marathi and Nepali and Bengali by Assamese, so those are labelled hi and bn; other tags are left alone. The 2026-11-12 rebuild changed 368 of 37,888 tags (341 en -> hi, 25 en-US -> hi, 2 en-US -> bn).
