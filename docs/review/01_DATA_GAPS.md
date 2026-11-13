# 01 Data gaps and assumptions

## Label mapping (assumed, fixed)
- LIAR: pants-fire,false -> FAKE; barely-true,half-true -> MISLEADING; mostly-true,true -> LEGIT.
- X-Fact: false -> FAKE; mostly false, partly true/misleading -> MISLEADING; mostly true,true -> LEGIT; complicated/hard to categorise, other -> INCONCLUSIVE (kept, `inconclusive_flag` in meta_json).
- FakeNewsNet PolitiFact: fake -> FAKE, real -> LEGIT (titles only; single split "all").
- PTC: doc-level PROPAGANDA if >=3 spans or >=1 span/100 whitespace tokens, else label5 NULL (spans still used).
- Fact-check sites (Interface/factchecks/*.jsonl): label5 = verdict_norm; claim only is input text; body/description only in meta_json/KB evidence. 60/20/20 by sha1(id). Falls back to articles_full.json if boomlive.jsonl is absent. ids are `<site>:<id>`.
- Table `gold_items_v1` is per-dataset write-once: a site whose row count later grows is skipped with a WARN (new sites are picked up automatically); write a v2 table to refresh.

## Gaps
- PTC / SemEval-2020 T11: only the **train** split (371 articles, 6129 technique spans) from GitHub mirror vrdn-23/SemEval-2020. Dev articles exist but no dev labels are public in any mirror found; the HF `sem_eval_2020_task_11` is a loader script pointing at the organisers' form-gated download. Test not obtained.
- QProp, IFND: skipped (economy; HF IFND has images/URLs only, no text; QProp needs a form). 
- X-Fact: official ood/zeroshot splits not fetched (only train/dev/test `.all.tsv`). Malformed TSV rows are dropped (on_bad_lines=skip).
- Web search: DuckDuckGo HTML endpoint returned a bot-challenge page (HTTP 202, "anomaly") on the live smoke call with our polite UA; `web_search` therefore fails soft to []. We did not spoof a browser UA. Needs another provider/API key for real use. `fetch_article` not live-tested.
- BOOM: 21 rows without verdict dropped; only False/Misleading present, so no LEGIT from fact-checkers.
- FakeNewsNet has no official split.
