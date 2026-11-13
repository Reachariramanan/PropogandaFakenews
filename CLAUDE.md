# FakeNewsPropoganda (FNP): hybrid LLM + metrics verdict system

The system classifies claims, posts and articles into **FAKE / PROPAGANDA / LEGIT / MISLEADING / INCONCLUSIVE**, with a "plausibly fake" tier.
- **Design notes:** `methodologoes/`.
- **Plan:** `docs/ARCHITECTURE.md`.
- **Maths and assumptions:** `docs/METHODS.md`.
- **Open problems for the owner to review:** `docs/review/`.

## Run
- `.venv/bin/python` (created with `--system-site-packages`, so torch/transformers come from `~/.local`).
- Tests: `.venv/bin/python -m pytest tests -q`. They are offline: Gemma is mocked and the maths is checked on synthetic data.
- Scripts in `scripts/` are numbered stages in run order. They run as a dry-run pilot by default and write only with `--apply`.

## Rules
- **Readers.** Only the L1 claim extractor, the L2 propaganda tagger and the L3 stance step read raw text. Everything downstream works on structured records (`fnp/schema.py`). The Gemma adjudicator never sees the raw article.
- **No double counting.** Every evidence unit carries `independence` from `fnp.math.independence` before any fusion.
- **Leakage.** For BOOM fact-checks, use only the `claim` field as input; the body is the debunk and leaks the label. When evaluating, retrieval must exclude the item's own KB entry (leave-one-out).
- **Constants.** Every tunable constant lives in `fnp/config.py:CONSTANTS` with its provenance, and the assumed ones are also listed in `docs/review/02_ASSUMPTIONS_REGISTER.md`.
- **EDA is read-only.** Use `fnp/eda_bridge.py` (parquet first, a DuckDB snapshot if the database is locked). Never write to EDA paths or to Postgres.
- **Gemma.** The endpoint is shared with the EDA session; keep `GEMMA_MAX_CONCURRENCY` ≤ 12. Responses are cached by sha1 of (prompt version, input) in `data/fnp.duckdb:gemma_cache`.
- **GPU.** Only about 7 GB is free (sglang Gemma holds about 40 GB). Laya training uses the multilingual 322M checkpoint, bf16, gradient checkpointing and small batches.
- **Storage.** Tables are versioned (`*_v1`) and never overwritten in place. Corpus-derived data stays in `data/` (gitignored).
- **Reporting.** No classifier is called SOUND without held-out or hand-judged precision with a Wilson CI.
