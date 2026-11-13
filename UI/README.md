# FNP dashboard

Local, offline dashboard for the FNP pipeline (stdlib `http.server` + duckdb, no CDN).

    .venv/bin/python UI/server.py [--host 0.0.0.0] [--port 8879]
    nohup .venv/bin/python UI/server.py --port 8879 > logs/ui_server.log 2>&1 &

Default port is 8878 (EDA uses 8877). Another process may hold 8878; use `--port`.
Binds 0.0.0.0 like the EDA server. Open http://localhost:PORT/.

Tabs: Overview, Methods, Data, Runs explorer, Evaluation, Verdicts, Laya & RL, Review, Try it.

Endpoints: `/api/overview[?tests=1]`, `/api/datasets`, `/api/units`, `/api/unit/<id>`, `/api/eval`,
`/api/verdicts`, `/api/docs[?path=]`, `/api/laya`, `/api/logs`, POST `/api/predict {text}`.

Notes
- Everything is read fresh per request. The units index only appends newly written lines.
- `data/fnp.duckdb` is opened read-only; if a pipeline holds the write lock, a copy at
  `data/cache/ui_snapshot.duckdb` (refreshed when older than 5 min) is read instead.
- `/api/predict` runs `fnp.pipeline.analyse_text` live (Gemma, 15-40 s). Gemma's own cache may write
  to `data/fnp.duckdb` or its `_gemma_cache.duckdb` sibling, which is existing pipeline behaviour.
- `?tests=1` runs `pytest tests -q` (120 s timeout, cached 5 min).
