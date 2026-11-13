"""Human annotation tool: incremental ingest from the news/social DBs, a Postgres annotation store,
a stratified annotator queue, and versioned training exports (claim extraction; propaganda spans +
techniques). The source databases are only ever read (read-only sessions)."""
