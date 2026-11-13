-- FNP annotation database, schema `annot`, v1.
-- Offsets are Unicode code-point offsets into item.text (half-open [start_char, end_char)),
-- i.e. Python str indices. item.text is immutable (trigger). label_event is append-only (trigger).

CREATE SCHEMA IF NOT EXISTS annot;
SET search_path = annot;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------- people + taxonomies
CREATE TABLE annotator (
    annotator_id  serial PRIMARY KEY,
    username      text NOT NULL UNIQUE CHECK (username ~ '^[A-Za-z0-9_.@-]{2,64}$'),
    display_name  text,
    role          text NOT NULL DEFAULT 'annotator' CHECK (role IN ('annotator','adjudicator','admin')),
    active        boolean NOT NULL DEFAULT true,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz
);

CREATE TABLE taxonomy_claim_type (
    code         text PRIMARY KEY,
    label        text NOT NULL,
    description  text NOT NULL DEFAULT '',
    sort_order   int  NOT NULL DEFAULT 0,
    active       boolean NOT NULL DEFAULT true
);

CREATE TABLE taxonomy_technique (
    code          text PRIMARY KEY,               -- one of the 14 merged PTC classes
    label         text NOT NULL,
    description   text NOT NULL DEFAULT '',
    ptc18_members text[] NOT NULL DEFAULT '{}',   -- original PTC names merged into this class
    sort_order    int  NOT NULL DEFAULT 0,
    active        boolean NOT NULL DEFAULT true
);

-- ---------------------------------------------------------------- ingestion
CREATE TABLE ingest_source (
    source_id     serial PRIMARY KEY,
    kind          text NOT NULL CHECK (kind IN ('news','social')),
    platform      text NOT NULL,                  -- outlet group for news, platform for social
    source_db     text NOT NULL,                  -- shaktidb_news | shaktidb_social
    source_table  text NOT NULL,                  -- schema-qualified
    enabled       boolean NOT NULL DEFAULT true,
    weight        double precision NOT NULL DEFAULT 1.0,   -- relative share in the stratified queue
    UNIQUE (kind, platform, source_table)
);

-- Newest-first watermark on the SOURCE's ingestion time (scraped_at), keyset (ts, pk) so ties are safe.
-- Late-arriving rows are caught by re-reading `lookback` behind the watermark (dedup by item unique key).
CREATE TABLE ingest_watermark (
    source_id        int PRIMARY KEY REFERENCES ingest_source ON DELETE CASCADE,
    newest_ts        timestamp,
    newest_pk        text,
    lookback         interval NOT NULL DEFAULT interval '48 hours',
    last_run_at      timestamptz,
    rows_seen        bigint NOT NULL DEFAULT 0
);

CREATE TABLE ingest_run (
    run_id       bigserial PRIMARY KEY,
    source_id    int NOT NULL REFERENCES ingest_source,
    mode         text NOT NULL CHECK (mode IN ('newest','backfill')),
    started_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    n_fetched    int NOT NULL DEFAULT 0,
    n_filtered   int NOT NULL DEFAULT 0,
    n_dup        int NOT NULL DEFAULT 0,
    n_inserted   int NOT NULL DEFAULT 0,
    params       jsonb NOT NULL DEFAULT '{}',
    error        text
);

-- ---------------------------------------------------------------- items
CREATE TABLE item (
    item_id       bigserial PRIMARY KEY,
    source_id     int NOT NULL REFERENCES ingest_source,
    source_pk     text NOT NULL,                  -- primary key of the row in the source table
    ingest_mode   text NOT NULL CHECK (ingest_mode IN ('newest','backfill')),
    run_id        bigint REFERENCES ingest_run,
    url           text,
    author        text,
    outlet        text,                           -- news outlet / channel / subreddit when known
    published_at  timestamp,
    scraped_at    timestamp,
    fetched_at    timestamptz NOT NULL DEFAULT now(),
    lang          text,
    lang_group    text NOT NULL CHECK (lang_group IN ('en','hi','other')),
    period        text NOT NULL,                  -- 'YYYY-MM' of published_at (or 'unknown')
    title         text,
    title_len     int NOT NULL DEFAULT 0,         -- text = title + E'\n\n' + body when title_len > 0
    text          text NOT NULL,                  -- IMMUTABLE; all span offsets index into this
    text_sha256   char(64) NOT NULL,
    norm_hash     char(40) NOT NULL,              -- sha1 of normalised text: exact-dup guard
    dup_key       char(40) NOT NULL,              -- near-dup key: split unit (no leakage across splits)
    is_double     boolean NOT NULL DEFAULT false, -- needs two independent annotators
    is_blind      boolean NOT NULL DEFAULT false, -- suggestions withheld (anchoring-bias control)
    status        text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','in_progress','done','excluded')),
    meta          jsonb NOT NULL DEFAULT '{}',
    UNIQUE (source_id, source_pk),
    UNIQUE (norm_hash)
);
CREATE INDEX item_strat_idx ON item (source_id, lang_group, period) WHERE status <> 'excluded';
CREATE INDEX item_dup_key_idx ON item (dup_key);
CREATE INDEX item_status_idx ON item (status);

CREATE FUNCTION forbid_item_text_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.text IS DISTINCT FROM OLD.text OR NEW.title IS DISTINCT FROM OLD.title
       OR NEW.title_len <> OLD.title_len OR NEW.text_sha256 <> OLD.text_sha256 THEN
        RAISE EXCEPTION 'annot.item.text is immutable (item_id=%)', OLD.item_id;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER item_text_immutable BEFORE UPDATE ON item
    FOR EACH ROW EXECUTE FUNCTION forbid_item_text_update();

-- Model pre-annotations, one row per suggested span. Never mixed with human labels.
CREATE TABLE suggestion (
    suggestion_id  bigserial PRIMARY KEY,
    item_id        bigint NOT NULL REFERENCES item ON DELETE CASCADE,
    layer          text NOT NULL CHECK (layer IN ('claim','propaganda')),
    model_version  text NOT NULL,
    start_char     int NOT NULL,
    end_char       int NOT NULL,
    span_text      text NOT NULL,
    label          text,                          -- suggested claim_type code or technique code
    confidence     double precision,
    payload        jsonb NOT NULL DEFAULT '{}',   -- e.g. normalised claim, speaker, check_worthy score
    created_at     timestamptz NOT NULL DEFAULT now(),
    CHECK (end_char > start_char)
);
CREATE INDEX suggestion_item_idx ON suggestion (item_id, layer);

-- ---------------------------------------------------------------- assignments (stratified queue is computed on demand)
CREATE TABLE assignment (
    assignment_id  bigserial PRIMARY KEY,
    item_id        bigint NOT NULL REFERENCES item ON DELETE CASCADE,
    annotator_id   int NOT NULL REFERENCES annotator,
    status         text NOT NULL DEFAULT 'assigned' CHECK (status IN ('assigned','in_progress','done','skipped')),
    skip_reason    text,
    assigned_at    timestamptz NOT NULL DEFAULT now(),
    leased_until   timestamptz,
    finished_at    timestamptz,
    active_ms      bigint NOT NULL DEFAULT 0,
    UNIQUE (item_id, annotator_id)
);
CREATE INDEX assignment_annotator_idx ON assignment (annotator_id, status);

-- Per item x annotator x layer outcome. 'none' is the explicit "no claims / no propaganda" label.
CREATE TABLE layer_result (
    result_id     bigserial PRIMARY KEY,
    item_id       bigint NOT NULL REFERENCES item ON DELETE CASCADE,
    annotator_id  int NOT NULL REFERENCES annotator,
    layer         text NOT NULL CHECK (layer IN ('claim','propaganda')),
    outcome       text NOT NULL CHECK (outcome IN ('annotated','none')),
    revision      int NOT NULL DEFAULT 1,
    submitted_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (item_id, annotator_id, layer)
);

-- ---------------------------------------------------------------- human labels (current state)
CREATE TABLE claim_span (
    span_id          bigserial PRIMARY KEY,
    item_id          bigint NOT NULL REFERENCES item ON DELETE CASCADE,
    annotator_id     int NOT NULL REFERENCES annotator,
    start_char       int NOT NULL,
    end_char         int NOT NULL,
    span_text        text NOT NULL,
    claim_type       text NOT NULL REFERENCES taxonomy_claim_type,
    check_worthy     text NOT NULL DEFAULT 'unsure' CHECK (check_worthy IN ('yes','no','unsure')),  -- separate from type
    normalised_claim text,
    speaker          text,
    origin           text NOT NULL DEFAULT 'manual' CHECK (origin IN ('manual','suggestion_accepted','suggestion_edited')),
    suggestion_id    bigint REFERENCES suggestion,
    revision         int NOT NULL DEFAULT 1,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    deleted_at       timestamptz,
    CHECK (end_char > start_char)
);
CREATE INDEX claim_span_item_idx ON claim_span (item_id, annotator_id) WHERE deleted_at IS NULL;

CREATE TABLE propaganda_span (
    span_id       bigserial PRIMARY KEY,
    item_id       bigint NOT NULL REFERENCES item ON DELETE CASCADE,
    annotator_id  int NOT NULL REFERENCES annotator,
    start_char    int NOT NULL,
    end_char      int NOT NULL,
    span_text     text NOT NULL,
    origin        text NOT NULL DEFAULT 'manual' CHECK (origin IN ('manual','suggestion_accepted','suggestion_edited')),
    suggestion_id bigint REFERENCES suggestion,
    revision      int NOT NULL DEFAULT 1,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    deleted_at    timestamptz,
    CHECK (end_char > start_char)
);
CREATE INDEX propaganda_span_item_idx ON propaganda_span (item_id, annotator_id) WHERE deleted_at IS NULL;

CREATE TABLE propaganda_span_technique (
    span_id    bigint NOT NULL REFERENCES propaganda_span ON DELETE CASCADE,
    technique  text   NOT NULL REFERENCES taxonomy_technique,
    PRIMARY KEY (span_id, technique)
);

-- ---------------------------------------------------------------- history (source of truth for provenance)
CREATE TABLE label_event (
    event_id      bigserial PRIMARY KEY,
    ts            timestamptz NOT NULL DEFAULT now(),
    item_id       bigint NOT NULL,
    annotator_id  int NOT NULL,
    layer         text CHECK (layer IN ('claim','propaganda')),
    entity_type   text NOT NULL CHECK (entity_type IN ('claim_span','propaganda_span','layer_result','suggestion','assignment')),
    entity_id     bigint,
    action        text NOT NULL CHECK (action IN ('create','edit','delete','accept_suggestion','reject_suggestion',
                                                  'submit','reopen','skip','start')),
    before        jsonb,
    after         jsonb
);
CREATE INDEX label_event_item_idx ON label_event (item_id, ts);

CREATE FUNCTION forbid_event_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'annot.label_event is append-only';
END $$;
CREATE TRIGGER label_event_append_only BEFORE UPDATE OR DELETE ON label_event
    FOR EACH ROW EXECUTE FUNCTION forbid_event_change();

-- ---------------------------------------------------------------- exports
CREATE TABLE export_run (
    export_id    bigserial PRIMARY KEY,
    version      text NOT NULL UNIQUE,            -- e.g. claims_v1, spans_v1
    kind         text NOT NULL CHECK (kind IN ('claims','propaganda')),
    created_at   timestamptz NOT NULL DEFAULT now(),
    created_by   text,
    config       jsonb NOT NULL DEFAULT '{}',
    counts       jsonb NOT NULL DEFAULT '{}',
    manifest_sha256 char(64)
);
CREATE TABLE export_split (
    export_id  bigint NOT NULL REFERENCES export_run ON DELETE CASCADE,
    dup_key    char(40) NOT NULL,
    split      text NOT NULL CHECK (split IN ('train','dev','test')),
    PRIMARY KEY (export_id, dup_key)
);
