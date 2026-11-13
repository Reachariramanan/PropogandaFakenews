-- Duplicate keys, exclusion reasons, selection provenance, gold items and adjudication assignments.
CREATE SCHEMA IF NOT EXISTS annot;
SET search_path = annot;

-- body_key: near-duplicate key of the BODY text (title excluded), used to drop reposts at ingest and in the pool.
-- (item.dup_key stays the train/dev/test split group.)
ALTER TABLE item ADD COLUMN IF NOT EXISTS body_key char(40);
CREATE INDEX IF NOT EXISTS item_body_key_idx ON item (body_key) WHERE body_key IS NOT NULL;
ALTER TABLE item ADD COLUMN IF NOT EXISTS excluded_reason text;   -- near_duplicate_of:<id> | low_substance | skipped_by_<n>_annotators | ...

-- how an assignment came about. NULL selected_by = built before this migration (provenance unknown).
ALTER TABLE assignment ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'normal'
    CHECK (kind IN ('normal','gold','adjudication'));
ALTER TABLE assignment ADD COLUMN IF NOT EXISTS selected_by text
    CHECK (selected_by IN ('steered','random','gold','adjudication'));
ALTER TABLE assignment ADD COLUMN IF NOT EXISTS sel_score real;      -- class-deficit score at selection time
ALTER TABLE assignment ADD COLUMN IF NOT EXISTS sel_meta jsonb;      -- predicted labels the score was based on, and the cell

-- Reference labels for calibration: an adjudicated item whose labels are frozen as a snapshot. Gold items are served to
-- annotators blind, inside their normal batches; their labels are scored against `ref` and never enter exports.
CREATE TABLE IF NOT EXISTS gold_item (
    item_id     bigint PRIMARY KEY REFERENCES item ON DELETE CASCADE,
    ref         jsonb NOT NULL,                   -- {"claim": {"outcome": .., "spans": [..]}, "propaganda": {...}}
    created_by  int REFERENCES annotator,
    created_at  timestamptz NOT NULL DEFAULT now(),
    active      boolean NOT NULL DEFAULT true,
    note        text
);
CREATE INDEX IF NOT EXISTS assignment_kind_idx ON assignment (kind) WHERE kind <> 'normal';
