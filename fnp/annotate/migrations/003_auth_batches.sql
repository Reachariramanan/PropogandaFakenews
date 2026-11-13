-- Password login, sessions, work-time tracking and per-annotator batches.
SET search_path = annot;

ALTER TABLE annotator
    ADD COLUMN password_hash        text,            -- scrypt$n$r$p$salt_hex$hash_hex ; NULL = cannot log in
    ADD COLUMN must_change_password boolean NOT NULL DEFAULT false,
    ADD COLUMN failed_logins        int     NOT NULL DEFAULT 0,
    ADD COLUMN locked_until         timestamptz,
    ADD COLUMN last_login_at        timestamptz,
    ADD COLUMN batch_size           int CHECK (batch_size IS NULL OR batch_size BETWEEN 1 AND 5000);  -- NULL = ann_batch_size

-- One row per login. Tokens are stored only as sha256 hashes.
CREATE TABLE auth_session (
    session_id    bigserial PRIMARY KEY,
    annotator_id  int NOT NULL REFERENCES annotator,
    token_hash    char(64) NOT NULL UNIQUE,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz NOT NULL DEFAULT now(),
    expires_at    timestamptz NOT NULL,
    ended_at      timestamptz,
    end_reason    text CHECK (end_reason IN ('logout','expired','replaced','disabled','password_changed')),
    ip            text,
    user_agent    text
);
CREATE INDEX auth_session_user_idx ON auth_session (annotator_id, created_at DESC);

-- Audit trail of authentication attempts (append-only by convention; the app never updates or deletes).
CREATE TABLE login_event (
    event_id      bigserial PRIMARY KEY,
    ts            timestamptz NOT NULL DEFAULT now(),
    username      text NOT NULL,
    annotator_id  int REFERENCES annotator,
    outcome       text NOT NULL CHECK (outcome IN ('ok','bad_password','unknown_user','locked','disabled','no_password',
                                                   'logout','password_changed','password_reset','user_created')),
    ip            text,
    detail        text
);
CREATE INDEX login_event_ts_idx ON login_event (ts DESC);

-- Work time: one row per (annotator, minute) in which the annotator was actively working (heartbeat with recent
-- input, or any label write). Counting minute buckets makes several tabs / sessions impossible to double count.
CREATE TABLE activity_minute (
    annotator_id  int NOT NULL REFERENCES annotator,
    minute        timestamptz NOT NULL,
    session_id    bigint REFERENCES auth_session,
    PRIMARY KEY (annotator_id, minute)
);

-- Batches: every annotator works through successive batches of N items (N = ann_batch_size).
CREATE TABLE batch (
    batch_id      bigserial PRIMARY KEY,
    annotator_id  int NOT NULL REFERENCES annotator,
    seq           int NOT NULL,
    size          int NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    completed_at  timestamptz,
    meta          jsonb NOT NULL DEFAULT '{}',
    UNIQUE (annotator_id, seq)
);
ALTER TABLE assignment
    ADD COLUMN batch_id  bigint REFERENCES batch,
    ADD COLUMN batch_pos int;
CREATE INDEX assignment_batch_idx ON assignment (batch_id, status);
CREATE INDEX item_pool_idx ON item (lang_group, status) WHERE status IN ('queued','in_progress');
