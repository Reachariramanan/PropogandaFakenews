-- Marks (item, model) pairs the pre-annotator already ran, including runs that produced zero suggestions.
SET search_path = annot;
CREATE TABLE suggestion_done (
    item_id        bigint NOT NULL REFERENCES item ON DELETE CASCADE,
    model_version  text   NOT NULL,
    done_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (item_id, model_version)
);
