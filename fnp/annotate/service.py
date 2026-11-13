"""Annotation logic on top of the annot schema: stratified queue, span CRUD, layer submission, history.

Invariants kept here (and checked again at export):
  * span offsets are code-point offsets into the immutable item.text, 0 <= start < end <= len(text);
    span_text is always taken from item.text server-side, never trusted from the client;
  * a layer_result 'annotated' has >= 1 live span of that layer, 'none' has 0; any later span edit
    re-opens the layer (the result row is removed and logged), so the annotator must re-submit;
  * claim and propaganda layers never share rows; check-worthiness is a separate column from claim type;
  * every mutation writes an append-only label_event with before/after snapshots.
"""
from __future__ import annotations

import json
import random
import re
from typing import Optional

from ..config import C
from . import adjudicate, batching
from .db import tx

LAYERS = ("claim", "propaganda")
CHECK_WORTHY = ("yes", "no", "unsure")


class AnnotError(ValueError):
    """Bad request from a client (maps to HTTP 400)."""


# ------------------------------------------------------------------ helpers
def _event(cur, item_id, annotator_id, layer, entity_type, entity_id, action, before=None, after=None):
    cur.execute("INSERT INTO label_event (item_id, annotator_id, layer, entity_type, entity_id, action, before, after) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (item_id, annotator_id, layer, entity_type, entity_id, action,
                 json.dumps(before, default=str) if before is not None else None,
                 json.dumps(after, default=str) if after is not None else None))


def get_or_create_annotator(username: str) -> dict:
    username = (username or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{2,64}", username):
        raise AnnotError("username must be 2-64 characters: letters, digits, _ . @ -")
    with tx() as cur:
        cur.execute("INSERT INTO annotator (username, display_name) VALUES (%s,%s) "
                    "ON CONFLICT (username) DO UPDATE SET last_seen_at = now() RETURNING *", (username, username))
        return dict(cur.fetchone())


def taxonomy() -> dict:
    with tx() as cur:
        cur.execute("SELECT code,label,description FROM taxonomy_claim_type WHERE active ORDER BY sort_order")
        ct = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT code,label,description FROM taxonomy_technique WHERE active ORDER BY sort_order")
        tq = [dict(r) for r in cur.fetchall()]
    return {"claim_types": ct, "techniques": tq, "check_worthy": list(CHECK_WORTHY)}


def _item_text(cur, item_id: int) -> dict:
    cur.execute("SELECT item_id, text, is_blind, is_double FROM item WHERE item_id=%s", (item_id,))
    r = cur.fetchone()
    if not r:
        raise AnnotError(f"unknown item {item_id}")
    return r


def _require_assignment(cur, item_id: int, annotator_id: int, writable: bool = True) -> None:
    """Annotators may only read / label items that sit in their own batch (blind items stay blind, no peeking ahead)."""
    cur.execute("SELECT status FROM assignment WHERE item_id=%s AND annotator_id=%s", (item_id, annotator_id))
    a = cur.fetchone()
    if not a:
        raise AnnotError("this item is not assigned to you")
    if writable and a["status"] == "skipped":
        raise AnnotError("this item was skipped: re-open it first")


def _validate_offsets(text: str, start, end) -> tuple[int, int]:
    if not isinstance(start, int) or not isinstance(end, int) or isinstance(start, bool) or isinstance(end, bool):
        raise AnnotError("start and end must be integers (code-point offsets)")
    if not (0 <= start < end <= len(text)):
        raise AnnotError(f"offsets out of range: need 0 <= start < end <= {len(text)}")
    if not text[start:end].strip():
        raise AnnotError("span is only whitespace")
    return start, end


def _touch(cur, item_id, annotator_id) -> None:
    """Refresh the lease and mark the assignment in progress."""
    cur.execute("UPDATE assignment SET status = CASE WHEN status='assigned' THEN 'in_progress' ELSE status END, "
                "leased_until = now() + make_interval(mins => %s) WHERE item_id=%s AND annotator_id=%s",
                (int(C("ann_lease_minutes")), item_id, annotator_id))
    cur.execute("UPDATE item SET status='in_progress' WHERE item_id=%s AND status='queued'", (item_id,))


def _reopen_layer(cur, item_id, annotator_id, layer) -> None:
    cur.execute("DELETE FROM layer_result WHERE item_id=%s AND annotator_id=%s AND layer=%s RETURNING outcome",
                (item_id, annotator_id, layer))
    old = cur.fetchone()
    if old:
        _event(cur, item_id, annotator_id, layer, "layer_result", None, "reopen", before=dict(old))
        cur.execute("UPDATE assignment SET status='in_progress', finished_at=NULL WHERE item_id=%s AND annotator_id=%s "
                    "AND status='done'", (item_id, annotator_id))
        _refresh_item_status(cur, item_id)


def _need(is_double: bool) -> int:
    return 2 if is_double else 1


def _refresh_item_status(cur, item_id) -> None:
    cur.execute("SELECT is_double FROM item WHERE item_id=%s", (item_id,))
    dbl = cur.fetchone()["is_double"]
    cur.execute("SELECT count(*) n FROM assignment WHERE item_id=%s AND status='done'", (item_id,))
    done = cur.fetchone()["n"]
    cur.execute("SELECT count(*) n FROM assignment WHERE item_id=%s AND status IN ('assigned','in_progress','done')", (item_id,))
    live = cur.fetchone()["n"]
    st = "done" if done >= _need(dbl) else "in_progress" if live else "queued"
    cur.execute("UPDATE item SET status=%s WHERE item_id=%s AND status <> 'excluded'", (st, item_id))


# ------------------------------------------------------------------ queue
def next_item(annotator_id: int, rng: Optional[random.Random] = None) -> Optional[int]:
    """The annotator's open item, else the next one of their batch, else the next batch (see batching.py)."""
    return batching.next_item(annotator_id, rng)


# ------------------------------------------------------------------ reading an item
def _claim_row(r) -> dict:
    return {"span_id": r["span_id"], "start": r["start_char"], "end": r["end_char"], "text": r["span_text"],
            "claim_type": r["claim_type"], "check_worthy": r["check_worthy"], "normalised_claim": r["normalised_claim"],
            "speaker": r["speaker"], "origin": r["origin"], "suggestion_id": r["suggestion_id"], "revision": r["revision"]}


def get_item(item_id: int, annotator_id: int) -> dict:
    with tx() as cur:
        cur.execute("""SELECT i.item_id, s.kind, s.platform, i.url, i.author, i.outlet, i.published_at, i.lang, i.lang_group,
                              i.period, i.title_len, i.text, i.is_blind, i.is_double, i.status
                       FROM item i JOIN ingest_source s USING (source_id) WHERE i.item_id=%s""", (item_id,))
        it = cur.fetchone()
        if not it:
            raise AnnotError(f"unknown item {item_id}")
        it = dict(it)
        cur.execute("SELECT status, skip_reason, kind FROM assignment WHERE item_id=%s AND annotator_id=%s", (item_id, annotator_id))
        asg = cur.fetchone()
        if not asg:
            raise AnnotError("this item is not assigned to you")
        asg = dict(asg)
        kind = asg.pop("kind")
        if kind == "gold":                      # gold items are served blind: the annotator is measured without model suggestions
            it["is_blind"] = True
        adjud = adjudicate.view(cur, item_id, annotator_id) if kind == "adjudication" else None
        cur.execute("SELECT * FROM claim_span WHERE item_id=%s AND annotator_id=%s AND deleted_at IS NULL ORDER BY start_char, span_id",
                    (item_id, annotator_id))
        claims = [_claim_row(r) for r in cur.fetchall()]
        cur.execute("""SELECT p.*, coalesce(array_agg(t.technique ORDER BY t.technique) FILTER (WHERE t.technique IS NOT NULL), '{}') techs
                       FROM propaganda_span p LEFT JOIN propaganda_span_technique t USING (span_id)
                       WHERE p.item_id=%s AND p.annotator_id=%s AND p.deleted_at IS NULL GROUP BY p.span_id ORDER BY p.start_char, p.span_id""",
                    (item_id, annotator_id))
        props = [{"span_id": r["span_id"], "start": r["start_char"], "end": r["end_char"], "text": r["span_text"],
                  "techniques": list(r["techs"]), "origin": r["origin"], "suggestion_id": r["suggestion_id"],
                  "revision": r["revision"]} for r in cur.fetchall()]
        cur.execute("SELECT layer, outcome FROM layer_result WHERE item_id=%s AND annotator_id=%s", (item_id, annotator_id))
        results = {r["layer"]: r["outcome"] for r in cur.fetchall()}
        sugg = []
        if not it["is_blind"]:
            cur.execute("""SELECT s.suggestion_id, s.layer, s.model_version, s.start_char, s.end_char, s.span_text, s.label,
                                  s.confidence, s.payload
                           FROM suggestion s WHERE s.item_id=%s
                             AND NOT EXISTS (SELECT 1 FROM label_event e WHERE e.entity_type='suggestion'
                                   AND e.entity_id=s.suggestion_id AND e.annotator_id=%s
                                   AND e.action IN ('reject_suggestion','accept_suggestion'))
                           ORDER BY s.start_char""", (item_id, annotator_id))
            sugg = [{"suggestion_id": r["suggestion_id"], "layer": r["layer"], "model": r["model_version"],
                     "start": r["start_char"], "end": r["end_char"], "text": r["span_text"], "label": r["label"],
                     "confidence": r["confidence"], "payload": r["payload"]} for r in cur.fetchall()]
    # is_blind is reported (the UI shows a neutral "no suggestions" state), is_double is not (annotators must not coordinate)
    del it["is_double"]
    out = {"item": it, "assignment": asg, "claims": claims, "propaganda": props, "results": results, "suggestions": sugg}
    if adjud is not None:                       # only adjudicators ever see the other annotators' labels
        out["adjudication"] = adjud
    return out


# ------------------------------------------------------------------ claim spans
def _check_claim_fields(cur, claim_type, check_worthy):
    cur.execute("SELECT 1 FROM taxonomy_claim_type WHERE code=%s AND active", (claim_type,))
    if not cur.fetchone():
        raise AnnotError(f"unknown claim_type {claim_type!r}")
    if check_worthy not in CHECK_WORTHY:
        raise AnnotError(f"check_worthy must be one of {CHECK_WORTHY}")


def _origin(cur, suggestion_id, layer, start, end, label_set) -> tuple[str, Optional[int]]:
    if not suggestion_id:
        return "manual", None
    cur.execute("SELECT start_char,end_char,label FROM suggestion WHERE suggestion_id=%s AND layer=%s", (suggestion_id, layer))
    s = cur.fetchone()
    if not s:
        raise AnnotError("unknown suggestion_id for this layer")
    same = s["start_char"] == start and s["end_char"] == end and (not s["label"] or s["label"] in label_set)
    return ("suggestion_accepted" if same else "suggestion_edited"), suggestion_id


def add_claim(item_id: int, annotator_id: int, start, end, claim_type, check_worthy="unsure",
              normalised_claim=None, speaker=None, suggestion_id=None) -> dict:
    with tx() as cur:
        it = _item_text(cur, item_id)
        _require_assignment(cur, item_id, annotator_id)
        start, end = _validate_offsets(it["text"], start, end)
        _check_claim_fields(cur, claim_type, check_worthy)
        origin, sid = _origin(cur, suggestion_id, "claim", start, end, {claim_type})
        cur.execute("""INSERT INTO claim_span (item_id, annotator_id, start_char, end_char, span_text, claim_type, check_worthy,
                       normalised_claim, speaker, origin, suggestion_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                    (item_id, annotator_id, start, end, it["text"][start:end], claim_type, check_worthy,
                     (normalised_claim or "").strip() or None, (speaker or "").strip() or None, origin, sid))
        row = _claim_row(cur.fetchone())
        _event(cur, item_id, annotator_id, "claim", "claim_span", row["span_id"], "create", after=row)
        if sid:
            _event(cur, item_id, annotator_id, "claim", "suggestion", sid, "accept_suggestion", after={"span_id": row["span_id"]})
        _reopen_layer(cur, item_id, annotator_id, "claim")
        _touch(cur, item_id, annotator_id)
        return row


def update_claim(span_id: int, annotator_id: int, **f) -> dict:
    with tx() as cur:
        cur.execute("SELECT * FROM claim_span WHERE span_id=%s AND annotator_id=%s AND deleted_at IS NULL FOR UPDATE", (span_id, annotator_id))
        old = cur.fetchone()
        if not old:
            raise AnnotError("no such claim span")
        it = _item_text(cur, old["item_id"])
        start = f.get("start", old["start_char"]); end = f.get("end", old["end_char"])
        start, end = _validate_offsets(it["text"], start, end)
        ct = f.get("claim_type", old["claim_type"]); cw = f.get("check_worthy", old["check_worthy"])
        _check_claim_fields(cur, ct, cw)
        nc = f["normalised_claim"] if "normalised_claim" in f else old["normalised_claim"]
        sp = f["speaker"] if "speaker" in f else old["speaker"]
        origin = old["origin"]
        if old["suggestion_id"] and origin == "suggestion_accepted":
            origin, _ = _origin(cur, old["suggestion_id"], "claim", start, end, {ct})
        cur.execute("""UPDATE claim_span SET start_char=%s, end_char=%s, span_text=%s, claim_type=%s, check_worthy=%s,
                       normalised_claim=%s, speaker=%s, origin=%s, revision=revision+1, updated_at=now()
                       WHERE span_id=%s RETURNING *""",
                    (start, end, it["text"][start:end], ct, cw, (nc or "").strip() or None, (sp or "").strip() or None,
                     origin, span_id))
        new = _claim_row(cur.fetchone())
        _event(cur, old["item_id"], annotator_id, "claim", "claim_span", span_id, "edit", before=_claim_row(old), after=new)
        _reopen_layer(cur, old["item_id"], annotator_id, "claim")
        _touch(cur, old["item_id"], annotator_id)
        return new


def delete_claim(span_id: int, annotator_id: int) -> None:
    with tx() as cur:
        cur.execute("UPDATE claim_span SET deleted_at=now() WHERE span_id=%s AND annotator_id=%s AND deleted_at IS NULL RETURNING *",
                    (span_id, annotator_id))
        old = cur.fetchone()
        if not old:
            raise AnnotError("no such claim span")
        _event(cur, old["item_id"], annotator_id, "claim", "claim_span", span_id, "delete", before=_claim_row(old))
        _reopen_layer(cur, old["item_id"], annotator_id, "claim")
        _touch(cur, old["item_id"], annotator_id)


# ------------------------------------------------------------------ propaganda spans
def _prop_row(cur, span_id) -> dict:
    cur.execute("""SELECT p.*, coalesce(array_agg(t.technique ORDER BY t.technique) FILTER (WHERE t.technique IS NOT NULL), '{}') techs
                   FROM propaganda_span p LEFT JOIN propaganda_span_technique t USING (span_id) WHERE p.span_id=%s GROUP BY p.span_id""",
                (span_id,))
    r = cur.fetchone()
    return {"span_id": r["span_id"], "start": r["start_char"], "end": r["end_char"], "text": r["span_text"],
            "techniques": list(r["techs"]), "origin": r["origin"], "suggestion_id": r["suggestion_id"], "revision": r["revision"]}


def _check_techniques(cur, techniques) -> list[str]:
    if not isinstance(techniques, (list, tuple)) or not techniques:
        raise AnnotError("pick at least one technique")
    techniques = sorted(set(techniques))
    cur.execute("SELECT code FROM taxonomy_technique WHERE active AND code = ANY(%s)", (techniques,))
    ok = {r["code"] for r in cur.fetchall()}
    bad = [t for t in techniques if t not in ok]
    if bad:
        raise AnnotError(f"unknown technique(s): {bad}")
    return techniques


def add_propaganda(item_id: int, annotator_id: int, start, end, techniques, suggestion_id=None) -> dict:
    with tx() as cur:
        it = _item_text(cur, item_id)
        _require_assignment(cur, item_id, annotator_id)
        start, end = _validate_offsets(it["text"], start, end)
        techniques = _check_techniques(cur, techniques)
        origin, sid = _origin(cur, suggestion_id, "propaganda", start, end, set(techniques))
        cur.execute("INSERT INTO propaganda_span (item_id, annotator_id, start_char, end_char, span_text, origin, suggestion_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING span_id",
                    (item_id, annotator_id, start, end, it["text"][start:end], origin, sid))
        span_id = cur.fetchone()["span_id"]
        for t in techniques:
            cur.execute("INSERT INTO propaganda_span_technique (span_id, technique) VALUES (%s,%s)", (span_id, t))
        row = _prop_row(cur, span_id)
        _event(cur, item_id, annotator_id, "propaganda", "propaganda_span", span_id, "create", after=row)
        if sid:
            _event(cur, item_id, annotator_id, "propaganda", "suggestion", sid, "accept_suggestion", after={"span_id": span_id})
        _reopen_layer(cur, item_id, annotator_id, "propaganda")
        _touch(cur, item_id, annotator_id)
        return row


def update_propaganda(span_id: int, annotator_id: int, **f) -> dict:
    with tx() as cur:
        cur.execute("SELECT * FROM propaganda_span WHERE span_id=%s AND annotator_id=%s AND deleted_at IS NULL FOR UPDATE",
                    (span_id, annotator_id))
        old = cur.fetchone()
        if not old:
            raise AnnotError("no such propaganda span")
        before = _prop_row(cur, span_id)
        it = _item_text(cur, old["item_id"])
        start, end = _validate_offsets(it["text"], f.get("start", old["start_char"]), f.get("end", old["end_char"]))
        techs = _check_techniques(cur, f["techniques"]) if "techniques" in f else before["techniques"]
        origin = old["origin"]
        if old["suggestion_id"] and origin == "suggestion_accepted":
            origin, _ = _origin(cur, old["suggestion_id"], "propaganda", start, end, set(techs))
        cur.execute("UPDATE propaganda_span SET start_char=%s, end_char=%s, span_text=%s, origin=%s, revision=revision+1, "
                    "updated_at=now() WHERE span_id=%s", (start, end, it["text"][start:end], origin, span_id))
        cur.execute("DELETE FROM propaganda_span_technique WHERE span_id=%s", (span_id,))
        for t in techs:
            cur.execute("INSERT INTO propaganda_span_technique (span_id, technique) VALUES (%s,%s)", (span_id, t))
        new = _prop_row(cur, span_id)
        _event(cur, old["item_id"], annotator_id, "propaganda", "propaganda_span", span_id, "edit", before=before, after=new)
        _reopen_layer(cur, old["item_id"], annotator_id, "propaganda")
        _touch(cur, old["item_id"], annotator_id)
        return new


def delete_propaganda(span_id: int, annotator_id: int) -> None:
    with tx() as cur:
        cur.execute("SELECT item_id FROM propaganda_span WHERE span_id=%s AND annotator_id=%s AND deleted_at IS NULL FOR UPDATE",
                    (span_id, annotator_id))
        r = cur.fetchone()
        if not r:
            raise AnnotError("no such propaganda span")
        before = _prop_row(cur, span_id)
        cur.execute("UPDATE propaganda_span SET deleted_at=now() WHERE span_id=%s", (span_id,))
        _event(cur, r["item_id"], annotator_id, "propaganda", "propaganda_span", span_id, "delete", before=before)
        _reopen_layer(cur, r["item_id"], annotator_id, "propaganda")
        _touch(cur, r["item_id"], annotator_id)


# ------------------------------------------------------------------ suggestions, submit, skip
def reject_suggestion(item_id: int, annotator_id: int, suggestion_id: int) -> None:
    with tx() as cur:
        _require_assignment(cur, item_id, annotator_id)
        cur.execute("SELECT layer FROM suggestion WHERE suggestion_id=%s AND item_id=%s", (suggestion_id, item_id))
        s = cur.fetchone()
        if not s:
            raise AnnotError("unknown suggestion")
        _event(cur, item_id, annotator_id, s["layer"], "suggestion", suggestion_id, "reject_suggestion")
        _touch(cur, item_id, annotator_id)


def submit_layer(item_id: int, annotator_id: int, layer: str, outcome: str, active_ms: int = 0) -> dict:
    """outcome 'annotated' (>=1 span) or 'none' (explicit no-claim / no-propaganda; 0 spans)."""
    if layer not in LAYERS or outcome not in ("annotated", "none"):
        raise AnnotError("layer must be claim|propaganda and outcome annotated|none")
    tbl = "claim_span" if layer == "claim" else "propaganda_span"
    bc = None
    with tx() as cur:
        cur.execute("SELECT status FROM assignment WHERE item_id=%s AND annotator_id=%s", (item_id, annotator_id))
        a = cur.fetchone()
        if not a or a["status"] == "skipped":
            raise AnnotError("item is not assigned to you (or was skipped)")
        cur.execute(f"SELECT count(*) n FROM {tbl} WHERE item_id=%s AND annotator_id=%s AND deleted_at IS NULL", (item_id, annotator_id))
        n = cur.fetchone()["n"]
        if outcome == "annotated" and n == 0:
            raise AnnotError("no spans marked: add a span, or submit 'none' for no claims / no propaganda")
        if outcome == "none" and n > 0:
            raise AnnotError(f"{n} span(s) exist: delete them first, or submit 'annotated'")
        cur.execute("""INSERT INTO layer_result (item_id, annotator_id, layer, outcome) VALUES (%s,%s,%s,%s)
                       ON CONFLICT (item_id, annotator_id, layer) DO UPDATE SET outcome=EXCLUDED.outcome,
                       revision=layer_result.revision+1, submitted_at=now() RETURNING outcome, revision""",
                    (item_id, annotator_id, layer, outcome))
        res = dict(cur.fetchone())
        _event(cur, item_id, annotator_id, layer, "layer_result", None, "submit", after={**res, "n_spans": n})
        cur.execute("SELECT count(*) n FROM layer_result WHERE item_id=%s AND annotator_id=%s", (item_id, annotator_id))
        complete = cur.fetchone()["n"] == len(LAYERS)
        cur.execute("UPDATE assignment SET active_ms = active_ms + %s WHERE item_id=%s AND annotator_id=%s",
                    (max(0, min(int(active_ms or 0), 3_600_000)), item_id, annotator_id))
        if complete:
            cur.execute("UPDATE assignment SET status='done', finished_at=now(), leased_until=NULL WHERE item_id=%s AND annotator_id=%s",
                        (item_id, annotator_id))
            _refresh_item_status(cur, item_id)
            bc = batching.refresh_batch(cur, annotator_id, item_id)
        else:
            _touch(cur, item_id, annotator_id)
        return {**res, "item_complete": complete, "batch_completed": bc["completed"] if complete and bc else None}


LANGUAGE_SKIP = "Language I cannot annotate"         # annotator-specific, so it never counts towards excluding the item


def _exclude_after_skips(cur, item_id: int) -> bool:
    """Take an item out of the pool once ann_skip_exclude_n DIFFERENT annotators skipped it for a content reason
    (unreadable / spam / offensive / other). Unstarted copies in other batches are released. -> True if excluded."""
    need = int(C("ann_skip_exclude_n"))
    cur.execute("SELECT count(DISTINCT annotator_id) n FROM assignment WHERE item_id=%s AND kind='normal' AND status='skipped' "
                "AND coalesce(skip_reason,'') NOT ILIKE %s", (item_id, LANGUAGE_SKIP + "%"))
    if cur.fetchone()["n"] < need:
        return False
    cur.execute("UPDATE item SET status='excluded', excluded_reason=%s WHERE item_id=%s AND status NOT IN ('done','excluded') RETURNING item_id",
                (f"skipped_by_{need}_annotators", item_id))
    if not cur.fetchone():
        return False
    cur.execute("""DELETE FROM assignment a WHERE a.item_id=%s AND a.status='assigned'
                   AND NOT EXISTS (SELECT 1 FROM claim_span c WHERE c.item_id=a.item_id AND c.annotator_id=a.annotator_id)
                   AND NOT EXISTS (SELECT 1 FROM propaganda_span p WHERE p.item_id=a.item_id AND p.annotator_id=a.annotator_id)
                   RETURNING a.batch_id""", (item_id,))
    for b in {r["batch_id"] for r in cur.fetchall() if r["batch_id"]}:
        cur.execute("UPDATE batch SET size = (SELECT count(*) FROM assignment WHERE batch_id=%s) WHERE batch_id=%s", (b, b))
    return True


def skip_item(item_id: int, annotator_id: int, reason: str) -> None:
    reason = (reason or "").strip()[:200]
    if not reason:
        raise AnnotError("give a reason for skipping")
    with tx() as cur:
        cur.execute("UPDATE assignment SET status='skipped', skip_reason=%s, finished_at=now(), leased_until=NULL "
                    "WHERE item_id=%s AND annotator_id=%s AND status IN ('assigned','in_progress') RETURNING assignment_id",
                    (reason, item_id, annotator_id))
        if not cur.fetchone():
            raise AnnotError("item is not open for you")
        _event(cur, item_id, annotator_id, None, "assignment", None, "skip", after={"reason": reason})
        _refresh_item_status(cur, item_id)
        _exclude_after_skips(cur, item_id)
        batching.refresh_batch(cur, annotator_id, item_id)


def reopen_item(item_id: int, annotator_id: int) -> None:
    """Re-open a finished (done/skipped) item of your own, to correct it."""
    with tx() as cur:
        cur.execute("UPDATE assignment SET status='in_progress', skip_reason=NULL, finished_at=NULL, "
                    "leased_until = now() + make_interval(mins => %s) WHERE item_id=%s AND annotator_id=%s "
                    "AND status IN ('done','skipped') RETURNING assignment_id", (int(C("ann_lease_minutes")), item_id, annotator_id))
        if not cur.fetchone():
            raise AnnotError("nothing to re-open")
        _event(cur, item_id, annotator_id, None, "assignment", None, "reopen")
        _refresh_item_status(cur, item_id)


# ------------------------------------------------------------------ lists + stats
def mine(annotator_id: int, limit: int = 100) -> list[dict]:
    with tx() as cur:
        cur.execute("""SELECT a.item_id, a.status, a.skip_reason, a.finished_at, a.batch_pos, b.seq AS batch_seq, s.platform, i.lang_group, left(i.text, 110) preview,
                              (SELECT count(*) FROM claim_span c WHERE c.item_id=a.item_id AND c.annotator_id=a.annotator_id AND c.deleted_at IS NULL) n_claims,
                              (SELECT count(*) FROM propaganda_span p WHERE p.item_id=a.item_id AND p.annotator_id=a.annotator_id AND p.deleted_at IS NULL) n_prop
                       FROM assignment a JOIN item i USING (item_id) JOIN ingest_source s USING (source_id) LEFT JOIN batch b USING (batch_id)
                       WHERE a.annotator_id=%s AND a.status <> 'assigned'
                       ORDER BY coalesce(a.finished_at, a.assigned_at) DESC LIMIT %s""", (annotator_id, limit))
        return [dict(r) for r in cur.fetchall()]


def stats() -> dict:
    with tx() as cur:
        out = {}
        cur.execute("SELECT status, count(*) n FROM item GROUP BY 1"); out["items_by_status"] = {r["status"]: r["n"] for r in cur.fetchall()}
        cur.execute("""SELECT s.platform, count(*) items, count(*) FILTER (WHERE i.status='done') done
                       FROM item i JOIN ingest_source s USING (source_id) GROUP BY 1 ORDER BY 2 DESC""")
        out["by_platform"] = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT lang_group, count(*) items, count(*) FILTER (WHERE status='done') done FROM item GROUP BY 1 ORDER BY 1")
        out["by_lang"] = [dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT n.username, count(*) FILTER (WHERE a.status='done') done, count(*) FILTER (WHERE a.status='skipped') skipped,
                              count(*) FILTER (WHERE a.status IN ('assigned','in_progress')) open,
                              coalesce(round(avg(a.active_ms) FILTER (WHERE a.status='done') / 1000.0), 0) avg_sec
                       FROM assignment a JOIN annotator n USING (annotator_id) GROUP BY 1 ORDER BY 2 DESC""")
        out["by_annotator"] = [dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT e.layer, count(*) FILTER (WHERE e.action='accept_suggestion' AND e.entity_type='suggestion') accepted,
                              count(*) FILTER (WHERE e.action='reject_suggestion') rejected
                       FROM label_event e JOIN item i USING (item_id) WHERE e.layer IS NOT NULL GROUP BY 1""")
        out["suggestion_feedback"] = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT count(*) FILTER (WHERE kind='claims') c, count(*) FILTER (WHERE kind='propaganda') p FROM export_run")
        out["exports"] = dict(cur.fetchone())
        cur.execute("""SELECT s.platform, w.newest_ts, w.last_run_at, w.rows_seen FROM ingest_watermark w JOIN ingest_source s USING (source_id)
                       ORDER BY s.platform""")
        out["watermarks"] = [dict(r) for r in cur.fetchall()]
        return out


def history(item_id: int, annotator_id: int) -> list[dict]:
    with tx() as cur:
        cur.execute("SELECT event_id, ts, layer, entity_type, entity_id, action, before, after FROM label_event "
                    "WHERE item_id=%s AND annotator_id=%s ORDER BY event_id", (item_id, annotator_id))
        return [dict(r) for r in cur.fetchall()]
