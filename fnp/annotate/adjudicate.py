"""Adjudication of disputed double-annotated items, and gold (calibration) items.

An item is DISPUTED in a layer when its two annotators' labels disagree under the `agree` policy (export.resolve_item): one
marked something the other did not, or they chose different techniques for the same span. Such items are left out of training
exports until an admin / adjudicator resolves them:

  start(item, adjudicator)         opens the item for the adjudicator (assignment.kind='adjudication'); the normal annotation UI
                                   shows both annotators' spans next to it;
  copy_from(item, adjudicator, a)  copies one annotator's spans of a layer as the adjudicator's starting point (then edit);
  the adjudicator submits both layers like any annotator; export.load_submissions marks that submission as the adjudication,
  which wins over the two annotators (resolve_item, policy `adjudicated`);
  mark_gold(item, adjudicator)     freezes the adjudicated labels as a gold item: future batches contain it blind
                                   (assignment.kind='gold') and quality.py scores the annotators against it.
"""
from __future__ import annotations

import json
from typing import Optional

from . import export, service as S
from .db import tx

LAYERS = ("claim", "propaganda")
ROLES = ("admin", "adjudicator")


def _need_role(cur, annotator_id: int) -> None:
    cur.execute("SELECT role, active FROM annotator WHERE annotator_id=%s", (annotator_id,))
    r = cur.fetchone()
    if not r or not r["active"] or r["role"] not in ROLES:
        raise S.AnnotError("only an admin or adjudicator can adjudicate")


# ------------------------------------------------------------------ the dispute queue
def disputes(limit: int = 300) -> dict:
    """-> {"items": [...], "counts": {...}}. state: pending | in_progress | adjudicated (originally disputed items only)."""
    per: dict[int, dict] = {}
    texts: dict[int, dict] = {}
    for layer in LAYERS:
        items, subs = export.load_submissions(layer)
        for iid, ss in subs.items():
            normal = [x for x in ss if x["role"] != "adjudicator"]
            adj = [x for x in ss if x["role"] == "adjudicator"]
            if len(normal) < 2:
                continue
            r = export.resolve_item(layer, normal, "agree")
            if r["status"] != "disputed":
                continue
            d = per.setdefault(iid, {"item_id": iid, "layers": {}, "annotators": sorted(x["username"] for x in normal), "adjudicated_layers": []})
            d["layers"][layer] = {"unmatched": r.get("unmatched"), "agreed_spans": len(r["spans"])}
            if adj:
                d["adjudicated_layers"].append(layer)
            texts[iid] = items[iid]
    if not per:
        return {"items": [], "counts": {"pending": 0, "in_progress": 0, "adjudicated": 0}}
    with tx() as cur:
        cur.execute("""SELECT x.item_id, n.username, x.status FROM assignment x JOIN annotator n USING (annotator_id)
                       WHERE x.kind='adjudication' AND x.item_id = ANY(%s)""", (list(per),))
        adj_asg = {r["item_id"]: r for r in cur.fetchall()}
        cur.execute("SELECT item_id FROM gold_item WHERE active AND item_id = ANY(%s)", (list(per),))
        gold = {r["item_id"] for r in cur.fetchall()}
    out = []
    for iid, d in sorted(per.items()):
        a = adj_asg.get(iid)
        done = set(d["adjudicated_layers"]) >= set(d["layers"])
        d.update(state="adjudicated" if done else "in_progress" if a else "pending", adjudicator=a["username"] if a else None,
                 gold=iid in gold, platform=texts[iid]["platform"], preview=texts[iid]["text"][:160])
        out.append(d)
    order = {"pending": 0, "in_progress": 1, "adjudicated": 2}
    out.sort(key=lambda d: (order[d["state"]], d["item_id"]))
    counts = {k: sum(d["state"] == k for d in out) for k in order}
    return {"items": out[:limit], "counts": counts}


# ------------------------------------------------------------------ opening, viewing, copying
def start(item_id: int, adjudicator_id: int) -> dict:
    with tx() as cur:
        _need_role(cur, adjudicator_id)
        cur.execute("SELECT count(DISTINCT annotator_id) n FROM assignment WHERE item_id=%s AND kind='normal' AND status='done'", (item_id,))
        if cur.fetchone()["n"] < 2:
            raise S.AnnotError("this item does not have two finished annotators: nothing to adjudicate")
        cur.execute("SELECT 1 FROM assignment WHERE item_id=%s AND annotator_id=%s AND kind='normal'", (item_id, adjudicator_id))
        if cur.fetchone():
            raise S.AnnotError("you annotated this item yourself: another adjudicator must resolve it")
        cur.execute("""INSERT INTO assignment (item_id, annotator_id, status, kind, selected_by) VALUES (%s,%s,'in_progress','adjudication','adjudication')
                       ON CONFLICT (item_id, annotator_id) DO UPDATE SET status = CASE WHEN assignment.status='skipped' THEN 'in_progress' ELSE assignment.status END
                       RETURNING status""", (item_id, adjudicator_id))
        return {"item_id": item_id, "status": cur.fetchone()["status"]}


def view(cur, item_id: int, adjudicator_id: int) -> Optional[dict]:
    """The annotators' spans for an item the caller is adjudicating; None for anything else."""
    cur.execute("SELECT 1 FROM assignment WHERE item_id=%s AND annotator_id=%s AND kind='adjudication'", (item_id, adjudicator_id))
    if not cur.fetchone():
        return None
    cur.execute("""SELECT r.annotator_id, n.username, r.layer, r.outcome FROM layer_result r JOIN annotator n USING (annotator_id)
                   JOIN assignment x ON x.item_id=r.item_id AND x.annotator_id=r.annotator_id AND x.kind='normal'
                   WHERE r.item_id=%s ORDER BY r.annotator_id, r.layer""", (item_id,))
    people: dict[int, dict] = {}
    for r in cur.fetchall():
        p = people.setdefault(r["annotator_id"], {"annotator_id": r["annotator_id"], "username": r["username"], "claim": None, "propaganda": None})
        p[r["layer"]] = {"outcome": r["outcome"], "spans": []}
    ids = list(people)
    cur.execute("SELECT * FROM claim_span WHERE item_id=%s AND annotator_id = ANY(%s) AND deleted_at IS NULL ORDER BY start_char", (item_id, ids))
    for r in cur.fetchall():
        if people[r["annotator_id"]]["claim"] is not None:
            people[r["annotator_id"]]["claim"]["spans"].append(
                {"start": r["start_char"], "end": r["end_char"], "text": r["span_text"], "claim_type": r["claim_type"], "check_worthy": r["check_worthy"]})
    cur.execute("""SELECT p.*, array_agg(t.technique ORDER BY t.technique) techs FROM propaganda_span p JOIN propaganda_span_technique t USING (span_id)
                   WHERE p.item_id=%s AND p.annotator_id = ANY(%s) AND p.deleted_at IS NULL GROUP BY p.span_id ORDER BY p.start_char""", (item_id, ids))
    for r in cur.fetchall():
        if people[r["annotator_id"]]["propaganda"] is not None:
            people[r["annotator_id"]]["propaganda"]["spans"].append(
                {"start": r["start_char"], "end": r["end_char"], "text": r["span_text"], "techniques": list(r["techs"])})
    cur.execute("SELECT 1 FROM gold_item WHERE item_id=%s AND active", (item_id,))
    return {"annotators": list(people.values()), "is_gold": bool(cur.fetchone())}


def copy_from(item_id: int, adjudicator_id: int, from_annotator_id: int, layer: str) -> None:
    """Replace the adjudicator's spans of `layer` by a copy of one annotator's (to be edited), as ordinary manual spans."""
    if layer not in LAYERS:
        raise S.AnnotError("layer must be claim|propaganda")
    with tx() as cur:
        v = view(cur, item_id, adjudicator_id)
        if v is None:
            raise S.AnnotError("open the item for adjudication first")
        src = next((p for p in v["annotators"] if p["annotator_id"] == from_annotator_id), None)
        if src is None or src[layer] is None:
            raise S.AnnotError("that annotator has not labelled this layer")
        tbl = "claim_span" if layer == "claim" else "propaganda_span"
        cur.execute(f"SELECT span_id FROM {tbl} WHERE item_id=%s AND annotator_id=%s AND deleted_at IS NULL", (item_id, adjudicator_id))
        mine = [r["span_id"] for r in cur.fetchall()]
    for sid in mine:
        (S.delete_claim if layer == "claim" else S.delete_propaganda)(sid, adjudicator_id)
    for sp in src[layer]["spans"]:
        if layer == "claim":
            S.add_claim(item_id, adjudicator_id, sp["start"], sp["end"], sp["claim_type"], sp["check_worthy"])
        else:
            S.add_propaganda(item_id, adjudicator_id, sp["start"], sp["end"], sp["techniques"])


# ------------------------------------------------------------------ gold
def mark_gold(item_id: int, by: int, note: Optional[str] = None) -> dict:
    """Freeze the adjudicated labels of an item as gold. Needs a finished adjudication (both layers)."""
    with tx() as cur:
        _need_role(cur, by)
        cur.execute("""SELECT x.annotator_id FROM assignment x WHERE x.item_id=%s AND x.kind='adjudication' AND x.status='done' ORDER BY x.finished_at DESC LIMIT 1""", (item_id,))
        a = cur.fetchone()
        if not a:
            raise S.AnnotError("adjudicate and submit both layers first")
        adj = a["annotator_id"]
        ref = {}
        cur.execute("SELECT layer, outcome FROM layer_result WHERE item_id=%s AND annotator_id=%s", (item_id, adj))
        for r in cur.fetchall():
            ref[r["layer"]] = {"outcome": r["outcome"], "spans": []}
        cur.execute("SELECT * FROM claim_span WHERE item_id=%s AND annotator_id=%s AND deleted_at IS NULL ORDER BY start_char", (item_id, adj))
        for r in cur.fetchall():
            ref["claim"]["spans"].append({"start": r["start_char"], "end": r["end_char"], "text": r["span_text"],
                                          "claim_type": r["claim_type"], "check_worthy": r["check_worthy"]})
        cur.execute("""SELECT p.*, array_agg(t.technique ORDER BY t.technique) techs FROM propaganda_span p JOIN propaganda_span_technique t USING (span_id)
                       WHERE p.item_id=%s AND p.annotator_id=%s AND p.deleted_at IS NULL GROUP BY p.span_id ORDER BY p.start_char""", (item_id, adj))
        for r in cur.fetchall():
            ref["propaganda"]["spans"].append({"start": r["start_char"], "end": r["end_char"], "text": r["span_text"], "techniques": list(r["techs"])})
        if set(ref) != set(LAYERS):
            raise S.AnnotError("both layers must be submitted")
        cur.execute("""INSERT INTO gold_item (item_id, ref, created_by, note) VALUES (%s,%s,%s,%s)
                       ON CONFLICT (item_id) DO UPDATE SET ref=EXCLUDED.ref, created_by=EXCLUDED.created_by, active=true, note=EXCLUDED.note""",
                    (item_id, json.dumps(ref), by, note))
        return {"item_id": item_id, "gold": True}


def unmark_gold(item_id: int, by: int) -> None:
    with tx() as cur:
        _need_role(cur, by)
        cur.execute("UPDATE gold_item SET active=false WHERE item_id=%s", (item_id,))


def gold_list() -> list[dict]:
    with tx() as cur:
        cur.execute("""SELECT g.item_id, g.created_at, g.active, n.username created_by, left(i.text, 140) preview,
                              (SELECT count(*) FROM assignment a WHERE a.item_id=g.item_id AND a.kind='gold') served,
                              (SELECT count(*) FROM assignment a WHERE a.item_id=g.item_id AND a.kind='gold' AND a.status='done') answered
                       FROM gold_item g JOIN item i USING (item_id) LEFT JOIN annotator n ON n.annotator_id=g.created_by ORDER BY g.created_at DESC""")
        return [dict(r) for r in cur.fetchall()]
