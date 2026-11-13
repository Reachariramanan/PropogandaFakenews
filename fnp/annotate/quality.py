"""Annotator quality: gold scores, agreement with partners, behaviour. Read-only; feeds the admin Quality tab and user page.

  gold       annotators label adjudicated reference items blind (assignment.kind='gold'). Span match: IoU >= 0.5 (and the
             same technique for propaganda); precision / recall / F1 per layer, claim-type accuracy on matched spans.
  agreement  on double-annotated items, each annotator against their partner (symmetric span F1 per layer).
  behaviour  share of 'none' outcomes against the team median, median active seconds per finished item, and how much
             they rely on model suggestions on assisted items (accept rate).
Flags are prompts for a human to look, not verdicts: small samples are reported with their n.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Optional

from ..config import C
from . import export
from .db import tx

LAYERS = ("claim", "propaganda")


def _prf(tp: int, n_ref: int, n_ann: int) -> dict:
    p = tp / n_ann if n_ann else None
    r = tp / n_ref if n_ref else None
    f = 2 * tp / (n_ref + n_ann) if (n_ref + n_ann) else None
    return {"precision": p, "recall": r, "f1": f, "n_ref": n_ref, "n_ann": n_ann, "tp": tp}


def match_against_ref(layer: str, ref: list[dict], ann: list[dict]) -> tuple[int, int, int, int]:
    """-> (tp, n_ref, n_ann, type_correct) with one-to-one IoU>=0.5 matching (technique must agree for propaganda)."""
    if layer == "propaganda":
        expand = lambda spans: [{**s, "technique": t} for s in spans for t in s["techniques"]]
        A, B = expand(ref), expand(ann)
        pairs = export.match_spans(A, B, export.IOU_MATCH, lambda p, q: p["technique"] == q["technique"])
        return len(pairs), len(A), len(B), 0
    pairs = export.match_spans(ref, ann, export.IOU_MATCH, None)
    return len(pairs), len(ref), len(ann), sum(ref[i]["claim_type"] == ann[j]["claim_type"] for i, j in pairs)


# ------------------------------------------------------------------ gold
def _submitted(cur, pairs: list[tuple[int, int]]) -> dict:
    """{(item_id, annotator_id): {"claim": {outcome, spans}, "propaganda": {...}}} for the given pairs."""
    out: dict = {}
    if not pairs:
        return out
    items = sorted({i for i, _ in pairs}); anns = sorted({a for _, a in pairs}); want = set(pairs)
    cur.execute("SELECT item_id, annotator_id, layer, outcome FROM layer_result WHERE item_id = ANY(%s) AND annotator_id = ANY(%s)", (items, anns))
    for r in cur.fetchall():
        if (r["item_id"], r["annotator_id"]) in want:
            out.setdefault((r["item_id"], r["annotator_id"]), {})[r["layer"]] = {"outcome": r["outcome"], "spans": []}
    cur.execute("SELECT * FROM claim_span WHERE item_id = ANY(%s) AND annotator_id = ANY(%s) AND deleted_at IS NULL ORDER BY start_char", (items, anns))
    for r in cur.fetchall():
        k = out.get((r["item_id"], r["annotator_id"]))
        if k and "claim" in k:
            k["claim"]["spans"].append({"start": r["start_char"], "end": r["end_char"], "claim_type": r["claim_type"], "check_worthy": r["check_worthy"]})
    cur.execute("""SELECT p.*, array_agg(t.technique ORDER BY t.technique) techs FROM propaganda_span p JOIN propaganda_span_technique t USING (span_id)
                   WHERE p.item_id = ANY(%s) AND p.annotator_id = ANY(%s) AND p.deleted_at IS NULL GROUP BY p.span_id ORDER BY p.start_char""", (items, anns))
    for r in cur.fetchall():
        k = out.get((r["item_id"], r["annotator_id"]))
        if k and "propaganda" in k:
            k["propaganda"]["spans"].append({"start": r["start_char"], "end": r["end_char"], "techniques": list(r["techs"])})
    return out


def gold_scores(annotator_id: Optional[int] = None) -> dict:
    """-> {annotator_id: {"n_items", "claim": prf + claim_type_accuracy, "propaganda": prf, "item_label_agreement"}}"""
    with tx() as cur:
        cur.execute("""SELECT x.item_id, x.annotator_id, g.ref FROM assignment x JOIN gold_item g USING (item_id)
                       WHERE x.kind='gold' AND x.status='done' AND (%s::int IS NULL OR x.annotator_id=%s)""", (annotator_id, annotator_id))
        rows = cur.fetchall()
        sub = _submitted(cur, [(r["item_id"], r["annotator_id"]) for r in rows])
    agg: dict = defaultdict(lambda: {"n_items": 0, **{l: {"tp": 0, "n_ref": 0, "n_ann": 0, "type_ok": 0} for l in LAYERS}, "agree": 0, "agree_n": 0})
    for r in rows:
        mine = sub.get((r["item_id"], r["annotator_id"]), {})
        a = agg[r["annotator_id"]]
        a["n_items"] += 1
        for layer in LAYERS:
            ref, got = r["ref"].get(layer), mine.get(layer)
            if not ref or not got:
                continue
            tp, nr, na, ok = match_against_ref(layer, ref["spans"], got["spans"])
            for k, v in (("tp", tp), ("n_ref", nr), ("n_ann", na), ("type_ok", ok)):
                a[layer][k] += v
            a["agree"] += bool(ref["spans"]) == bool(got["spans"]); a["agree_n"] += 1
    out = {}
    for aid, a in agg.items():
        d = {"n_items": a["n_items"], "item_label_agreement": a["agree"] / a["agree_n"] if a["agree_n"] else None}
        for layer in LAYERS:
            d[layer] = _prf(a[layer]["tp"], a[layer]["n_ref"], a[layer]["n_ann"])
        d["claim"]["claim_type_accuracy"] = a["claim"]["type_ok"] / a["claim"]["tp"] if a["claim"]["tp"] else None
        out[aid] = d
    return out


# ------------------------------------------------------------------ agreement with partners
def partner_agreement() -> dict:
    """On double items (two normal annotators, adjudications and gold ignored): per annotator, symmetric span F1 against the partner."""
    acc: dict = defaultdict(lambda: {"items": set(), **{l: {"tp": 0, "self": 0, "other": 0} for l in LAYERS}, "agree": 0, "n": 0})
    for layer in LAYERS:
        items, subs = export.load_submissions(layer)
        for iid, ss in subs.items():
            s = sorted([x for x in ss if x["role"] != "adjudicator"], key=lambda x: x["annotator_id"])
            if len(s) < 2:
                continue
            a, b = s[0], s[1]
            if layer == "propaganda":
                ex = lambda x: [{**sp, "technique": t} for sp in x["spans"] for t in sp["techniques"]]
                A, B = ex(a), ex(b)
                pairs = export.match_spans(A, B, export.IOU_MATCH, lambda p, q: p["technique"] == q["technique"])
            else:
                A, B = a["spans"], b["spans"]
                pairs = export.match_spans(A, B, export.IOU_MATCH, None)
            for me, (n_me, n_ot) in ((a, (len(A), len(B))), (b, (len(B), len(A)))):
                r = acc[me["annotator_id"]]
                r["items"].add(iid); r[layer]["tp"] += len(pairs); r[layer]["self"] += n_me; r[layer]["other"] += n_ot
                r["agree"] += bool(a["spans"]) == bool(b["spans"]); r["n"] += 1
    out = {}
    for aid, r in acc.items():
        d = {"n_items": len(r["items"]), "item_label_agreement": r["agree"] / r["n"] if r["n"] else None}
        for layer in LAYERS:
            t = r[layer]
            d[layer + "_f1"] = 2 * t["tp"] / (t["self"] + t["other"]) if (t["self"] + t["other"]) else None
        out[aid] = d
    return out


# ------------------------------------------------------------------ behaviour
def behaviour() -> dict:
    with tx() as cur:
        cur.execute("""SELECT x.annotator_id, count(*) FILTER (WHERE x.status='done') done,
                              percentile_cont(0.5) WITHIN GROUP (ORDER BY x.active_ms / 1000.0) FILTER (WHERE x.status='done' AND x.active_ms > 0) median_sec
                       FROM assignment x WHERE x.kind='normal' GROUP BY 1""")
        base = {r["annotator_id"]: dict(r) for r in cur.fetchall()}
        cur.execute("""SELECT r.annotator_id, r.layer, count(*) n, count(*) FILTER (WHERE r.outcome='none') none
                       FROM layer_result r JOIN assignment x ON x.item_id=r.item_id AND x.annotator_id=r.annotator_id AND x.kind='normal' GROUP BY 1,2""")
        for r in cur.fetchall():
            b = base.setdefault(r["annotator_id"], {"annotator_id": r["annotator_id"], "done": 0, "median_sec": None})
            b[f"{r['layer']}_none_rate"] = r["none"] / r["n"] if r["n"] else None
            b[f"{r['layer']}_n"] = r["n"]
        cur.execute("""SELECT e.annotator_id, count(*) FILTER (WHERE e.action='accept_suggestion' AND e.entity_type='suggestion') acc,
                              count(*) FILTER (WHERE e.action='reject_suggestion') rej FROM label_event e GROUP BY 1""")
        for r in cur.fetchall():
            if r["annotator_id"] in base and (r["acc"] + r["rej"]):
                base[r["annotator_id"]]["suggestion_accept_rate"] = r["acc"] / (r["acc"] + r["rej"])
                base[r["annotator_id"]]["suggestions_judged"] = r["acc"] + r["rej"]
    return base


# ------------------------------------------------------------------ combined view
def team_quality() -> dict:
    """-> {"rows": [per annotator], "team": {...}} with flags; one row per annotator that has any finished work."""
    with tx() as cur:
        cur.execute("SELECT annotator_id, username, role, active FROM annotator ORDER BY username")
        people = [dict(r) for r in cur.fetchall()]
    gold, part, beh = gold_scores(), partner_agreement(), behaviour()
    minn = int(C("ann_q_min_items"))
    nones = [b.get("claim_none_rate") for b in beh.values() if b.get("claim_n", 0) >= minn and b.get("claim_none_rate") is not None]
    team_none = median(nones) if nones else None
    rows = []
    for p in people:
        b = beh.get(p["annotator_id"])
        if not b or not (b.get("done") or p["annotator_id"] in gold):
            continue
        g, pa = gold.get(p["annotator_id"]), part.get(p["annotator_id"])
        flags = []
        if g and g["n_items"] >= int(C("ann_gold_min_items")):
            spans_f1 = [g[l]["f1"] for l in LAYERS if g[l]["f1"] is not None]
            if spans_f1 and sum(spans_f1) / len(spans_f1) < C("ann_gold_min_f1"):
                flags.append(f"gold span F1 {sum(spans_f1) / len(spans_f1):.2f} < {C('ann_gold_min_f1')}")
        if pa and pa["n_items"] >= int(C("ann_gold_min_items")):
            f = [pa[l + "_f1"] for l in LAYERS if pa[l + "_f1"] is not None]
            if f and sum(f) / len(f) < C("ann_q_min_agree_f1"):
                flags.append(f"agreement with partners F1 {sum(f) / len(f):.2f} < {C('ann_q_min_agree_f1')}")
        if b.get("done", 0) >= minn:
            nr = b.get("claim_none_rate")
            if nr is not None and team_none is not None and abs(nr - team_none) > C("ann_q_none_dev"):
                flags.append(f"'no claim' share {nr:.0%} vs team median {team_none:.0%}")
            if b.get("median_sec") is not None and b["median_sec"] < C("ann_q_fast_sec"):
                flags.append(f"median {b['median_sec']:.0f} s per item")
        rows.append({**p, "items_done": b.get("done", 0), "median_sec": b.get("median_sec"),
                     "claim_none_rate": b.get("claim_none_rate"), "propaganda_none_rate": b.get("propaganda_none_rate"),
                     "suggestion_accept_rate": b.get("suggestion_accept_rate"), "gold": g, "agreement": pa, "flags": flags})
    return {"rows": rows, "team": {"claim_none_rate_median": team_none, "annotators": len(rows)},
            "thresholds": {k: C(k) for k in ("ann_gold_min_items", "ann_gold_min_f1", "ann_q_min_items", "ann_q_none_dev", "ann_q_fast_sec", "ann_q_min_agree_f1")}}


def user_quality(annotator_id: int) -> dict:
    t = team_quality()
    row = next((r for r in t["rows"] if r["annotator_id"] == annotator_id), None)
    return {"row": row, "team": t["team"], "thresholds": t["thresholds"]}
