"""Per-annotator batches (ann_batch_size items each) built from a stratified, class-balanced selection.

An annotator always works inside one open batch. When every item of it is finished (done or skipped) the next batch
is built and added automatically ("finish 250, get the next 250"). Items are reserved to the annotator when the batch
is built, so two annotators never hold the same single-annotation item; a double-annotation item is reserved to two
different annotators. Serving order inside a batch is the (shuffled) selection order.

Selection, for every item of the batch:
  1. cell   : least-covered source (by ingest_source.weight) -> language (ann_share_*) -> month, using coverage of
              all reservations and finished items so far (cells = the stratification the owner asked for);
  2. item   : draw ann_balance_sample * k candidates from that cell and take the k with the highest class-deficit
              score. Score = how far the classes the models predict for the item (enc_v1 techniques, gemma claim
              types) are from their target (metrics.targets), counting items already reserved. Items the models
              have not seen score neutral (0.5); items they saw with no spans score the explicit-negative deficit.
              Predictions only steer *which* items are served; they are never shown on blind items.
  3. slice  : ann_random_frac of each batch is NOT steered (a random candidate of its cell) and flagged selected_by='random':
              an unbiased slice for evaluation and for checking the steering. Every assignment records how it was selected
              (steered | random | gold), the score and the predicted labels the score used (assignment.sel_*).
  4. gold   : ann_gold_frac of each batch are gold items (adjudicated reference labels, assignment.kind='gold'), served
              blind and scored in quality.py; their labels never enter exports. Only once gold items exist.
Class targets are capped at what the pool can plausibly supply (metrics.feasible_targets): imbalance that remains is
accepted and corrected by class weights in the export.
Languages with ann_share_* = 0 are never served. Batch building is serialised with an advisory lock.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Optional

from ..config import C
from . import metrics
from .db import tx

LOCK_KEY = 0x464E5041          # "FNPA": serialises batch construction across processes
NEUTRAL = 0.5


def allowed_langs() -> dict[str, float]:
    sh = {"en": C("ann_share_en"), "hi": C("ann_share_hi"), "other": C("ann_share_other")}
    return {g: v for g, v in sh.items() if v > 0}


def batch_size_for(cur, annotator_id: int) -> int:
    cur.execute("SELECT batch_size FROM annotator WHERE annotator_id=%s", (annotator_id,))
    r = cur.fetchone()
    return int(r["batch_size"] or C("ann_batch_size"))


# ------------------------------------------------------------------ scoring
class Deficit:
    """Class-deficit weights w_c = max(0, 1 - (finished + reserved)/target); updated as items are picked."""

    def __init__(self, cur):
        t = metrics.targets(cur)
        done, fl = metrics.class_counts(cur), metrics.inflight_predictions(cur)
        self.nominal = t
        ft = metrics.feasible_targets(cur, t, done)        # accept imbalance where the pool cannot supply a class
        self.target = {l: {c: v for c, v in ft[l].items() if v} for l in metrics.LAYERS}
        self.have = {l: defaultdict(float) for l in metrics.LAYERS}
        for l in metrics.LAYERS:
            for c in self.target[l]:
                self.have[l][c] = done[l].get(c, 0) + fl[l].get(c, 0)
        self.neg_target = {l: max(1, t["neg"]) for l in metrics.LAYERS}
        self.neg_have = {l: float(done["none"][l]) for l in metrics.LAYERS}

    def w(self, layer: str, code: str) -> float:
        tg = self.target[layer].get(code)
        return 0.0 if not tg else max(0.0, 1.0 - self.have[layer][code] / tg)

    def w_neg(self, layer: str) -> float:
        return max(0.0, 1.0 - self.neg_have[layer] / self.neg_target[layer])

    def layer_score(self, layer: str, labels: list[str], ran: bool) -> float:
        if not ran:
            return NEUTRAL
        ws = sorted((self.w(layer, c) for c in set(labels) if c in self.target[layer]), reverse=True)
        if not labels:
            return self.w_neg(layer)
        if not ws:
            return 0.0                              # only catch-all classes predicted (OTHER/UNCLEAR)
        return ws[0] + 0.25 * sum(ws[1:])          # best class counts fully, extra rare classes add a little

    def score(self, cand: dict) -> float:
        return (self.layer_score("claim", cand["claim_labels"], cand["claim_ran"])
                + self.layer_score("propaganda", cand["prop_labels"], cand["prop_ran"]))

    def take(self, cand: dict) -> None:
        for layer, labels, ran in (("claim", cand["claim_labels"], cand["claim_ran"]), ("propaganda", cand["prop_labels"], cand["prop_ran"])):
            if not ran:
                continue
            if not labels:
                self.neg_have[layer] += 1
            for c in set(labels):
                if c in self.target[layer]:
                    self.have[layer][c] += 1


# ------------------------------------------------------------------ building
POOL_WHERE = """i.status IN ('queued','in_progress') AND i.lang_group = ANY(%(langs)s)
    AND coalesce(t.n, 0) < CASE WHEN i.is_double THEN 2 ELSE 1 END
    AND NOT EXISTS (SELECT 1 FROM assignment a WHERE a.item_id=i.item_id AND a.annotator_id=%(a)s)"""
TAKEN = "taken AS (SELECT item_id, count(*) n FROM assignment WHERE status IN ('assigned','in_progress','done') GROUP BY 1)"


def _pick_gold(cur, annotator_id: int, k: int) -> list[int]:
    """Up to k gold items this annotator has never been assigned (least-served first, then random)."""
    if k <= 0:
        return []
    cur.execute("""SELECT g.item_id FROM gold_item g
                   WHERE g.active AND NOT EXISTS (SELECT 1 FROM assignment a WHERE a.item_id=g.item_id AND a.annotator_id=%s)
                   ORDER BY (SELECT count(*) FROM assignment a WHERE a.item_id=g.item_id AND a.kind='gold'), random() LIMIT %s""",
                (annotator_id, k))
    return [r["item_id"] for r in cur.fetchall()]


def build_batch(cur, annotator_id: int, rng: Optional[random.Random] = None) -> Optional[int]:
    """Create the annotator's next batch inside the caller's transaction. -> batch_id, or None when the pool is empty."""
    rng = rng or random.Random()
    cur.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
    size = batch_size_for(cur, annotator_id)
    gold = _pick_gold(cur, annotator_id, int(round(C("ann_gold_frac") * size)))
    size_n = size - len(gold)                              # ordinary items; gold fills the rest of the batch
    shares = allowed_langs()
    langs = sorted(shares)
    prm = {"langs": langs, "a": annotator_id}
    cur.execute(f"""WITH {TAKEN} SELECT i.source_id, i.lang_group, i.period, count(*) avail
                    FROM item i LEFT JOIN taken t USING (item_id) WHERE {POOL_WHERE} GROUP BY 1,2,3""", prm)
    avail = {(r["source_id"], r["lang_group"], r["period"]): r["avail"] for r in cur.fetchall()}
    if not avail:
        return None
    cur.execute("""SELECT i.source_id, i.lang_group, i.period, count(*) n FROM assignment a JOIN item i USING (item_id)
                   WHERE a.status IN ('done','in_progress','assigned') GROUP BY 1,2,3""")
    cov = defaultdict(int)
    for r in cur.fetchall():
        cov[(r["source_id"], r["lang_group"], r["period"])] += r["n"]
    cur.execute("SELECT source_id, platform, weight FROM ingest_source")
    src = {r["source_id"]: r for r in cur.fetchall()}

    # ---- 1. cell sequence (hierarchical least-covered), simulated on local copies
    def pick(cands, score):
        best = min(score(c) for c in cands)
        return rng.choice([c for c in cands if score(c) == best])

    left = dict(avail)
    seq: list[tuple] = []
    for _ in range(size_n):
        live = [k for k, v in left.items() if v > 0]
        if not live:
            break
        s = pick(sorted({k[0] for k in live}),
                 lambda sid: sum(v for k, v in cov.items() if k[0] == sid) / max(float(src[sid]["weight"]), 1e-6))
        lg = pick(sorted({k[1] for k in live if k[0] == s}),
                  lambda g: sum(v for k, v in cov.items() if k[0] == s and k[1] == g) / max(shares.get(g, 0.1), 1e-6))
        per = pick(sorted({k[2] for k in live if k[0] == s and k[1] == lg}), lambda p: cov.get((s, lg, p), 0))
        cell = (s, lg, per)
        seq.append(cell)
        left[cell] -= 1
        cov[cell] += 1
    if not seq:
        return None

    # ---- 2. candidates per cell (random draw, with the models' predicted classes)
    need = defaultdict(int)
    for c in seq:
        need[c] += 1
    m = max(1, int(C("ann_balance_sample")))
    cands: dict[tuple, list[dict]] = {}
    for (s, lg, per), k in need.items():
        cur.execute(f"""WITH {TAKEN},
            cand AS (SELECT i.item_id FROM item i LEFT JOIN taken t USING (item_id)
                     WHERE {POOL_WHERE} AND i.source_id=%(s)s AND i.lang_group=%(l)s AND i.period=%(p)s
                     ORDER BY random() LIMIT %(lim)s)
            SELECT c.item_id,
                   coalesce(array_agg(DISTINCT s.label) FILTER (WHERE s.layer='claim' AND s.label IS NOT NULL), '{{}}') AS claim_labels,
                   coalesce(array_agg(DISTINCT s.label) FILTER (WHERE s.layer='propaganda' AND s.label IS NOT NULL), '{{}}') AS prop_labels,
                   EXISTS (SELECT 1 FROM suggestion_done d WHERE d.item_id=c.item_id AND d.model_version LIKE 'gemma\\_%%') AS claim_ran,
                   EXISTS (SELECT 1 FROM suggestion_done d WHERE d.item_id=c.item_id AND d.model_version LIKE 'enc\\_%%') AS prop_ran
            FROM cand c LEFT JOIN suggestion s ON s.item_id=c.item_id GROUP BY c.item_id""",
                    {**prm, "s": s, "l": lg, "p": per, "lim": k * m})
        cands[(s, lg, per)] = [dict(r) for r in cur.fetchall()]

    # ---- 3. pick best-scored candidates, updating class deficits as the batch fills
    deficit = Deficit(cur)
    chosen: list[tuple] = []                               # (item_id, cell, selected_by, score, meta)
    used: set[int] = set()
    scored = 0
    n_rand = int(round(C("ann_random_frac") * len(seq)))
    rand_idx = set(rng.sample(range(len(seq)), n_rand)) if n_rand else set()
    for idx, cell in enumerate(seq):
        pool = [c for c in cands.get(cell, []) if c["item_id"] not in used]
        if not pool:
            continue
        how = "random" if idx in rand_idx else "steered"
        best = rng.choice(pool) if how == "random" else max(pool, key=lambda c: deficit.score(c) + rng.random() * 0.02)
        sc = deficit.score(best)                          # score the item had when it was picked (before it counts)
        used.add(best["item_id"])
        deficit.take(best)
        scored += bool(best["claim_ran"] or best["prop_ran"])
        chosen.append((best["item_id"], cell, how, sc,
                       {"claim_labels": best["claim_labels"], "prop_labels": best["prop_labels"], "claim_ran": best["claim_ran"],
                        "prop_ran": best["prop_ran"], "cell": list(cell)}))
    if not chosen and not gold:
        return None
    rng.shuffle(chosen)                                   # serving order: not grouped by source / language
    for g in gold:                                        # gold items at random positions, indistinguishable from the rest
        chosen.insert(rng.randint(0, len(chosen)), (g, None, "gold", None, None))

    cur.execute("SELECT coalesce(max(seq), 0) + 1 AS seq FROM batch WHERE annotator_id=%s", (annotator_id,))
    seqno = cur.fetchone()["seq"]
    by_p, by_l = defaultdict(int), defaultdict(int)
    for _, cell, how, _sc, _m in chosen:
        if cell is not None:
            by_p[src[cell[0]]["platform"]] += 1
            by_l[cell[1]] += 1
    cur.execute("INSERT INTO batch (annotator_id, seq, size, meta) VALUES (%s,%s,%s,%s) RETURNING batch_id",
                (annotator_id, seqno, len(chosen), json.dumps({"by_platform": dict(by_p), "by_lang": dict(by_l), "n_with_model_predictions": scored,
                                                               "requested": size, "n_random": sum(c[2] == "random" for c in chosen),
                                                               "n_gold": len(gold)})))
    bid = cur.fetchone()["batch_id"]
    for pos, (iid, _cell, how, sc, meta) in enumerate(chosen, 1):
        cur.execute("INSERT INTO assignment (item_id, annotator_id, status, batch_id, batch_pos, kind, selected_by, sel_score, sel_meta) "
                    "VALUES (%s,%s,'assigned',%s,%s,%s,%s,%s,%s)",
                    (iid, annotator_id, bid, pos, "gold" if how == "gold" else "normal", how, sc,
                     json.dumps(meta) if meta is not None else None))
    cur.execute("UPDATE item SET status='in_progress' WHERE item_id = ANY(%s) AND status='queued'", ([c[0] for c in chosen],))
    cur.execute("INSERT INTO label_event (item_id, annotator_id, layer, entity_type, entity_id, action, after) "
                "VALUES (%s,%s,NULL,'assignment',%s,'start',%s)",
                (chosen[0][0], annotator_id, bid, json.dumps({"batch_id": bid, "seq": seqno, "size": len(chosen)})))
    return bid


def refresh_batch(cur, annotator_id: int, item_id: int) -> Optional[dict]:
    """Mark the item's batch completed as soon as nothing in it is still open and, right away, add the next batch
    ("finish 250, get the next 250"). -> {"completed": seq, "next_batch": id|None} when a batch just completed, else None."""
    cur.execute("SELECT batch_id FROM assignment WHERE item_id=%s AND annotator_id=%s", (item_id, annotator_id))
    r = cur.fetchone()
    if not r or not r["batch_id"]:
        return None
    cur.execute("UPDATE batch SET completed_at = now() WHERE batch_id=%s AND completed_at IS NULL AND NOT EXISTS "
                "(SELECT 1 FROM assignment WHERE batch_id=%s AND status IN ('assigned','in_progress')) RETURNING seq",
                (r["batch_id"], r["batch_id"]))
    done = cur.fetchone()
    if not done:
        return None
    nxt = None
    cur.execute("SAVEPOINT next_batch")                 # a failure here must never lose the annotator's submission:
    try:                                                # next_item() builds the batch lazily if this did not
        cur.execute("SELECT active FROM annotator WHERE annotator_id=%s", (annotator_id,))
        if cur.fetchone()["active"]:
            nxt = build_batch(cur, annotator_id)
        cur.execute("RELEASE SAVEPOINT next_batch")
    except Exception:                                   # noqa: BLE001
        cur.execute("ROLLBACK TO SAVEPOINT next_batch")
    return {"completed": done["seq"], "next_batch": nxt}


def release_unstarted(annotator_id: int) -> int:
    """Give an annotator's not-yet-opened items back to the pool (admin action, also used when disabling an account)."""
    with tx() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
        cur.execute("""DELETE FROM assignment a WHERE a.annotator_id=%s AND a.status='assigned'
                       AND NOT EXISTS (SELECT 1 FROM claim_span c WHERE c.item_id=a.item_id AND c.annotator_id=a.annotator_id)
                       AND NOT EXISTS (SELECT 1 FROM propaganda_span p WHERE p.item_id=a.item_id AND p.annotator_id=a.annotator_id)
                       RETURNING a.item_id, a.batch_id""", (annotator_id,))
        rows = cur.fetchall()
        for r in rows:
            cur.execute("""UPDATE item SET status = CASE WHEN (SELECT count(*) FROM assignment WHERE item_id=%s AND status='done') >=
                                                              CASE WHEN is_double THEN 2 ELSE 1 END THEN 'done'
                                                         WHEN EXISTS (SELECT 1 FROM assignment WHERE item_id=%s AND status IN ('assigned','in_progress')) THEN 'in_progress'
                                                         ELSE 'queued' END
                           WHERE item_id=%s AND status <> 'excluded'""", (r["item_id"], r["item_id"], r["item_id"]))
        for b in {r["batch_id"] for r in rows if r["batch_id"]}:
            cur.execute("UPDATE batch SET size = (SELECT count(*) FROM assignment WHERE batch_id=%s) WHERE batch_id=%s", (b, b))
        return len(rows)


# ------------------------------------------------------------------ serving
def next_item(annotator_id: int, rng: Optional[random.Random] = None) -> Optional[int]:
    """Resume the open item; else the next unopened item of the open batch; else (batch finished) build the next batch."""
    from .service import _event, _touch                  # late import: service imports this module
    with tx() as cur:
        cur.execute("SELECT annotator_id FROM annotator WHERE annotator_id=%s FOR UPDATE", (annotator_id,))   # one request at a time per user
        cur.execute("""SELECT item_id FROM assignment WHERE annotator_id=%s AND (status='in_progress' OR (status='assigned' AND batch_id IS NULL))
                       ORDER BY coalesce(batch_pos, 0), assignment_id LIMIT 1""", (annotator_id,))
        r = cur.fetchone()
        if r:
            _touch(cur, r["item_id"], annotator_id)
            return r["item_id"]
        cur.execute("SELECT batch_id FROM batch WHERE annotator_id=%s AND completed_at IS NULL ORDER BY seq DESC LIMIT 1", (annotator_id,))
        b = cur.fetchone()
        if b:
            cur.execute("SELECT item_id FROM assignment WHERE batch_id=%s AND status='assigned' ORDER BY batch_pos LIMIT 1", (b["batch_id"],))
            r = cur.fetchone()
            if r:
                _touch(cur, r["item_id"], annotator_id)
                return r["item_id"]
            cur.execute("UPDATE batch SET completed_at=now() WHERE batch_id=%s AND completed_at IS NULL", (b["batch_id"],))
        bid = build_batch(cur, annotator_id, rng)
        if not bid:
            return None
        cur.execute("SELECT item_id FROM assignment WHERE batch_id=%s ORDER BY batch_pos LIMIT 1", (bid,))
        iid = cur.fetchone()["item_id"]
        _touch(cur, iid, annotator_id)
        return iid
