"""Read-only metrics over the annotation DB: progress towards the target, class / source / language / month balance,
team monitoring (who logged in, minutes worked, items finished) and per-user detail.

"Balanced" here means, per task (claims, propaganda):
  * class: every balanced class is present in at least  target_items * (1 - neg_frac) / n_classes  finished items
    (OTHER and UNCLEAR are catch-alls: counted, never steered towards);
  * explicit negatives: neg_frac of the target items carry a 'none' label;
  * source, language and month: the stratified queue (batching.py) keeps the cells level; the tables here show it.
Item-level counts use the first submitter of each (item, layer), so a double-annotated item is never counted twice.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from ..config import C
from .auth import TZ
from .db import tx

LAYERS = ("claim", "propaganda")
NO_TARGET = {"OTHER", "UNCLEAR"}          # claim-type catch-alls


# ------------------------------------------------------------------ targets + class counts
def _classes(cur) -> dict[str, list[str]]:
    cur.execute("SELECT code FROM taxonomy_claim_type WHERE active ORDER BY sort_order")
    ct = [r["code"] for r in cur.fetchall()]
    cur.execute("SELECT code FROM taxonomy_technique WHERE active ORDER BY sort_order")
    return {"claim": ct, "propaganda": [r["code"] for r in cur.fetchall()]}


def supply(cur) -> dict:
    """What the models say the remaining pool can supply, per class: items (queued / in progress, English or Hindi) with a
    predicted span of that class, scaled up by how much of the pool the model has seen so far.
    -> {"claim": {code: est_items}, "propaganda": {...}}"""
    langs = [g for g, k in (("en", "ann_share_en"), ("hi", "ann_share_hi"), ("other", "ann_share_other")) if C(k) > 0]
    cur.execute("SELECT count(*) n FROM item WHERE status IN ('queued','in_progress') AND lang_group = ANY(%s)", (langs,))
    pool = max(1, cur.fetchone()["n"])
    out: dict = {}
    for layer, like in (("claim", "gemma\\_%"), ("propaganda", "enc\\_%")):
        cur.execute("""SELECT count(*) n FROM item i WHERE i.status IN ('queued','in_progress') AND i.lang_group = ANY(%s)
                       AND EXISTS (SELECT 1 FROM suggestion_done d WHERE d.item_id=i.item_id AND d.model_version LIKE %s)""", (langs, like))
        cov = max(0.02, cur.fetchone()["n"] / pool)                      # fraction of the pool the model has processed
        cur.execute("""SELECT s.label, count(DISTINCT s.item_id) n FROM suggestion s JOIN item i USING (item_id)
                       WHERE s.layer=%s AND s.label IS NOT NULL AND i.status IN ('queued','in_progress') AND i.lang_group = ANY(%s)
                       GROUP BY 1""", (layer, langs))
        out[layer] = {r["label"]: r["n"] / cov for r in cur.fetchall()}
    return out


def feasible_targets(cur, t: Optional[dict] = None, cc: Optional[dict] = None, sup: Optional[dict] = None) -> dict:
    """Class targets capped at what the pool can plausibly deliver: min(target, finished + ann_target_supply_frac x predicted supply).
    A class the models never predict gets target = finished, i.e. no steering towards it. Imbalance that remains is accepted
    and handled by class weights in the export (see export.class_weights)."""
    t = t or targets(cur)
    cc = cc or class_counts(cur)
    sup = sup or supply(cur)
    frac = C("ann_target_supply_frac")
    out = {"items": t["items"], "neg": t["neg"]}
    for layer in LAYERS:
        out[layer] = {}
        for code, tg in t[layer].items():
            if not tg:
                out[layer][code] = tg
                continue
            have = cc[layer].get(code, 0)
            out[layer][code] = int(min(tg, have + frac * sup[layer].get(code, 0)))
    return out


def targets(cur) -> dict:
    total = C("ann_target_items")
    pos = total * (1 - C("ann_target_neg_frac"))
    cls = _classes(cur)
    bal_claim = [c for c in cls["claim"] if c not in NO_TARGET]
    return {"items": int(total), "neg": int(total * C("ann_target_neg_frac")),
            "claim": {c: (int(pos / len(bal_claim)) if c in bal_claim else None) for c in cls["claim"]},
            "propaganda": {c: int(pos / len(cls["propaganda"])) for c in cls["propaganda"]}}


FIRST = """first AS (SELECT DISTINCT ON (item_id, layer) item_id, layer, annotator_id, outcome
                     FROM layer_result ORDER BY item_id, layer, submitted_at, annotator_id)"""


def class_counts(cur) -> dict:
    """Finished items per class (first submitter only) and per explicit-negative outcome."""
    cur.execute(f"""WITH {FIRST}
        SELECT c.claim_type code, count(DISTINCT c.item_id) n FROM first f
        JOIN claim_span c ON c.item_id=f.item_id AND c.annotator_id=f.annotator_id AND c.deleted_at IS NULL
        WHERE f.layer='claim' AND f.outcome='annotated' GROUP BY 1""")
    claim = {r["code"]: r["n"] for r in cur.fetchall()}
    cur.execute(f"""WITH {FIRST}
        SELECT t.technique code, count(DISTINCT p.item_id) n FROM first f
        JOIN propaganda_span p ON p.item_id=f.item_id AND p.annotator_id=f.annotator_id AND p.deleted_at IS NULL
        JOIN propaganda_span_technique t USING (span_id)
        WHERE f.layer='propaganda' AND f.outcome='annotated' GROUP BY 1""")
    prop = {r["code"]: r["n"] for r in cur.fetchall()}
    cur.execute(f"WITH {FIRST} SELECT layer, outcome, count(*) n FROM first GROUP BY 1,2")
    lay = {(r["layer"], r["outcome"]): r["n"] for r in cur.fetchall()}
    return {"claim": claim, "propaganda": prop,
            "none": {l: lay.get((l, "none"), 0) for l in LAYERS},
            "annotated": {l: lay.get((l, "annotated"), 0) for l in LAYERS}}


def inflight_predictions(cur) -> dict:
    """Model-predicted classes of items already reserved in batches but not finished: they will soon count."""
    cur.execute("""SELECT s.layer, s.label, count(DISTINCT s.item_id) n FROM suggestion s
                   WHERE s.label IS NOT NULL AND s.item_id IN (SELECT item_id FROM assignment WHERE status IN ('assigned','in_progress'))
                   GROUP BY 1,2""")
    out = {"claim": {}, "propaganda": {}}
    for r in cur.fetchall():
        out[r["layer"]][r["label"]] = r["n"]
    return out


# ------------------------------------------------------------------ coverage report (admin)
def _share_rows(rows: list[dict], key: str) -> list[dict]:
    tot = sum(r["done"] for r in rows) or 1
    for r in rows:
        r["share_done"] = round(r["done"] / tot, 4)
    return rows


def coverage() -> dict:
    with tx() as cur:
        t = targets(cur)
        cc = class_counts(cur)
        cur.execute("SELECT status, count(*) n FROM item GROUP BY 1")
        st = {r["status"]: r["n"] for r in cur.fetchall()}
        langs = [g for g, k in (("en", "ann_share_en"), ("hi", "ann_share_hi"), ("other", "ann_share_other")) if C(k) > 0]
        cur.execute("""WITH taken AS (SELECT item_id, count(*) n FROM assignment WHERE status IN ('assigned','in_progress','done') GROUP BY 1)
                       SELECT i.lang_group, count(*) items,
                              count(*) FILTER (WHERE i.status IN ('queued','in_progress') AND coalesce(t.n,0) = 0) unassigned
                       FROM item i LEFT JOIN taken t USING (item_id) WHERE i.status <> 'excluded' GROUP BY 1 ORDER BY 1""")
        pool = [dict(r) for r in cur.fetchall() if r["lang_group"] in langs]
        cur.execute("""SELECT count(*) FILTER (WHERE EXISTS (SELECT 1 FROM suggestion_done d WHERE d.item_id=i.item_id AND d.model_version LIKE 'enc\\_%%')) enc,
                              count(*) FILTER (WHERE EXISTS (SELECT 1 FROM suggestion_done d WHERE d.item_id=i.item_id AND d.model_version LIKE 'gemma\\_%%')) gemma,
                              count(*) n
                       FROM item i WHERE i.status IN ('queued','in_progress') AND i.lang_group = ANY(%s)""", (langs,))
        pre = dict(cur.fetchone())
        done = st.get("done", 0)
        cur.execute("""SELECT s.platform, s.weight, count(*) FILTER (WHERE i.status='done') done,
                              count(*) FILTER (WHERE i.status <> 'excluded') items
                       FROM item i JOIN ingest_source s USING (source_id) GROUP BY 1,2 ORDER BY 3 DESC, 1""")
        by_platform = _share_rows([dict(r) for r in cur.fetchall()], "platform")
        cur.execute("SELECT lang_group, count(*) FILTER (WHERE status='done') done, count(*) FILTER (WHERE status<>'excluded') items "
                    "FROM item GROUP BY 1 ORDER BY 1")
        by_lang = _share_rows([dict(r) for r in cur.fetchall()], "lang_group")
        cur.execute("SELECT period, count(*) FILTER (WHERE status='done') done, count(*) FILTER (WHERE status<>'excluded') items "
                    "FROM item GROUP BY 1 ORDER BY 1 DESC LIMIT 36")
        by_period = _share_rows([dict(r) for r in cur.fetchall()], "period")
        cur.execute("""SELECT count(*) FILTER (WHERE a.finished_at >= now() - interval '7 days' AND a.status='done') done7
                       FROM assignment a""")
        done7 = cur.fetchone()["done7"]
        cur.execute("SELECT count(*) m FROM activity_minute WHERE minute >= now() - interval '7 days'")
        min7 = cur.fetchone()["m"]
        fl = inflight_predictions(cur)
        ft = feasible_targets(cur, t, cc)
        cur.execute("SELECT split_part(excluded_reason, ':', 1) reason, count(*) n FROM item WHERE status='excluded' AND excluded_reason IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")
        excluded = {r["reason"]: r["n"] for r in cur.fetchall()}
        cur.execute("SELECT selected_by, count(*) n FROM assignment WHERE kind='normal' GROUP BY 1")
        selection = {(r["selected_by"] or "unknown"): r["n"] for r in cur.fetchall()}
    rate = done7 / (min7 / 60) if min7 >= 60 else None
    remaining = max(0, t["items"] - done)
    out = {"target_items": t["items"], "target_neg": t["neg"], "done_items": done, "items_by_status": st, "pool": pool,
           "pre_annotated": pre, "by_platform": by_platform, "by_lang": by_lang, "by_period": by_period,
           "items_per_active_hour_7d": round(rate, 1) if rate else None,
           "annotator_hours_remaining": round(remaining / rate, 0) if rate else None, "tasks": {},
           "excluded_by_reason": excluded, "selection": selection}
    for layer in LAYERS:
        rows = []
        for code, tg in t[layer].items():
            n = cc[layer].get(code, 0)
            rows.append({"code": code, "target": tg, "feasible": ft[layer].get(code), "done": n, "in_flight_predicted": fl[layer].get(code, 0),
                         "pct": round(100 * n / tg, 1) if tg else None})
        out["tasks"][layer] = {"classes": rows, "annotated_items": cc["annotated"][layer], "none_items": cc["none"][layer],
                               "none_pct_of_finished": round(100 * cc["none"][layer] / max(1, cc["none"][layer] + cc["annotated"][layer]), 1)}
    return out


# ------------------------------------------------------------------ team monitoring
def _since(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days) if days and days > 0 else datetime(1970, 1, 1, tzinfo=timezone.utc)


def team(days: int = 30) -> dict:
    since = _since(days)
    with tx() as cur:
        cur.execute("""
        SELECT a.annotator_id, a.username, a.display_name, a.role, a.active, a.created_at, a.last_login_at, a.last_seen_at,
               a.must_change_password, a.locked_until > now() AS locked, a.batch_size,
               EXISTS (SELECT 1 FROM auth_session s WHERE s.annotator_id=a.annotator_id AND s.ended_at IS NULL
                       AND s.expires_at > now() AND s.last_seen_at > now() - interval '2 minutes') AS online,
               (SELECT count(*) FROM auth_session s WHERE s.annotator_id=a.annotator_id AND s.created_at >= %(since)s) AS logins,
               (SELECT coalesce(sum(extract(epoch FROM (coalesce(s.ended_at, s.last_seen_at) - s.created_at))), 0)
                  FROM auth_session s WHERE s.annotator_id=a.annotator_id AND s.created_at >= %(since)s) AS logged_in_seconds,
               (SELECT count(*) FROM activity_minute m WHERE m.annotator_id=a.annotator_id AND m.minute >= %(since)s) AS active_minutes,
               (SELECT count(*) FROM activity_minute m WHERE m.annotator_id=a.annotator_id
                   AND (m.minute AT TIME ZONE %(tz)s)::date = (now() AT TIME ZONE %(tz)s)::date) AS active_minutes_today,
               (SELECT count(*) FROM assignment x WHERE x.annotator_id=a.annotator_id AND x.status='done' AND x.finished_at >= %(since)s) AS items_done,
               (SELECT count(*) FROM assignment x WHERE x.annotator_id=a.annotator_id AND x.status='done'
                   AND (x.finished_at AT TIME ZONE %(tz)s)::date = (now() AT TIME ZONE %(tz)s)::date) AS items_done_today,
               (SELECT count(*) FROM assignment x WHERE x.annotator_id=a.annotator_id AND x.status='done') AS items_done_total,
               (SELECT count(*) FROM assignment x WHERE x.annotator_id=a.annotator_id AND x.status='skipped' AND x.finished_at >= %(since)s) AS skipped,
               (SELECT count(*) FROM claim_span c WHERE c.annotator_id=a.annotator_id AND c.deleted_at IS NULL AND c.created_at >= %(since)s) AS claim_spans,
               (SELECT count(*) FROM propaganda_span p WHERE p.annotator_id=a.annotator_id AND p.deleted_at IS NULL AND p.created_at >= %(since)s) AS propaganda_spans,
               (SELECT count(*) FROM layer_result r WHERE r.annotator_id=a.annotator_id AND r.outcome='none' AND r.submitted_at >= %(since)s) AS none_labels,
               (SELECT count(*) FROM label_event e WHERE e.annotator_id=a.annotator_id AND e.ts >= %(since)s AND e.action IN ('create','edit','delete')) AS label_edits,
               (SELECT count(*) FROM batch b WHERE b.annotator_id=a.annotator_id AND b.completed_at IS NOT NULL) AS batches_completed,
               (SELECT row_to_json(q) FROM (SELECT b.seq, b.size,
                        (SELECT count(*) FROM assignment x WHERE x.batch_id=b.batch_id AND x.status IN ('done','skipped')) AS finished
                      FROM batch b WHERE b.annotator_id=a.annotator_id AND b.completed_at IS NULL ORDER BY b.seq DESC LIMIT 1) q) AS batch
        FROM annotator a ORDER BY a.active DESC, a.role, a.username""", {"since": since, "tz": TZ})
        rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["logged_in_seconds"] = int(r["logged_in_seconds"])
        hrs = r["active_minutes"] / 60
        r["items_per_active_hour"] = round(r["items_done"] / hrs, 1) if hrs >= 0.25 else None
        r["avg_seconds_per_item"] = round(r["active_minutes"] * 60 / r["items_done"]) if r["items_done"] else None
    tot = {k: sum(r[k] for r in rows) for k in ("logins", "active_minutes", "items_done", "claim_spans", "propaganda_spans", "skipped")}
    tot["online"] = sum(1 for r in rows if r["online"])
    return {"days": days, "tz": TZ, "annotators": rows, "totals": tot}


def user_detail(annotator_id: int, days: int = 30) -> dict:
    since = _since(days)
    with tx() as cur:
        cur.execute("SELECT annotator_id, username, display_name, role, active, created_at, last_login_at, must_change_password, "
                    "failed_logins, locked_until, batch_size FROM annotator WHERE annotator_id=%s", (annotator_id,))
        a = cur.fetchone()
        if not a:
            return {}
        cur.execute("""SELECT s.session_id, s.created_at, s.last_seen_at, s.ended_at, s.end_reason, s.ip, left(s.user_agent, 80) ua,
                              extract(epoch FROM (coalesce(s.ended_at, s.last_seen_at) - s.created_at))::int AS seconds,
                              (SELECT count(*) FROM activity_minute m WHERE m.session_id=s.session_id) AS active_minutes,
                              (s.ended_at IS NULL AND s.expires_at > now() AND s.last_seen_at > now() - interval '2 minutes') AS online
                       FROM auth_session s WHERE s.annotator_id=%s AND s.created_at >= %s ORDER BY s.created_at DESC LIMIT 60""",
                    (annotator_id, since))
        sessions = [dict(r) for r in cur.fetchall()]
        day = lambda col: f"(({col}) AT TIME ZONE %(tz)s)::date"
        p = {"a": annotator_id, "since": since, "tz": TZ}
        series: dict = {}

        def add(sql, key):
            cur.execute(sql, p)
            for r in cur.fetchall():
                series.setdefault(r["d"], {"day": r["d"]})[key] = r["n"]
        add(f"SELECT {day('minute')} d, count(*) n FROM activity_minute WHERE annotator_id=%(a)s AND minute >= %(since)s GROUP BY 1", "active_minutes")
        add(f"SELECT {day('finished_at')} d, count(*) n FROM assignment WHERE annotator_id=%(a)s AND status='done' AND finished_at >= %(since)s GROUP BY 1", "items_done")
        add(f"SELECT {day('created_at')} d, count(*) n FROM claim_span WHERE annotator_id=%(a)s AND deleted_at IS NULL AND created_at >= %(since)s GROUP BY 1", "claim_spans")
        add(f"SELECT {day('created_at')} d, count(*) n FROM propaganda_span WHERE annotator_id=%(a)s AND deleted_at IS NULL AND created_at >= %(since)s GROUP BY 1", "propaganda_spans")
        add(f"SELECT {day('created_at')} d, count(*) n FROM auth_session WHERE annotator_id=%(a)s AND created_at >= %(since)s GROUP BY 1", "logins")
        cur.execute("SELECT claim_type code, count(*) n FROM claim_span WHERE annotator_id=%s AND deleted_at IS NULL GROUP BY 1 ORDER BY 2 DESC", (annotator_id,))
        claim_types = [dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT t.technique code, count(*) n FROM propaganda_span p JOIN propaganda_span_technique t USING (span_id)
                       WHERE p.annotator_id=%s AND p.deleted_at IS NULL GROUP BY 1 ORDER BY 2 DESC""", (annotator_id,))
        techniques = [dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT b.seq, b.size, b.created_at, b.completed_at,
                              count(*) FILTER (WHERE x.status='done') done, count(*) FILTER (WHERE x.status='skipped') skipped
                       FROM batch b LEFT JOIN assignment x USING (batch_id) WHERE b.annotator_id=%s GROUP BY b.batch_id ORDER BY b.seq DESC LIMIT 20""",
                    (annotator_id,))
        batches = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT ts, outcome, ip, detail FROM login_event WHERE annotator_id=%s ORDER BY ts DESC LIMIT 25", (annotator_id,))
        events = [dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT skip_reason, count(*) n FROM assignment WHERE annotator_id=%s AND status='skipped' GROUP BY 1 ORDER BY 2 DESC""", (annotator_id,))
        skips = [dict(r) for r in cur.fetchall()]
    return {"annotator": dict(a), "sessions": sessions, "daily": [series[k] for k in sorted(series, reverse=True)],
            "claim_types": claim_types, "techniques": techniques, "batches": batches, "login_events": events, "skip_reasons": skips,
            "days": days, "tz": TZ}


def login_events(limit: int = 100, only_problems: bool = False) -> list[dict]:
    with tx() as cur:
        cur.execute("""SELECT ts, username, outcome, ip, detail FROM login_event
                       WHERE (NOT %s OR outcome IN ('bad_password','unknown_user','locked','disabled','no_password'))
                       ORDER BY ts DESC LIMIT %s""", (only_problems, limit))
        return [dict(r) for r in cur.fetchall()]


def my_progress(annotator_id: int) -> dict:
    """What an annotator sees about themselves: current batch, totals, today's minutes."""
    with tx() as cur:
        cur.execute("""SELECT b.batch_id, b.seq, b.size,
                              (SELECT count(*) FROM assignment x WHERE x.batch_id=b.batch_id AND x.status IN ('done','skipped')) AS finished
                       FROM batch b WHERE b.annotator_id=%s AND b.completed_at IS NULL ORDER BY b.seq DESC LIMIT 1""", (annotator_id,))
        b = cur.fetchone()
        cur.execute("""SELECT count(*) FILTER (WHERE status='done') done, count(*) FILTER (WHERE status='skipped') skipped,
                              count(*) FILTER (WHERE status='done' AND (finished_at AT TIME ZONE %(tz)s)::date = (now() AT TIME ZONE %(tz)s)::date) done_today
                       FROM assignment WHERE annotator_id=%(a)s""", {"tz": TZ, "a": annotator_id})
        t = dict(cur.fetchone())
        cur.execute("SELECT count(*) n FROM batch WHERE annotator_id=%s AND completed_at IS NOT NULL", (annotator_id,))
        t["batches_completed"] = cur.fetchone()["n"]
        cur.execute("""SELECT count(*) FILTER (WHERE (minute AT TIME ZONE %(tz)s)::date = (now() AT TIME ZONE %(tz)s)::date) today, count(*) total
                       FROM activity_minute WHERE annotator_id=%(a)s""", {"tz": TZ, "a": annotator_id})
        m = dict(cur.fetchone())
        cur.execute("SELECT count(*) n FROM item WHERE status='done'")
        team_done = cur.fetchone()["n"]
    return {"batch": dict(b) if b else None, **t, "minutes_today": m["today"], "minutes_total": m["total"],
            "team_done": team_done, "target_items": int(C("ann_target_items"))}
