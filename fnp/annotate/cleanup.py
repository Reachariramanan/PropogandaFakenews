"""Pool hygiene: near-duplicate and link-only items.

  * backfill item.body_key (the near-duplicate key of the body text, see sources.body_key) for older rows;
  * per body_key group keep ONE item and set the others to status='excluded' (excluded_reason='near_duplicate_of:<id>');
  * set link-/mention-only items to excluded (excluded_reason='low_substance').
Items that already have any assignment are never touched (they are in somebody's batch or labelled); among the rest the
kept item is the one with a model pass (more useful for batch steering), then the earliest ingested. Nothing is deleted.
Excluded items keep their body_key, so the same repost is not ingested again.

  .venv/bin/python scripts/33_annot_dedupe.py            # dry run
  .venv/bin/python scripts/33_annot_dedupe.py --apply
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..config import C
from . import sources as S
from .db import tx


def backfill_body_keys(batch: int = 5000) -> int:
    n = 0
    while True:
        with tx() as cur:
            cur.execute("SELECT item_id, text, title_len FROM item WHERE body_key IS NULL ORDER BY item_id LIMIT %s", (batch,))
            rows = cur.fetchall()
            if not rows:
                return n
            for r in rows:
                cur.execute("UPDATE item SET body_key=%s WHERE item_id=%s", (S.body_key(r["text"], r["title_len"]), r["item_id"]))
            n += len(rows)


def plan() -> dict:
    """-> {"duplicates": [(drop_id, keep_id)], "low_substance": [id], "report": {...}} for items not yet excluded."""
    with tx() as cur:
        cur.execute("""SELECT i.item_id, i.body_key, i.text, i.title_len, s.platform,
                              EXISTS (SELECT 1 FROM assignment a WHERE a.item_id=i.item_id) AS used,
                              EXISTS (SELECT 1 FROM suggestion_done d WHERE d.item_id=i.item_id AND d.model_version LIKE 'gemma\\_%%') AS gemma
                       FROM item i JOIN ingest_source s USING (source_id)
                       WHERE i.status <> 'excluded' AND i.body_key IS NOT NULL""")
        rows = cur.fetchall()
    lo = C("ann_min_substance_chars")
    low = [r["item_id"] for r in rows if not r["used"] and S.substance(r["text"][r["title_len"]:]) < lo]
    low_set = set(low)
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["item_id"] not in low_set:
            groups[r["body_key"]].append(r)
    dups, by_platform, sizes = [], Counter(), Counter()
    for g in groups.values():
        if len(g) < 2:
            continue
        sizes[len(g)] += 1
        used = [r for r in g if r["used"]]
        keep = used[0] if used else min(g, key=lambda r: (not r["gemma"], r["item_id"]))
        for r in g:
            if r["used"] or r["item_id"] == keep["item_id"]:
                continue
            dups.append((r["item_id"], keep["item_id"]))
            by_platform[r["platform"]] += 1
    return {"duplicates": dups, "low_substance": low,
            "report": {"items_scanned": len(rows), "duplicate_groups": sum(sizes.values()), "to_exclude_duplicates": len(dups),
                       "to_exclude_low_substance": len(low), "largest_group": max(sizes) if sizes else 0,
                       "duplicates_by_platform": dict(by_platform.most_common())}}


def apply(p: dict) -> dict:
    with tx() as cur:
        for drop, keep in p["duplicates"]:
            cur.execute("UPDATE item SET status='excluded', excluded_reason=%s WHERE item_id=%s AND status <> 'excluded' "
                        "AND NOT EXISTS (SELECT 1 FROM assignment a WHERE a.item_id=item.item_id)", (f"near_duplicate_of:{keep}", drop))
        cur.execute("UPDATE item SET status='excluded', excluded_reason='low_substance' WHERE item_id = ANY(%s) AND status <> 'excluded' "
                    "AND NOT EXISTS (SELECT 1 FROM assignment a WHERE a.item_id=item.item_id)", (p["low_substance"],))
    return {"excluded": len(p["duplicates"]) + len(p["low_substance"])}
