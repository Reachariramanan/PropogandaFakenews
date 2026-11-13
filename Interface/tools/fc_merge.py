#!/usr/bin/env python3
"""Merge Interface/data/fc_shards/*.jsonl into Interface/data/factchecks_3y.jsonl (dedupe by id, then by url; keep the row with the longest body)."""
import json, glob
from pathlib import Path
D = Path(__file__).resolve().parent.parent / 'data'
best = {}
for f in sorted(glob.glob(str(D / 'fc_shards' / '*.jsonl'))):
    for l in open(f, encoding='utf-8'):
        try: r = json.loads(l)
        except Exception: continue
        for k in (r['id'], r['url'].rstrip('/')):
            if k in best and len(best[k]['body']) >= len(r['body']): break
        else:
            best[r['id']] = r; continue
        if k == r['id'] and k not in best: best[k] = r
uniq = {}
for r in best.values():
    u = r['url'].rstrip('/')
    if u not in uniq or len(uniq[u]['body']) < len(r['body']): uniq[u] = r
rows = sorted(uniq.values(), key=lambda r: (r['site'], r['published']), reverse=False)
with open(D / 'factchecks_3y.jsonl', 'w', encoding='utf-8') as o:
    for r in rows: o.write(json.dumps(r, ensure_ascii=False) + '\n')
print(len(rows), 'rows ->', D / 'factchecks_3y.jsonl')
