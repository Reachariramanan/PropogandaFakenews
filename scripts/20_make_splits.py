"""Stage 20: frozen article-level splits for PTC (the only technique-labelled corpus we have).

sha1(article_id) % 10: 0-6 train, 7 dev, 8-9 test  (70/10/20). Deterministic, so new
articles added later land in a stable split. Writes data/splits/ptc_v1.json (write-once).

  .venv/bin/python scripts/20_make_splits.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fnp.datasets import load_ptc  # noqa: E402

OUT = ROOT / "data" / "splits" / "ptc_v1.json"


def bucket(aid: str) -> str:
    b = int(hashlib.sha1(aid.encode()).hexdigest(), 16) % 10
    return "train" if b <= 6 else "dev" if b == 7 else "test"


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT} exists; splits are write-once (make ptc_v2 to change).")
    arts, spans = load_ptc()
    split = {a: bucket(a) for a in arts.article_id}
    n_sp = Counter(split[a] for a in spans.article_id if a in split)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"version": "ptc_v1", "rule": "sha1(id)%10: 0-6 train,7 dev,8-9 test",
                               "split": split}, indent=0, sort_keys=True))
    print("articles", dict(Counter(split.values())), "spans", dict(n_sp))


if __name__ == "__main__":
    main()
