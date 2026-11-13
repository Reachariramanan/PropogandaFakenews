"""Tiny fact-check corpus + builders shared by the build_index and server tests. Everything lives in a temp dir."""
import contextlib
import io
import json
import os
import pickle
from unittest import mock

import helpers  # noqa: F401
import build_index

ROWS = [
    {"id": "fc1", "site": "BoomLive", "url": "https://www.boomlive.in/fact-check/flood-video/", "claim":
     "Viral video shows Delhi flood this week", "verdict_raw": "False", "verdict_norm": "FAKE", "rating_value": "1",
     "published": "2026-10-01T10:00:00Z", "lang": "en", "title": "Old flood video is not from Delhi",
     "body": "The footage is from 2018.", "description": "Old footage shared as new."},
    {"id": "fc2", "site": "altnews", "url": "https://altnews.in/free-electricity", "claim":
     "PM announces free electricity for all households", "verdict_raw": "Misleading", "verdict_norm": "MISLEADING",
     "rating_value": "", "published": "2026-09-15", "lang": "en", "title": "Free electricity claim is misleading",
     "body": "Only a limited subsidy exists.", "description": ""},
    {"id": "fc3", "site": "factly", "url": "https://factly.in/highway", "claim":
     "New highway opens in Kerala", "verdict_raw": "True", "verdict_norm": "LEGIT", "rating_value": "5",
     "published": "2026-08-01T00:00:00+05:30", "lang": "en", "title": "Kerala highway opening is true",
     "body": "Confirmed by the ministry.", "description": ""},
    {"id": "fc4", "site": "boomlive", "url": "https://boomlive.in/hindi-claim", "claim":
     "दिल्ली में बाढ़ का पुराना वीडियो", "verdict_raw": "", "verdict_norm": "BOGUS",
     "published": "not-a-date", "lang": "hi", "title": "पुराना वीडियो", "body": "", "description": ""},
    # same URL as fc1 once normalised (www, trailing slash, query): the longer body must win
    {"id": "fc1-dup", "site": "boomlive", "url": "https://boomlive.in/fact-check/flood-video?utm=x", "claim":
     "Viral video shows Delhi flood this week", "verdict_raw": "False", "verdict_norm": "FAKE",
     "published": "2026-10-01", "lang": "en", "title": "Old flood video is not from Delhi",
     "body": "The footage is from 2018. " * 5, "description": ""},
]
CANDIDATES = [
    {"id": "c1", "platform": "x_posts", "url": "https://x.com/u/status/1", "text":
     "Shocking: free electricity for everyone, share now!", "published": "2026-10-20T00:00:00Z", "lang": "en"},
]
FACTCHECKERS = [
    {"id": "f1", "site": "pibfactcheck", "platform": "Instagram", "url": "https://instagram.com/p/1", "claim":
     "Fake notice about bank holiday", "verdict_norm": "FAKE", "published": "2026-10-10T00:00:00Z", "lang": "en"},
    {"id": "f2", "site": "x", "platform": "telegram", "url": "https://t.me/x/2", "claim": "not a fact check",
     "is_fact_check": "False", "verdict_norm": "FAKE"},
    {"id": "f3", "site": "y", "platform": "whatsapp", "url": "https://wa.me/3", "claim": "A true statement",
     "verdict_norm": "TRUE", "published": "2026-10-11T00:00:00Z", "lang": "en"},
]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_inputs(root):
    """Lay out root/factchecks, root/data like the real Interface/ input contract."""
    write_jsonl(os.path.join(root, "factchecks", "a.jsonl"), ROWS)
    write_jsonl(os.path.join(root, "data", "db_claim_candidates.jsonl"), CANDIDATES)
    write_jsonl(os.path.join(root, "data", "factcheckers.jsonl"), FACTCHECKERS)


def patched_build(root, engine_jsonl=None):
    """Patch build_index globals so nothing outside `root` is read or written."""
    return mock.patch.multiple(
        build_index, HERE=root, BUILD=os.path.join(root, "build"),
        DUCK=os.path.join(root, "absent.duckdb"), JSONL=engine_jsonl or os.path.join(root, "absent.jsonl"))


def run_build(root, engine_jsonl=None):
    """Run build_index.main() against `root`; returns (items, meta, index)."""
    with patched_build(root, engine_jsonl), contextlib.redirect_stdout(io.StringIO()):
        build_index.main()
    bdir = os.path.join(root, "build")
    with open(os.path.join(bdir, "items.json"), encoding="utf-8") as f:
        d = json.load(f)
    with open(os.path.join(bdir, "index.pkl"), "rb") as f:
        idx = pickle.load(f)
    return d["items"], d["meta"], idx
