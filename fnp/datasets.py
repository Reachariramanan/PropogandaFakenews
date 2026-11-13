"""Loaders for the public gold datasets and the local BOOM fact-checks.

Downloaded data is untrusted: we only parse it (csv/tsv/json/txt), never execute it.
All loaders return DataFrames with GOLD_COLS (or the PTC article/span frames).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .config import BOOM_DIR, DATASETS_DIR, FNP_ROOT

FACTCHECK_DIR = FNP_ROOT / "Interface" / "factchecks"

GOLD_COLS = ["item_id", "dataset", "split", "text", "claim", "lang", "label5",
             "raw_label", "speaker", "date", "url", "meta_json"]

LIAR_MAP = {"pants-fire": "FAKE", "false": "FAKE", "barely-true": "MISLEADING",
            "half-true": "MISLEADING", "mostly-true": "LEGIT", "true": "LEGIT"}
XFACT_MAP = {"false": "FAKE", "mostly false": "MISLEADING", "partly true/misleading": "MISLEADING",
             "mostly true": "LEGIT", "true": "LEGIT",
             "complicated/hard to categorise": "INCONCLUSIVE", "other": "INCONCLUSIVE"}
FNN_MAP = {"fake": "FAKE", "real": "LEGIT"}
BOOM_MAP = {"false": "FAKE", "misleading": "MISLEADING"}
QPROP_MAP = {"propaganda": "PROPAGANDA"}  # anything else -> LEGIT (no QProp data fetched yet)


def map_label(mapping: dict, raw) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return mapping.get(str(raw).strip().lower())


def qprop_label(raw) -> str:
    return QPROP_MAP.get(str(raw).strip().lower(), "LEGIT")


def ptc_doc_label(n_spans: int, n_tokens: int) -> str | None:
    """PROPAGANDA if >=3 spans or density >=1 span/100 tokens; else None (unlabelled-for-doc)."""
    if n_spans >= 3 or (n_tokens > 0 and 100.0 * n_spans / n_tokens >= 1.0):
        return "PROPAGANDA"
    return None


def boom_split(item_id) -> str:
    """Deterministic 60/20/20 by sha1(id)."""
    b = int(hashlib.sha1(str(item_id).encode()).hexdigest(), 16) % 100
    return "train" if b < 60 else ("dev" if b < 80 else "test")


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=GOLD_COLS)
    return df


def _js(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, default=str)


def _nn(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


# ---------------------------------------------------------------------------
def load_liar(root: Path | None = None) -> pd.DataFrame:
    root = Path(root or DATASETS_DIR / "liar" / "x")
    cols = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
            "c_barely", "c_false", "c_half", "c_mostly", "c_pants", "context"]
    rows = []
    for fn, split in (("train.tsv", "train"), ("valid.tsv", "dev"), ("test.tsv", "test")):
        p = root / fn
        if not p.exists():
            continue
        d = pd.read_csv(p, sep="\t", header=None, names=cols, quoting=3, dtype=str,
                        keep_default_na=False)
        for r in d.itertuples():
            rows.append(dict(
                item_id=f"liar:{r.id.replace('.json', '')}", dataset="liar", split=split,
                text=r.statement, claim=r.statement, lang="en", label5=map_label(LIAR_MAP, r.label),
                raw_label=r.label, speaker=r.speaker or None, date=None, url=None,
                meta_json=_js(dict(subject=r.subject, job=r.job, state=r.state, party=r.party,
                                   context=r.context))))
    return _frame(rows)


def load_xfact(root: Path | None = None) -> pd.DataFrame:
    root = Path(root or DATASETS_DIR / "xfact")
    rows = []
    for fn, split in (("train.all.tsv", "train"), ("dev.all.tsv", "dev"), ("test.all.tsv", "test")):
        p = root / fn
        if not p.exists():
            continue
        d = pd.read_csv(p, sep="\t", quoting=3, on_bad_lines="skip", dtype=str, keep_default_na=False)
        for i, r in enumerate(d.itertuples()):
            lab = map_label(XFACT_MAP, r.label)
            if lab is None:
                continue
            rows.append(dict(
                item_id=f"xfact:{split}:{i}", dataset="xfact", split=split, text=r.claim,
                claim=r.claim, lang=r.language, label5=lab, raw_label=r.label,
                speaker=r.claimant or None, date=(None if r.claimDate in ("", "none") else r.claimDate),
                url=r.link_1 or None,
                meta_json=_js(dict(site=r.site, review_date=r.reviewDate,
                                   inconclusive_flag=(lab == "INCONCLUSIVE")))))
    return _frame(rows)


def load_fakenewsnet(root: Path | None = None) -> pd.DataFrame:
    root = Path(root or DATASETS_DIR / "fnn")
    rows = []
    for name, lab in (("politifact_fake", "fake"), ("politifact_real", "real")):
        p = root / f"{name}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype=str, keep_default_na=False)
        for r in d.itertuples():
            if not r.title.strip():
                continue
            rows.append(dict(
                item_id=f"fnn:{r.id}", dataset="fakenewsnet", split="all", text=r.title,
                claim=None, lang="en", label5=map_label(FNN_MAP, lab), raw_label=lab,
                speaker=None, date=None, url=r.news_url or None,
                meta_json=_js(dict(n_tweets=len([t for t in r.tweet_ids.split("\t") if t])))))
    return _frame(rows)


def load_boom(path: Path | None = None) -> pd.DataFrame:
    """BOOM fact-checks. Input text is ONLY `claim` (title fallback). `text`/body is the
    debunk and leaks the label, so it is never read. Missing/unmapped verdict -> skipped."""
    path = Path(path or BOOM_DIR / "articles_full.json")
    if not path.exists():
        return _frame([])
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for a in data:
        lab = map_label(BOOM_MAP, a.get("verdict"))
        if lab is None:
            continue
        claim = (a.get("claim") or "").strip() or (a.get("title") or "").strip()
        if not claim:
            continue
        iid = str(a["id"])
        rows.append(dict(
            item_id=f"boomlive:{iid}", dataset="boomlive", split=boom_split(iid), text=claim, claim=claim,
            lang="en", label5=lab, raw_label=a.get("verdict"), speaker=None,
            date=a.get("published"), url=a.get("url"),
            meta_json=_js(dict(title=a.get("title"), rating=a.get("rating")))))
    return _frame(rows)


def load_factchecks(folder: Path | None = None) -> pd.DataFrame:
    """Glob Interface/factchecks/*.jsonl (read-only). Input text = `claim` only; `body` and
    `description` are debunk text and go to meta_json['body'/'description'] as evidence
    passages. label5 = verdict_norm. 60/20/20 split by sha1(id); deduped by id. If
    boomlive.jsonl is absent, falls back to articles_full.json (load_boom)."""
    use_fallback = folder is None
    folder = Path(folder or FACTCHECK_DIR)
    rows, seen = [], set()
    files = sorted(folder.glob("*.jsonl")) if folder.is_dir() else []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                a = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = str(a.get("id") or "")
            lab = (a.get("verdict_norm") or "").upper() or None
            claim = (a.get("claim") or "").strip() or (a.get("title") or "").strip()
            if not iid or iid in seen or lab not in ("FAKE", "MISLEADING", "LEGIT", "INCONCLUSIVE") or not claim:
                continue
            seen.add(iid)
            site = a.get("site") or iid.split(":")[0]
            rows.append(dict(
                item_id=iid, dataset=site, split=boom_split(iid), text=claim, claim=claim,
                lang=a.get("lang") or "en", label5=lab, raw_label=a.get("verdict_raw"),
                speaker=None, date=a.get("published"), url=a.get("url"),
                meta_json=_js(dict(title=a.get("title"), rating=a.get("rating_value"),
                                   description=a.get("description"),
                                   body=(a.get("body") or "")[:4000]))))
    if use_fallback and not any(f.stem == "boomlive" for f in files):
        b = load_boom()
        rows += [r for r in b.to_dict("records") if r["item_id"] not in seen]
    return _frame(rows)


# ---------------------------------------------------------------------------
# PTC / SemEval-2020 T11
def load_ptc(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (articles[article_id, split, text], spans[article_id, split, start, end, technique]).

    Only the train split has span labels in the mirror we could reach (dev labels were
    withheld by the organisers' mirror); dev articles are skipped as they cannot be scored.
    Article text is read with newline='' so character offsets match the label files.
    """
    root = Path(root or DATASETS_DIR / "ptc" / "vrdn" / "datasets")
    arts, spans = [], []
    lab_dir = root / "train-labels-task2-technique-classification"
    art_dir = root / "train-articles"
    if not art_dir.is_dir():
        return (pd.DataFrame(columns=["article_id", "split", "text"]),
                pd.DataFrame(columns=["article_id", "split", "start", "end", "technique"]))
    for p in sorted(art_dir.glob("article*.txt")):
        aid = re.sub(r"^article", "", p.stem)
        with open(p, encoding="utf-8", newline="") as f:
            arts.append(dict(article_id=aid, split="train", text=f.read()))
    known = {a["article_id"] for a in arts}
    for p in sorted(lab_dir.glob("*.labels")):
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) != 4 or parts[0] not in known:
                continue
            try:
                s, e = int(parts[2]), int(parts[3])
            except ValueError:
                continue
            spans.append(dict(article_id=parts[0], split="train", start=s, end=e, technique=parts[1]))
    return pd.DataFrame(arts), pd.DataFrame(spans)


def ptc_gold(articles: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    n = spans.groupby("article_id").size().to_dict() if len(spans) else {}
    rows = []
    for r in articles.itertuples():
        k = n.get(r.article_id, 0)
        lab = ptc_doc_label(k, len(r.text.split()))
        rows.append(dict(item_id=f"ptc:{r.article_id}", dataset="ptc", split=r.split, text=r.text,
                         claim=None, lang="en", label5=lab, raw_label=f"{k}_spans", speaker=None,
                         date=None, url=None, meta_json=_js(dict(n_spans=k))))
    return _frame(rows)


def load_all_gold() -> pd.DataFrame:
    arts, spans = load_ptc()
    parts = [load_liar(), load_xfact(), load_fakenewsnet(), load_factchecks(), ptc_gold(arts, spans)]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)
