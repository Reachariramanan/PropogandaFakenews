#!/usr/bin/env python3
"""Merge fact-check JSONL sources, score items, build TF-IDF index -> Interface/build/."""
import glob
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fnlib  # noqa: E402

BUILD = os.path.join(HERE, "build")
DUCK = os.path.join(os.path.dirname(HERE), "data", "fnp.duckdb")
JSONL = os.path.join(os.path.dirname(HERE), "reports", "verdict_v1_docs.jsonl")
LABEL_SCORE = {"FAKE": 0.95, "MISLEADING": 0.70, "INCONCLUSIVE": 0.50, "LEGIT": 0.05}
BODY_MAX = 2500


def norm_url(u):
    try:
        p = urlsplit((u or "").strip())
        return (p.netloc.lower().removeprefix("www.") + p.path.rstrip("/")).lower()
    except Exception:
        return (u or "").strip().lower()


# Script-based language repair. Some sources tag Hindi (and other Indic) titles as English or leave the tag empty, so when the
# existing tag is en / und / empty and more than half of the letters belong to one script, the script's language is used.
# Devanagari is labelled hi (it is shared with Marathi and Nepali), Bengali bn (shared with Assamese), Arabic script ur.
SCRIPT_LANG = (
    ("hi", ((0x0900, 0x097F),)), ("bn", ((0x0980, 0x09FF),)), ("gu", ((0x0A80, 0x0AFF),)), ("ta", ((0x0B80, 0x0BFF),)),
    ("te", ((0x0C00, 0x0C7F),)), ("kn", ((0x0C80, 0x0CFF),)),
    ("ur", ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
)
SCRIPT_SHARE = 0.5


def script_lang(text, current=None):
    """The language implied by the script of `text`, or `current` unchanged. Only overrides en-*, und and empty tags."""
    base = str(current or "").strip().lower().replace("_", "-").split("-")[0]
    if base not in ("", "en", "und"):
        return current
    letters, counts = 0, Counter()
    for ch in text or "":
        if not ch.isalpha():
            continue
        letters += 1
        o = ord(ch)
        for code, ranges in SCRIPT_LANG:
            if any(lo <= o <= hi for lo, hi in ranges):
                counts[code] += 1
                break
    if letters:
        code, n = counts.most_common(1)[0] if counts else (None, 0)
        if code and n / letters > SCRIPT_SHARE:
            return code
    return current


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    if isinstance(d, dict):
                        out.append(d)
                except Exception:
                    pass
    return out


def parse_date(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except Exception:
        return None


def to_float(x):
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else None
    except Exception:
        return None


def label_score(verdict, rating):
    base = LABEL_SCORE.get(verdict)
    if base is None:
        return None
    r = to_float(rating)
    # numeric 1..5 ratings (1 = false): mild nudge, only when direction agrees with the verdict
    if r is not None and 1 <= r <= 5:
        rf = (5 - r) / 4
        agrees = (verdict in ("FAKE", "MISLEADING") and rf >= 0.5) or (verdict == "LEGIT" and rf <= 0.5) \
            or verdict == "INCONCLUSIVE"
        if agrees:
            base = 0.85 * base + 0.15 * rf
    return round(min(max(base, 0.0), 1.0), 4)


def load_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "factchecks", "*.jsonl"))):
        for d in read_jsonl(p):
            d["_kind"] = "factcheck"
            rows.append(d)
    p = os.path.join(HERE, "data", "factchecks_3y.jsonl")
    if os.path.exists(p):
        for d in read_jsonl(p):
            d["_kind"] = "factcheck"
            rows.append(d)
    # fact-checkers' own social-media accounts (Instagram / Telegram / WhatsApp)
    p = os.path.join(HERE, "data", "factcheckers.jsonl")
    if os.path.exists(p):
        for d in read_jsonl(p):
            if str(d.get("is_fact_check", "True")).lower() == "false":
                continue
            if d.get("site") == "pibfactcheck":
                d["site"] = "pib_factcheck"
            v = (d.get("verdict_norm") or "").upper()
            v = "LEGIT" if v == "TRUE" else v
            d["verdict_norm"] = v
            d["channel"] = (d.get("platform") or "social").lower()
            d["_kind"] = "factcheck" if v in LABEL_SCORE and v != "INCONCLUSIVE" else "candidate"
            rows.append(d)
    p = os.path.join(HERE, "data", "db_factchecks.jsonl")
    if os.path.exists(p):
        for d in read_jsonl(p):
            d["_kind"] = "factcheck"
            rows.append(d)
    p = os.path.join(HERE, "data", "db_claim_candidates.jsonl")
    if os.path.exists(p):
        for d in read_jsonl(p):
            d["_kind"] = "candidate"
            d.setdefault("claim", d.get("text"))
            d.setdefault("site", d.get("platform") or "candidate")
            rows.append(d)
    return rows


def load_engine():
    """Return ({doc_id: engine dict}, status). Never raises."""
    status = {"available": False, "rows": 0, "matched": 0, "note": "engine scores pending"}
    recs = None
    if os.path.exists(JSONL):
        try:
            recs = read_jsonl(JSONL)
        except Exception:
            recs = None
    if recs is None:
        if not os.path.exists(DUCK):
            return {}, status
        try:
            import duckdb
        except Exception:
            status["note"] = "engine scores pending (duckdb module not installed)"
            return {}, status
        tmp = None
        try:
            try:
                con = duckdb.connect(DUCK, read_only=True)
            except Exception:
                tmp = tempfile.mkdtemp()
                cp = os.path.join(tmp, "fnp.duckdb")
                shutil.copy(DUCK, cp)
                con = duckdb.connect(cp, read_only=True)
            cur = con.execute("SELECT * FROM verdict_v1")
            cols = [c[0] for c in cur.description]
            recs = [dict(zip(cols, r)) for r in cur.fetchall()]
            con.close()
        except Exception:
            return {}, status
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    def js(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return None
        return v

    doc_rows, claim_rows = {}, defaultdict(list)
    for r in recs:
        did = r.get("doc_id") or r.get("unit_id")
        if r.get("unit") == "claim" and r.get("doc_id"):
            claim_rows[r["doc_id"]].append(r)
        else:
            doc_rows[did] = r
    out = {}

    def fake_p(r):
        probs = js(r.get("probs")) or {}
        if not probs:
            return None, probs  # no engine probabilities: no score, not 0.0
        try:
            return float(probs.get("FAKE", 0)) + float(probs.get("MISLEADING", 0)), probs
        except Exception:
            return None, probs

    for did in set(doc_rows) | set(claim_rows):
        r = doc_rows.get(did)
        level = "doc"
        if r is None:  # fall back to the most suspicious claim-level row
            cands = [(fake_p(c)[0] or 0, c) for c in claim_rows[did]]
            r = max(cands, key=lambda x: x[0])[1]
            level = "claim"
        p, probs = fake_p(r)
        if p is None:
            continue
        out[did] = {
            "level": level, "label": r.get("label"), "p_fake_or_misleading": round(p, 4),
            "probs": probs, "plausibly_fake": r.get("plausibly_fake"),
            "secondary": r.get("secondary"), "prediction_set": js(r.get("prediction_set")),
            "features": js(r.get("features")), "decider": r.get("decider"),
            "rationale": r.get("rationale"), "evidence_ids": js(r.get("evidence_ids")),
            "claims_scored": len(claim_rows.get(did, [])),
        }
    status.update(available=True, rows=len(recs), note="engine scores loaded")
    return out, status


def main():
    os.makedirs(BUILD, exist_ok=True)
    rows = load_rows()
    # ---- dedupe by url (prefer richer row)
    by_url = {}
    for d in rows:
        key = norm_url(d.get("url")) or ("id:" + str(d.get("id")))
        old = by_url.get(key)
        rich = lambda x: (x["_kind"] == "factcheck", len(x.get("body") or ""), len(x.get("claim") or ""))
        if old is None or rich(d) > rich(old):
            by_url[key] = d
    items, seen_ids = [], set()
    for i, d in enumerate(by_url.values()):
        iid = str(d.get("id") or d.get("url") or f"row{i}")
        if iid in seen_ids:
            iid = f"{iid}#{i}"
        seen_ids.add(iid)
        kind = d["_kind"]
        verdict = (d.get("verdict_norm") or "").upper() if kind == "factcheck" else "UNRATED"
        if kind == "factcheck" and verdict not in LABEL_SCORE:
            verdict = "INCONCLUSIVE"
        body = (d.get("body") or "")[:BODY_MAX]
        claim = (d.get("claim") or "").strip()
        title = (d.get("title") or "").strip()
        if kind == "candidate" and not title:
            title = claim[:110]
        pub = parse_date(d.get("published"))
        items.append({
            "id": iid, "kind": kind, "site": (d.get("site") or "unknown").lower(), "url": d.get("url"),
            "claim": claim, "title": title, "verdict_raw": d.get("verdict_raw"), "verdict_norm": verdict,
            "rating_value": d.get("rating_value"), "published": pub.isoformat() if pub else None,
            "lang": script_lang((title + " " + claim).strip(), d.get("lang") or "und"), "description": (d.get("description") or "").strip(),
            "body": body,
            "engagement": d.get("engagement"),
            "channel": d.get("channel") or (str(d.get("platform") or "").removesuffix("_posts") if d["_kind"] == "candidate" and d.get("platform") else "web"),
            "org": d.get("org") or None, "country": d.get("country") or None,
            "source_tier": d.get("source_tier") or None,
            "claim_basis": d.get("claim_basis") or None, "verdict_basis": d.get("verdict_basis") or None,
            "categories": (d.get("categories") if isinstance(d.get("categories"), list) else [])[:5],
            "claimant": (d.get("claimant") or None), "claim_date": (d.get("claim_date") or None),
            "author": d.get("author") or None,
            "_pub": pub,
        })

    # ---- retrieval index over factchecks (claim + title)
    fc = [it for it in items if it["kind"] == "factcheck"]
    texts = [(it["title"] + " . " + it["claim"]).strip() for it in fc]
    index = fnlib.TfidfIndex().fit(texts)
    nbrs = index.neighbors(k=5, thr=0.30) if fc else []
    now = datetime.now(timezone.utc)
    DUP = 0.5

    engine, engine_status = load_engine()
    matched = 0
    pos = {it["id"]: n for n, it in enumerate(fc)}
    for it in items:
        s = fnlib.signals(it["claim"] or it["title"])
        it["claim_len"] = s["chars"]
        it["caps_ratio"] = s["caps_ratio"]
        it["exclamation_ratio"] = s["exclamation_ratio"]
        it["sensational"] = s["sensational"]
        age = (now - it["_pub"]).days if it["_pub"] else None
        it["age_days"] = age
        it["recency"] = round(math.exp(-max(age, 0) / 180), 4) if age is not None else None
        if it["id"] in pos:
            nb = nbrs[pos[it["id"]]]
            self_sites = {it["site"]}
        else:
            txt = (it["title"] + " . " + it["claim"]).strip()
            nb = [(j, sc) for j, sc in index.query(txt, 5)]
            self_sites = set()
        dup = [(fc[j], sc) for j, sc in nb if sc >= DUP and fc[j]["id"] != it["id"]]
        sites = sorted(self_sites | {m["site"] for m, _ in dup})
        it["cross_checker_agreement"] = len(sites)
        it["cross_checker_sites"] = sites
        it["cross_checker_verdicts"] = dict(Counter(m["verdict_norm"] for m, _ in dup))
        it["neighbors"] = [{"id": fc[j]["id"], "site": fc[j]["site"], "title": fc[j]["title"],
                            "verdict_norm": fc[j]["verdict_norm"], "similarity": round(sc, 3)}
                           for j, sc in nb if sc >= 0.35 and fc[j]["id"] != it["id"]][:3]
        # ---- plausibility
        lab = label_score(it["verdict_norm"], it["rating_value"]) if it["kind"] == "factcheck" else None
        # verdicts inferred from categories/tags/titles (not ClaimReview) are weaker evidence: shrink toward 0.5
        if lab is not None and it.get("verdict_basis") and it["verdict_basis"] != "claimreview":
            lab = round(0.5 + 0.6 * (lab - 0.5), 4)
        eng = engine.get(it["id"])
        it["engine"] = eng
        if eng:
            matched += 1
        if lab is not None and eng:
            it["plaus"], it["score_source"] = round(0.5 * lab + 0.5 * eng["p_fake_or_misleading"], 4), "blend"
        elif eng:
            it["plaus"], it["score_source"] = eng["p_fake_or_misleading"], "engine"
        elif lab is not None:
            it["plaus"], it["score_source"] = lab, "fact-checker-label"
        else:
            it["plaus"], it["score_source"] = None, "none"
        it["label_score"] = lab
    engine_status["matched"] = matched

    # ---- per-site verdict mix
    mix = defaultdict(Counter)
    for it in fc:
        mix[it["site"]][it["verdict_norm"]] += 1
    site_mix = {s: dict(c) for s, c in mix.items()}
    for it in items:
        c = mix.get(it["site"])
        it["site_fake_rate"] = round((c["FAKE"] + c["MISLEADING"]) / sum(c.values()), 3) if c else None
        it.pop("_pub", None)

    items.sort(key=lambda x: (x["plaus"] is None, -(x["plaus"] or 0), x["published"] or ""))
    meta = {"built": now.isoformat(), "n_items": len(items), "n_factchecks": len(fc),
            "n_candidates": len(items) - len(fc), "rows_read": len(rows),
            "sources": sorted({it["site"] for it in items}), "site_verdict_mix": site_mix,
            "engine": engine_status, "tfidf_backend": index.backend,
            "index_ids": [it["id"] for it in fc]}
    with open(os.path.join(BUILD, "items.json.tmp"), "w", encoding="utf-8") as f:
        json.dump(fnlib.clean_json({"meta": meta, "items": items}), f, ensure_ascii=False, allow_nan=False)
    os.replace(os.path.join(BUILD, "items.json.tmp"), os.path.join(BUILD, "items.json"))
    with open(os.path.join(BUILD, "index.pkl.tmp"), "wb") as f:
        pickle.dump(index, f, protocol=4)
    os.replace(os.path.join(BUILD, "index.pkl.tmp"), os.path.join(BUILD, "index.pkl"))
    print(f"indexed {len(items)} items ({len(fc)} fact-checks, {len(items) - len(fc)} candidates) "
          f"from {len(rows)} rows; backend={index.backend}; engine={engine_status['note']}")


if __name__ == "__main__":
    main()
