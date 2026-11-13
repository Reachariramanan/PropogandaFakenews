"""Stage 31: Propaganda Lab aggregates -> reports/prop_lab/v1/ (the contract read by Evidence Desk).

Inputs : data/prop_lab/v1/parts/*.parquet (stage 30), data/prop_lab/v1/gemma_sample.jsonl (stage 33,
         optional), EDA geo_ontology (topics, entities; read-only), data/geo/* (gazetteer, registry,
         world paths).
Outputs: meta.json, world_paths.json, countries.json, heatmap.json, country/<ISO3>.json,
         posts/<mode>/<ISO3>.jsonl   -- strict JSON (no NaN/Infinity). Schema: docs/PROP_LAB_CONTRACT.md
Rules (also written into meta.json):
  signal  = not too_short and >= 1 strong span (mean token prob >= max(class thr, 0.5))
  debunk  = nearest kb_v2 fact-check is FAKE/MISLEADING, bge-reranker >= 0.8 and cosine >= 0.72
  windows = 7d / 30d / all, relative to the newest post in the data (not wall-clock)
  lift    = empirical-Bayes (Beta-binomial, method of moments) shrunk rate / global rate
  insufficient = n < 30 posts
  .venv/bin/python scripts/31_prop_lab_aggregate.py [--parts DIR] [--out DIR] [--sample N]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fnp.config import DATA_DIR, FNP_ROOT, REPORTS_DIR  # noqa: E402
from fnp.eda_bridge import open_eda_db  # noqa: E402
from fnp.geo.countries import GAZ_PATH  # noqa: E402
from fnp.geo.origin import load_registry, origin_of  # noqa: E402
from fnp.math.calibration import wilson_ci  # noqa: E402

PARTS = DATA_DIR / "prop_lab" / "v1" / "parts"
GEMMA_VERDICT = DATA_DIR / "prop_lab" / "v1" / "gemma_verdict.jsonl"
GEMMA_SPANS = DATA_DIR / "prop_lab" / "v1" / "gemma_spans.jsonl"
OUT = REPORTS_DIR / "prop_lab" / "v1"
WORLD = DATA_DIR / "geo" / "world_paths_v1.json"
GEO_DB = FNP_ROOT.parent / "EDA" / "data" / "geo_ontology.duckdb"
TECH = ['Loaded_Language', 'Name_Calling,Labeling', 'Repetition', 'Exaggeration,Minimisation', 'Doubt',
        'Appeal_to_fear-prejudice', 'Flag-Waving', 'Causal_Oversimplification', 'Slogans',
        'Appeal_to_Authority', 'Black-and-White_Fallacy', 'Thought-terminating_Cliches',
        'Whataboutism,Straw_Men,Red_Herring', 'Bandwagon,Reductio_ad_hitlerum']
WINDOWS = {"7d": 7, "30d": 30, "all": None}
SOURCES = ("social", "news", "all")
MODES = ("about", "origin")
MIN_N = 30
LIFT_PSEUDO = 10.0
DEBUNK_RR, DEBUNK_COS = 0.8, 0.72
POSTS_PER_COUNTRY = 150
UNL = "UNL"


def finite(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.floating,)):
        return finite(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(obj), ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def r4(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 4)


# ----------------------------------------------------------------- load
def load_posts(parts: Path, sample: int | None) -> pd.DataFrame:
    fs = sorted(glob.glob(str(parts / "*.parquet")))
    if not fs:
        raise SystemExit(f"no parts in {parts}")
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df = df.drop_duplicates("post_uid", keep="first")
    if sample:
        df = df.head(sample)
    df["posted"] = pd.to_datetime(df.posted_at, utc=True, errors="coerce")
    df["signal"] = (~df.too_short) & (df.n_strong >= 1)
    df["debunk"] = (df.kb_label.isin(["FAKE", "MISLEADING"]) & (df.kb_rerank.fillna(0) >= DEBUNK_RR)
                    & (df.kb_cos >= DEBUNK_COS))
    df["strong_set"] = df.strong_techniques.map(json.loads)
    df["about"] = df.countries.map(lambda s: [(c["iso3"], c["weight"]) for c in json.loads(s)])
    reg = load_registry()
    df["origin"] = [([(o[0], 1.0)] if (o := origin_of(p, a, reg)) else [])
                    for p, a in zip(df.platform, df.author_name.fillna(""))]
    def last_ok(path):
        out = {}
        for line in path.open() if path.exists() else []:
            r = json.loads(line)
            if r.get("ok"):
                out[r["post_uid"]] = r
        return out
    gv, gs = last_ok(GEMMA_VERDICT), last_ok(GEMMA_SPANS)
    df["gemma"] = df.post_uid.map(gv)                 # verdict (fake/misleading layer)
    df["gspans"] = df.post_uid.map(gs)                # Gemma span check (encoder agreement)
    return df


NARR = DATA_DIR / "prop_lab" / "v1" / "narratives.parquet"


def load_narratives() -> dict:
    """(mode, iso3) -> [cluster dict] from stage 34 (empty if not built)."""
    if not NARR.exists():
        return {}
    out = {}
    for r in pd.read_parquet(NARR).itertuples():
        out.setdefault((r.mode, r.iso3), []).append(
            {"cluster": int(r.cluster), "name": r.name or "", "summary": r.summary or "", "terms": r.terms,
             "cohesion": float(r.cohesion), "post_uids": set(r.post_uids), "exemplar_uids": list(r.exemplar_uids)})
    return out


def eda_context(uids: set) -> pd.DataFrame:
    """Named entities per post from EDA geo_ontology (read-only; covers posts up to EDA's last run)."""
    try:
        con = open_eda_db(GEO_DB)
        ent = con.execute("SELECT m.post_uid, d.canonical_name AS name, m.entity_type AS type "
                          "FROM entity_mentions m JOIN entity_dim d USING (entity_uid)").df()
        con.close()
    except Exception as e:                                   # EDA busy/missing: context is optional
        print(f"EDA context unavailable: {type(e).__name__}: {e}", flush=True)
        return pd.DataFrame(columns=["post_uid", "name", "type"])
    return ent[ent.post_uid.isin(uids)]


# ----------------------------------------------------------------- stats
def eb_prior(k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """Beta(a, b) by method of moments over units with n >= MIN_N (binomial noise removed)."""
    m = n >= MIN_N
    if m.sum() < 3:
        return 1.0, 1.0
    p = k[m] / n[m]
    mu = float(np.average(p, weights=n[m]))
    var = float(np.average((p - mu) ** 2, weights=n[m])) - mu * (1 - mu) * float(np.mean(1 / n[m]))
    if var <= 1e-6:
        s = 1000.0
    else:
        s = max(mu * (1 - mu) / var - 1, 2.0)
    return mu * s, (1 - mu) * s


def tech_profile(sub: pd.DataFrame) -> dict:
    sig = sub[sub.signal]
    c = Counter(t for s in sig.strong_set for t in s)
    n = max(len(sig), 1)
    return {t: c.get(t, 0) / n for t in TECH}, {t: c.get(t, 0) for t in TECH}


def explode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    e = df[["post_uid", mode]].explode(mode)
    e["iso3"] = e[mode].map(lambda x: x[0] if isinstance(x, tuple) else UNL)
    e["w"] = e[mode].map(lambda x: x[1] if isinstance(x, tuple) else 1.0)
    return e[["post_uid", "iso3", "w"]]


def window_mask(df: pd.DataFrame, w: str, end: pd.Timestamp) -> pd.Series:
    d = WINDOWS[w]
    return pd.Series(True, index=df.index) if d is None else (df.posted >= end - pd.Timedelta(days=d))


def country_table(df: pd.DataFrame, ex: pd.DataFrame, glob_prof: dict) -> tuple[dict, dict]:
    """Per-country stats. `lift` is LENGTH-ADJUSTED: observed signal posts / expected given each
    post's token-length decile (long posts both mention more countries and carry more spans),
    shrunk toward 1 with LIFT_PSEUDO expected posts. `lift_raw` is the EB-shrunk rate / global."""
    if len(df):
        dec = pd.qcut(df.n_tokens.rank(method="first"), q=min(10, len(df)), labels=False)
        p_len = df.groupby(dec).signal.transform("mean")
        exp_map = dict(zip(df.post_uid, p_len))
    else:
        exp_map = {}
    j = ex.merge(df[["post_uid", "signal", "debunk", "strong_set", "gemma", "gspans"]], on="post_uid")
    j["expected"] = j.post_uid.map(exp_map)
    gk, gn = int(df.signal.sum()), len(df)
    g_rate = gk / gn if gn else None
    agg = j.groupby("iso3").agg(n=("post_uid", "nunique"), n_w=("w", "sum"), k=("signal", "sum"),
                                d=("debunk", "sum"), e=("expected", "sum"))
    a, b = eb_prior(agg.k.values.astype(float), agg.n.values.astype(float))
    out = {}
    for iso, r in agg.iterrows():
        n, k, d = int(r.n), int(r.k), int(r.d)
        lo, hi = wilson_ci(k, n)
        dlo, dhi = wilson_ci(d, n)
        sub = j[j.iso3 == iso]
        prof, cnt = tech_profile(sub)
        top = max(TECH, key=lambda t: cnt[t]) if any(cnt.values()) else None
        ver = [g for g in sub.gemma if isinstance(g, dict)]
        out[iso] = {"n": n, "n_weighted": r4(r.n_w), "n_signal": k, "rate": r4(k / n), "ci": [r4(lo), r4(hi)],
                    "lift": r4((k + LIFT_PSEUDO) / (float(r.e) + LIFT_PSEUDO)),
                    "expected_signal": r4(r.e),
                    "lift_raw": r4(((k + a) / (n + a + b)) / g_rate) if g_rate else None,
                    "insufficient": n < MIN_N, "top_technique": top,
                    "top_technique_ratio": r4(prof[top] / glob_prof[top]) if top and glob_prof.get(top) else None,
                    "debunk_n": d, "debunk_rate": r4(d / n), "debunk_ci": [r4(dlo), r4(dhi)],
                    "verified_n": len(ver),
                    "verified_fake": sum(g.get("label") == "FAKE" for g in ver),
                    "verified_misleading": sum(g.get("label") == "MISLEADING" for g in ver),
                    "verified_plausibly_fake": sum(bool(g.get("plausibly_fake")) for g in ver),
                    **agreement_stats(sub.gspans)}
    glo = {"n": gn, "n_signal": gk, "rate": r4(g_rate) if g_rate is not None else None,
           "ci": [r4(x) for x in wilson_ci(gk, gn)], "debunk_n": int(df.debunk.sum()),
           "debunk_rate": r4(df.debunk.mean()) if gn else None, "eb_prior": [r4(a), r4(b)]}
    return out, glo


def agreement_stats(gs) -> dict:
    """Encoder vs Gemma on the per-country span sample: share of sampled signal posts Gemma also
    flags, and mean share of encoder spans overlapping a Gemma span."""
    a = [g["agreement"] for g in gs if isinstance(g, dict) and g.get("agreement", {}).get("encoder_flags")]
    if not a:
        return {"agreement_n": 0, "agreement_post": None, "agreement_span": None}
    ov = [x["overlap"] for x in a if x.get("overlap") is not None]
    return {"agreement_n": len(a), "agreement_post": r4(sum(x["gemma_flags"] for x in a) / len(a)),
            "agreement_span": r4(sum(ov) / len(ov)) if ov else None}


# ----------------------------------------------------------------- post records
def post_record(r, mode_isos=None) -> dict:
    spans = [s for s in json.loads(r.spans) if s[2] in r.strong_set][:12]
    g = r.gemma if isinstance(r.gemma, dict) else None
    gs = r.gspans if isinstance(r.gspans, dict) else None
    return {"post_uid": r.post_uid, "source": r.source, "platform": r.platform,
            "url": r.url if isinstance(r.url, str) and r.url.startswith("http") else None,
            "author": r.author_name if isinstance(r.author_name, str) else None,
            "posted_at": r.posted.isoformat() if pd.notna(r.posted) else None,
            "lang": r.lang if isinstance(r.lang, str) and r.lang else "und",
            "text": r.text[:900], "spans": spans, "techniques": sorted(r.strong_set),
            "signal": bool(r.signal), "density": r4(r.density), "n_strong": int(r.n_strong),
            "about": [c[0] for c in r.about], "origin": [c[0] for c in r.origin],
            "debunk": ({"kb_claim": r.kb_claim, "kb_label": r.kb_label, "kb_url": r.kb_url,
                        "rerank": r4(r.kb_rerank), "cos": r4(r.kb_cos)} if r.debunk else None),
            "verified": ({"label": g.get("label"), "probs": g.get("probs"), "fake_score": g.get("fake_score"),
                          "basis": g.get("basis"), "plausibly_fake": g.get("plausibly_fake"),
                          "rationale": (g.get("rationale") or "")[:400]} if g else None),
            "gemma_check": ({"gemma_spans": gs.get("gemma_spans", [])[:12], "agreement": gs.get("agreement")}
                            if gs else None)}


def post_score(r) -> float:
    return (2.0 if isinstance(r.gemma, dict) else 0) + (1.0 if isinstance(r.gspans, dict) else 0) \
        + (1.5 if r.debunk else 0) \
        + min(r.n_strong, 6) * 0.3 + min(r.density, 20) * 0.02


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=str(PARTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sample", type=int, help="use only the newest N posts (sample build)")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    tmp = out.with_name(out.name + ".building")
    shutil.rmtree(tmp, ignore_errors=True)
    df = load_posts(Path(a.parts), a.sample)
    end = df.posted.max()
    print(f"{len(df)} posts through {end}", flush=True)
    ent = eda_context(set(df.post_uid))
    narr = load_narratives()
    gaz = json.loads(GAZ_PATH.read_text())["countries"]
    names = {k: v["name"] for k, v in gaz.items()}
    world = json.loads(WORLD.read_text())
    for k, v in world["countries"].items():
        names.setdefault(k, v["name"])
    names[UNL] = "Unlocated"

    exp = {m: explode(df, m) for m in MODES}
    countries, heat, globals_ = {}, {}, {}
    for m in MODES:
        countries[m], heat[m], globals_[m] = {}, {}, {}
        for w in WINDOWS:
            countries[m][w], heat[m][w], globals_[m][w] = {}, {}, {}
            wm = window_mask(df, w, end)
            for s in SOURCES:
                sub = df[wm & ((df.source == s) if s != "all" else True)]
                gp, gc = tech_profile(sub)
                ex = exp[m][exp[m].post_uid.isin(set(sub.post_uid))]
                tab, glo = country_table(sub, ex, gp)
                glo["technique_share"] = {t: r4(v) for t, v in gp.items()}
                glo["located"] = int(ex[ex.iso3 != UNL].post_uid.nunique())
                countries[m][w][s], globals_[m][w][s] = tab, glo
                rows = sorted([i for i, r in tab.items() if i != UNL and r["n"] >= MIN_N],
                              key=lambda i: -tab[i]["n"])
                share, ratio = [], []
                for i in rows:
                    pr, _ = tech_profile(sub[sub.post_uid.isin(set(ex[ex.iso3 == i].post_uid))])
                    share.append([r4(pr[t]) for t in TECH])
                    ratio.append([r4(pr[t] / gp[t]) if gp[t] else None for t in TECH])
                heat[m][w][s] = {"countries": rows, "share": share, "ratio": ratio}
        print(f"mode {m} aggregated {time.time()-t0:.0f}s", flush=True)

    # ---------------- country detail + posts
    dfi = df.set_index("post_uid", drop=False)
    dfi.index.name = None              # keep post_uid as a column only (merge keys)
    detail_isos = {i for m in MODES for i, r in countries[m]["all"]["all"].items() if r["n"] >= 5}
    week0 = end - pd.Timedelta(weeks=26)
    for iso in sorted(detail_isos):
        det = {"iso3": iso, "name": names.get(iso, iso), "modes": {}}
        for m in MODES:
            ids = set(exp[m][exp[m].iso3 == iso].post_uid)
            if not ids:
                continue
            P = dfi.loc[list(ids)]
            det["modes"][m] = {}
            for w in WINDOWS:
                Pw = P[window_mask(P, w, end)]
                if Pw.empty:
                    continue
                gp = {t: globals_[m][w]["all"]["technique_share"][t] for t in TECH}
                pr, cnt = tech_profile(Pw)
                block = {"kpis": {s: countries[m][w][s].get(iso) for s in SOURCES},
                         "techniques": [{"technique": t, "n": cnt[t], "share": r4(pr[t]), "global_share": gp[t],
                                         "ratio": r4(pr[t] / gp[t]) if gp[t] else None} for t in TECH],
                         "platforms": Pw.platform.value_counts().head(8).to_dict(),
                         "languages": Pw.lang.fillna("und").value_counts().head(8).to_dict()}
                wids = set(Pw.post_uid)
                nar, covered = [], set()
                for c in narr.get((m, iso), []):
                    mem = c["post_uids"] & wids
                    if len(mem) < 5:
                        continue
                    covered |= mem
                    sp = Pw.loc[list(mem)]
                    _, tc = tech_profile(sp)
                    sig_ids = sp[sp.signal].sort_values("n_strong", ascending=False).post_uid.tolist()
                    ex_ids = (sig_ids[:2] or [u for u in c["exemplar_uids"] if u in mem][:2])
                    n_unique = int(sp.text.str.slice(0, 120).str.casefold().str.replace(r"\W+", " ", regex=True)
                                   .str.strip().nunique())
                    nar.append({"cluster": c["cluster"], "name": c["name"] or c["terms"], "summary": c["summary"],
                                "terms": c["terms"], "cohesion": r4(c["cohesion"]), "n": len(mem),
                                "n_unique": n_unique, "templated": n_unique < 0.3 * len(mem),
                                "n_signal": int(sp.signal.sum()), "rate": r4(sp.signal.mean()),
                                "top_technique": max(TECH, key=lambda t: tc[t]) if any(tc.values()) else None,
                                "exemplars": [post_record(dfi.loc[x]) for x in ex_ids]})
                # rank by signal rate x log(unique texts): template floods (one account posting the same
                # boilerplate thousands of times) must not top the list
                nar.sort(key=lambda x: -(x["rate"] or 0) * math.log1p(x["n_unique"]))
                block["narratives"] = nar[:12]
                block["narratives_coverage"] = r4(len(covered) / len(Pw))
                en = ent[ent.post_uid.isin(set(Pw[Pw.signal].post_uid))]
                en = en[~en.name.str.casefold().isin({names.get(iso, "").casefold()})]
                block["entities"] = [{"name": n, "type": t, "n": int(c)} for (n, t), c in
                                     en.groupby(["name", "type"]).size().sort_values(ascending=False).head(12).items()]
                deb = Pw[Pw.debunk].sort_values("kb_rerank", ascending=False)
                ver = [g for g in Pw.gemma if isinstance(g, dict)]
                block["fake_layer"] = {
                    "debunk_n": int(Pw.debunk.sum()), "n": len(Pw),
                    "matches": [{"post_uid": r.post_uid, "text": r.text[:300], "kb_claim": r.kb_claim,
                                 "kb_label": r.kb_label, "kb_url": r.kb_url, "rerank": r4(r.kb_rerank)}
                                for r in deb.head(10).itertuples()],
                    "verified": {"n": len(ver), **Counter(g.get("label") for g in ver)},
                    "verified_posts": [post_record(dfi.loc[g["post_uid"]]) for g in ver
                                       if g.get("label") in ("FAKE", "MISLEADING") or g.get("plausibly_fake")][:10]}
                if w == "all":
                    tw = Pw[Pw.posted >= week0].copy()
                    tw["week"] = tw.posted.dt.tz_localize(None).dt.to_period("W").dt.start_time
                    tr = tw.groupby("week").agg(n=("post_uid", "size"), k=("signal", "sum"))
                    block["trend"] = [{"week": str(i.date()), "n": int(r.n), "n_signal": int(r.k),
                                       "rate": r4(r.k / r.n)} for i, r in tr.iterrows()]
                det["modes"][m][w] = block
            # posts file (all windows; UI filters by posted_at / technique)
            Pp = P[P.signal | P.debunk | P.gemma.map(lambda g: isinstance(g, dict))
                   | P.gspans.map(lambda g: isinstance(g, dict))]
            Pp = Pp.assign(_s=[post_score(r) for r in Pp.itertuples()]).sort_values("_s", ascending=False)
            f = tmp / "posts" / m / f"{iso}.jsonl"
            f.parent.mkdir(parents=True, exist_ok=True)
            with f.open("w") as fh:
                for r in Pp.head(POSTS_PER_COUNTRY).itertuples():
                    fh.write(json.dumps(finite(post_record(r)), ensure_ascii=False, allow_nan=False) + "\n")
        dump(det, tmp / "country" / f"{iso}.json")
    print(f"details {len(detail_isos)} countries {time.time()-t0:.0f}s", flush=True)

    meta = {"version": "prop_lab_v1", "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_through": end.isoformat() if pd.notna(end) else None,
            "n_posts": int((df.source == "social").sum()), "n_articles": int((df.source == "news").sum()),
            "n_located_about": int(df.about.map(bool).sum()), "n_located_origin": int(df.origin.map(bool).sum()),
            "n_gemma_verified": int(df.gemma.map(lambda g: isinstance(g, dict)).sum()),
            "n_gemma_span_checked": int(df.gspans.map(lambda g: isinstance(g, dict)).sum()),
            "windows": list(WINDOWS), "sources": list(SOURCES), "modes": list(MODES), "techniques": TECH,
            "min_n": MIN_N, "unlocated_key": UNL, "names": {k: names.get(k, k) for k in
                                                           set(names) & ({UNL} | {i for m in MODES for i in countries[m]["all"]["all"]})},
            "global": globals_,
            "method": {
                "engine": "span encoder prop_span_v1 (mmBERT/Laya multilingual, 14 PTC classes; PTC test "
                          "span+technique F1 0.268); Gemma chunk_v2 + FNP verdict on a per-country sample",
                "signal_rule": "post >= 12 tokens with >= 1 strong span (mean prob >= max(class threshold, 0.5))",
                "debunk_rule": f"nearest kb_v2 fact-check is FAKE/MISLEADING, reranker >= {DEBUNK_RR}, cosine >= {DEBUNK_COS}",
                "about": "countries named in the text (gazetteer: names, demonyms, capitals, cities >= 1M, "
                         "multilingual aliases, flag emoji); k countries -> weight 1/k each",
                "origin": "publisher/channel country from a reviewed channel registry (social) or outlet/ccTLD "
                          "(news); never from language",
                "lift": "length-adjusted: (observed signal posts + 10) / (expected given token-length decile + 10); "
                        "lift_raw = empirical-Bayes shrunk rate / global rate",
                "narratives": "per-country spherical k-means on bge-m3 post vectors (stage 34), c-TF-IDF terms, "
                              "Gemma-written neutral names",
                "windows": "relative to data_through, not wall-clock"},
            "caveats": ["Propaganda SIGNAL, not a verdict: the encoder is not validated on social media and "
                        "over-fires on short or emotive posts; compare countries by lift, not raw rate.",
                        "Origin is known only for registered broadcast channels; most posts are unlocated in origin mode.",
                        "News articles: 0 until shaktidb_news is reachable."]}
    dump(meta, tmp / "meta.json")
    dump({"countries": countries}, tmp / "countries.json")
    dump({"techniques": TECH, "views": heat}, tmp / "heatmap.json")
    shutil.copy(WORLD, tmp / "world_paths.json")
    if out.exists():
        shutil.rmtree(out.with_name(out.name + ".prev"), ignore_errors=True)
        out.rename(out.with_name(out.name + ".prev"))
    tmp.rename(out)
    print(f"wrote {out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
