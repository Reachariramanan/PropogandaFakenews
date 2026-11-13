"""Stage 34: per-country narratives for the Propaganda Lab.

EDA's topic v4 clusters are too noisy to reuse here (nearest-centroid assignment matched off-topic
posts even at cosine 0.75) and stop at 2026-07-29, so narratives are clustered per country:
  * posts of a country (mode about / origin, n >= MIN_N, newest MAX_POSTS) with an EDA bge-m3 vector;
  * spherical k-means, k = clamp(round(sqrt(n / 25)), 3, 12); clusters < MIN_CLUSTER posts dropped;
  * terms: c-TF-IDF (cluster vs the country's other posts), unigrams + bigrams;
  * name: Gemma, from the 8 posts nearest the centroid (cached; --name, run when Gemma is free);
    without --name the terms are the label.
Output: data/prop_lab/v1/narratives.parquet
  (mode, iso3, cluster, n, terms, name, summary, post_uids[list], exemplar_uids[list])
  .venv/bin/python scripts/34_prop_lab_narratives.py [--name] [--concurrency 6]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

from fnp.config import DATA_DIR, FNP_ROOT  # noqa: E402
from fnp.geo.origin import load_registry, origin_of  # noqa: E402

PARTS = DATA_DIR / "prop_lab" / "v1" / "parts"
OUT = DATA_DIR / "prop_lab" / "v1" / "narratives.parquet"
EMB_GLOB = str(FNP_ROOT.parent / "EDA" / "data" / "parquet" / "post_embeddings" / "**" / "*.parquet")
MIN_N, MAX_POSTS, MIN_CLUSTER = 30, 30000, 10
URL_RE = re.compile(r"https?://\S+|www\.\S+|@\w+|#")
NAME_SYSTEM = ("You name news/social-media narratives. Given sample posts from one cluster, return ONLY JSON "
               "{\"name\": \"<3-7 word neutral English title of the shared story line>\", "
               "\"summary\": \"<one neutral sentence>\"}. Describe the topic, do not judge truth or intent.")


def load() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f, columns=["post_uid", "platform", "author_name", "text", "posted_at",
                                                "countries"]) for f in sorted(glob.glob(str(PARTS / "*.parquet")))])
    df = df.drop_duplicates("post_uid").reset_index(drop=True)
    reg = load_registry()
    df["about"] = df.countries.map(lambda s: [c["iso3"] for c in json.loads(s)])
    df["origin"] = [[o[0]] if (o := origin_of(p, a or "", reg)) else [] for p, a in zip(df.platform, df.author_name)]
    return df


def vectors(uids: list[str]) -> dict:
    con = duckdb.connect()
    con.register("u", pd.DataFrame({"post_uid": uids}))
    e = con.execute(f"SELECT e.post_uid, e.embedding FROM read_parquet('{EMB_GLOB}', union_by_name=true) e "
                    "JOIN u USING (post_uid)").df()
    con.close()
    return {u: np.asarray(v, dtype=np.float32) for u, v in zip(e.post_uid, e.embedding)}


def ctfidf_terms(texts: list[str], labels: np.ndarray, k: int, top: int = 6) -> list[str]:
    docs = [" ".join(URL_RE.sub(" ", t) for t, l in zip(texts, labels) if l == c) for c in range(k)]
    try:
        cv = CountVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000, stop_words="english",
                             token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
        X = cv.fit_transform(docs).astype(float).toarray()
    except ValueError:
        return [""] * k
    tf = X / np.maximum(X.sum(1, keepdims=True), 1)
    idf = np.log(1 + X.sum() / np.maximum(X.sum(0), 1))
    s = tf * idf
    voc = np.array(cv.get_feature_names_out())
    return [", ".join(voc[np.argsort(-s[c])[:top]]) for c in range(k)]


def name_cluster(texts: list[str]) -> dict:
    from fnp.llm.cache import cached_call

    def val(o):
        return None if isinstance(o, dict) and isinstance(o.get("name"), str) and o["name"].strip() else "need name"
    r = cached_call("narrative_name", "narrative_name_v1", NAME_SYSTEM,
                    json.dumps({"posts": [t[:400] for t in texts]}, ensure_ascii=False), val, 200)
    return r["obj"] if r.get("ok") else {"name": "", "summary": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="store_true", help="name clusters with Gemma")
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    df = load()
    V = vectors(df.post_uid.tolist())
    print(f"{len(df)} posts, {len(V)} with vectors", flush=True)
    rows = []
    for mode in ("about", "origin"):
        ex = df[["post_uid", mode, "posted_at", "text"]].explode(mode).dropna(subset=[mode])
        for iso, g in ex.groupby(mode):
            g = g[g.post_uid.isin(V)].sort_values("posted_at", ascending=False).head(MAX_POSTS)
            if len(g) < MIN_N:
                continue
            X = np.stack([V[u] for u in g.post_uid])
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
            k = int(np.clip(round(np.sqrt(len(g) / 25)), 3, 12))
            km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(X)
            lab = km.labels_
            C = km.cluster_centers_ / np.maximum(np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True), 1e-9)
            terms = ctfidf_terms(g.text.fillna("").tolist(), lab, k)
            uids = g.post_uid.to_numpy()
            for c in range(k):
                m = lab == c
                if m.sum() < MIN_CLUSTER:
                    continue
                sim = X[m] @ C[c]
                rows.append({"mode": mode, "iso3": iso, "cluster": c, "n": int(m.sum()), "terms": terms[c],
                             "cohesion": round(float(sim.mean()), 4), "name": "", "summary": "",
                             "post_uids": uids[m].tolist(),
                             "exemplar_uids": uids[m][np.argsort(-sim)[:8]].tolist()})
        print(f"mode {mode}: {len(rows)} clusters so far", flush=True)
    out = pd.DataFrame(rows)
    if a.name and len(out):
        text = dict(zip(df.post_uid, df.text.fillna("")))
        with ThreadPoolExecutor(a.concurrency) as ex:
            names = list(ex.map(lambda us: name_cluster([text[u] for u in us]), out.exemplar_uids))
        out["name"] = [n.get("name", "")[:80] for n in names]
        out["summary"] = [n.get("summary", "")[:300] for n in names]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {len(out)} narratives ({out.iso3.nunique() if len(out) else 0} countries) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
