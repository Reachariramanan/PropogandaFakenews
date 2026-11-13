"""Stage 30: Propaganda Lab batch inference (stored, not real-time).

For every social post (EDA incoming queue + post_facts parquet; news when the DB is reachable):
  * propaganda spans from the fine-tuned span encoder (data/models/prop_span_v1, 14 PTC classes),
    batched on GPU (bf16) with the checkpoint's per-class thresholds;
  * "about" countries from the gazetteer matcher (fnp/geo/countries.py);
  * nearest fact-check in kb_v2 (bge-m3 cosine; EDA's stored bge-m3 vectors are reused, missing
    ones are embedded) and, for cos >= RERANK_MIN_COS, the bge-reranker score -> debunk match.
Output: data/prop_lab/v1/parts/part_NNNN.parquet (resumable: existing parts are skipped).
  .venv/bin/python scripts/30_prop_lab_infer.py [--limit N] [--part-size 4000] [--apply]
Without --apply it runs one 500-post part into a scratch dir and prints throughput + GPU peak.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fnp.config import DATA_DIR, FNP_ROOT  # noqa: E402
from fnp.eda_bridge import read_parquet  # noqa: E402
from fnp.geo.countries import matcher  # noqa: E402
from fnp.live.sources import INCOMING, MIN_CHARS  # noqa: E402

OUT = DATA_DIR / "prop_lab" / "v1" / "parts"
EMB_GLOB = str(FNP_ROOT.parent / "EDA" / "data" / "parquet" / "post_embeddings" / "**" / "*.parquet")
MAX_LEN = 512
TOKEN_BUDGET = 48_000          # tokens per encoder batch (padded)
MIN_TOKENS = 12                # shorter posts: too_short (encoder over-fires on titles)
STRONG_P = 0.5                 # a span is "strong" if its mean token prob >= max(thr, STRONG_P)
RERANK_MIN_COS = 0.72
MAX_SPANS_STORED = 40


def load_items(limit=None) -> pd.DataFrame:
    cols = ["post_uid", "platform", "url", "author_name", "text", "lang", "posted_at"]
    t = read_parquet("post_facts")[cols]
    if INCOMING.exists():
        con = duckdb.connect(str(INCOMING), read_only=True)
        q = con.execute("SELECT post_uid, platform, NULL AS url, NULL AS author_name, text, lang, posted_at "
                        "FROM incoming_posts").df()
        con.close()
        t = pd.concat([q, t]).drop_duplicates("post_uid", keep="last")
    t = t[t.text.fillna("").str.strip().str.len() >= MIN_CHARS]
    t = t.sort_values("posted_at", ascending=False).reset_index(drop=True)
    t["source"] = "social"
    return t.head(limit) if limit else t


def load_encoder(dev):
    from fnp.llm.propaganda_enc import _load
    m, model, tok, thr = _load()
    return m, model.to(dev).eval(), tok, np.asarray(thr, dtype=np.float32)


@torch.no_grad()
def encode_spans(texts: list[str], m, model, tok, thr, dev) -> list[dict]:
    enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
    wins = []                                    # (doc, a, b, token ids)
    for d, ids in enumerate(enc["input_ids"]):
        for a, b in m.windows(len(ids), MAX_LEN - 2, (MAX_LEN - 2) // 2):
            wins.append((d, a, b, [tok.cls_token_id] + ids[a:b] + [tok.sep_token_id]))
    acc = [np.zeros((len(ids), len(thr)), np.float32) for ids in enc["input_ids"]]
    cnt = [np.zeros(len(ids), np.float32) for ids in enc["input_ids"]]
    order = sorted(range(len(wins)), key=lambda i: len(wins[i][3]))
    i = 0
    while i < len(order):
        L = len(wins[order[i]][3])
        j = i
        while j < len(order) and (j - i + 1) * len(wins[order[j]][3]) <= TOKEN_BUDGET:
            j += 1
        j = max(j, i + 1)
        batch = [wins[k] for k in order[i:j]]
        L = max(len(w[3]) for w in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for r, w in enumerate(batch):
            ids[r, :len(w[3])] = torch.tensor(w[3])
            mask[r, :len(w[3])] = 1
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            p = torch.sigmoid(model(ids.to(dev), mask.to(dev)).float()).cpu().numpy()
        for r, (d, a, b, _) in enumerate(batch):
            acc[d][a:b] += p[r, 1:1 + (b - a)]
            cnt[d][a:b] += 1
        i = j
    out = []
    for d, offs in enumerate(enc["offset_mapping"]):
        P = acc[d] / np.maximum(cnt[d], 1)[:, None]
        n_tok = len(offs)
        spans = []
        for s, e, t in m.decode(P, offs, thr):
            k = m.CIDX[t]
            tok_in = [x for x, (a, b) in enumerate(offs) if b > s and a < e]
            pr = float(P[tok_in, k].mean()) if tok_in else 0.0
            spans.append([int(s), int(e), t, round(pr, 3)])
        strong = [sp for sp in spans if sp[3] >= max(float(thr[m.CIDX[sp[2]]]), STRONG_P)]
        tech = {}
        for sp in spans:
            tech[sp[2]] = tech.get(sp[2], 0) + 1
        out.append({"n_tokens": n_tok, "too_short": n_tok < MIN_TOKENS, "n_spans": len(spans),
                    "n_strong": len(strong), "density": round(100 * len(spans) / max(n_tok, 1), 3),
                    "techniques": json.dumps(tech),
                    "strong_techniques": json.dumps(sorted({sp[2] for sp in strong})),
                    "tech_max": json.dumps([round(float(x), 3) for x in (P.max(0) if n_tok else np.zeros(len(thr)))]),
                    "spans": json.dumps(sorted(spans, key=lambda x: -x[3])[:MAX_SPANS_STORED])})
    return out


class KBMatch:
    def __init__(self, dev):
        from fnp.retrieval.kb import KB
        self.kb = KB.load()
        self.E = torch.tensor(self.kb.emb, device=dev, dtype=torch.float16)
        self.E = self.E / self.E.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.dev = dev

    def top1(self, V: np.ndarray):
        q = torch.tensor(V, device=self.dev, dtype=torch.float16)
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-6)
        s, i = (q @ self.E.T).max(dim=1)
        return s.float().cpu().numpy(), i.cpu().numpy()

    def rerank(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        from fnp.retrieval.kb import get_reranker
        if not pairs:
            return np.zeros(0)
        sc = get_reranker().predict(pairs, batch_size=32, show_progress_bar=False)
        return 1 / (1 + np.exp(-np.asarray(sc, dtype=np.float32))) if np.asarray(sc).max() > 1 else np.asarray(sc)


def eda_vectors(uids: list[str]) -> dict:
    con = duckdb.connect()
    con.register("u", pd.DataFrame({"post_uid": uids}))
    df = con.execute(f"SELECT e.post_uid, e.embedding FROM read_parquet('{EMB_GLOB}', union_by_name=true) e "
                     "JOIN u USING (post_uid)").df()
    con.close()
    return {r.post_uid: np.asarray(r.embedding, dtype=np.float32) for r in df.itertuples()}


def run_part(df: pd.DataFrame, enc, kbm: KBMatch, dev) -> pd.DataFrame:
    m, model, tok, thr = enc
    texts = df.text.fillna("").str.slice(0, 6000).tolist()
    sig = encode_spans(texts, m, model, tok, thr, dev)
    gm = matcher()
    countries = [json.dumps(gm.match(t)) for t in texts]
    vec = eda_vectors(df.post_uid.tolist())
    miss = [k for k, u in enumerate(df.post_uid) if u not in vec]
    if miss:
        from fnp.retrieval.kb import embed_texts
        V = embed_texts([texts[k][:1500] for k in miss], batch_size=32)
        for k, v in zip(miss, V):
            vec[df.post_uid.iat[k]] = v
    V = np.stack([vec[u] for u in df.post_uid])
    cos, idx = kbm.top1(V)
    meta = kbm.kb.meta
    rr = np.full(len(df), np.nan, dtype=np.float32)
    cand = np.where(cos >= RERANK_MIN_COS)[0]
    if len(cand):
        rr[cand] = kbm.rerank([(texts[k][:1000], meta.claim.iat[idx[k]]) for k in cand])
    out = df[["post_uid", "source", "platform", "url", "author_name", "lang", "posted_at"]].copy()
    out["text"] = [t[:1200] for t in texts]
    for k in sig[0]:
        out[k] = [s[k] for s in sig]
    out["countries"] = countries
    out["kb_cos"] = np.round(cos, 4)
    out["kb_id"] = [meta.kb_id.iat[i] for i in idx]
    out["kb_label"] = [meta.label5.iat[i] for i in idx]
    out["kb_claim"] = [meta.claim.iat[i][:300] for i in idx]
    out["kb_url"] = [meta.url.iat[i] for i in idx]
    out["kb_rerank"] = rr
    out["posted_at"] = out.posted_at.astype(str)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--part-size", type=int, default=4000)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**30 > 4 else "cpu"
    items = load_items(a.limit if a.apply else 500)
    out_dir = OUT if a.apply else DATA_DIR / "prop_lab" / "dryrun"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = a.part_size if a.apply else 500
    print(f"{len(items)} items, device {dev}, parts of {size} -> {out_dir}", flush=True)
    enc = load_encoder(dev)
    kbm = KBMatch(dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    for p in range(0, len(items), size):
        f = out_dir / f"part_{p // size:04d}.parquet"
        if f.exists() and a.apply:
            continue
        t0 = time.time()
        res = run_part(items.iloc[p:p + size].reset_index(drop=True), enc, kbm, dev)
        res.to_parquet(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").rename(f)
        pk = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        print(f"part {p // size} {len(res)} posts {time.time()-t0:.0f}s "
              f"({len(res)/(time.time()-t0):.0f}/s) gpu_peak {pk:.2f}GB "
              f"spans>0 {(res.n_spans > 0).mean():.0%} strong>0 {(res.n_strong > 0).mean():.0%} "
              f"located {(res.countries != '[]').mean():.0%} kb_rr>=0.5 {(res.kb_rerank >= 0.5).mean():.1%}",
              flush=True)
    print(f"done in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
