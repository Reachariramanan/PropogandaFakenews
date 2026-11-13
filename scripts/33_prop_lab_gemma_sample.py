"""Stage 33: Gemma checks for the Propaganda Lab (targeted samples, stored).

Gemma is shared and slow under load (a full propaganda_v2 + verdict pass was ~3 posts/min), so
its budget goes where it adds information:
  * spans   -- SPAN_PER_COUNTRY posts per country (about mode, n >= MIN_N; 3 strongest encoder
               signal + 2 random signal posts): Gemma propaganda_v2 spans -> agreement with the
               encoder. Output data/prop_lab/v1/gemma_spans.jsonl
  * verdict -- posts with a candidate fact-check match (kb_cos >= VERDICT_MIN_COS), best reranked
               first, up to VERDICT_PER_COUNTRY per country: full FNP verdict (claims -> KB evidence
               -> maths -> stacker), first VERDICT_CHARS chars. Output data/prop_lab/v1/gemma_verdict.jsonl
               Without KB evidence the pipeline returns INCONCLUSIVE, so posts with no candidate
               match are not worth the Gemma calls.
Fake/misleading call (`fake_call`): per claim, PROPAGANDA is removed (the stacker's PROPAGANDA
probability is a known length artefact, docs/review/05 #1/#7) but only if the remaining mass is
>= MIN_NONPROP_MASS -- otherwise the stacker made no factual call and the Gemma adjudicator's
probabilities are used under the same rule, else INCONCLUSIVE.
Rows with ok=false are retried on the next run (last row per post_uid wins).
  .venv/bin/python scripts/33_prop_lab_gemma_sample.py [--task spans|verdict|both] [--limit N] [--apply]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from fnp.config import DATA_DIR, gemma_config  # noqa: E402
from fnp.llm import prompts as P  # noqa: E402
from fnp.llm import validators as V  # noqa: E402
from fnp.llm.cache import cached_call  # noqa: E402
from fnp.llm.propaganda_v2 import CHUNK_CHARS, MAX_TOKENS, chunk_windows, system_v2  # noqa: E402
from fnp.triage import fake_call, fm_probs as _fm  # noqa: E402

PARTS = DATA_DIR / "prop_lab" / "v1" / "parts"
OUT = {"spans": DATA_DIR / "prop_lab" / "v1" / "gemma_spans.jsonl",
       "verdict": DATA_DIR / "prop_lab" / "v1" / "gemma_verdict.jsonl"}
VERSION = {"spans": "spans_v1", "verdict": "verdict_v2"}
MIN_N = 30
SPAN_PER_COUNTRY = 5
VERDICT_PER_COUNTRY = 15
VERDICT_MIN_COS = 0.72
VERDICT_CHARS = 1500
SPAN_CHARS = 3000


def gemma_spans(doc_id: str, text: str) -> tuple[list, bool]:
    system, version = system_v2(), f"propaganda_v2_v2_c{CHUNK_CHARS}"
    spans, ok = [], True
    for off, chunk in chunk_windows(text):
        user = P.render(P.PROPAGANDA_USER, NOTE="", TEXT=chunk)
        res = cached_call("propaganda_v2", version, system, user, V.validate_propaganda, MAX_TOKENS,
                          retry_failed=True)
        if not res["ok"]:
            ok = False
            continue
        got, _, _, _ = V.parse_spans(res["obj"], doc_id, chunk)
        spans += [[s["start"] + off, s["end"] + off, s["technique"]] for s in got]
    return sorted(spans), ok


def agreement(enc: list, gem: list) -> dict:
    """Share of encoder strong spans overlapping any Gemma span (and with the same technique)."""
    def ov(a, b):
        return a[0] < b[1] and b[0] < a[1]
    n = len(enc)
    hit = sum(any(ov(e, g) for g in gem) for e in enc)
    tech = sum(any(ov(e, g) and e[2] == g[2] for g in gem) for e in enc)
    return {"enc_spans": n, "gemma_spans": len(gem), "overlap": round(hit / n, 3) if n else None,
            "same_technique": round(tech / n, 3) if n else None,
            "gemma_flags": bool(gem), "encoder_flags": bool(n)}


def by_country(df: pd.DataFrame) -> dict:
    out = {}
    for uid, cs in zip(df.post_uid, df.countries):
        for c in json.loads(cs):
            out.setdefault(c["iso3"], []).append(uid)
    return out


def pick_spans(df: pd.DataFrame, n_about: Counter, rng: random.Random) -> list[str]:
    sig = df[df.signal]
    rank = dict(zip(sig.post_uid, sig.n_strong * 10 + sig.density.clip(upper=20)))
    chosen = []
    for iso, uids in by_country(sig).items():
        if n_about[iso] < MIN_N:
            continue
        uids = sorted(set(uids), key=lambda u: -rank[u])
        top = uids[:3]
        rest = uids[3:]
        chosen += top + rng.sample(rest, min(len(rest), SPAN_PER_COUNTRY - len(top)))
    return list(dict.fromkeys(chosen))


def pick_verdict(df: pd.DataFrame) -> list[str]:
    cand = df[df.kb_cos >= VERDICT_MIN_COS].sort_values(["kb_rerank", "kb_cos"], ascending=False)
    chosen = []
    for iso, uids in by_country(cand).items():
        chosen += list(dict.fromkeys(uids))[:VERDICT_PER_COUNTRY]
    unl = cand[cand.countries == "[]"].post_uid.head(VERDICT_PER_COUNTRY * 2).tolist()
    return list(dict.fromkeys(chosen + unl))


def done_ids(task: str) -> set:
    st = {}
    for line in OUT[task].open() if OUT[task].exists() else []:
        r = json.loads(line)
        st[r["post_uid"]] = bool(r.get("ok")) and r.get("version") == VERSION[task]
    return {u for u, ok in st.items() if ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both", choices=["spans", "verdict", "both"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cols = ["post_uid", "text", "lang", "countries", "too_short", "n_strong", "density", "spans",
            "strong_techniques", "kb_cos", "kb_rerank"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(PARTS / "*.parquet")))])
    df = df.drop_duplicates("post_uid").reset_index(drop=True)
    df["signal"] = (~df.too_short) & (df.n_strong >= 1)
    n_about = Counter(c["iso3"] for cs in df.countries for c in json.loads(cs))
    jobs = []
    if a.task in ("spans", "both"):
        jobs += [("spans", u) for u in pick_spans(df, n_about, random.Random(0)) if u not in done_ids("spans")]
    if a.task in ("verdict", "both"):
        jobs += [("verdict", u) for u in pick_verdict(df) if u not in done_ids("verdict")]
    jobs = jobs[: a.limit]
    print(f"todo {Counter(t for t, _ in jobs)}", flush=True)
    if not a.apply or not jobs:
        return
    D = df.set_index("post_uid")
    deciders = None
    if any(t == "verdict" for t, _ in jobs):
        from fnp.pipeline import load_deciders
        deciders = load_deciders(None)

    def run(job):
        task, uid = job
        r = D.loc[uid]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if task == "spans":
            gs, ok = gemma_spans(uid, r.text[:SPAN_CHARS])
            strong = set(json.loads(r.strong_techniques))
            enc = [s for s in json.loads(r.spans) if s[2] in strong and s[0] < SPAN_CHARS]
            return {"post_uid": uid, "version": VERSION[task], "ok": ok, "gemma_spans": gs,
                    "agreement": agreement(enc, gs), "at": now}
        from fnp.pipeline import analyse_text
        res = analyse_text(f"plab:{uid}", r.text[:VERDICT_CHARS], r.lang, deciders)
        cl = res["claims"]
        ok = bool(cl) and any(c["gemma_adj"].get("ok") for c in cl)
        per = [{"claim": (c["claim"].get("claim") if isinstance(c["claim"], dict) else c["claim"])[:300],
                "stacker": c["verdict"]["probs"],
                "gemma": c["gemma_adj"]["probs"] if c["gemma_adj"].get("ok") else None,
                "rationale": (c["verdict"].get("rationale") or "")[:400]} for c in cl]
        fm = fake_call(per)
        top = max(per, key=lambda c: (_fm(c["stacker"]) or _fm(c["gemma"]) or {"fake_score": 0})["fake_score"],
                  default=None)
        return {"post_uid": uid, "version": VERSION[task], "ok": ok, "label": fm["label"], "probs": fm["probs"],
                "fake_score": fm["fake_score"], "basis": fm["basis"],
                "plausibly_fake": fm["label"] in ("FAKE", "MISLEADING") or 0.5 <= fm["fake_score"] < 0.8,
                "rationale": (top or {}).get("rationale", ""), "n_claims": len(cl), "claims": per[:5], "at": now}

    t0 = time.time()
    for p in OUT.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    fh = {k: p.open("a") for k, p in OUT.items()}
    try:
        with ThreadPoolExecutor(gemma_config()["concurrency"]) as ex:
            for k, (job, fut) in enumerate(zip(jobs, [ex.submit(run, j) for j in jobs]), 1):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"post_uid": job[1], "version": VERSION[job[0]], "ok": False,
                           "error": f"{type(e).__name__}: {e}"[:200]}
                fh[job[0]].write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                fh[job[0]].flush()
                if k % 25 == 0:
                    print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    finally:
        for f in fh.values():
            f.close()
    print(f"done {len(jobs)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
