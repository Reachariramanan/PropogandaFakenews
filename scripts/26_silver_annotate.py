"""Stage 26: Gemma silver annotation of the social_v1 pack with a 3-voter agreement filter.

Voters: Gemma chunked prompt v2, Gemma chunked prompt v1, fine-tuned encoder. A silver span is a
cluster of same-technique (14-class) spans from >=2 voters whose overlap is >=50% of the shorter
span; its text range is the union. Writes data/annotation/social_v1/silver_spans.jsonl and
review_sample.json (100 posts for human checking: 70 stratified random + 30 most-disputed).
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fnp import gemma  # noqa: E402
from fnp.eval.ptc_scorer import to14  # noqa: E402
from fnp.llm.cache import GemmaCache  # noqa: E402
from fnp.llm.propaganda_enc import tag_spans_enc  # noqa: E402
from fnp.llm.propaganda_v2 import tag_spans_v2  # noqa: E402

PK = ROOT / "data/annotation/social_v1"


def overlap(a, b):
    o = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return o / max(1, min(a[1] - a[0], b[1] - b[0]))


def cluster(votes: dict[str, list]) -> list[dict]:
    allsp = [(s, e, t, v) for v, L in votes.items() for s, e, t in L]
    allsp.sort(key=lambda x: (x[2], x[0]))
    out = []
    for t in sorted({x[2] for x in allsp}):
        cl: list[list] = []
        for s, e, _, v in (x for x in allsp if x[2] == t):
            for c in cl:
                if overlap((s, e), (c[0], c[1])) >= 0.5:
                    c[0], c[1] = min(c[0], s), max(c[1], e); c[2].add(v); break
            else:
                cl.append([s, e, {v}])
        out += [{"start": c[0], "end": c[1], "technique": t, "voters": sorted(c[2]), "n_votes": len(c[2])} for c in cl]
    return sorted(out, key=lambda x: (x["start"], x["end"]))


def main() -> None:
    items = [json.loads(l) for l in open(PK / "items.jsonl")]
    cache = GemmaCache(ROOT / "data/cache/prop_exp.duckdb")
    print("gemma host:", gemma.endpoint_host(), "| items", len(items), flush=True)

    def run(it):
        t, d = it["text"], it["item_id"]
        return {"gemma_v2": [(s["start"], s["end"], to14(s["technique"])) for s in tag_spans_v2(d, t, prompt="v2", cache=cache)],
                "gemma_v1": [(s["start"], s["end"], to14(s["technique"])) for s in tag_spans_v2(d, t, prompt="v1", cache=cache)]}
    g = gemma.run_batch(items, run, 8)
    stats, out = Counter(), []
    for it, votes in zip(items, g):
        votes["encoder"] = [(s["start"], s["end"], to14(s["technique"])) for s in tag_spans_enc(it["item_id"], it["text"])]
        cl = cluster(votes)
        for c in cl:
            c["text"] = it["text"][c["start"]:c["end"]]
            stats[f"votes_{c['n_votes']}"] += 1
        keep = [c for c in cl if c["n_votes"] >= 2]
        out.append({"item_id": it["item_id"], "platform": it["platform"], "lang_group": it["lang_group"],
                    "silver": keep, "disputed": [c for c in cl if c["n_votes"] == 1],
                    "n_per_voter": {k: len(v) for k, v in votes.items()}})
    with open(PK / "silver_spans.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    rng = random.Random(5)
    by = defaultdict(list)
    for o in out:
        by[(o["platform"], o["lang_group"] == "hi")].append(o["item_id"])
    pick = []
    for k, v in by.items():
        rng.shuffle(v); pick += v[:max(1, round(70 * len(v) / len(out)))]
    pick = pick[:70]
    disputed = sorted((o for o in out if o["item_id"] not in pick), key=lambda o: -len(o["disputed"]))
    pick += [o["item_id"] for o in disputed[:30]]
    (PK / "review_sample.json").write_text(json.dumps(pick))
    tech = Counter(c["technique"] for o in out for c in o["silver"])
    print(json.dumps({"clusters_by_votes": stats, "posts_with_silver_span": sum(bool(o["silver"]) for o in out),
                      "silver_spans": sum(len(o["silver"]) for o in out), "by_technique": tech.most_common(),
                      "review_sample": len(pick)}, indent=1))


if __name__ == "__main__":
    main()
