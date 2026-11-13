"""Stage 22: score Gemma tagger variants (chunking / recall-oriented prompt) on a PTC split.

  .venv/bin/python scripts/22_prop_variants.py --split dev --variants chunk_v1 chunk_v2
Spans are cached in data/cache/prop_exp.duckdb, so reruns are free. Writes
reports/EVAL_prop_v1_<split>.json / .md next to the stage-21 baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fnp import gemma  # noqa: E402
from fnp.datasets import load_ptc  # noqa: E402
from fnp.eval.ptc_scorer import CLASSES14, score  # noqa: E402
from fnp.llm.cache import GemmaCache  # noqa: E402
from fnp.llm.propaganda_v2 import tag_spans_v2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--variants", nargs="+", default=["chunk_v1", "chunk_v2"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    split = json.loads((ROOT / "data/splits/ptc_v1.json").read_text())["split"]
    arts, spans = load_ptc()
    text = dict(zip(arts.article_id, arts.text))
    gold = defaultdict(list)
    for r in spans.itertuples():
        gold[r.article_id].append((r.start, r.end, r.technique))
    docs = sorted(d for d, s in split.items() if s == a.split)
    if a.limit:
        docs = docs[:a.limit]
    cache = GemmaCache(ROOT / "data/cache/prop_exp.duckdb")
    print("gemma host:", gemma.endpoint_host(), "| docs:", len(docs), flush=True)
    results = {}
    for v in a.variants:
        prompt = v.split("_")[1]
        preds = gemma.run_batch(docs, lambda d: [(s["start"], s["end"], s["technique"])
                                                  for s in tag_spans_v2(d, text[d], prompt=prompt, cache=cache)],
                                a.concurrency)
        pred = dict(zip(docs, preds))
        results[v] = score(gold, pred, docs)
        r = results[v]
        print(f"{v}: SI P/R/F1 {r['SI']['precision']:.3f}/{r['SI']['recall']:.3f}/{r['SI']['f1']:.3f}  "
              f"span+tech {r['TC_span']['precision']:.3f}/{r['TC_span']['recall']:.3f}/{r['TC_span']['f1']:.3f}  "
              f"pred spans {r['SI']['n_pred']}", flush=True)
    out = ROOT / f"reports/EVAL_prop_v1_{a.split}.json"
    out.write_text(json.dumps(results, indent=1))
    L = [f"# Gemma tagger variants on PTC {a.split} ({len(docs)} articles)", "",
         "| variant | SI P | SI R | SI F1 | span+tech P | R | F1 |", "|---|---|---|---|---|---|---|"]
    for v, r in results.items():
        x, y = r["SI"], r["TC_span"]
        L.append(f"| {v} | {x['precision']:.3f} | {x['recall']:.3f} | {x['f1']:.3f} | "
                 f"{y['precision']:.3f} | {y['recall']:.3f} | {y['f1']:.3f} |")
    L += ["", "| technique | gold | " + " | ".join(f"{v} F1 (pred)" for v in results) + " |",
          "|---|---|" + "---|" * len(results)]
    for c in CLASSES14:
        g = next(iter(results.values()))["per_technique"][c]["n_gold"]
        L.append(f"| {c} | {g} | " + " | ".join(
            f"{r['per_technique'][c]['f1']:.3f} ({r['per_technique'][c]['n_pred']})" for r in results.values()) + " |")
    (ROOT / f"reports/EVAL_prop_v1_{a.split}.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
