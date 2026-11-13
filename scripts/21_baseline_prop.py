"""Stage 21: baseline of the current Gemma propaganda tagger on the frozen PTC splits.

Reads the Gemma spans already stored in data/runs/units_v1.jsonl (no new model calls),
scores them with fnp.eval.ptc_scorer against PTC gold, per split, and writes
reports/EVAL_prop_v0.json and reports/EVAL_prop_v0.md.

  .venv/bin/python scripts/21_baseline_prop.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fnp.datasets import load_ptc  # noqa: E402
from fnp.eval.ptc_scorer import CLASSES14, score  # noqa: E402


def main() -> None:
    split = json.loads((ROOT / "data/splits/ptc_v1.json").read_text())["split"]
    arts, spans = load_ptc()
    gold = defaultdict(list)
    for r in spans.itertuples():
        gold[r.article_id].append((r.start, r.end, r.technique))
    pred, seen = {}, set()
    for line in open(ROOT / "data/runs/units_v1.jsonl"):
        d = json.loads(line)
        if d["dataset"] != "ptc":
            continue
        aid = d["item_id"].split(":", 1)[1]
        if aid in seen:        # spans are document-level; keep one unit per article
            continue
        seen.add(aid)
        pred[aid] = [(s["start"], s["end"], s["technique"]) for s in (d.get("spans") or [])]
    res = {}
    for sp in ("train", "dev", "test"):
        docs = [a for a, s in split.items() if s == sp and a in pred]
        res[sp] = score(gold, pred, docs)
        res[sp]["n_docs_in_split"] = sum(1 for s in split.values() if s == sp)
    (ROOT / "reports/EVAL_prop_v0.json").write_text(json.dumps(res, indent=1))

    L = ["# Propaganda tagger baseline v0 (Gemma, units_v1)", "",
         "Scorer: `fnp/eval/ptc_scorer.py` (SemEval-2020 T11 style, fractional char overlap, "
         "14 merged classes). Splits: `data/splits/ptc_v1.json`. Spans come from the stored "
         "run, so every split is scored only on the articles that run covers.", "",
         "| split | docs scored / in split | SI P | SI R | SI F1 | span+technique P | R | F1 |",
         "|---|---|---|---|---|---|---|---|"]
    for sp, r in res.items():
        a, b = r["SI"], r["TC_span"]
        L.append(f"| {sp} | {r['n_docs']}/{r['n_docs_in_split']} | {a['precision']:.3f} | "
                 f"{a['recall']:.3f} | {a['f1']:.3f} | {b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} |")
    L += ["", "## Per technique (test split)", "",
          "| technique | gold spans | pred spans | P | R | F1 |", "|---|---|---|---|---|---|"]
    for c in CLASSES14:
        v = res["test"]["per_technique"][c]
        L.append(f"| {c} | {v['n_gold']} | {v['n_pred']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    L += ["", "Notes: `train` rows are not training data for Gemma (it is prompted, not trained); "
          "all three splits are valid evaluation slices. Test is the headline number."]
    (ROOT / "reports/EVAL_prop_v0.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
