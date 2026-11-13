"""Stage 24: combine the fine-tuned span encoder (stage 23) with the Gemma chunk_v2 tagger.

Per technique, the source (encoder | gemma | union) is chosen on DEV only; test is scored once.
Writes reports/EVAL_prop_v2.md (final comparison incl. the stage-21 baseline).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("s23", ROOT / "scripts/23_train_span_encoder.py")
s23 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s23)
from fnp.datasets import load_ptc  # noqa: E402
from fnp.eval.ptc_scorer import CLASSES14, to14, score  # noqa: E402
from fnp.llm.cache import GemmaCache  # noqa: E402
from fnp.llm.propaganda_v2 import tag_spans_v2  # noqa: E402


def main() -> None:
    split = json.loads((ROOT / "data/splits/ptc_v1.json").read_text())["split"]
    arts, spans = load_ptc()
    texts = dict(zip(arts.article_id, arts.text))
    gold = defaultdict(list)
    for r in spans.itertuples():
        gold[r.article_id].append((r.start, r.end, r.technique))
    D = {s: sorted(d for d, x in split.items() if x == s) for s in ("dev", "test")}
    ck = torch.load(ROOT / "data/models/prop_span_v1/model.pt", map_location="cpu")
    model = s23.SpanTagger(); model.encoder.load_state_dict(ck["encoder"]); model.head.load_state_dict(ck["head"])
    plan = {"device": "cpu", "max_len": 512, "amp": False}
    thr = ck["thr"]
    cache = GemmaCache(ROOT / "data/cache/prop_exp.duckdb")
    enc, gem = {}, {}
    for sp in D:
        P = s23.predict_probs(model, s23.AutoTokenizer.from_pretrained(f"{s23.BASE}/tokenizer"), texts, D[sp], plan)
        for d in D[sp]:
            enc[d] = [(s, e, to14(t)) for s, e, t in s23.decode(*P[d], thr)]
            gem[d] = [(s["start"], s["end"], to14(s["technique"]))
                      for s in tag_spans_v2(d, texts[d], prompt="v2", cache=cache)]   # cached
    def combine(choice):
        out = {}
        for d in enc:
            o = []
            for s, e, t in enc[d]:
                if choice[t] in ("enc", "union"):
                    o.append((s, e, t))
            for s, e, t in gem[d]:
                if t in choice and choice[t] in ("gem", "union"):
                    o.append((s, e, t))
            out[d] = o
        return out
    def per(sp, ch):
        return score(gold, combine(ch), D[sp])
    res = {}
    base = {c: "enc" for c in CLASSES14}
    allv = {name: {c: name for c in CLASSES14} for name in ("enc", "gem", "union")}
    for name, ch in allv.items():
        res[f"{name}_only" if name != "union" else "union_all"] = {sp: per(sp, ch) for sp in D}
    best = {}
    dev_scores = {n: per("dev", ch)["per_technique"] for n, ch in allv.items()}
    for c in CLASSES14:
        best[c] = max(("enc", "gem", "union"), key=lambda n: dev_scores[n][c]["f1"])
    res["per_class_best(dev-selected)"] = {sp: per(sp, best) for sp in D}
    res["choice"] = best
    (ROOT / "reports/EVAL_prop_v2_ensemble.json").write_text(json.dumps(res, indent=1))
    b0 = json.loads((ROOT / "reports/EVAL_prop_v0.json").read_text())
    L = ["# Propaganda span/technique tagger: final comparison (PTC held-out)", "",
         "Scorer `fnp/eval/ptc_scorer.py`; splits `data/splits/ptc_v1.json`. Dev was used for thresholds / per-class source choice; **test scored once**.", "",
         "| system | split | SI P | SI R | SI F1 | span+tech P | R | F1 |", "|---|---|---|---|---|---|---|---|"]
    def row(name, sp, r):
        x, y = r["SI"], r["TC_span"]
        L.append(f"| {name} | {sp} | {x['precision']:.3f} | {x['recall']:.3f} | {x['f1']:.3f} | {y['precision']:.3f} | {y['recall']:.3f} | {y['f1']:.3f} |")
    for sp in ("dev", "test"):
        row("v0 Gemma (orig. prompt, 8k trunc)", sp, b0[sp])
    for k, v in res.items():
        if k == "choice": continue
        for sp in ("dev", "test"):
            row(k, sp, v[sp])
    L += ["", "## Per technique, test F1 (gold spans)", "", "| technique | gold | v0 | enc | gem | union_all | per-class best (src) |", "|---|---|---|---|---|---|---|"]
    for c in CLASSES14:
        g = res["enc_only"]["test"]["per_technique"][c]["n_gold"]
        f = lambda r: f"{r['per_technique'][c]['f1']:.3f}"
        L.append(f"| {c} | {g} | {f(b0['test'])} | {f(res['enc_only']['test'])} | {f(res['gem_only']['test'])} | {f(res['union_all']['test'])} | {f(res['per_class_best(dev-selected)']['test'])} ({best[c]}) |")
    (ROOT / "reports/EVAL_prop_v2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
