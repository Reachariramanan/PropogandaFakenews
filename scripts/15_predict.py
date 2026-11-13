"""Stage 15: verdict for arbitrary input (text, file or URL).

  .venv/bin/python scripts/15_predict.py --text "Video shows ..."
  .venv/bin/python scripts/15_predict.py --url https://example.com/story
  .venv/bin/python scripts/15_predict.py --file post.txt [--laya data/models/laya_fnp_vN]

Prints JSON: doc verdict (label, probs, plausibly_fake, secondary) and per-claim verdicts
with the evidence trail. Uses the stacker + conformal saved by 14_evaluate --apply when
present, otherwise the maths rule head alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnp.pipeline import analyse_text, load_deciders  # noqa: E402
from fnp.text_utils import sha16  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file")
    g.add_argument("--url")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--laya", default=None)
    a = ap.parse_args()

    if a.url:
        from fnp.retrieval.web import fetch_article
        art = fetch_article(a.url)
        if not art or not art.get("text"):
            sys.exit(f"could not fetch {a.url}")
        text, doc_id = art["text"], f"url:{sha16(a.url)}"
    else:
        text = a.text if a.text else Path(a.file).read_text()
        doc_id = f"input:{sha16(text)}"

    res = analyse_text(doc_id, text, a.lang, load_deciders(a.laya))
    d = res["doc"]
    out = {"doc_id": doc_id, "label": d["label"], "probs": d["probs"],
           "plausibly_fake": d.get("plausibly_fake"), "secondary": d.get("secondary"),
           "propaganda": res["spans_summary"],
           "claims": [{"claim": r["claim"]["claim"], "label": r["verdict"]["label"],
                       "probs": {k: round(v, 3) for k, v in r["verdict"]["probs"].items()},
                       "prediction_set": r["verdict"]["prediction_set"],
                       "decider": r["verdict"]["decider"],
                       "rationale": r["verdict"]["rationale"],
                       "key_features": {k: r["features"].get(k) for k in
                                        ("p_true", "p_true_lo", "p_true_hi", "ds_uncertainty",
                                         "n_eff", "kb_match", "p_prop")},
                       "evidence": [{"source": e["source"], "stance": e["stance"],
                                     "kb_claim": e["text"][:200], "url": e.get("url")}
                                    for e in r["evidence"]]}
                      for r in res["claims"]]}
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
