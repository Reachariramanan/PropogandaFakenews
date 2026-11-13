"""Fine-tuned span encoder (scripts/23_train_span_encoder.py) as a drop-in propaganda tagger.

tag_spans_enc(doc_id, text) -> list[Span] in the same dict shape as fnp.llm.propaganda.tag_spans
(technique names are the 14 merged PTC classes). Local CPU/GPU, no Gemma calls. Opt-in: the
pipeline still uses tag_spans until the owner switches it.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "data/models/prop_span_v1/model.pt"


@lru_cache(maxsize=1)
def _load():
    spec = importlib.util.spec_from_file_location("prop_s23", ROOT / "scripts/23_train_span_encoder.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    ck = torch.load(CKPT, map_location="cpu")
    model = m.SpanTagger()
    model.encoder.load_state_dict(ck["encoder"])
    model.head.load_state_dict(ck["head"])
    model.eval()
    tok = m.AutoTokenizer.from_pretrained(f"{m.BASE}/tokenizer")
    return m, model, tok, ck["thr"]


MIN_TOKENS = 12     # same as scripts/30_prop_lab_infer.py: shorter texts are "too_short"
STRONG_P = 0.5      # a span is "strong" if its mean token prob >= max(class threshold, STRONG_P)


def tag_spans_enc(doc_id: str, text: str) -> list[dict]:
    return analyse_enc(doc_id, text)["spans"]


def analyse_enc(doc_id: str, text: str) -> dict:
    """One encoder pass -> {spans, technique_probs, p_prop, too_short, n_tokens}.

    technique_probs: all 14 merged PTC classes, each the MAX token probability over the text
    (0..1, whether or not it passed the class threshold). p_prop: highest mean-probability among
    STRONG spans (strong = mean token prob >= max(class threshold, 0.5)); 0.0 when there is none,
    None when the text is too_short (< 12 encoder tokens, where the encoder over-fires). It is an
    uncalibrated signal score, not a probability that the text is propaganda.
    """
    m, model, tok, thr = _load()
    plan = {"device": "cpu", "max_len": 512, "amp": False}
    P = m.predict_probs(model, tok, {doc_id: text}, [doc_id], plan)
    probs, offs = P[doc_id]
    n_tok = len(offs)
    out = []
    for s, e, t in m.decode(probs, offs, thr):
        k = m.CIDX[t]
        while s < len(text) and text[s].isspace():        # snap to whole words, no stray spaces
            s += 1
        while s > 0 and text[s - 1].isalnum() and text[s].isalnum():
            s -= 1
        while e > s and text[e - 1].isspace():
            e -= 1
        while e < len(text) and text[e - 1].isalnum() and text[e].isalnum():
            e += 1
        if e <= s:
            continue
        sel = [i for i, (a, b) in enumerate(offs) if b > s and a < e]
        conf = float(probs[sel, k].mean()) if sel else 0.5
        out.append({"doc_id": doc_id, "start": s, "end": e, "text": text[s:e], "technique": t, "confidence": conf,
                    "strong": conf >= max(float(thr[k]), STRONG_P)})
    spans = sorted(out, key=lambda x: (x["start"], x["end"]))
    tp = {c: (round(float(probs[:, k].max()), 4) if n_tok else 0.0) for c, k in m.CIDX.items()}
    too_short = n_tok < MIN_TOKENS
    strong = [x["confidence"] for x in spans if x["strong"]]
    return {"spans": spans, "technique_probs": tp, "n_tokens": n_tok, "too_short": too_short,
            "p_prop": None if too_short else round(max(strong, default=0.0), 4)}
