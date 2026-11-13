import json
from pathlib import Path

import numpy as np
import pandas as pd

from fnp import datasets as D
from fnp.retrieval import web
from fnp.retrieval.kb import KB


def test_label_maps():
    assert D.map_label(D.LIAR_MAP, "pants-fire") == "FAKE"
    assert D.map_label(D.LIAR_MAP, "half-true") == "MISLEADING"
    assert D.map_label(D.LIAR_MAP, "mostly-true") == "LEGIT"
    assert D.map_label(D.XFACT_MAP, "complicated/hard to categorise") == "INCONCLUSIVE"
    assert D.map_label(D.XFACT_MAP, "mostly false") == "MISLEADING"
    assert D.map_label(D.FNN_MAP, "real") == "LEGIT"
    assert D.map_label(D.LIAR_MAP, None) is None
    assert D.qprop_label("propaganda") == "PROPAGANDA" and D.qprop_label("non") == "LEGIT"
    assert D.ptc_doc_label(3, 10000) == "PROPAGANDA"
    assert D.ptc_doc_label(1, 100) == "PROPAGANDA"
    assert D.ptc_doc_label(1, 1000) is None


def test_boom_leakage_and_split(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([
        {"id": "1", "claim": "C1", "title": "T1", "verdict": "False", "text": "LEAK debunk"},
        {"id": "2", "claim": "", "title": "T2", "verdict": "Misleading", "text": "LEAK"},
        {"id": "3", "claim": "C3", "verdict": "", "text": "x"}]))
    d = D.load_boom(p)
    assert list(d.text) == ["C1", "T2"] and list(d.label5) == ["FAKE", "MISLEADING"]
    assert not d.text.str.contains("LEAK").any() and not d.meta_json.str.contains("LEAK").any()
    assert D.boom_split("1") == D.boom_split("1") in ("train", "dev", "test")


def test_factchecks_loader(tmp_path):
    rows = [dict(id="s:1", site="s", claim="c1", verdict_norm="FAKE", body="B", description="D"),
            dict(id="s:1", site="s", claim="dup", verdict_norm="FAKE"),
            dict(id="s:2", site="s", claim="c2", verdict_norm="BAD")]
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    d = D.load_factchecks(tmp_path)
    assert len(d) == 1 and d.text[0] == "c1" and "B" in d.meta_json[0]


def test_kb_search_synthetic():
    vocab = {"a": 0, "b": 1, "c": 2}
    def emb(ts):
        v = np.zeros((len(ts), 3), np.float32)
        for i, t in enumerate(ts):
            v[i, vocab[t[0]]] = 1
        return v
    meta = pd.DataFrame(dict(kb_id=["x1", "x2", "x3"], claim=["a1", "b1", "a2"], label5=["FAKE"] * 3,
                             raw_label=["f"] * 3, source=["t"] * 3, url=[None] * 3))
    kb = KB(emb(["a", "b", "a"]), meta, embed_fn=emb, rerank_fn=lambda pairs: [-i for i in range(len(pairs))])
    r = kb.search("a?", k=2, rerank=False)
    assert {x["kb_id"] for x in r} == {"x1", "x3"}
    r = kb.search("a?", k=2, exclude_ids=("x1",))
    assert r[0]["kb_id"] == "x3" and r[0]["rerank_score"] is not None
    assert len(kb.search(["a?", "b?"], k=1, rerank=False)) == 2


def test_web_parser():
    html = (Path(__file__).parent / "fixtures" / "ddg.html").read_text()
    r = web.parse_ddg_html(html, 5)
    assert len(r) == 2
    assert r[0] == dict(title="First result & more", url="https://example.com/a?x=1", snippet="Snippet with bold text.")
    assert r[1]["url"] == "https://example.org/b"
