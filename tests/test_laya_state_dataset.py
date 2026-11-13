import json, math
from fnp.laya.state_builder import QUESTIONS, build_state, MAX_STATE_CHARS
from fnp.laya.dataset import make_row, split_of, split_rows, write_jsonl
from fnp.laya.synth import synth_records
from fnp.schema import LABELS


def test_state_deterministic_and_bounded():
    ev = [{"text": "x" * 900, "stance": "refute", "evidence_type": "FACT_CHECK", "reliability": 0.9 - i / 20} for i in range(9)]
    feats = {"p_true": 0.2345, "n_eff": math.nan, "p_prop": 0.5}
    args = ({"claim": "c " * 600}, feats, ev, {"n": 2, "tech": ["Doubt"]}, {"burst_intensity": 3.0})
    s = build_state(*args)
    assert s == build_state(*args) and len(s) <= MAX_STATE_CHARS
    assert "p_true=0.23" in s and "n_eff=unknown" in s and "top 5 of 9" in s
    assert build_state({"claim": "x"}, {}, [])  # empty features/evidence work


def test_questions_and_dataset_roundtrip(tmp_path):
    assert tuple(QUESTIONS["verdict"]["criteria"]) == LABELS
    rows = synth_records(40)
    p = tmp_path / "r.jsonl"
    write_jsonl(rows, p)
    from laya.train import read_jsonl, items_from_rows
    from transformers import AutoTokenizer
    from laya.train import resolve_checkpoint_dir
    back = read_jsonl(str(p))
    assert back == json.loads(json.dumps(rows))
    r = make_row("s", gold="FAKE", gemma_probs={"LEGIT": 1.0})
    pr = r["gold"]["verdict"]["probabilities"]
    assert abs(pr["FAKE"] - 0.7) < 1e-9 and abs(pr["LEGIT"] - 0.3) < 1e-9
    assert split_of("a") == split_of("a") and sum(map(len, split_rows(rows).values())) == 40
    try:
        tok = AutoTokenizer.from_pretrained(resolve_checkpoint_dir("multilingual") + "/tokenizer")
    except Exception:
        return  # tokenizer not cached offline
    items, skipped = items_from_rows(tok, back[:5], 512, 256)
    assert len(items) == 20 and not skipped
