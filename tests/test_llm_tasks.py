import math

import pytest

from fnp import gemma
from fnp.llm import adjudicate, cache as C, claims, propaganda, redteam, silver, stance, timeline
from fnp.llm import prompts as P
from fnp.llm import validators as V
from fnp.schema import LABELS, PROPAGANDA_TECHNIQUES


class Fake:
    """Stands in for gemma.chat_json: applies the caller's validator like the real one."""
    def __init__(self, obj):
        self.obj, self.calls = obj, []

    def __call__(self, system, user, *, max_tokens=1200, validate=None, retries=1):
        self.calls.append((system, user, max_tokens))
        bad = validate(self.obj) if validate else None
        if bad:
            return {"ok": False, "obj": None, "error": "invalid:" + bad, "tokens": 3, "latency_s": 0, "attempts": 2}
        return {"ok": True, "obj": self.obj, "error": None, "tokens": 7, "latency_s": 0.1, "attempts": 1}


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path):
    C.reset_default_cache(tmp_path / "c.duckdb")
    yield
    C.reset_default_cache()


def use(monkeypatch, obj):
    f = Fake(obj)
    monkeypatch.setattr(gemma, "chat_json", f)
    return f


# ------------------------------------------------------------------ prompts
def test_prompts_cover_everything():
    for t in PROPAGANDA_TECHNIQUES:
        assert t in P.PROPAGANDA_SYSTEM
    for l in LABELS:
        assert l in P.ADJUDICATE_SYSTEM and l in P.SILVER_SYSTEM and l in P.REDTEAM_SYSTEM
    assert "0.99" in P.ADJUDICATE_SYSTEM
    assert P.render("a {{X}} {b}", X="{y}") == "a {y} {b}"
    assert len(set(P.PROMPT_VERSIONS.values())) == 7


# ------------------------------------------------------------------ claims
TEXT_HI = "प्रधानमंत्री ने कहा कि 2024 में भारत की जीडीपी 7% बढ़ी। यह एक शानदार सरकार है।"


def claim_obj(**kw):
    c = {"claim": "India's GDP grew 7% in 2024, per the PM", "claim_type": "statistical",
         "speaker": "Prime Minister", "quoted_or_reported": "reported", "check_worthy": 0.9,
         "date": "2024", "entities": ["India"], "evidence_quote": "2024 में भारत की जीडीपी 7% बढ़ी"}
    c.update(kw)
    return c


def test_claims_good_and_nonverbatim_dropped(monkeypatch):
    f = use(monkeypatch, {"language": "hi", "claims": [
        claim_obj(),
        claim_obj(claim="Fabricated claim", evidence_quote="India GDP grew 9%"),   # not in text
        claim_obj(claim="Empty quote", evidence_quote="")]})
    out = claims.extract_claims("d1", TEXT_HI)
    assert out.ok and len(out) == 1
    c = out[0]
    assert c["doc_id"] == "d1" and c["lang"] == "hi" and c["claim_type"] == "statistical"
    assert c["claim_id"] == V.sha16("d1", c["claim"])
    assert out.dropped["non_verbatim"] == 2
    assert f.calls[0][2] >= 1500


def test_claims_whitespace_case_tolerant_and_cap(monkeypatch):
    text = "Alpha  said that the bridge opened on 5 May.\nBeta ran."
    many = [{"claim": f"claim {i}", "claim_type": "bogus", "check_worthy": i / 11,
             "evidence_quote": "ALPHA said that the bridge opened"} for i in range(12)]
    use(monkeypatch, {"claims": many})
    out = claims.extract_claims("d", text, lang="en")
    assert len(out) == V.MAX_CLAIMS == 8 and out.dropped["over_cap"] == 4
    assert out[0]["check_worthy"] >= out[-1]["check_worthy"] > 0.2     # lowest-scoring dropped
    assert all(c["claim_type"] == "other" and c["lang"] == "en" for c in out)


def test_claims_validator_rejects_bad():
    assert V.validate_claims({"claims": []}) is None
    assert V.validate_claims({"claims": "x"}) is not None
    assert V.validate_claims({"claims": [{"claim": "a"}]}) is not None                  # no quote
    assert V.validate_claims({"claims": [{"claim": "", "evidence_quote": "x"}]}) is not None
    assert V.validate_claims({"claims": [claim_obj(check_worthy=3)]}) is not None


def test_claims_failure_reports_error(monkeypatch):
    use(monkeypatch, {"claims": "nope"})
    out = claims.extract_claims("d", "some text here")
    assert not out.ok and out.error.startswith("invalid") and len(out) == 0


def test_claims_truncation_note(monkeypatch):
    f = use(monkeypatch, {"claims": []})
    out = claims.extract_claims("d", "x" * 7000)
    assert out.truncated and "truncated" in f.calls[0][1] and len(f.calls[0][1]) < 6600


# ------------------------------------------------------------------ propaganda
def test_spans_offsets_and_drops(monkeypatch):
    text = "The corrupt elite are destroying our country. Everyone knows it. Plain fact."
    use(monkeypatch, {"spans": [
        {"fragment": "corrupt elite", "technique": "Name_Calling,Labeling", "confidence": 0.9},
        {"fragment": "Everyone   knows it", "technique": "bandwagon", "confidence": 0.7},   # ws-flex, case-insens technique
        {"fragment": "not in the text", "technique": "Doubt", "confidence": 0.5},
        {"fragment": "Plain fact", "technique": "Made_Up", "confidence": 0.5}],
        "one_sided": 0.8, "framing": "Elite-blaming"})
    out = propaganda.tag_spans("d", text)
    assert out.ok and len(out) == 2
    for s in out:
        assert text[s["start"]:s["end"]] == s["text"]
    assert out[0]["text"] == "corrupt elite" and out[0]["start"] == 4
    assert out[1]["technique"] == "Bandwagon" and out[1]["text"] == "Everyone knows it"
    assert out.dropped["non_verbatim"] == 1 and out.dropped["bad_technique"] == 1
    assert out.one_sided == 0.8 and out.framing == "Elite-blaming"


def test_spans_unicode_offsets(monkeypatch):
    text = "ये गद्दार देश को बर्बाद कर रहे हैं।"
    use(monkeypatch, {"spans": [{"fragment": "गद्दार", "technique": "Name_Calling,Labeling", "confidence": 1}],
                      "one_sided": 0.5, "framing": "x"})
    out = propaganda.tag_spans("d", text)
    assert out[0]["text"] == "गद्दार" and text[out[0]["start"]:out[0]["end"]] == "गद्दार"


def test_propaganda_validator():
    ok = {"spans": [], "one_sided": 0.2, "framing": "neutral"}
    assert V.validate_propaganda(ok) is None
    assert V.validate_propaganda({"spans": [], "framing": "x"}) is not None
    assert V.validate_propaganda({**ok, "one_sided": 1.5}) is not None
    assert V.validate_propaganda({**ok, "spans": [{"technique": "Doubt"}]}) is not None


# ------------------------------------------------------------------ stance
CLAIM = {"claim_id": "c1", "claim": "Video shows X in 2025", "date": "2025", "speaker": None}


def test_stance_good_and_mismatch(monkeypatch):
    use(monkeypatch, {"stance": "Context_Mismatch", "stance_conf": 0.8, "evidence_type": "fact_check",
                      "relevance": 0.9, "context_mismatch_reason": "Footage is from 2019", "rationale": "r"})
    r = stance.judge(CLAIM, "The video is from 2019.", {"source": "BOOM"})
    assert r["ok"] and r["stance"] == "context_mismatch" and r["evidence_type"] == "FACT_CHECK"
    assert r["context_mismatch_reason"] == "Footage is from 2019"


def test_stance_validator_rejects():
    base = {"stance": "support", "stance_conf": 0.5, "evidence_type": "UNKNOWN", "relevance": 0.5}
    assert V.validate_stance(base) is None
    assert V.validate_stance({**base, "stance": "maybe"}) is not None
    assert V.validate_stance({**base, "stance_conf": 2}) is not None
    assert V.validate_stance({**base, "stance": "context_mismatch"}) is not None   # needs reason
    assert V.validate_stance({**base, "relevance": None}) is not None
    assert V.parse_stance({**base, "evidence_type": "weird"})["evidence_type"] == "UNKNOWN"
    assert V.parse_stance({**base, "context_mismatch_reason": "x"})["context_mismatch_reason"] is None


def test_stance_failure_and_empty(monkeypatch):
    use(monkeypatch, {"stance": "bad"})
    r = stance.judge(CLAIM, "text")
    assert not r["ok"] and r["stance"] == "neutral" and r["stance_conf"] == 0
    assert stance.judge(CLAIM, "  ")["error"] == "empty_evidence"


def test_judge_many_order(monkeypatch):
    use(monkeypatch, {"stance": "support", "stance_conf": 1, "evidence_type": "MEDIA_REPORT", "relevance": 1})
    res = stance.judge_many([(CLAIM, f"ev {i}", {"source": "s"}) for i in range(5)], concurrency=3)
    assert len(res) == 5 and all(r["ok"] for r in res)


# ------------------------------------------------------------------ timeline
def test_timeline(monkeypatch):
    cs = [{"claim_id": "a", "claim": "x", "date": "2024"}, {"claim_id": "b", "claim": "y", "date": "2019"}]
    use(monkeypatch, {"flags": [
        {"flag": "Media Predates Event", "prob": 0.9, "claim_ids": ["a", "zzz"], "reason": "r"},
        {"flag": "other", "prob": 0.5, "claim_ids": ["unknown"], "reason": "r"}]})
    out = timeline.check(cs)
    assert out == [{"flag": "media_predates_event", "prob": 0.9, "claim_ids": ["a"], "reason": "r"}]
    assert timeline.check([]) == []
    assert V.validate_timeline({"flags": [{"flag": "x", "prob": 7, "claim_ids": []}]}) is not None


# ------------------------------------------------------------------ adjudicate / silver / redteam
STATE = {"claims": [{"claim_id": "c1", "claim": "x"}], "features": {"p_true": 0.2},
         "evidence": [{"evidence_id": "e1", "stance": "refute", "evidence_type": "FACT_CHECK", "reliability": 0.9}],
         "spans_summary": {"n": 0}, "propagation": {}}


def test_probs_renormalised_and_label_argmax(monkeypatch):
    use(monkeypatch, {"probs": {"fake": 6, "LEGIT": 2, "misleading": 2}, "label": "LEGIT",
                      "rationale": "r", "key_evidence_ids": ["e1", "nope"]})
    r = adjudicate.decide(STATE)
    assert r["ok"] and math.isclose(sum(r["probs"].values()), 1.0)
    assert list(r["probs"]) == list(LABELS) and r["label"] == "FAKE"       # argmax overrides stated label
    assert r["key_evidence_ids"] == ["e1"]
    assert r["probs"]["PROPAGANDA"] > 0                                    # floored, finite logs


def test_adjudicator_rejects_raw_text_state():
    with pytest.raises(ValueError):
        adjudicate.decide({**STATE, "article_text": "x"} | {"text": "full article"})


def test_adjudicator_truncates_long_strings(monkeypatch):
    f = use(monkeypatch, {"probs": {"INCONCLUSIVE": 1}})
    adjudicate.decide({**STATE, "note": "z" * 5000})
    assert "z" * 600 not in f.calls[0][1]


def test_verdict_validator():
    assert V.validate_verdict({"probs": {"FAKE": 0.5, "LEGIT": 0.5}}) is None
    assert V.validate_verdict({"probs": {"FAKE": 0, "LEGIT": 0}}) is not None
    assert V.validate_verdict({"probs": {"FAKE": -1}}) is not None
    assert V.validate_verdict({"probs": {"TRUE": 1}}) is not None
    assert V.validate_verdict({"probs": "x"}) is not None
    assert V.validate_verdict({"probs": {"FAKE": 1}, "label": "MAYBE"}) is not None


def test_adjudicate_failure_is_flagged(monkeypatch):
    use(monkeypatch, {"probs": {"FAKE": 0, "LEGIT": 0}})
    r = adjudicate.decide(STATE)
    assert not r["ok"] and r["label"] == "INCONCLUSIVE" and r["probs"]["INCONCLUSIVE"] == 1.0


def test_silver(monkeypatch):
    use(monkeypatch, {"probs": {"FAKE": 3, "MISLEADING": 1}, "label": "FAKE", "rationale": "old video"})
    r = silver.label("Video shows something")
    assert r["ok"] and r["label"] == "FAKE" and math.isclose(sum(r["probs"].values()), 1)
    assert "key_evidence_ids" not in r
    assert silver.label(CLAIM)["ok"]


def test_redteam(monkeypatch):
    use(monkeypatch, {"issues": [{"issue": "Evidence loop", "severity": "HIGH"}],
                      "adjusted_probs": {"FAKE": 2, "INCONCLUSIVE": 2}, "confidence_adjustment": -0.3})
    r = redteam.review(STATE, {"probs": {"FAKE": 0.9, "INCONCLUSIVE": 0.1}, "label": "FAKE"})
    assert r["ok"] and r["issues"] == [{"issue": "Evidence loop", "severity": "high"}]
    assert math.isclose(sum(r["adjusted_probs"].values()), 1) and r["confidence_adjustment"] == -0.3


def test_redteam_validator_and_failure(monkeypatch):
    good = {"issues": [], "adjusted_probs": {"FAKE": 1}, "confidence_adjustment": 0}
    assert V.validate_redteam(good) is None
    assert V.validate_redteam({**good, "issues": [{"issue": "x", "severity": "dire"}]}) is not None
    assert V.validate_redteam({**good, "confidence_adjustment": "a"}) is not None
    assert V.validate_redteam({**good, "confidence_adjustment": 5}) is not None
    use(monkeypatch, {"issues": 3})
    r = redteam.review(STATE, {"probs": {"FAKE": 3, "LEGIT": 1}})
    assert not r["ok"] and math.isclose(r["adjusted_probs"]["FAKE"], 0.75, abs_tol=0.01)


def test_task_cache_hit(monkeypatch):
    f = use(monkeypatch, {"probs": {"FAKE": 1}})
    silver.label("same text")
    r = silver.label("same text")
    assert len(f.calls) == 1 and r["meta"]["cached"] is True
