"""features, rule head, stacker, conformal, calibration, roll-up."""
import numpy as np
import pytest

from fnp.math.calibration import apply_temperature, brier, ece, fit_temperature, risk_coverage, wilson_ci
from fnp.math.conformal import SplitConformal, decide_label
from fnp.math.features import claim_features, doc_rollup, story_graph
from fnp.math.verdict import Stacker, final_verdict, rule_head
from fnp.schema import DECIDABLE, FEATURE_NAMES, LABELS


def ev(stance, origin, et="AGENCY_FINDING"):
    return dict(stance=stance, origin_key=origin, evidence_type=et, stance_conf=0.95, relevance=1.0, source=origin)


CLAIM = {"claim_id": "c1", "check_worthy": 0.9}


def test_claim_features_keys_and_cases():
    f = claim_features(CLAIM, [ev("refute", "a", "FACT_CHECK"), ev("refute", "b", "COURT_FINDING"), ev("refute", "b")],
                       spans=[], kb_hits=[{"score": 0.9}], propagation={"coordination": 0.1})
    assert tuple(f) == FEATURE_NAMES
    assert f["p_true"] < 0.1 and f["n_evidence"] == 3 and f["n_eff"] < 2.01 and f["kb_match"] == 1
    assert f["p_prop"] == 0 and np.isnan(f["burst_intensity"])
    rp = rule_head(f)
    assert max(rp, key=rp.get) == "FAKE" and abs(sum(rp.values()) - 1) < 1e-9
    empty = claim_features(CLAIM, [])
    assert empty["n_eff"] == 0 and np.isnan(empty["p_prop"])
    assert max(rule_head(empty), key=rule_head(empty).get) == "INCONCLUSIVE"


def test_rule_head_legit_propaganda_misleading():
    sup = claim_features(CLAIM, [ev("support", f"o{i}") for i in range(5)], spans=[])
    assert max(rule_head(sup), key=rule_head(sup).get) == "LEGIT"
    spans = [dict(technique="Loaded_Language", confidence=0.9)] * 4
    prop = claim_features(CLAIM, [ev("support", "o1", "MEDIA_REPORT")], spans=spans)
    assert max(rule_head(prop), key=rule_head(prop).get) == "PROPAGANDA"
    cm = claim_features(CLAIM, [ev("context_mismatch", "a", "FACT_CHECK"), ev("support", "b", "SOCIAL_POST")], spans=[])
    assert max(rule_head(cm), key=rule_head(cm).get) == "MISLEADING"


def test_story_graph_and_rollup():
    g = story_graph({"c1": 0.9, "c2": 0.5}, [("c1", "c2", 1, 1.0)])
    assert g["bp_marginal"]["c2"] > 0.5
    v_fake = final_verdict("c1", claim_features(CLAIM, [ev("refute", "a", "FACT_CHECK"), ev("refute", "b", "COURT_FINDING")], spans=[]))
    v_leg = final_verdict("c2", claim_features({"claim_id": "c2", "check_worthy": 0.9}, [ev("support", f"o{i}") for i in range(5)], spans=[]))
    assert v_fake["label"] == "FAKE" and v_leg["label"] == "LEGIT"
    d = doc_rollup([v_fake, v_leg], "d1")
    assert d["label"] == "FAKE" and "LEGIT" in d["secondary"]
    assert doc_rollup([v_leg], "d2")["label"] == "LEGIT" and doc_rollup([], "d3")["label"] == "INCONCLUSIVE"


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    P = rng.dirichlet(np.ones(4), n) * 0.6
    P[np.arange(n), y] += 0.4 * rng.random(n) + 0.2 * (rng.random(n) < 0.7)
    return P / P.sum(1, keepdims=True), y


@pytest.mark.parametrize("method", ["lac", "aps"])
def test_conformal_coverage(method):
    Pc, yc = _synthetic(2000, 1)
    Pt, yt = _synthetic(4000, 2)
    cp = SplitConformal(method, alpha=0.1, randomized=(method == "aps")).fit(Pc, yc)
    cov = cp.predict_mask(Pt)[np.arange(len(yt)), yt].mean()
    assert 0.88 <= cov <= 0.93


def test_decide_label():
    p = {"FAKE": 0.5, "PROPAGANDA": 0.4, "LEGIT": 0.05, "MISLEADING": 0.05}
    d = decide_label(p, ["FAKE", "PROPAGANDA"])
    assert d["label"] == "FAKE" and "PROPAGANDA" in d["secondary"]
    assert decide_label(p, ["FAKE", "LEGIT"])["label"] == "INCONCLUSIVE"
    assert decide_label(p, [])["label"] == "INCONCLUSIVE"
    q = decide_label({"FAKE": 0.35, "PROPAGANDA": 0.0, "LEGIT": 0.35, "MISLEADING": 0.3}, ["FAKE", "LEGIT"])
    assert q["plausibly_fake"] and q["fake_tier"] == "plausible"


def test_stacker_learns_and_calibrates():
    rng = np.random.default_rng(0)
    feats, ys = [], []
    for i in range(300):
        k = i % 4
        f = {n: float(rng.normal()) for n in FEATURE_NAMES}
        f["p_true"] = [0.05, 0.6, 0.95, 0.4][k] + 0.05 * rng.normal()
        f["p_prop"] = [0.2, 0.9, 0.1, 0.2][k] + 0.05 * rng.normal()
        f["context_mismatch"] = [0, 0, 0, 1.0][k] + 0.1 * rng.normal()
        if i % 7 == 0:
            f["i2"] = float("nan")
        feats.append(f); ys.append(DECIDABLE[k])
    st = Stacker().fit(feats[:240], ys[:240], gemma_probs=None)
    P = st.predict_proba(feats[240:])
    acc = (P.argmax(1) == np.array([DECIDABLE.index(v) for v in ys[240:]])).mean()
    assert acc > 0.9 and np.allclose(P.sum(1), 1) and st.T > 0
    v = final_verdict("x", feats[0], stacker=st)
    assert v["decider"] == "stacker" and set(v["probs"]) == set(LABELS)


def test_calibration_tools():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, (3000, 4))
    true_p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=p) for p in true_p])
    over = apply_temperature(true_p, 0.5)
    T = fit_temperature(over, y)
    assert T == pytest.approx(2.0, rel=0.15)
    assert ece(true_p, y) < 0.03 < ece(over, y)
    assert brier(true_p, y) < brier(over, y)
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    rc = risk_coverage(true_p, y)
    assert rc["risk"][len(y) // 10] < rc["risk"][-1]
