"""truth_discovery, bp, centrality, tensor_cp, event_bn, propaganda_score, propagation."""
import numpy as np
import pytest

from fnp.math.bp import exact_marginals, loopy_bp
from fnp.math.centrality import claim_centrality
from fnp.math.event_bn import timeline_inconsistency
from fnp.math.propaganda_score import p_prop, span_prf, technique_entropy
from fnp.math.propagation import propagation_features
from fnp.math.tensor_cp import build_tensor, coordination_score
from fnp.math.truth_discovery import dawid_skene, truthfinder


def _planted(seed=0, S=8, I=400, K=2):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, K, I)
    acc = np.linspace(0.55, 0.95, S)
    obs = np.where(rng.random((S, I)) < acc[:, None], z, 1 - z)
    obs[rng.random((S, I)) < 0.2] = -1
    return z, acc, obs


def test_dawid_skene_recovers_confusions():
    z, acc, obs = _planted()
    r = dawid_skene(obs)
    assert (r["posterior"].argmax(1) == z).mean() > 0.95
    assert np.allclose(r["reliability"], acc, atol=0.07)
    assert r["converged"]


def test_truthfinder_runs():
    z, acc, obs = _planted(1)
    r = truthfinder(obs)
    assert (r["posterior"].argmax(1) == z).mean() > 0.9
    assert np.corrcoef(r["trust"], acc)[0, 1] > 0.8


def test_bp_exact_on_tree_and_loop():
    p = [0.9, 0.5, 0.3, 0.6, 0.5]
    tree = [(0, 1, 1, 1.0), (1, 2, -1, 0.7), (1, 3, 1, 0.4), (3, 4, -1, 1.2)]
    r = loopy_bp(p, tree)
    assert r["converged"] and np.allclose(r["marginals"], exact_marginals(p, tree), atol=1e-6)
    loop = tree + [(4, 0, 1, 0.5)]
    r2 = loopy_bp(p, loop)
    assert r2["converged"] and np.allclose(r2["marginals"], exact_marginals(p, loop), atol=0.08)


def test_centrality():
    e = [dict(stance="support", origin_key="a", reliability=0.9, independence=1, relevance=1, stance_conf=1),
         dict(stance="refute", origin_key="b", reliability=0.2, independence=1, relevance=1, stance_conf=1)]
    c = claim_centrality({"x": e, "y": []})
    assert c["x"] > 0.5 and c["y"] == 0.5


def test_cp_planted_coordinated_block():
    rng = np.random.default_rng(0)
    S, I, T, E = 30, 10, 12, 3
    X = rng.poisson(0.3, (S, I, T, E)).astype(float)
    null = coordination_score(X.copy())
    X[np.ix_([3, 7, 11, 19], [2, 5], [6], [0])] += 15
    r = coordination_score(X)
    assert set(r["top_sources"]) >= {3, 7, 11, 19} and len(r["top_sources"]) <= 6
    assert r["score"] > 0.4 and r["score"] > null["score"] + 0.2
    assert build_tensor([(0, 0, 0, 0), (0, 0, 0, 0, 2)], (1, 1, 1, 1))[0, 0, 0, 0] == 3


def test_noisy_or_timeline():
    assert timeline_inconsistency([]) == pytest.approx(0.02)
    p = timeline_inconsistency([{"name": "media_older_than_event", "p": 1.0},
                                {"name": "date_order_impossible", "p": 0.5}])
    assert p == pytest.approx(1 - 0.98 * 0.1 * (1 - 0.475))


def test_propaganda_scores_and_span_f1():
    spans = [dict(technique="Loaded_Language", confidence=1.0), dict(technique="Doubt", confidence=0.5)]
    assert p_prop(spans) == pytest.approx(1 - 0.5 * 0.75)
    assert technique_entropy(spans) == pytest.approx(1.0)
    gold = [dict(doc_id="d", start=10, end=20, technique="Loaded_Language")]
    pred = [dict(doc_id="d", start=15, end=25, technique="Loaded_Language"),
            dict(doc_id="d", start=40, end=45, technique="Doubt")]
    r = span_prf(pred, gold)
    assert r["precision"] == pytest.approx(0.5 / 2) and r["recall"] == pytest.approx(0.5)
    pred[0]["technique"] = "Doubt"
    assert span_prf(pred, gold)["f1"] == 0
    assert span_prf(pred, gold, match_technique=False)["recall"] == pytest.approx(0.5)
    assert span_prf([dict(doc_id="e", start=10, end=20)], gold, False)["f1"] == 0


def test_propagation_nan_safe():
    f = propagation_features({"dup_rate": 1.4, "burst_intensity": None})
    assert f["dup_rate"] == 1.0 and np.isnan(f["burst_intensity"]) and np.isnan(f["coordination"])
