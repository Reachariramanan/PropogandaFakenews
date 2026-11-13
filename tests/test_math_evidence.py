"""independence, reliability, woe, dempster_shafer, meta, infogain."""
import numpy as np
import pytest

from fnp.math import dempster_shafer as ds
from fnp.math.independence import (assign_independence, cluster_evidence, cluster_n_eff,
                                   copy_probability, kish_n_eff)
from fnp.math.infogain import kl_bernoulli_bits, leave_one_out_bits
from fnp.math.meta import fixed_and_random, meta_pool
from fnp.math.reliability import reliability_posterior
from fnp.math.woe import fit_kappa, woe_posterior


def ev(stance="support", origin=None, url=None, et="MEDIA_REPORT", **kw):
    d = dict(stance=stance, origin_key=origin, url=url, evidence_type=et, stance_conf=1.0,
             relevance=1.0, source=kw.pop("source", "s"))
    d.update(kw)
    return d


def test_clustering_union_find_and_cosine():
    evs = [ev(origin="A"), ev(origin="A", url="http://x.com/1"), ev(url="https://www.x.com/1/"),
           ev(origin="B"), ev()]
    emb = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0.05, 1]], float)
    lab = cluster_evidence(evs)
    assert lab[0] == lab[1] == lab[2] and lab[3] != lab[0] and lab[4] != lab[3]
    lab2 = cluster_evidence(evs, emb)
    assert lab2[3] == lab2[4]
    assert kish_n_eff([1, 1, 1, 1]) == 4
    assert cluster_n_eff(lab, np.ones(5)) == pytest.approx(25 / 11)


def test_woe_duplicates_do_not_add():
    one = assign_independence([ev(origin="CBI", et="AGENCY_FINDING")])
    five = assign_independence([ev(origin="CBI", et="AGENCY_FINDING", source=f"s{i}") for i in range(5)])
    p1, p5 = woe_posterior(one)["p_true"], woe_posterior(five)["p_true"]
    assert abs(p5 - p1) < 1e-9 and p1 > 0.8
    m1, m5 = meta_pool(one), meta_pool(five)
    assert m5["fe"] == pytest.approx(m1["fe"]) and m5["fe_se"] == pytest.approx(m1["fe_se"], rel=0.05)  # eps adds k*eps precision
    indep = assign_independence([ev(origin=f"o{i}", et="AGENCY_FINDING") for i in range(5)])
    assert woe_posterior(indep)["p_true"] > p1


def test_woe_direction_and_fit_kappa():
    evs = assign_independence([ev("support", "a"), ev("refute", "b", et="COURT_FINDING"),
                               ev("context_mismatch", "c")])
    w = woe_posterior(evs)
    assert w["p_true"] < 0.5 and w["context_mismatch"] > 0
    rng = np.random.default_rng(0)
    net = rng.normal(0, 1.5, 2000)
    y = rng.random(2000) < 1 / (1 + np.exp(-(0.4 + 1.7 * net)))
    f = fit_kappa(net, y)
    assert f["kappa"] == pytest.approx(1.7, abs=0.2)
    assert f["prior"] == pytest.approx(1 / (1 + np.exp(-0.4)), abs=0.05)


def test_reliability_beta():
    r0 = reliability_posterior("MEDIA_REPORT")
    assert r0["mean"] == pytest.approx(0.6) and r0["lo"] < 0.6 < r0["hi"]
    r = reliability_posterior("MEDIA_REPORT", k=2, n=20)
    assert r["mean"] == pytest.approx((2.4 + 2) / 24) and r["hi"] - r["lo"] < r0["hi"] - r0["lo"]


def test_copy_probability_rare_false_values():
    truth = {i: 0 for i in range(20)}
    a = {i: (1 if i < 6 else 0) for i in range(20)}     # 6 shared false values
    b = dict(a)
    c = {i: (2 if i < 6 else 0) for i in range(20)}     # same accuracy, different errors
    assert copy_probability(a, b, truth, 0.7, 0.7)["p_copy"] > 0.95
    assert copy_probability(a, c, truth, 0.7, 0.7)["p_copy"] < 0.2


def test_zadeh_paradox_dempster_vs_yager():
    A, B, Cc = frozenset("A"), frozenset("B"), frozenset("C")
    frame = A | B | Cc
    m1, m2 = {A: 0.99, B: 0.01}, {Cc: 0.99, B: 0.01}
    d = ds.combine([m1, m2], frame, "dempster")
    y = ds.combine([m1, m2], frame, "yager")
    assert d["m"][B] == pytest.approx(1.0) and d["K"] == pytest.approx(0.9999)
    assert y["m"][B] == pytest.approx(1e-4) and y["m"][frame] == pytest.approx(0.9999)


def test_ds_claim_fusion():
    evs = assign_independence([ev("support", "a", et="COURT_FINDING"), ev("refute", "b", et="SOCIAL_POST")])
    r = ds.ds_fuse(evs)
    assert r["bel_T"] > r["bel_F"] > 0 and r["conflict"] > 0
    assert r["pl_T"] - r["bel_T"] == pytest.approx(r["uncertainty"])
    assert ds.ds_fuse([])["uncertainty"] == 1.0


def test_dersimonian_laird_hand_and_bcg():
    y = np.array([0.1, 0.3, 0.35, 0.65, 0.45, 0.15]); v = np.array([0.03, 0.03, 0.05, 0.01, 0.05, 0.02])
    w = 1 / v
    fe = (w * y).sum() / w.sum()
    Q = (w * (y - fe) ** 2).sum()
    tau2 = max(0, (Q - 5) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    r = fixed_and_random(y, v)
    assert r["fe"] == pytest.approx(fe) and r["Q"] == pytest.approx(Q) and r["tau2"] == pytest.approx(tau2)
    assert r["re"] == pytest.approx((ws * y).sum() / ws.sum()) and r["I2"] == pytest.approx((Q - 5) / Q)
    # BCG vaccine trials (Colditz 1994; metafor dat.bcg log RR), DL: mu=-0.7141 se=0.1787 tau2=0.3088 Q=152.23
    yb = [-0.8893, -1.5854, -1.3481, -1.4416, -0.2175, -0.7861, -1.6209, 0.0120, -0.4694, -1.3713, -0.3394, 0.4459, -0.0173]
    vb = [0.3256, 0.1946, 0.4154, 0.0200, 0.0512, 0.0069, 0.2230, 0.0040, 0.0564, 0.0730, 0.0124, 0.5325, 0.0714]
    b = fixed_and_random(yb, vb)
    assert b["re"] == pytest.approx(-0.7141, abs=2e-3) and b["re_se"] == pytest.approx(0.1787, abs=2e-3)
    assert b["tau2"] == pytest.approx(0.3088, abs=3e-3) and b["Q"] == pytest.approx(152.23, abs=0.5)


def test_meta_pool_interval():
    evs = assign_independence([ev("support", f"o{i}", et="AGENCY_FINDING") for i in range(6)])
    m = meta_pool(evs)
    assert 0.5 < m["p_true_lo"] < m["p_pooled"] < m["p_true_hi"] <= 1
    assert meta_pool([])["p_true_lo"] == 0 and meta_pool([])["p_true_hi"] == 1


def test_infogain():
    assert kl_bernoulli_bits(0.5, 0.5) == pytest.approx(0)
    assert kl_bernoulli_bits(1.0, 0.5) == pytest.approx(1.0, abs=1e-6)
    evs = assign_independence([ev("support", "a", et="COURT_FINDING"), ev("support", "b", et="SOCIAL_POST")])
    bits = leave_one_out_bits(evs)
    assert bits[0] > bits[1] > 0
