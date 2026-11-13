import math
import numpy as np
from fnp.rl.routing_bandit import RoutingBandit, fallback_policy, context_vector, COSTS
from fnp.rl.voi_stop import evoi, should_continue


def test_bandit_converges_to_best_arm(tmp_path):
    rng = np.random.default_rng(0)
    # reward: laya good when confident, gemma flat-mediocre, human costs 1 -> best arm depends on x
    def rewards(x):
        return np.array([-0.001 - 2.0 * (1 - x[0]), -0.05 - 0.4, -1.0]) + rng.normal(0, 0.05, 3)
    X = np.array([[c, 0, 0, 0, 0, 0, 0, 1.0] for c in rng.uniform(0, 1, 600)])
    R = np.array([rewards(x) for x in X])
    b = RoutingBandit(seed=1).fit_replay(X, R, epochs=1)
    assert b.choose(np.array([0.99, 0, 0, 0, 0, 0, 0, 1.0])) == "laya"
    assert b.choose(np.array([0.2, 0, 0, 0, 0, 0, 0, 1.0])) == "gemma"
    b.save(tmp_path / "b.json")
    assert RoutingBandit.load(tmp_path / "b.json").choose(np.array([0.99, 0, 0, 0, 0, 0, 0, 1.0])) == "laya"
    assert b.choose(np.ones(8), explore=True) in b.arms
    assert fallback_policy(0.9, 2) == "gemma" and fallback_policy(0.9, 1) == "laya" and fallback_policy(0.5, 1) == "gemma"
    assert context_vector([.6, .1, .1, .1, .1]).shape == (8,)


def test_voi_analytic_and_stops_when_decisive():
    # p=0.5; support LR=4 / refute LR=1/4 with prob .5 each -> posteriors .8/.2 -> EVOI = 0.5 - 0.2
    assert abs(evoi(0.0, [0.5, 0.5], [4, 0.25]) - 0.3) < 1e-9
    assert should_continue(0.0, [0.5, 0.5], [4, 0.25], cost=0.05)
    # decisive belief (p~0.998): no single unit can flip it -> EVOI 0 -> stop
    assert evoi(6.0, [0.5, 0.5], [4, 0.25]) == 0.0
    assert not should_continue(6.0, [0.5, 0.5], [4, 0.25], cost=0.01)
