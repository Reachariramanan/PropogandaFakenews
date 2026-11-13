from fnp.eval.ptc_scorer import score

G = {"a": [(0, 10, "Loaded_Language"), (20, 30, "Straw_Men")]}


def test_perfect():
    r = score(G, G)
    assert r["SI"]["f1"] == 1.0 and r["TC_span"]["f1"] == 1.0


def test_wrong_technique_keeps_si_zeroes_tc():
    p = {"a": [(0, 10, "Doubt"), (20, 30, "Doubt")]}
    r = score(G, p)
    assert r["SI"]["f1"] == 1.0 and r["TC_span"]["f1"] == 0.0


def test_merged_class_counts():
    p = {"a": [(0, 10, "Loaded_Language"), (20, 30, "Red_Herring")]}
    assert score(G, p)["TC_span"]["f1"] == 1.0


def test_partial_overlap_and_empty():
    p = {"a": [(0, 5, "Loaded_Language")]}
    r = score(G, p)
    assert abs(r["SI"]["precision"] - 1.0) < 1e-9 and abs(r["SI"]["recall"] - 0.25) < 1e-9
    assert score(G, {})["SI"]["f1"] == 0.0
