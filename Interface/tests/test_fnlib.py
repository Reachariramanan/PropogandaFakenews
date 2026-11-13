import json
import unittest

import helpers  # noqa: F401  (sys.path / env setup)
import fnlib


def strict_dumps(v):
    return json.dumps(v, allow_nan=False)


class CleanJson(unittest.TestCase):
    def test_nan_inf_become_none(self):
        self.assertIsNone(fnlib.clean_json(float("nan")))
        self.assertIsNone(fnlib.clean_json(float("inf")))
        self.assertIsNone(fnlib.clean_json(float("-inf")))

    def test_finite_and_scalars_unchanged(self):
        for v in (0.0, -1.5, 3, 10 ** 30, True, False, None, "x", "", "NaN"):
            self.assertEqual(fnlib.clean_json(v), v)
            self.assertIs(type(fnlib.clean_json(v)), type(v))

    def test_nested_structures(self):
        nan, inf = float("nan"), float("inf")
        src = {"a": [1, nan, {"b": inf, "c": [(2.0, -inf)], "d": "ok"}], "e": {"f": {"g": nan}}, "h": (nan, 1)}
        out = fnlib.clean_json(src)
        self.assertEqual(out, {"a": [1, None, {"b": None, "c": [[2.0, None]], "d": "ok"}],
                               "e": {"f": {"g": None}}, "h": [None, 1]})
        strict_dumps(out)  # would raise ValueError if any NaN/inf survived

    def test_raw_output_is_not_strict_but_cleaned_is(self):
        src = {"x": [float("nan")]}
        with self.assertRaises(ValueError):
            strict_dumps(src)
        self.assertEqual(strict_dumps(fnlib.clean_json(src)), '{"x": [null]}')

    def test_does_not_mutate_input(self):
        src = {"a": [float("nan")]}
        fnlib.clean_json(src)
        self.assertTrue(src["a"][0] != src["a"][0])  # still NaN

    def test_empty_containers(self):
        self.assertEqual(fnlib.clean_json({}), {})
        self.assertEqual(fnlib.clean_json([]), [])


class Tokenizer(unittest.TestCase):
    def test_unigrams_bigrams_stopwords(self):
        t = fnlib.tok("The viral flood in Delhi 2018")
        self.assertIn("flood", t)
        self.assertIn("delhi", t)
        self.assertIn("flood_delhi", t)  # "in" is a stopword, so the bigram skips it
        self.assertNotIn("the", t)
        self.assertNotIn("2018", t)  # pure digits dropped

    def test_indic_script_kept(self):
        self.assertIn("दिल्ली", fnlib.tok("दिल्ली में बाढ़"))

    def test_empty(self):
        self.assertEqual(fnlib.tok(""), [])
        self.assertEqual(fnlib.tok(None), [])


class Signals(unittest.TestCase):
    def test_empty_text_has_zero_signals(self):
        s = fnlib.signals("")
        self.assertEqual(s["chars"], 0)
        self.assertEqual(s["sensational"], 0.0)
        strict_dumps(s)

    def test_shouty_text_scores_higher_than_calm(self):
        calm = fnlib.signals("The ministry published its annual report on Tuesday.")
        loud = fnlib.signals("BREAKING!!! SHARE this URGENT video before it is deleted!!!")
        self.assertGreater(loud["sensational"], calm["sensational"])
        self.assertIn("share", loud["urgency_terms"])
        self.assertLessEqual(loud["sensational"], 1.0)


class TfidfIndexBackends(unittest.TestCase):
    TEXTS = ["flood video delhi old footage", "free electricity scheme announced prime minister",
             "highway inaugurated kerala", "cricket world cup final result"]

    def check(self, backend):
        idx = fnlib.TfidfIndex()
        idx.backend = backend
        idx.fit(self.TEXTS)
        hits = idx.query("old flood footage from delhi", 3)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], 0)
        self.assertTrue(all(0 < s <= 1.0001 for _, s in hits))
        self.assertEqual(idx.query("zzzz qqqq", 3), [])
        nb = idx.neighbors(k=2, thr=0.05)
        self.assertEqual(len(nb), len(self.TEXTS))
        self.assertTrue(all(j != i for i, row in enumerate(nb) for j, _ in row))  # never its own neighbour

    def test_python_backend(self):
        self.check("python")

    @unittest.skipUnless(fnlib.HAVE_SKLEARN, "scikit-learn not installed")
    def test_sklearn_backend(self):
        self.check("sklearn")

    def test_empty_index(self):
        idx = fnlib.TfidfIndex()
        idx.fit([])
        self.assertEqual(idx.query("anything", 3), [])


if __name__ == "__main__":
    unittest.main()
