import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import claimcheck  # noqa: E402


class AxisLabel(unittest.TestCase):
    def test_propaganda_top_class_falls_back_to_best_fake_axis_class(self):
        v = {"label": "PROPAGANDA", "probs": {"FAKE": 0.36, "LEGIT": 0.30, "MISLEADING": 0.26, "PROPAGANDA": 0.08}}
        self.assertEqual(claimcheck.verdict_axis_label(v), "FAKE")

    def test_other_labels_unchanged(self):
        self.assertEqual(claimcheck.verdict_axis_label({"label": "LEGIT", "probs": {"LEGIT": 0.9}}), "LEGIT")

    def test_propaganda_without_probs_is_inconclusive(self):
        self.assertEqual(claimcheck.verdict_axis_label({"label": "PROPAGANDA"}), "INCONCLUSIVE")

    def test_combine_uses_axis_label_for_model_estimate(self):
        eng = {"state": "ok", "verdict": {"label": "PROPAGANDA", "plausibly_fake": True,
                                           "probs": {"FAKE": 0.36, "LEGIT": 0.30, "MISLEADING": 0.26, "PROPAGANDA": 0.08}},
               "propaganda": {"techniques": [], "one_sided": False}}
        step1, _ = claimcheck.combine({"matches": [], "score": None}, eng, False)
        self.assertEqual(step1["label"], "FAKE")


if __name__ == "__main__":
    unittest.main()
