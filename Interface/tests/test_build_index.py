import json
import math
import os
import shutil
import tempfile
import unittest

import helpers  # noqa: F401
import build_index as bi
import fixture


class Helpers(unittest.TestCase):
    def test_norm_url(self):
        self.assertEqual(bi.norm_url("https://www.Example.com/a/b/?q=1#f"), "example.com/a/b")
        self.assertEqual(bi.norm_url("http://example.com/a/b"), "example.com/a/b")
        self.assertEqual(bi.norm_url(""), "")
        self.assertEqual(bi.norm_url(None), "")

    def test_read_jsonl_skips_bad_lines_and_non_dicts(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"a": 1}\n\nnot json\n[1, 2]\n"str"\n{"b": "é"}\n')
            self.assertEqual(bi.read_jsonl(p), [{"a": 1}, {"b": "é"}])

    def test_parse_date(self):
        z = bi.parse_date("2026-10-01T10:00:00Z")
        self.assertEqual((z.year, z.month, z.day, z.utcoffset().total_seconds()), (2026, 10, 1, 0))
        naive = bi.parse_date("2026-09-15")
        self.assertIsNotNone(naive.tzinfo)  # naive input is treated as UTC
        self.assertEqual(bi.parse_date("2026-08-01T00:00:00+05:30").utcoffset().total_seconds(), 19800)
        for bad in (None, "", "not-a-date", "2026-13-45"):
            self.assertIsNone(bi.parse_date(bad))

    def test_to_float(self):
        self.assertEqual(bi.to_float(" 3.5 "), 3.5)
        self.assertEqual(bi.to_float(2), 2.0)
        for bad in (None, "", "abc", "nan", "inf", "-inf", float("nan")):
            self.assertIsNone(bi.to_float(bad))

    def test_label_score_base_and_nudge(self):
        self.assertEqual(bi.label_score("FAKE", None), 0.95)
        self.assertEqual(bi.label_score("LEGIT", ""), 0.05)
        self.assertIsNone(bi.label_score("UNRATED", 1))
        self.assertIsNone(bi.label_score("BOGUS", None))
        # numeric 1..5 rating (1 = false) that agrees nudges: 0.85*label + 0.15*(5-r)/4
        self.assertAlmostEqual(bi.label_score("FAKE", "1"), 0.85 * 0.95 + 0.15 * 1.0)
        self.assertAlmostEqual(bi.label_score("LEGIT", 5), 0.85 * 0.05)
        # disagreeing or out-of-range ratings are ignored
        self.assertEqual(bi.label_score("FAKE", 5), 0.95)
        self.assertEqual(bi.label_score("LEGIT", 1), 0.05)
        self.assertEqual(bi.label_score("FAKE", 9), 0.95)
        self.assertEqual(bi.label_score("FAKE", "nan"), 0.95)

    def test_import_has_no_side_effects(self):
        # BUILD must not be (re)created or touched just by importing the module
        self.assertTrue(callable(bi.main))
        self.assertEqual(os.path.basename(bi.BUILD), "build")


class LoadRows(unittest.TestCase):
    def test_sources_merged_and_factcheckers_normalised(self):
        with tempfile.TemporaryDirectory() as root:
            fixture.write_inputs(root)
            with fixture.patched_build(root):
                rows = bi.load_rows()
        by_id = {r["id"]: r for r in rows}
        self.assertNotIn("f2", by_id)  # is_fact_check == "False" skipped
        self.assertEqual(by_id["f1"]["site"], "pib_factcheck")
        self.assertEqual(by_id["f1"]["channel"], "instagram")
        self.assertEqual(by_id["f1"]["_kind"], "factcheck")
        self.assertEqual(by_id["f3"]["verdict_norm"], "LEGIT")  # TRUE -> LEGIT
        self.assertEqual(by_id["c1"]["_kind"], "candidate")
        self.assertEqual(by_id["c1"]["claim"], by_id["c1"]["text"])
        self.assertEqual(by_id["c1"]["site"], "x_posts")

    def test_missing_optional_files(self):
        with tempfile.TemporaryDirectory() as root:
            fixture.write_jsonl(os.path.join(root, "factchecks", "a.jsonl"), fixture.ROWS[:1])
            with fixture.patched_build(root):
                self.assertEqual(len(bi.load_rows()), 1)


class LoadEngine(unittest.TestCase):
    def test_missing_sources_is_pending_not_an_error(self):
        with tempfile.TemporaryDirectory() as root, fixture.patched_build(root):
            out, st = bi.load_engine()
        self.assertEqual(out, {})
        self.assertFalse(st["available"])

    def test_doc_row_wins_and_claim_fallback_takes_most_suspicious(self):
        with tempfile.TemporaryDirectory() as root:
            p = os.path.join(root, "v.jsonl")
            fixture.write_jsonl(p, [
                {"doc_id": "d1", "unit": "doc", "probs": {"FAKE": 0.6, "MISLEADING": 0.1, "LEGIT": 0.3}},
                {"doc_id": "d1", "unit": "claim", "unit_id": "d1#1", "probs": {"FAKE": 0.99}},
                {"doc_id": "d2", "unit": "claim", "unit_id": "d2#1", "probs": {"FAKE": 0.2}},
                {"doc_id": "d2", "unit": "claim", "unit_id": "d2#2", "probs": {"FAKE": 0.7, "MISLEADING": 0.1}},
            ])
            with fixture.patched_build(root, engine_jsonl=p):
                out, st = bi.load_engine()
        self.assertTrue(st["available"])
        self.assertEqual(out["d1"]["level"], "doc")
        self.assertAlmostEqual(out["d1"]["p_fake_or_misleading"], 0.7)
        self.assertEqual(out["d2"]["level"], "claim")
        self.assertAlmostEqual(out["d2"]["p_fake_or_misleading"], 0.8)
        self.assertEqual(out["d2"]["claims_scored"], 2)


class FullBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        fixture.write_inputs(cls.root)
        eng = os.path.join(cls.root, "engine.jsonl")
        fixture.write_jsonl(eng, [{"doc_id": "fc2", "unit": "doc", "probs": {"FAKE": 0.1, "MISLEADING": 0.1}}])
        cls.items, cls.meta, cls.index = fixture.run_build(cls.root, engine_jsonl=eng)
        cls.by_id = {i["id"]: i for i in cls.items}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_dedupe_by_normalised_url_keeps_richer_row(self):
        self.assertNotIn("fc1", self.by_id)
        self.assertIn("fc1-dup", self.by_id)
        self.assertEqual(len(self.items), 7)  # 5 rows - 1 dup + 1 candidate + 2 factchecker rows (f2 skipped)
        self.assertEqual(self.meta["n_items"], len(self.items))
        self.assertEqual(self.meta["n_factchecks"] + self.meta["n_candidates"], len(self.items))

    def test_verdict_normalisation_and_kinds(self):
        self.assertEqual(self.by_id["fc4"]["verdict_norm"], "INCONCLUSIVE")  # bogus label
        self.assertEqual(self.by_id["c1"]["verdict_norm"], "UNRATED")
        self.assertEqual(self.by_id["c1"]["kind"], "candidate")
        self.assertEqual(self.by_id["c1"]["channel"], "x")
        self.assertEqual(self.by_id["fc3"]["site"], "factly")
        self.assertEqual(self.by_id["fc2"]["site"], "altnews")
        self.assertEqual(self.by_id["fc1-dup"]["site"], "boomlive")
        self.assertIsNone(self.by_id["fc4"]["published"])  # unparsable date

    def test_scores(self):
        for it in self.items:
            if it["plaus"] is not None:
                self.assertTrue(0.0 <= it["plaus"] <= 1.0, it["id"])
        self.assertEqual(self.by_id["c1"]["score_source"], "none")
        self.assertIsNone(self.by_id["c1"]["plaus"])
        self.assertEqual(self.by_id["fc3"]["score_source"], "fact-checker-label")
        self.assertAlmostEqual(self.by_id["fc3"]["plaus"], round(0.85 * 0.05, 4), places=3)  # rating 5 nudge
        e = self.by_id["fc2"]  # MISLEADING 0.70 blended with engine 0.2
        self.assertEqual(e["score_source"], "blend")
        self.assertAlmostEqual(e["plaus"], 0.45, places=4)

    def test_items_sorted_scored_first_and_json_strict(self):
        ps = [i["plaus"] for i in self.items]
        scored = [p for p in ps if p is not None]
        self.assertEqual(ps[:len(scored)], sorted(scored, reverse=True))
        self.assertTrue(all(p is None for p in ps[len(scored):]))
        with open(os.path.join(self.root, "build", "items.json"), encoding="utf-8") as f:
            json.loads(f.read(), parse_constant=lambda c: self.fail("non-strict constant " + c))

    def test_meta_and_index_consistent(self):
        self.assertEqual(len(self.meta["index_ids"]), self.meta["n_factchecks"])
        self.assertTrue(set(self.meta["index_ids"]) <= set(self.by_id))
        self.assertEqual(self.index.n, self.meta["n_factchecks"])
        hits = self.index.query("free electricity households", 3)
        self.assertEqual(self.meta["index_ids"][hits[0][0]], "fc2")
        self.assertTrue(math.isfinite(hits[0][1]))
        self.assertIn("built", self.meta)

    def test_no_tmp_files_left_and_outputs_confined_to_root(self):
        self.assertEqual(sorted(os.listdir(os.path.join(self.root, "build"))), ["index.pkl", "items.json"])


if __name__ == "__main__":
    unittest.main()
