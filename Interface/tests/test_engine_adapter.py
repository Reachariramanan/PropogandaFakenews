"""engine_adapter: both response shapes, 18 -> 14 bucket mapping, span repair, transport errors."""
import json
import os
import unittest

import helpers  # noqa: F401
import engine_adapter as ea
import fake_engine

ROOT = os.path.dirname(helpers.INTERFACE)
HEATMAP = os.path.join(ROOT, "reports", "prop_lab", "v1", "heatmap.json")
EIGHTEEN = ("Loaded_Language", "Name_Calling,Labeling", "Repetition", "Exaggeration,Minimisation", "Doubt",
            "Appeal_to_fear-prejudice", "Flag-Waving", "Causal_Oversimplification", "Slogans", "Appeal_to_Authority",
            "Black-and-White_Fallacy", "Thought-terminating_Cliches", "Whataboutism", "Reductio_ad_hitlerum",
            "Red_Herring", "Bandwagon", "Obfuscation,Intentional_Vagueness,Confusion", "Straw_Men")
TEXT = "The corrupt traitors are destroying our glorious nation, everyone knows it."


class Buckets(unittest.TestCase):
    def test_fourteen_buckets_equal_heatmap(self):
        if not os.path.isfile(HEATMAP):
            self.skipTest("reports/prop_lab/v1/heatmap.json not built")
        with open(HEATMAP, encoding="utf-8") as f:
            self.assertEqual(list(ea.BUCKETS), json.load(f)["techniques"])

    def test_eighteen_map_into_buckets(self):
        got = {t: ea.merge_technique(t) for t in EIGHTEEN}
        self.assertEqual(set(v for v in got.values() if v), set(ea.BUCKETS))
        self.assertIsNone(got["Obfuscation,Intentional_Vagueness,Confusion"])
        for t in ("Whataboutism", "Straw_Men", "Red_Herring"):
            self.assertEqual(got[t], "Whataboutism,Straw_Men,Red_Herring")
        for t in ("Bandwagon", "Reductio_ad_hitlerum"):
            self.assertEqual(got[t], "Bandwagon,Reductio_ad_hitlerum")
        for b in ea.BUCKETS:
            self.assertEqual(ea.merge_technique(b), b)  # merged names map to themselves

    def test_display_spellings_and_junk(self):
        self.assertEqual(ea.merge_technique("Name Calling/Labeling"), "Name_Calling,Labeling")
        self.assertEqual(ea.merge_technique("Appeal to Fear/Prejudice"), "Appeal_to_fear-prejudice")
        self.assertEqual(ea.merge_technique("Whataboutism/Straw Men/Red Herring"), "Whataboutism,Straw_Men,Red_Herring")
        self.assertIsNone(ea.merge_technique("nonsense"))
        self.assertIsNone(ea.merge_technique(None))


class Spans(unittest.TestCase):
    def test_offsets_kept_when_consistent(self):
        s = ea.repair_span({"technique": "Loaded_Language", "start": 4, "end": 11, "text": "corrupt", "confidence": 0.8}, TEXT)
        self.assertEqual((s["start"], s["end"], s["text"], s["confidence"]), (4, 11, "corrupt", 0.8))

    def test_missing_offsets_located_by_first_match(self):
        s = ea.repair_span({"technique": "Name_Calling,Labeling", "text": "traitors"}, TEXT)
        self.assertEqual(TEXT[s["start"]:s["end"]], "traitors")
        self.assertIsNone(s["confidence"])

    def test_wrong_offsets_relocated(self):
        s = ea.repair_span({"technique": "Loaded_Language", "start": 0, "end": 3, "text": "glorious"}, TEXT)
        self.assertEqual(TEXT[s["start"]:s["end"]], "glorious")

    def test_case_insensitive_fallback(self):
        s = ea.repair_span({"technique": "Doubt", "text": "GLORIOUS"}, TEXT)
        self.assertEqual(TEXT[s["start"]:s["end"]], "glorious")

    def test_unplaceable_dropped(self):
        for sp in ({"technique": "Doubt", "text": "absent phrase"}, {"technique": "Doubt"},
                   {"technique": "Doubt", "start": 5, "end": 500}, {"technique": "Obfuscation", "text": "corrupt"},
                   {"technique": "Bogus", "text": "corrupt"}, "x"):
            self.assertIsNone(ea.repair_span(sp, TEXT), sp)

    def test_offsets_without_text_filled_from_slice(self):
        s = ea.repair_span({"technique": "Slogans", "start": 4, "end": 11}, TEXT)
        self.assertEqual(s["text"], "corrupt")

    def test_dedupe_sort_and_drop_count(self):
        kept, dropped = ea.repair_spans([
            {"technique": "Doubt", "text": "everyone knows it"}, {"technique": "Loaded_Language", "text": "corrupt"},
            {"technique": "Loaded_Language", "text": "corrupt"}, {"technique": "Doubt", "text": "nope"}], TEXT)
        self.assertEqual([k["technique"] for k in kept], ["Loaded_Language", "Doubt"])
        self.assertEqual(dropped, 1)


class Shapes(unittest.TestCase):
    def test_predict_v0(self):
        spans = [{"technique": "Name_Calling,Labeling", "text": "traitors", "start": 12, "end": 20, "confidence": 0.6},
                 {"technique": "Straw_Men", "text": "everyone knows it"},
                 {"technique": "Obfuscation,Intentional_Vagueness,Confusion", "text": "nation"}]
        raw = fake_engine.v0_response("MISLEADING", spans=spans)
        raw["spans_summary"]["one_sided"] = True
        r = ea.normalise(raw, TEXT)
        self.assertEqual(r["engine"], "predict_v0")
        self.assertEqual(set(r["verdict"]), {"label", "probs", "plausibly_fake", "rationale", "decider"})
        self.assertEqual((r["verdict"]["label"], r["verdict"]["rationale"], r["verdict"]["decider"]),
                         ("MISLEADING", "Gemma says so.", "stacker"))
        p = r["propaganda"]
        self.assertIsNone(p["is_propaganda_p"])
        self.assertTrue(p["one_sided"])
        self.assertEqual([t["technique"] for t in p["techniques"]],
                         ["Name_Calling,Labeling", "Whataboutism,Straw_Men,Red_Herring"])
        self.assertEqual(p["dropped"], 1)
        for t in p["techniques"]:
            self.assertEqual(TEXT[t["start"]:t["end"]], t["text"])
        self.assertEqual(r["secs"], 1.2)

    def test_predict_v0_propaganda_block_with_span_objects(self):
        raw = fake_engine.v0_response()
        raw.pop("spans")
        raw["propaganda"] = {"n": 1, "one_sided": False, "techniques": [{"technique": "Doubt", "text": "everyone knows it"}]}
        self.assertEqual(ea.normalise(raw, TEXT)["propaganda"]["techniques"][0]["technique"], "Doubt")
        raw["propaganda"]["techniques"] = ["Doubt"]  # names only: nothing placeable
        self.assertEqual(ea.normalise(raw, TEXT)["propaganda"]["techniques"], [])

    def test_check_v1(self):
        raw = {"verdict": {"label": "FAKE", "probs": {"FAKE": 0.9, "LEGIT": 0.1}, "plausibly_fake": True,
                           "prediction_set": ["FAKE"], "rationale": "r", "decider": "stacker_laya"},
               "propaganda": {"is_propaganda_p": 0.77, "one_sided": True,
                              "techniques": [{"technique": "Loaded_Language", "start": 4, "end": 11, "text": "corrupt",
                                              "confidence": 0.9}]},
               "model_versions": {"laya": "rl_v1"}, "secs": 2.5}
        r = ea.normalise(raw, TEXT)
        self.assertEqual(r["engine"], "check_v1")
        self.assertEqual(r["propaganda"]["is_propaganda_p"], 0.77)
        self.assertEqual(r["model_versions"], {"laya": "rl_v1"})
        self.assertNotIn("prediction_set", r["verdict"])
        self.assertTrue(r["verdict"]["plausibly_fake"])
        self.assertEqual(r["propaganda"]["techniques"][0]["text"], "corrupt")

    def test_non_finite_numbers_dropped(self):
        raw = fake_engine.v0_response(probs={"FAKE": float("nan"), "LEGIT": 0.4}, secs=float("inf"))
        r = ea.normalise(raw, TEXT)
        self.assertEqual(r["verdict"]["probs"], {"LEGIT": 0.4})
        self.assertIsNone(r["secs"])

    def test_bad_shapes_raise(self):
        for bad in (None, [], {}, {"error": "boom"}, {"doc": {"probs": {}}}, {"verdict": {}}):
            with self.assertRaises(ea.AdapterError, msg=repr(bad)):
                ea.normalise(bad, TEXT)


class Transport(unittest.TestCase):
    def test_unreachable(self):
        with self.assertRaises(ea.EngineError) as c:
            ea.call_engine(helpers.DEAD_ENGINE_URL, "hello", None, 1)
        self.assertEqual(c.exception.state, "unavailable")

    def test_timeout_and_http_error_and_garbage(self):
        with fake_engine.FakeEngine(delay=0.8) as e:
            with self.assertRaises(ea.EngineError) as c:
                ea.call_engine(e.url, "hello", None, 0.2)
            self.assertEqual(c.exception.state, "timeout")
        with fake_engine.FakeEngine(status=500) as e:
            with self.assertRaises(ea.EngineError) as c:
                ea.call_engine(e.url, "hello", None, 2)
            self.assertEqual(c.exception.state, "error")
        with fake_engine.FakeEngine(status=503) as e:
            with self.assertRaises(ea.EngineError) as c:
                ea.call_engine(e.url, "hello", None, 2)
            self.assertEqual(c.exception.state, "busy")


if __name__ == "__main__":
    unittest.main()
