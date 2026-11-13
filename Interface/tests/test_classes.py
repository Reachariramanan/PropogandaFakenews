"""Always-both flow (E4b): propaganda is never gated; step1.classes and step2.propaganda.classes on every analysis,
on cached Workbench results and on /api/rt/item analysis rows. No claim is made that either is a calibrated probability."""
import json
import os
import unittest

import claimcheck
import engine_adapter as ea
import fake_engine
import rt_jobs
import test_claimcheck as tc
import test_rt_workbench as wb
from fake_engine import v1_response
from test_claimcheck import FLOOD, LEGIT, NOVEL

setUpModule, tearDownModule = tc.setUpModule, tc.tearDownModule

HEATMAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports", "prop_lab", "v1", "heatmap.json")
ALL14 = {b: 0.0 for b in ea.BUCKETS}


def span(tech, conf, start=0, text=None):
    text = text or NOVEL[start:start + 5]
    return {"technique": tech, "start": start, "end": start + len(text), "text": text, "confidence": conf}


def strict(obj):
    def bad(c):
        raise ValueError(c)
    return json.loads(json.dumps(obj, allow_nan=False), parse_constant=bad)


class Buckets(unittest.TestCase):
    def test_names_equal_the_prop_lab_heatmap(self):
        if not os.path.exists(HEATMAP):
            self.skipTest("heatmap.json not present")
        with open(HEATMAP) as f:
            self.assertEqual(list(ea.BUCKETS), json.load(f)["techniques"])
        self.assertEqual(set(ea.BUCKETS), {v for v in ea.PTC_MERGE.values() if v})

    def test_all_14_present_with_explicit_zeros_when_no_spans(self):
        classes, basis, note = ea.propaganda_classes({"techniques": []})
        self.assertEqual(len(classes), 14)
        self.assertEqual({c["technique"] for c in classes}, set(ea.BUCKETS))
        self.assertTrue(all(c["probability"] == 0.0 and c["basis"] == "span_max" for c in classes))
        self.assertEqual((basis, note), ("span_max", "highest span confidence, not a calibrated probability"))
        self.assertEqual([c["technique"] for c in classes], list(ea.BUCKETS))  # ties keep the fixed bucket order

    def test_span_max_takes_the_highest_confidence_per_bucket_sorted_desc(self):
        prop = {"techniques": [span("Doubt", 0.4), span("Doubt", 0.7, 6), span("Loaded_Language", 0.55, 12),
                               span("Whataboutism", 0.2, 18), span("Straw_Men", 0.3, 24)]}
        classes, _, _ = ea.propaganda_classes(prop)
        probs = [c["probability"] for c in classes]
        self.assertEqual(probs, sorted(probs, reverse=True))
        got = {c["technique"]: c["probability"] for c in classes}
        self.assertEqual(got["Doubt"], 0.7)
        self.assertEqual(got["Loaded_Language"], 0.55)
        self.assertEqual(got["Whataboutism,Straw_Men,Red_Herring"], 0.3)  # merged bucket, max of its members
        self.assertEqual([c["technique"] for c in classes][:2], ["Doubt", "Loaded_Language"])
        self.assertEqual(sum(1 for p in probs if p == 0.0), 11)

    def test_engine_technique_probs_win_and_switch_the_basis(self):
        tp = dict(ALL14, Slogans=0.8, Doubt=0.1)
        prop = {"techniques": [span("Doubt", 0.9)], "technique_probs": tp}
        classes, basis, note = ea.propaganda_classes(prop)
        self.assertEqual((classes[0]["technique"], classes[0]["probability"]), ("Slogans", 0.8))
        self.assertEqual(classes[1]["technique"], "Doubt")
        self.assertEqual(classes[1]["probability"], 0.1)  # the engine's number, not the span confidence
        self.assertTrue(all(c["basis"] == "engine" for c in classes))
        self.assertEqual((basis, note), ("engine", ea.BASIS_NOTES["engine"]))
        self.assertEqual(len(classes), 14)

    def test_incomplete_or_malformed_technique_probs_are_ignored(self):
        partial = {k: 0.5 for k in list(ea.BUCKETS)[:13]}
        for bad in (partial, dict(ALL14, Doubt=1.5), dict(ALL14, Doubt=float("nan")), dict(ALL14, Doubt="x"),
                    dict(ALL14, Doubt=True), [1, 2], "x"):
            self.assertIsNone(ea.parse_technique_probs(bad), bad)
            self.assertEqual(ea.propaganda_classes({"techniques": [], "technique_probs": bad})[1], "span_max")

    def test_ptc_names_in_technique_probs_are_merged(self):
        raw = {k: 0.0 for k in ea.PTC_MERGE if k in ea.BUCKETS or ea.PTC_MERGE[k]}
        raw["Straw_Men"] = 0.6
        raw["Whataboutism"] = 0.2
        tp = ea.parse_technique_probs(raw)
        self.assertEqual(tp["Whataboutism,Straw_Men,Red_Herring"], 0.6)

    def test_confidence_none_counts_as_zero_and_is_clamped(self):
        classes, _, _ = ea.propaganda_classes({"techniques": [span("Doubt", None), span("Slogans", 1.4, 6)]})
        got = {c["technique"]: c["probability"] for c in classes}
        self.assertEqual((got["Doubt"], got["Slogans"]), (0.0, 1.0))

    def test_verdict_classes(self):
        fc = {"FAKE": 0.9, "MISLEADING": 0.03, "LEGIT": 0.05, "INCONCLUSIVE": 0.02}
        cl = ea.verdict_classes(fc, 0.6)
        self.assertEqual([c["label"] for c in cl], ["FAKE", "MISLEADING", "LEGIT", "INCONCLUSIVE", "PROPAGANDA"])
        self.assertEqual([c["secondary"] for c in cl], [False] * 4 + [True])
        self.assertEqual(cl[-1]["probability"], 0.6)
        self.assertEqual([c["label"] for c in ea.verdict_classes(fc)], ["FAKE", "MISLEADING", "LEGIT", "INCONCLUSIVE"])
        self.assertEqual(ea.verdict_classes({}), [])
        self.assertEqual(ea.verdict_classes(None, 0.5), [])
        self.assertEqual(ea.verdict_classes({"FAKE": 1.0})[1]["probability"], 0.0)  # absent classes are explicit 0s


class AnalyseBlocks(tc.Base):
    def engine1(self, make):
        return self.engine(make=make, path="/api/check_v1")

    def test_legit_leaning_item_still_returns_propaganda_and_both_blocks(self):
        spans = [span("Loaded_Language", 0.62)]
        with self.engine1(lambda t: v1_response("LEGIT", 0.05, spans=spans)):
            st, _, b = self.analyse(NOVEL)
        self.assertEqual(st, 200)
        self.assertEqual(b["step1"]["label"], "LEGIT")
        s2 = b["step2"]
        self.assertEqual((s2["shown"], s2["gated_out"], s2["forced"], s2["available"]), (True, False, False, True))
        cl = s2["propaganda"]["classes"]
        self.assertEqual(len(cl), 14)
        self.assertEqual((cl[0]["technique"], cl[0]["probability"], cl[0]["basis"]), ("Loaded_Language", 0.62, "span_max"))
        self.assertEqual(s2["propaganda"]["basis"], "span_max")
        self.assertIn("not a calibrated probability", s2["propaganda"]["basis_note"])
        self.assertNotIn("classes", b["engine"])

    def test_step1_classes_come_from_fake_call_with_raw_propaganda_secondary(self):
        with self.engine1(lambda t: v1_response("FAKE", 0.93)):
            _, _, b = self.analyse(NOVEL)
        cl = b["step1"]["classes"]
        self.assertEqual([c["label"] for c in cl], ["FAKE", "MISLEADING", "LEGIT", "INCONCLUSIVE", "PROPAGANDA"])
        self.assertEqual({c["label"]: c["probability"] for c in cl},
                         {"FAKE": 0.9, "MISLEADING": 0.03, "LEGIT": 0.05, "INCONCLUSIVE": 0.02, "PROPAGANDA": 0.6})
        self.assertEqual([c["label"] for c in cl if c["secondary"]], ["PROPAGANDA"])
        self.assertIn("uncalibrated", b["step1"]["classes_note"])

    def test_strong_index_match_still_gets_engine_classes(self):
        with self.engine1(lambda t: v1_response("LEGIT", 0.1)):
            _, _, b = self.analyse(FLOOD)
        self.assertEqual(b["step1"]["lead"], "index")
        self.assertEqual(len(b["step1"]["classes"]), 5)
        self.assertEqual(len(b["step2"]["propaganda"]["classes"]), 14)

    def test_technique_probs_from_the_engine_switch_basis_over_http(self):
        tp = dict(ALL14, Doubt=0.77)
        with self.engine1(lambda t: v1_response("FAKE", 0.9, spans=[span("Doubt", 0.4)], technique_probs=tp)):
            _, _, b = self.analyse(NOVEL)
        p = b["step2"]["propaganda"]
        self.assertEqual((p["basis"], p["basis_note"]), ("engine", ea.BASIS_NOTES["engine"]))
        self.assertEqual((p["classes"][0]["technique"], p["classes"][0]["probability"]), ("Doubt", 0.77))
        self.assertEqual({c["basis"] for c in p["classes"]}, {"engine"})

    def test_no_propaganda_block_means_no_classes_and_not_shown(self):
        with self.engine1(lambda t: {**v1_response("FAKE", 0.9), "propaganda": None}):
            _, _, b = self.analyse(NOVEL)
        self.assertEqual((b["step2"]["shown"], b["step2"]["available"], b["step2"]["propaganda"]), (False, False, None))

    def test_engine_down_has_empty_step1_classes(self):
        _, _, b = self.analyse(NOVEL)
        self.assertEqual(b["step1"]["classes"], [])
        self.assertEqual((b["step2"]["shown"], b["step2"]["gated_out"]), (False, False))

    def test_degraded_still_shows_propaganda_classes(self):
        with self.engine1(lambda t: v1_response(degraded=True, spans=[span("Slogans", 0.5)])):
            _, _, b = self.analyse(NOVEL)
        self.assertEqual(b["engine"]["state"], "degraded")
        self.assertTrue(b["step2"]["shown"])
        self.assertEqual(b["step2"]["propaganda"]["classes"][0]["technique"], "Slogans")

    def test_predict_v0_shape_also_gets_both_blocks(self):
        with self.engine(make=lambda t: fake_engine.v0_response("LEGIT", spans=[{"technique": "Doubt", "text": "Zorbleton",
                                                                                   "confidence": 0.5}])):
            _, _, b = self.analyse(NOVEL)
        self.assertEqual(len(b["step2"]["propaganda"]["classes"]), 14)
        self.assertEqual(b["step1"]["classes"][-1]["label"], "PROPAGANDA")

    def test_response_is_strict_json(self):
        with self.engine1(lambda t: v1_response("FAKE", 0.9, spans=[span("Doubt", float("nan"))])):
            st, _, b = self.analyse(NOVEL)
        self.assertEqual(st, 200)
        strict(b)
        self.assertEqual(len(b["step2"]["propaganda"]["classes"]), 14)

    def test_probabilities_are_never_called_calibrated_or_validated(self):
        with self.engine1(lambda t: v1_response("FAKE", 0.9)):
            _, _, b = self.analyse(NOVEL)
        for note in (b["step1"]["classes_note"], b["step2"]["propaganda"]["basis_note"]):
            self.assertNotIn("validated", note)
            self.assertTrue("not a calibrated" in note or "uncalibrated" in note)


class WithBlocks(unittest.TestCase):
    OLD = {"engine": {"state": "ok", "verdict": {"label": "LEGIT", "probs": {"LEGIT": 0.8, "FAKE": 0.2}}},
           "step1": {"label": "LEGIT", "lead": "engine"},
           "step2": {"shown": False, "gated_out": True, "forced": False, "available": True,
                     "propaganda": {"techniques": [span("Doubt", 0.4)], "n": 1}}}

    def test_old_gated_result_is_upgraded_without_mutation(self):
        old = json.loads(json.dumps(self.OLD))
        new = claimcheck.with_blocks(old)
        self.assertEqual(old, self.OLD)
        self.assertEqual((new["step2"]["shown"], new["step2"]["gated_out"]), (True, False))
        self.assertEqual(len(new["step2"]["propaganda"]["classes"]), 14)
        self.assertEqual([c["label"] for c in new["step1"]["classes"]], ["FAKE", "MISLEADING", "LEGIT", "INCONCLUSIVE"])
        self.assertIs(claimcheck.with_blocks(new), new)  # idempotent

    def test_non_bodies_pass_through(self):
        for v in (None, {}, {"step1": 1}, "x"):
            self.assertIs(claimcheck.with_blocks(v), v)


class WorkbenchBlocks(wb.JobBase):
    def test_fresh_and_cached_results_carry_the_blocks(self):
        with self.engine(make=lambda t: v1_response("LEGIT", 0.1, spans=[span("Doubt", 0.5, 0, t[:5])])) as e:
            first = self.finish(self.jobs.submit([wb.NEWS2])["job_id"])
            second = self.finish(self.jobs.submit([wb.NEWS2])["job_id"])
        self.assertEqual(e.count, 1)
        for v, state in ((first, "done"), (second, "cached")):
            it = v["items"][0]
            self.assertEqual(it["state"], state)
            self.assertEqual(len(it["result"]["step2"]["propaganda"]["classes"]), 14)
            self.assertEqual(len(it["result"]["step1"]["classes"]), 5)
            self.assertFalse(it["result"]["step2"]["gated_out"])

    def test_a_result_stored_before_the_flow_change_is_upgraded_on_a_cache_hit(self):
        with self.engine() as e:
            self.finish(self.jobs.submit([wb.NEWS2])["job_id"])
            rj = next(r[2] for r in self.analyses() if r[0] == wb.NEWS2)
            res = json.loads(rj)
            for k in ("classes", "classes_note"):
                res["step1"].pop(k, None)
            res["step2"]["propaganda"].pop("classes", None)
            res["step2"].update(shown=False, gated_out=True)
            import sqlite3
            con = sqlite3.connect(self.db)
            con.execute("UPDATE analyses SET result_json = ? WHERE item_id = ?", (json.dumps(res), wb.NEWS2))
            con.commit()
            con.close()
            v = self.finish(self.jobs.submit([wb.NEWS2])["job_id"])
        it = v["items"][0]
        self.assertEqual((it["state"], e.count), ("cached", 1))
        self.assertTrue(it["result"]["step2"]["shown"])
        self.assertEqual(len(it["result"]["step2"]["propaganda"]["classes"]), 14)
        self.assertEqual(len(it["result"]["step1"]["classes"]), 5)


class RtItemBlocks(wb.WbHttp):
    def test_item_analysis_rows_have_the_blocks(self):
        with self.engine() as e:
            v = self.post("/api/rt/analyse", {"ids": [wb.NEWS2]})[2]
            self.poll(v["job_id"])
        st, _, item = self.get("/api/rt/item/" + wb.NEWS2)
        self.assertEqual(st, 200)
        res = item["analyses"][0]["result"]
        self.assertEqual(len(res["step2"]["propaganda"]["classes"]), 14)
        self.assertEqual(len(res["step1"]["classes"]), 5)
        strict(item)


if __name__ == "__main__":
    unittest.main()
