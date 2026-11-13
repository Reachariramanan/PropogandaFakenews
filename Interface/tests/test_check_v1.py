"""check_v1 engine shape end to end: adapter + /api/analyse against a fake check_v1 server on an ephemeral port."""
import unittest

import claimcheck
import engine_adapter as ea
import fake_engine
import test_claimcheck as tc
from fake_engine import v1_response
from test_claimcheck import FLOOD, NOVEL

setUpModule, tearDownModule = tc.setUpModule, tc.tearDownModule
SPANS = [{"technique": "Loaded_Language", "start": 0, "end": 5, "text": NOVEL[:5], "confidence": 0.6}]


class AdapterV1(unittest.TestCase):
    def test_detected_by_keys_not_url_and_uses_fake_call(self):
        r = ea.normalise(v1_response("MISLEADING", 0.7, raw_label="PROPAGANDA"), NOVEL)
        self.assertEqual(r["engine"], "check_v1")
        self.assertEqual(r["verdict"]["label"], "MISLEADING")
        self.assertEqual(r["verdict"]["fake_score"], 0.7)
        self.assertEqual(r["verdict"]["rationale"], "stacker rationale")
        self.assertEqual(r["laya"]["label"], "FAKE")
        self.assertFalse(r["degraded"])

    def test_basis_none_is_inconclusive(self):
        r = ea.normalise(v1_response("LEGIT", 0.4, basis="none"), NOVEL)
        self.assertEqual(r["verdict"]["label"], "INCONCLUSIVE")
        self.assertIsNone(r["verdict"]["fake_score"])
        self.assertFalse(r["verdict"]["plausibly_fake"])

    def test_degraded_without_verdict_is_accepted(self):
        raw = v1_response(degraded=True, spans=SPANS)
        raw.pop("fake_call")
        r = ea.normalise(raw, NOVEL)
        self.assertTrue(r["degraded"])
        self.assertIsNone(r["verdict"])
        self.assertEqual(len(r["propaganda"]["techniques"]), 1)
        self.assertIn("timeout", r["error"])

    def test_503_carries_retry_after(self):
        with fake_engine.FakeEngine(status=503, headers={"Retry-After": "30"}, path="/api/check_v1") as e:
            with self.assertRaises(ea.EngineError) as c:
                ea.call_engine(e.url, "hello", None, 2)
        self.assertEqual((c.exception.state, c.exception.retry_after), ("busy", 30))


class AnalyseV1(tc.Base):
    def engine1(self, make, **kw):
        return self.engine(make=make, path="/api/check_v1", **kw)

    def test_normal(self):
        with self.engine1(lambda t: v1_response("FAKE", 0.93, spans=SPANS)):
            st, _, b = self.analyse(NOVEL)
        e = b["engine"]
        self.assertEqual((st, e["state"], e["engine"], e["degraded"]), (200, "ok", "check_v1", False))
        self.assertEqual(e["laya"]["label"], "FAKE")
        self.assertEqual(b["step1"]["source"], "model estimate")
        self.assertEqual(b["step1"]["label"], "FAKE")
        self.assertEqual(b["step1"]["plausibility"], 0.93)
        self.assertTrue(b["step2"]["shown"])
        self.assertEqual(b["step2"]["propaganda"]["n"], 1)

    def test_propaganda_top_class_on_raw_verdict_does_not_leak(self):
        with self.engine1(lambda t: v1_response("MISLEADING", 0.66, raw_label="PROPAGANDA")):
            _, _, b = self.analyse(NOVEL)
        self.assertEqual(b["step1"]["label"], "MISLEADING")
        self.assertEqual(b["engine"]["verdict"]["label"], "MISLEADING")
        self.assertAlmostEqual(b["step1"]["plausibility"], 0.66)

    def test_basis_none_gives_inconclusive(self):
        with self.engine1(lambda t: v1_response("FAKE", 0.99, basis="none")):
            _, _, b = self.analyse(NOVEL)
        self.assertEqual(b["step1"]["label"], "INCONCLUSIVE")
        self.assertLess(b["step1"]["plausibility"], 0.5)

    def test_degraded_with_fake_call(self):
        with self.engine1(lambda t: v1_response("FAKE", 0.8, degraded=True, spans=SPANS)) as e:
            _, _, b = self.analyse(NOVEL)
            _, _, b2 = self.analyse(NOVEL)  # degraded is never cached
        self.assertEqual(e.count, 2)
        en = b["engine"]
        self.assertEqual((en["state"], en["degraded"]), ("degraded", True))
        self.assertIn("timed out", en["reason"])
        self.assertNotIn("propaganda", en)
        self.assertEqual(b["step1"]["label"], "FAKE")
        self.assertTrue(b["step2"]["shown"])
        self.assertEqual(b["step2"]["propaganda"]["n"], 1)
        self.assertEqual(b2["engine"]["state"], "degraded")

    def test_degraded_basis_none_presents_no_verdict(self):
        with self.engine1(lambda t: v1_response("FAKE", 0.9, basis="none", degraded=True, spans=SPANS)):
            _, _, b = self.analyse(NOVEL)
        self.assertEqual(b["engine"]["state"], "degraded")
        self.assertIsNone(b["engine"]["verdict"])
        self.assertIsNone(b["step1"]["label"])
        self.assertEqual(b["step1"]["source"], "none")
        self.assertTrue(b["step2"]["shown"])

    def test_degraded_strong_index_keeps_factcheck_lead(self):
        with self.engine1(lambda t: v1_response("LEGIT", 0.1, degraded=True)):
            _, _, b = self.analyse(FLOOD)
        self.assertEqual(b["step1"]["source"], "fact-check")
        self.assertEqual(b["engine"]["state"], "degraded")

    def test_busy_503_uses_retry_after_header(self):
        with self.engine1(lambda t: {"error": "busy"}, status=503, headers={"Retry-After": "30"}):
            st, hd, b = self.analyse(NOVEL)
        self.assertEqual((st, b["engine"]["state"]), (200, "busy"))
        self.assertEqual(hd["retry-after"], "30")
        self.assertEqual(b["engine"]["retry_after_s"], 30)

    def test_413_is_an_error_state(self):
        with self.engine1(lambda t: {"error": "too large"}, status=413):
            st, _, b = self.analyse(NOVEL)
        self.assertEqual((st, b["engine"]["state"]), (200, "error"))
        self.assertIn("413", b["engine"]["reason"])

    def test_predict_v0_fallback_still_works(self):
        with self.engine(make=lambda t: fake_engine.v0_response("FAKE")):
            _, _, b = self.analyse(NOVEL)
        self.assertEqual((b["engine"]["state"], b["engine"]["engine"]), ("ok", "predict_v0"))
        self.assertEqual(b["step1"]["label"], "FAKE")
        self.assertIsNone(b["engine"].get("laya"))

    def test_default_url(self):
        self.assertTrue(claimcheck.ENGINE_URL.endswith("/api/check_v1") or "FNP_ENGINE_URL" in __import__("os").environ)


if __name__ == "__main__":
    unittest.main()
