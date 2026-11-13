"""Claim Check step 2: propaganda signal, Laya view, degraded reasons, with_blocks upgrade (synthetic texts only)."""
import os
import re
import unittest

import helpers  # noqa: F401
import claimcheck
import engine_adapter as ea
import test_claimcheck as tc
from fake_engine import v0_response, v1_response

setUpModule, tearDownModule = tc.setUpModule, tc.tearDownModule
SHORT = "The harbour authority published the ferry timetable."
LOUD = "Anyone who disagrees with this plan is plainly an enemy of the people. " + "Nobody serious could object. " * 20
def SP(c, t="harbour"):  # offsets omitted: the adapter locates the text
    return {"technique": "Name_Calling,Labeling", "text": t, "confidence": c}
WEB = os.path.join(helpers.INTERFACE, "web")


def norm(text=SHORT, **kw):
    return ea.normalise(v1_response(**kw), text)


class AdapterFields(unittest.TestCase):
    def test_laya_raw_passthrough(self):
        for raw, want in ((0.0, 0.0), (0.7, 0.7), (None, None), (1.4, None), (True, None), (-0.1, None), ("x", None)):
            self.assertEqual(norm(laya_is_propaganda_raw=raw)["propaganda"]["laya_is_propaganda_raw"], want, raw)

    def test_one_sided_float_and_bool(self):
        for raw, score, flag in ((0.0, 0.0, False), (1.0, 1.0, True), (0.5, 0.5, True), (True, 1.0, True),
                                 (False, 0.0, False), (None, None, None), (3, None, None)):
            p = norm(one_sided=raw)["propaganda"]
            self.assertEqual((p["one_sided_score"], p["one_sided"]), (score, flag), raw)
            self.assertIn(p["one_sided"], (True, False, None))

    def test_text_words_and_n_claims(self):
        r = norm(claims=[{"claim": "x"}, {"claim": "y"}])
        self.assertEqual((r["propaganda"]["text_words"], r["n_claims"]), (7, 2))
        self.assertIsNone(norm(claims=None)["n_claims"])

    def test_degraded_why(self):
        self.assertIsNone(norm()["degraded_why"])
        self.assertEqual(norm(degraded_reason="no_claims_old")["degraded_why"], "no_claims")
        self.assertEqual(norm(degraded=True)["degraded_why"], "timeout")
        self.assertEqual(norm(degraded=True, error="timed out waiting")["degraded_why"], "timeout")
        self.assertEqual(norm(degraded=True, error="gemma 500")["degraded_why"], "unavailable")
        # no claims but an error text is not the no_claims case
        self.assertEqual(norm(degraded=True, claims=[], error="boom")["degraded_why"], "unavailable")

    def test_predict_v0_keeps_new_keys(self):
        from fake_engine import v0_response
        r = ea.normalise(v0_response(), SHORT)
        p = r["propaganda"]
        self.assertIsNone(p["laya_is_propaganda_raw"])
        self.assertEqual(p["text_words"], 7)
        self.assertIsNone(r["n_claims"])
        self.assertIsNone(r["degraded_why"])


class Signal(unittest.TestCase):
    def sig(self, *confs):
        return ea.propaganda_signal({"techniques": [dict(SP(c), start=i, end=i + 1, text="x") for i, c in enumerate(confs)]})

    def test_boundaries(self):
        self.assertEqual(self.sig(0.49)["level"], "weak")
        self.assertEqual(self.sig(0.50)["level"], "strong")
        self.assertEqual(self.sig()["level"], "none")
        self.assertEqual(ea.propaganda_signal(None)["level"], "none")

    def test_counts(self):
        s = self.sig(0.9, 0.3, 0.5)
        self.assertEqual((s["strong_n"], s["weak_n"], s["max_confidence"], s["threshold"]), (2, 1, 0.9, 0.5))
        self.assertEqual(s["techniques"], ["Name_Calling,Labeling"])


class LayaView(unittest.TestCase):
    def view(self, text=SHORT, **kw):
        n = norm(text, **kw)
        return ea.laya_view(n["laya"], n["propaganda"], n["n_claims"], n["degraded_why"])

    def test_ok(self):
        v = self.view(claims=[{}])
        self.assertEqual((v["state"], v["available"], v["label"], v["independent"]), ("ok", True, "FAKE", False))
        self.assertEqual(v["propaganda_head"], 0.0)
        self.assertTrue(v["head_in_range"])
        self.assertTrue(all(isinstance(p["probability"], float) for p in v["probs"]))

    def test_states(self):
        self.assertEqual(self.view(claims=[], laya=None)["state"], "not_run_no_claims")
        self.assertEqual(self.view(degraded_reason="no_claims_old")["state"], "not_run_no_claims")
        self.assertEqual(self.view(degraded=True, claims=[{}], laya=None)["state"], "timeout")
        self.assertEqual(self.view(claims=[{}], laya=None)["state"], "not_returned")
        self.assertEqual(ea.laya_view(None, None, None, None)["state"], "not_returned")
        self.assertIsNone(self.view(claims=[{}], laya_is_propaganda_raw=None)["propaganda_head"])

    def test_compare(self):
        strong, weak = [SP(0.9)], []
        v = self.view(claims=[{}], spans=strong, laya_is_propaganda_raw=0.0)
        self.assertEqual(v["compared"], "differs")
        self.assertIn("article-style", v["compared_note"])
        self.assertIn("0.00", v["compared_note"])
        long_text = SHORT + " word" * 80
        v = self.view(long_text, claims=[{}], spans=strong, laya_is_propaganda_raw=0.0)
        self.assertEqual(v["compared"], "differs")
        self.assertFalse(v["short_text"])
        self.assertEqual(self.view(claims=[{}], spans=strong, laya_is_propaganda_raw=0.8)["compared"], "agree")
        v = self.view(claims=[{}], spans=weak, laya_is_propaganda_raw=0.0)
        self.assertEqual((v["compared"], v["compared_note"]), ("agree", "Agrees: both low."))
        v = self.view(claims=[{}], spans=weak, laya_is_propaganda_raw=0.9)
        self.assertEqual(v["compared"], "differs")
        self.assertEqual(self.view(claims=[{}], laya_is_propaganda_raw=None)["compared"], "not_compared")

    def test_short_boundary(self):
        self.assertTrue(self.view("w " * 59, claims=[{}])["short_text"])
        self.assertFalse(self.view("w " * 60, claims=[{}])["short_text"])


class NewFields(unittest.TestCase):
    def test_engine_flags_pass_through(self):
        r = norm(p_prop=0.74, too_short=False, n_tokens=32, laya_reason=None,
                 spans=[dict(SP(0.56), strong=True), dict(SP(0.3, "ferry"), strong=False)])
        p = r["propaganda"]
        self.assertEqual((p["p_prop"], p["too_short"], p["n_tokens"]), (0.74, False, 32))
        self.assertEqual([s["strong"] for s in p["techniques"]], [True, False])
        sg = ea.propaganda_signal(p)
        self.assertEqual((sg["level"], sg["basis"], sg["p_prop"]), ("strong", "engine", 0.74))

    def test_engine_strong_flag_beats_threshold(self):
        # 0.6 but flagged not strong (below its class threshold) -> weak; 0.4 flagged strong -> strong
        weak = ea.propaganda_signal({"techniques": [dict(SP(0.6), strong=False, start=0, end=1)]})
        self.assertEqual((weak["level"], weak["basis"]), ("weak", "engine"))
        strong = ea.propaganda_signal({"techniques": [dict(SP(0.4), strong=True, start=0, end=1)]})
        self.assertEqual(strong["level"], "strong")
        mixed = ea.propaganda_signal({"techniques": [dict(SP(0.6), strong=True), SP(0.9)]})  # one lacks the flag: fallback
        self.assertEqual(mixed["basis"], "confidence")

    def test_absent_and_bad_fields(self):
        p = norm()["propaganda"]
        self.assertEqual((p["p_prop"], p["too_short"], p["n_tokens"]), (None, None, None))
        p = norm(p_prop=None, too_short="yes", n_tokens=True)["propaganda"]
        self.assertEqual((p["p_prop"], p["too_short"], p["n_tokens"]), (None, None, None))

    def test_too_short_overrides_word_proxy(self):
        n = norm("w " * 80, claims=[{}], too_short=True)
        self.assertTrue(ea.laya_view(n["laya"], n["propaganda"], 1, None)["short_text"])
        n = norm(SHORT, claims=[{}], too_short=False)
        self.assertFalse(ea.laya_view(n["laya"], n["propaganda"], 1, None)["short_text"])

    def test_reasons(self):
        self.assertEqual(norm(degraded_reason="no_claims")["degraded_why"], "no_claims")
        self.assertFalse(norm(degraded_reason="no_claims")["degraded"])
        self.assertEqual(norm(degraded_reason="no_claims")["laya_reason"], "no_claims")
        self.assertEqual(norm(degraded_reason="timeout")["degraded_why"], "timeout")
        self.assertEqual(norm(degraded_reason="gemma_error")["degraded_why"], "unavailable")
        n = norm(claims=[{}], laya=None, laya_reason="timeout")
        self.assertEqual(ea.laya_view(n["laya"], n["propaganda"], 1, None, n["laya_reason"])["state"], "timeout")

    def test_technique_probs_give_engine_basis(self):
        tp = {b: 0.05 for b in ea.BUCKETS}
        tp["Doubt"] = 0.3
        n = norm(technique_probs=tp)
        _, basis, note = ea.propaganda_classes(n["propaganda"])
        self.assertEqual(basis, "engine")
        self.assertIn("max token probability", note)


class OverHttp(tc.Base):
    def test_new_no_claims_ok_state(self):
        with self.engine1(lambda t: v1_response(degraded_reason="no_claims", spans=[SP(0.9, "enemy of the people")])) as e:
            _, _, b = self.analyse(LOUD)
            _, _, b2 = self.analyse(LOUD)
        self.assertEqual((b["engine"]["state"], b["engine"]["degraded_why"]), ("ok", "no_claims"))
        self.assertEqual(b["step2"]["laya"]["state"], "not_run_no_claims")
        self.assertIn("no checkable", b["step1"]["note"])
        self.assertEqual(e.count, 1)
        self.assertTrue(b2["engine"]["cached"])

    def test_gemma_error_not_cached(self):
        with self.engine1(lambda t: v1_response(degraded_reason="gemma_error", spans=[], claims=[{}], laya=None)) as e:
            _, _, b = self.analyse(SHORT)
            self.analyse(SHORT)
        self.assertEqual(e.count, 2)
        self.assertEqual(b["engine"]["state"], "degraded")
        self.assertNotIn("timed out", b["engine"]["reason"])

    def engine1(self, make):
        return self.engine(make=make, path="/api/check_v1")

    def test_short_zero_spans(self):
        with self.engine1(lambda t: v1_response("LEGIT", 0.2, one_sided=0.0, claims=[{}])):
            _, _, b = self.analyse(SHORT)
        p = b["step2"]["propaganda"]
        self.assertEqual(p["signal"]["level"], "none")
        self.assertEqual(b["step2"]["laya"]["propaganda_head"], 0.0)
        self.assertTrue(b["step2"]["laya"]["short_text"])
        self.assertEqual(len(p["classes"]), 14)
        self.assertIs(p["one_sided"], False)

    def test_loud_one_sided(self):
        with self.engine1(lambda t: v1_response("FAKE", 0.9, spans=[SP(0.91, "enemy of the people"), SP(0.2, "Nobody serious")], one_sided=1.0, claims=[{}])):
            _, _, b = self.analyse(LOUD)
        p = b["step2"]["propaganda"]
        self.assertIs(p["one_sided"], True)
        self.assertEqual(p["one_sided_score"], 1.0)
        self.assertEqual((p["signal"]["level"], p["signal"]["strong_n"], p["signal"]["weak_n"]), ("strong", 1, 1))
        self.assertEqual(b["step2"]["laya"]["state"], "ok")

    def test_no_claims_old_degraded_shape(self):
        with self.engine1(lambda t: v1_response(degraded_reason="no_claims_old", spans=[SP(0.9, "enemy of the people")])) as e:
            _, _, b = self.analyse(LOUD)
            _, _, b2 = self.analyse(LOUD)
        en = b["engine"]
        self.assertEqual((en["state"], en["degraded_why"], en["n_claims"]), ("degraded", "no_claims", 0))
        self.assertNotIn("timed out", en["reason"])
        self.assertNotIn("timed out", b["step1"]["note"])
        self.assertIn("no checkable", en["reason"])
        self.assertTrue(b["step2"]["shown"])
        self.assertEqual(b["step2"]["laya"]["state"], "not_run_no_claims")
        self.assertNotIn("timed out", b["step2"]["reason"])
        self.assertEqual(e.count, 1)  # deterministic, so cached
        self.assertEqual((b2["engine"]["state"], b2["engine"]["cached"], b2["engine"]["degraded_why"]), ("degraded", True, "no_claims"))
        self.assertTrue(b2["step2"]["shown"])

    def test_timeout_degraded_still_uncached(self):
        with self.engine1(lambda t: v1_response("FAKE", 0.8, degraded=True, spans=[SP(0.9)], laya=None, claims=[{}])) as e:
            _, _, b = self.analyse(SHORT)
            self.analyse(SHORT)
        self.assertEqual(e.count, 2)
        self.assertEqual(b["engine"]["degraded_why"], "timeout")
        self.assertIn("timed out", b["engine"]["reason"])
        self.assertEqual(b["step2"]["laya"]["state"], "timeout")

    def test_other_degraded_is_unavailable(self):
        with self.engine1(lambda t: v1_response("FAKE", 0.8, degraded=True, error="gemma 500", claims=[{}], laya=None)):
            _, _, b = self.analyse(SHORT)
        self.assertEqual(b["engine"]["degraded_why"], "unavailable")
        self.assertNotIn("timed out", b["engine"]["reason"])

    def test_v0_still_works(self):
        with self.engine(make=lambda t: v0_response(spans=[])):
            _, _, b = self.analyse(SHORT)
        self.assertEqual(b["step2"]["laya"]["state"], "not_returned")
        self.assertEqual(b["step2"]["propaganda"]["signal"]["level"], "none")


class WithBlocks(unittest.TestCase):
    def test_adds_laya_and_signal_to_old_body(self):
        body = {"step1": {"classes": []}, "step2": {"shown": True, "propaganda": {"techniques": [SP(0.7)], "classes": [], "n": 1}},
                "engine": {"state": "ok", "laya": {"label": "FAKE", "probs": {"FAKE": 0.6}, "confidence": 0.6}}}
        out = claimcheck.with_blocks(body)
        self.assertEqual(out["step2"]["propaganda"]["signal"]["level"], "strong")
        self.assertEqual(out["step2"]["laya"]["state"], "ok")
        self.assertIsNone(out["step2"]["laya"]["propaganda_head"])  # not stored
        self.assertNotIn("laya", body["step2"])
        self.assertIs(claimcheck.with_blocks(out), out)

    def test_no_propaganda_unchanged(self):
        body = {"step1": {"classes": []}, "step2": {"shown": False, "propaganda": None}}
        self.assertIs(claimcheck.with_blocks(body), body)


class Static(unittest.TestCase):
    def test_every_term_key_resolves(self):
        labels = open(os.path.join(WEB, "labels.js"), encoding="utf-8").read()
        js = open(os.path.join(WEB, "claimcheck.js"), encoding="utf-8").read()
        for key in set(re.findall(r"term\('([^']+)'", js)):
            self.assertIn("'%s':" % key, labels, key)

    def test_no_detected_wording_for_empty(self):
        js = open(os.path.join(WEB, "claimcheck.js"), encoding="utf-8").read()
        self.assertNotIn("No propaganda techniques detected", js)
        self.assertIn("No strong propaganda spans found", js)


if __name__ == "__main__":
    unittest.main()
