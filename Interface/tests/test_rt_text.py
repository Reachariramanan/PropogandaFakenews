import random
import unittest

import helpers  # noqa: F401  (sys.path / env setup)
import rt_store
from rt_text import clean_text, hamming, normalize_lang, simhash64, to_signed, to_unsigned


class CleanText(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("   \n\t "), "")

    def test_strips_html_scripts_and_entities(self):
        s = "<p>Hello&nbsp;<b>world</b> &amp; friends</p><script>alert(1)</script><style>p{x:y}</style><!-- c -->done"
        out = clean_text(s)
        self.assertEqual(out, "Hello world & friends done")
        self.assertNotIn("<", out)

    def test_plain_less_than_survives(self):
        self.assertEqual(clean_text("if a < b and c > d then I <3 you"), "if a < b and c > d then I <3 you")

    def test_control_and_zero_width_chars_removed(self):
        out = clean_text("a\x00b\x07c​d﻿e\x85f\ng\t h")
        self.assertEqual(out, "abcdef g h")

    def test_whitespace_collapsed(self):
        self.assertEqual(clean_text("one   two\n\n\nthree four"), "one two three four")

    def test_truncates_on_word_boundary_within_limit(self):
        text = " ".join("word%03d" % i for i in range(1000))
        out = clean_text(text, 1500)
        self.assertLessEqual(len(out), 1500)
        self.assertTrue(out.endswith("…"))
        last = out[:-1].split(" ")[-1]
        self.assertIn(last, text.split(" "))  # no half word at the cut

    def test_short_text_untouched_and_no_ellipsis(self):
        self.assertEqual(clean_text("short text", 50), "short text")

    def test_unbroken_text_hard_cut(self):
        out = clean_text("x" * 5000, 100)
        self.assertEqual(len(out), 100)

    def test_devanagari_and_cjk_survive(self):
        self.assertEqual(clean_text("<div>नमस्ते दुनिया 你好</div>"), "नमस्ते दुनिया 你好")


class Lang(unittest.TestCase):
    def test_normalize(self):
        for raw, want in [(None, "und"), ("", "und"), ("unknown", "und"), ("EN", "en"), ("in", "id"), ("hi ", "hi"),
                          ("zh_CN", "zh-cn"), ("qme", "qme")]:
            self.assertEqual(normalize_lang(raw), want, raw)


BASE = ("Heavy floods hit the river valley on Monday and thousands of residents were evacuated from low lying villages "
        "while rescue teams worked through the night to reach families stranded on rooftops and bridges")


class Simhash(unittest.TestCase):
    def test_deterministic_and_64_bit(self):
        a = simhash64("Title", BASE)
        self.assertEqual(a, simhash64("Title", BASE))
        self.assertTrue(0 <= a < 2 ** 64)

    def test_too_short_is_none(self):
        self.assertIsNone(simhash64("hi there", ""))
        self.assertIsNone(simhash64(None, None))

    def test_case_and_punctuation_insensitive(self):
        self.assertEqual(simhash64("Floods!", BASE), simhash64("floods", BASE.upper()))

    def test_near_text_close_unrelated_far(self):
        a = simhash64("Floods in valley", BASE)
        b = simhash64("Floods in valley", BASE + " officials said")
        other = simhash64("Cricket final", "The home side won the toss and elected to bat first on a dry pitch "
                          "before scoring four hundred runs in the opening innings of the final test match")
        self.assertLessEqual(hamming(a, b), 12)
        self.assertGreater(hamming(a, other), 12)

    def test_signed_roundtrip(self):
        for u in (0, 1, 2 ** 63 - 1, 2 ** 63, 2 ** 64 - 1):
            s = to_signed(u)
            self.assertTrue(-2 ** 63 <= s < 2 ** 63)
            self.assertEqual(to_unsigned(s), u)


class IndexExactness(unittest.TestCase):
    def test_index_matches_brute_force(self):
        rnd = random.Random(7)
        hashes = {"id%d" % i: rnd.getrandbits(64) for i in range(300)}
        base = rnd.getrandbits(64)
        for i, flips in enumerate([0, 1, 2, 3, 3, 4, 5]):  # planted neighbours of `base`
            h = base
            for b in rnd.sample(range(64), flips):
                h ^= 1 << b
            hashes["near%d" % i] = h
        hashes["base"] = base
        for k in (1, 2, 3, 5):
            idx = rt_store.SimhashIndex(k)
            for i, h in hashes.items():
                idx.add(i, h)
            for probe in ("base", "id0", "id5"):
                want = sorted((o, hamming(hashes[probe], h)) for o, h in hashes.items()
                              if o != probe and hamming(hashes[probe], h) <= k)
                got = sorted(idx.query(hashes[probe], exclude=probe))
                self.assertEqual(got, want, (k, probe))


if __name__ == "__main__":
    unittest.main()
