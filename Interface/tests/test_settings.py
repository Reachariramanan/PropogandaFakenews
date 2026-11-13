import io
import os
import stat
import tempfile
import unittest

import helpers  # noqa: F401  (sets up sys.path and offline env)
import settings


class SettingsTests(unittest.TestCase):
    def write(self, text):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "interface.env")
        with open(p, "w") as f:
            f.write(text)
        return p

    def load(self, text, env=None):
        env = {} if env is None else env
        logs = []
        res = settings.load_settings(self.write(text), env=env, log=logs.append)
        return res, env, logs

    def test_parse_basic_comments_blank(self):
        res, env, _ = self.load("# c\n\nFNP_HOST=1.2.3.4\n  FNP_PORT = 8766  \nFNP_RT_SYNC=1 # not a comment\n")
        self.assertEqual(env["FNP_HOST"], "1.2.3.4")
        self.assertEqual(env["FNP_PORT"], "8766")
        self.assertEqual(env["FNP_RT_SYNC"], "1 # not a comment")  # no shell semantics
        self.assertEqual(len(res["applied"]), 3)

    def test_quotes_and_no_expansion(self):
        _, env, _ = self.load("FNP_A=\"x y\"\nFNP_B='z'\nFNP_C=$HOME/x\nFNP_D=\"unterminated\nFNP_E=\n")
        self.assertEqual(env["FNP_A"], "x y")
        self.assertEqual(env["FNP_B"], "z")
        self.assertEqual(env["FNP_C"], "$HOME/x")
        self.assertEqual(env["FNP_D"], "\"unterminated")
        self.assertEqual(env["FNP_E"], "")

    def test_real_env_wins(self):
        res, env, _ = self.load("FNP_PORT=1\nFNP_HOST=h\n", {"FNP_PORT": "9"})
        self.assertEqual(env["FNP_PORT"], "9")
        self.assertEqual(env["FNP_HOST"], "h")
        self.assertEqual(res["kept"], ["FNP_PORT"])
        self.assertEqual(res["applied"], ["FNP_HOST"])

    def test_non_fnp_keys_ignored_and_counted(self):
        res, env, _ = self.load("PATH=/evil\nHOME=/x\nnoequals\nFNP_OK=1\n")
        self.assertEqual(env, {"FNP_OK": "1"})
        self.assertEqual(res["ignored"], 3)

    def test_export_prefix_tolerated(self):
        _, env, _ = self.load("export FNP_X=1\n")
        self.assertEqual(env, {"FNP_X": "1"})

    def test_missing_file_is_normal(self):
        env = {}
        logs = []
        res = settings.load_settings(os.path.join(tempfile.mkdtemp(), "nope.env"), env=env, log=logs.append)
        self.assertFalse(res["found"])
        self.assertIsNone(res["error"])
        self.assertEqual((env, logs), ({}, []))

    def test_unreadable_file(self):
        p = self.write("FNP_X=1\n")
        os.chmod(p, 0)
        if os.access(p, os.R_OK):  # running as root
            self.skipTest("file still readable")
        env, logs = {}, []
        res = settings.load_settings(p, env=env, log=logs.append)
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
        self.assertEqual(res["error"], "PermissionError")
        self.assertEqual(env, {})
        self.assertEqual(len(logs), 1)

    def test_undecodable_file(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "x.env")
        with open(p, "wb") as f:
            f.write(b"FNP_X=\xff\xfe\n")
        res = settings.load_settings(p, env={}, log=lambda m: None)
        self.assertEqual(res["error"], "UnicodeDecodeError")

    def test_values_never_in_log_or_summary(self):
        secret = "s3cr3t-VALUE-xyz"
        res, _, logs = self.load("FNP_AUTH_PASS=%s\nFNP_AUTH_USER=\"%s\"\nOTHER=%s\n" % (secret, secret, secret))
        self.assertTrue(logs)
        for m in logs:
            self.assertNotIn(secret, m)
        self.assertNotIn(secret, repr(res))
        self.assertIn("FNP_AUTH_PASS", logs[0])  # key names only

    def test_default_logging_goes_to_stderr_without_values(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            settings.load_settings(self.write("FNP_Q=topsecret\n"), env={})
        self.assertNotIn("topsecret", buf.getvalue())
        self.assertIn("FNP_Q", buf.getvalue())

    def test_import_server_does_not_read_real_file(self):
        import server  # noqa: F401  (loader only runs under __main__)
        self.assertNotIn("FNP_RT_ALLOW_OPEN", os.environ)


if __name__ == "__main__":
    unittest.main()
