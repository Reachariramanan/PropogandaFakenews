"""Runs the node-based tests of the pure front-end helpers (tests/js/*.test.js) and the JS syntax check.
Skipped cleanly when `node` is not installed."""
import glob
import os
import shutil
import subprocess
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(os.path.dirname(HERE), "web")
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "node is not installed")
class JsHelpers(unittest.TestCase):
    def test_every_js_file_parses(self):
        for f in sorted(glob.glob(os.path.join(WEB, "*.js"))):
            r = subprocess.run([NODE, "--check", f], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, "%s: %s" % (f, r.stderr))

    def test_node_unit_tests(self):
        files = sorted(glob.glob(os.path.join(HERE, "js", "*.test.js")))
        self.assertTrue(files)
        for f in files:
            r = subprocess.run([NODE, "--test", f], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, "%s\n%s\n%s" % (f, r.stdout[-3000:], r.stderr[-1500:]))


if __name__ == "__main__":
    unittest.main()
