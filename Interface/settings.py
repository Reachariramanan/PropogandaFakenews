"""Settings file loader for the Evidence Desk server (stdlib only).

Reads ~/.config/fnp/interface.env (KEY=VALUE per line, '#' comments, blank lines, optional surrounding single or
double quotes, no shell expansion) and applies each FNP_* key to os.environ ONLY when the variable is not already
set, so a real environment variable always wins. Values are never printed or logged: only key names and counts.
A missing file is normal."""
import os
import sys

DEFAULT_PATH = os.path.join(os.path.expanduser("~"), ".config", "fnp", "interface.env")
PREFIX = "FNP_"


def parse(text):
    """Parse env-file text into ({key: value}, ignored_count). Ignored = non-FNP_ keys and malformed lines."""
    out, ignored = {}, 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not key or not key.startswith(PREFIX):
            ignored += 1
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out, ignored


def load_settings(path=None, env=None, log=None):
    """Apply the file to env (default os.environ) for unset keys. Returns a summary dict of counts and key names
    (never values): {"path", "found", "applied": [keys], "kept": [keys already in env], "ignored": n, "error"}."""
    env = os.environ if env is None else env
    log = (lambda m: print(m, file=sys.stderr, flush=True)) if log is None else log
    path = path or DEFAULT_PATH
    res = {"path": path, "found": False, "applied": [], "kept": [], "ignored": 0, "error": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return res
    except (OSError, UnicodeDecodeError) as e:
        res["error"] = type(e).__name__
        log("settings: could not read %s (%s); continuing with the environment only" % (path, res["error"]))
        return res
    res["found"] = True
    pairs, res["ignored"] = parse(text)
    for k, v in pairs.items():
        if k in env:
            res["kept"].append(k)
        else:
            env[k] = v
            res["applied"].append(k)
    log("settings: %s: %d applied (%s), %d kept from environment, %d ignored" % (
        path, len(res["applied"]), ", ".join(res["applied"]) or "-", len(res["kept"]), res["ignored"]))
    return res
