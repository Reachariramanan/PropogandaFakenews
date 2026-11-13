"""Manage annotation accounts from the command line (the first admin must be created here; everything else can be
done in the UI under Admin > Users).

  .venv/bin/python scripts/31_annot_users.py list
  .venv/bin/python scripts/31_annot_users.py create priya --role admin            # prompts for a password
  .venv/bin/python scripts/31_annot_users.py create ravi --generate               # random password, shown once
  .venv/bin/python scripts/31_annot_users.py reset ravi --generate                # new temporary password
  .venv/bin/python scripts/31_annot_users.py disable ravi | enable ravi
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fnp.annotate import auth  # noqa: E402
from fnp.annotate.db import tx  # noqa: E402


def _id(name: str) -> int:
    with tx() as cur:
        cur.execute("SELECT annotator_id FROM annotator WHERE lower(username)=lower(%s)", (name,))
        r = cur.fetchone()
    if not r:
        sys.exit(f"no such user: {name}")
    return r["annotator_id"]


def _pw(a) -> tuple[str | None, bool]:
    if a.generate:
        return None, True
    p1 = getpass.getpass("password: ")
    if p1 != getpass.getpass("again: "):
        sys.exit("passwords differ")
    return p1, False


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("create"); c.add_argument("username"); c.add_argument("--role", default="annotator", choices=auth.ROLES)
    c.add_argument("--generate", action="store_true"); c.add_argument("--no-force-change", action="store_true")
    r = sub.add_parser("reset"); r.add_argument("username"); r.add_argument("--generate", action="store_true")
    r.add_argument("--no-force-change", action="store_true")
    for n in ("enable", "disable"):
        sub.add_parser(n).add_argument("username")
    a = ap.parse_args()
    if a.cmd == "list":
        with tx() as cur:
            cur.execute("SELECT username, role, active, password_hash IS NOT NULL AS has_pw, last_login_at FROM annotator ORDER BY username")
            for x in cur.fetchall():
                print(f"{x['username']:24s} {x['role']:12s} {'active' if x['active'] else 'DISABLED':9s} "
                      f"{'pw' if x['has_pw'] else 'NO-PASSWORD':12s} last login: {x['last_login_at'] or '-'}")
    elif a.cmd == "create":
        pw, gen = _pw(a)
        res = auth.create_user(a.username, pw, a.role, must_change=not a.no_force_change, actor="cli")
        print(f"created {res['annotator']['username']} ({a.role})")
        if res["password"]:
            print(f"temporary password (shown once): {res['password']}")
    elif a.cmd == "reset":
        pw, gen = _pw(a)
        new = auth.set_password(_id(a.username), pw, not a.no_force_change, actor="cli")
        print("password reset; all open sessions ended" + (f"\ntemporary password (shown once): {new}" if new else ""))
    else:
        auth.set_active(_id(a.username), a.cmd == "enable")
        print(f"{a.username}: {a.cmd}d")


if __name__ == "__main__":
    main()
