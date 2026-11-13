"""Triage priority order: fresher first inside a tier, then stronger signals (synthetic data, fixed clock)."""
import os
import sqlite3
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rt_triage  # noqa: E402

NOW = datetime(2026, 11, 12, 12, 0, 0, tzinfo=timezone.utc).timestamp()
DAY = 86400.0


def iso(days_ago):
    return datetime.fromtimestamp(NOW - days_ago * DAY, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AgeBand(unittest.TestCase):
    def test_band_edges(self):
        edges = rt_triage.AGE_DAYS
        for i, d in enumerate(edges):
            self.assertEqual(rt_triage.age_band(NOW - (d * DAY - 60), NOW), i)   # a minute younger than the edge: this band
            self.assertEqual(rt_triage.age_band(NOW - d * DAY, NOW), i + 1)      # exactly on the edge: the next band
        self.assertEqual(rt_triage.age_band(NOW - 400 * DAY, NOW), len(edges))
        self.assertEqual(rt_triage.age_band(None, NOW), len(edges) + 1)
        self.assertEqual(list(edges), sorted(edges))


class SqlOrder(unittest.TestCase):
    """Run the real ORDER BY expression over a tiny fixture shaped like the queue's joins (aliases i, d, e and a tier column t)."""

    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        c = self.con
        c.execute("CREATE TABLE i (id TEXT, published_at TEXT, engagement INTEGER, tier INTEGER)")
        c.execute("CREATE TABLE d (id TEXT, d_pfm REAL)")
        c.execute("CREATE TABLE e (id TEXT, fc_rerank REAL, fc_cos REAL, p_prop REAL)")

    def add(self, iid, days_ago, tier, p_prop, eng=0):
        self.con.execute("INSERT INTO i VALUES (?,?,?,?)", (iid, iso(days_ago), eng, tier))
        self.con.execute("INSERT INTO e VALUES (?,?,?,?)", (iid, None, None, p_prop))

    def ranked(self):
        order = rt_triage.order_priority(NOW)
        q = ("SELECT i.id, i.tier AS t FROM i LEFT JOIN d ON d.id = i.id LEFT JOIN e ON e.id = i.id ORDER BY %s" % order)
        return [r[0] for r in self.con.execute(q)]

    def test_fresher_beats_stronger_within_a_tier(self):
        self.add("old_strong", 40, 3, 0.99)
        self.add("new_weak", 0.2, 3, 0.81)
        self.add("mid", 5, 3, 0.9)
        self.assertEqual(self.ranked(), ["new_weak", "mid", "old_strong"])

    def test_same_band_stronger_first_then_newest(self):
        self.add("a", 0.5, 3, 0.85)
        self.add("b", 0.4, 3, 0.95)
        self.add("c", 0.3, 3, 0.95)
        self.assertEqual(self.ranked(), ["c", "b", "a"])   # same band: higher signal first; equal signal: newer first

    def test_tier_still_dominates_freshness(self):
        self.add("high_old", 50, 3, 0.9)
        self.add("medium_new", 0.1, 2, 0.5)
        self.assertEqual(self.ranked(), ["high_old", "medium_new"])


if __name__ == "__main__":
    unittest.main()
