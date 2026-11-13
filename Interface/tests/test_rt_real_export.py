"""S1: the REAL FNP realtime export format (meta.json without a file list, news/ and social/ part_NNNN.jsonl, rows with
technique_probs / spans as offsets / countries as [{iso3,...}] / nearest_factcheck {kb_id,cosine,rerank}).
Everything is SYNTHETIC: invented ids and invented text in the real schema, never a real post. Offline, temp dirs only."""
import json
import os
import sqlite3
import time
import unittest
from unittest import mock

import helpers  # noqa: F401  (sys.path + offline env)
import rt_api
import rt_enrich
from build_index import norm_url
from test_rt_api import NOW, Fixture, item  # noqa: F401
from rt_text import simhash64  # noqa: F401

CLASSES = rt_enrich.TECHNIQUE_CLASSES
BUILT = "2026-11-12T10:00:00+00:00"
NEXT = "2026-11-12T13:00:00+00:00"
CAVEAT = "Synthetic caveat: a ranking signal, not a calibrated probability. <b>not html</b>"
FC_URL = "https://factcheck.example.org/fc/flood-claim/"


def probs(**kw):
    d = {c: 0.01 for c in CLASSES}
    for k, v in kw.items():
        d[{"LL": CLASSES[0], "Doubt": CLASSES[4], "Rep": CLASSES[2], "Slog": CLASSES[8]}[k]] = v
    return d


def real_row(iid, too_short=False, p_prop=0.8, tp=None, strong=("Loaded_Language",), spans=None, countries=None,
             nf="default", **kw):
    d = dict(id=iid, kind="news" if iid.startswith("news:") else "social", platform="news", source="example.org",
             published_at="2026-11-12T08:00:00Z", lang="en", too_short=too_short, n_tokens=40,
             p_prop=None if too_short else p_prop, n_spans=len(spans or []), n_strong=len(strong), density=0.2,
             technique_probs=tp if tp is not None else probs(LL=0.97, Doubt=0.4),
             strong_techniques=list(strong), spans=spans or [],
             countries=[{"iso3": "IND", "weight": 0.5, "methods": ["name"], "terms": ["invented place"]}] if countries is None else countries,
             nearest_factcheck=dict(kb_id="kb:test-1", label="FAKE", claim="Invented claim about a flood <script>x</script>",
                                    url=FC_URL, cosine=0.61, rerank=None) if nf == "default" else nf,
             scored_at="2026-11-12T09:59:00+00:00", model_version="prop_span_test+kb_test")
    d.update(kw)
    return d


def write_real(base, parts, built_at=BUILT, meta_extra=None, n_total=None, raw=None):
    """parts: {"news/part_0000.jsonl": [rows]}; meta.json is written last and has NO `files` list (like FNP's)."""
    os.makedirs(base, exist_ok=True)
    total = 0
    for rel, rows in parts.items():
        p = os.path.join(base, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        lines = [json.dumps(r, allow_nan=False) for r in rows] + list((raw or {}).get(rel, []))
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        total += len(lines)
    meta = dict(built_at=built_at, n_items=dict(news=0, social=0), n_items_total=total if n_total is None else n_total,
                n_in_window=1000, n_unscored_in_window=1000 - total, last_item_published_at={"news": "2026-11-12T08:37:00Z"},
                next_run_at=NEXT, interval_hours=3, n_scored_this_run=total, n_failed_this_run=0, run_seconds=5,
                model_version="prop_span_test+kb_test", technique_classes=list(CLASSES),
                caveat=CAVEAT, analysed_text="title + '\\n' + text_excerpt; span offsets index into this string")
    meta.update(meta_extra or {})
    tmp = os.path.join(base, "meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, os.path.join(base, "meta.json"))
    return meta


def full_text(it):
    return (it["title"] or "") + "\n" + (it["text_excerpt"] or "")


class RealFixture(Fixture):
    """Fixture items: reddit_posts:10 has title 'Rumour about vaccine' and text 'they say the vaccine is a plot'."""

    def setUp(self):
        super().setUp()
        self.names_path = os.path.join(self.tmp, "pl_meta.json")
        with open(self.names_path, "w") as f:
            json.dump({"names": {"IND": "India", "GBR": "United Kingdom"}}, f)
        self.desk = {norm_url(FC_URL): "db:desk-777"}
        self.api = rt_api.RtApi(db_path=self.db, enrich=self.enrich, clock=lambda: NOW,
                                desk_url=lambda u: self.desk.get(norm_url(u)),
                                country_names=rt_api.CountryNames(self.names_path))
        it = [i for i in __import__("test_rt_api").build_items() if i["id"] == "reddit_posts:10"][0]
        self.t10 = full_text(it)
        self.title10 = it["title"]

    def span(self, phrase, tech="Loaded_Language", p=0.9, nth=0):
        a = -1
        for _ in range(nth + 1):
            a = self.t10.index(phrase, a + 1)
        return [a, a + len(phrase), tech, p]

    def default_parts(self):
        sp = [self.span("vaccine", p=0.95, nth=1), self.span("Rumour", "Doubt", 0.5), [9999, 10005, "Slogans", 0.7],
              [self.t10.index("\n") - 3, self.t10.index("\n") + 4, "Repetition", 0.6]]  # last: straddles title/text
        return {"news/part_0000.jsonl": [real_row("news:scraped_articles:1", spans=[]),
                                         real_row("news:scraped_articles:2", too_short=True, p_prop=None,
                                                  strong=(), spans=[[0, 5, "Doubt", 0.4]])],
                "social/part_0000.jsonl": [real_row("reddit_posts:10", spans=sp, p_prop=0.9,
                                                    nf=dict(kb_id="kb:test-2", label="MISLEADING", claim="Invented second claim",
                                                            url=FC_URL, cosine=0.81, rerank=0.66)),
                                           real_row("reddit_posts:11", nf=None, countries=[]),
                                           real_row("telegram_posts:12", strong=("Loaded_Language", "Doubt"),
                                                    spans=[[0, 4, "Loaded_Language", 0.9], [5, 8, "Loaded_Language", 0.8],
                                                           [9, 12, "Doubt", 0.7]])]}


class Validate(unittest.TestCase):
    def test_good_row(self):
        iid, r = rt_enrich.validate_record(real_row("reddit_posts:1", spans=[[0, 4, "Loaded_Language", 0.9],
                                                                            [5, 8, "Loaded_Language", 0.8]]))
        self.assertEqual(iid, "reddit_posts:1")
        tp = json.loads(r["tp"])
        self.assertEqual(len(tp), 14)
        self.assertEqual(tp[0], 0.97)
        self.assertEqual((r["too_short"], r["n_spans"], r["n_strong"], r["fc_id"], r["fc_cos"], r["fc_rerank"]),
                         (0, 2, 1, "kb:test-1", 0.61, None))
        self.assertEqual(r["fc"][3], FC_URL)
        ex = json.loads(r["extra"])
        self.assertEqual(ex["techniques"], [{"technique": "Loaded_Language", "count": 2, "max_confidence": 0.97}])
        self.assertEqual(ex["rspans"][0][2], 0)  # technique stored as its index in the 14-class taxonomy

    def test_too_short_has_no_p_prop_and_no_techniques(self):
        _, r = rt_enrich.validate_record(real_row("reddit_posts:2", too_short=True, spans=[[0, 3, "Doubt", 0.4]]))
        self.assertEqual((r["too_short"], r["p_prop"]), (1, None))
        self.assertEqual(json.loads(r["extra"])["techniques"], [])

    def test_missing_keys_are_tolerated_never_guessed(self):
        _, r = rt_enrich.validate_record({"id": "reddit_posts:3", "scored_at": "2026-11-12T09:00:00+00:00"})
        self.assertEqual((r["p_prop"], r["tp"], r["fc_id"], r["n_spans"]), (None, None, None, 0))
        _, r = rt_enrich.validate_record(real_row("reddit_posts:4", tp={CLASSES[0]: 0.5, "Unknown_Class": 0.9}))
        self.assertEqual(json.loads(r["tp"])[:2], [0.5, None])   # missing buckets stay null, never 0

    def test_bad_rows_raise(self):
        bad = [real_row("x:1", p_prop=1.7), real_row("x:2", tp=[0.1] * 14), real_row("x:3", tp={CLASSES[0]: 2.0}),
               real_row("x:4", spans="nope"), real_row("x:5", nf="a string"), real_row("x:6", too_short="yes"),
               real_row("x:7", strong_techniques="Loaded_Language"), real_row("x:8", countries={"IND": 1}),
               real_row("x:9", tp={"Nothing_Known": 0.2}), {"scored_at": "2026-11-12T09:00:00+00:00"},
               real_row("x:10", nf=dict(kb_id="kb:1", cosine=3.0)), real_row("x:11", n_spans=-1)]
        for d in bad:
            with self.assertRaises(ValueError, msg=str(d)[:60]):
                rt_enrich.validate_record(d)

    def test_malformed_spans_and_countries_are_dropped_not_fatal(self):
        rep = {}
        _, r = rt_enrich.validate_record(real_row("x:1", spans=[[3, 1, "Doubt", 0.5], [1, 2, "Doubt"], "s", [0, 2, "Doubt", 9],
                                                                [0, 2, "Doubt", 0.5]],
                                                  countries=[{"iso3": "TOOLONG"}, "x", {"iso3": "gbr", "weight": 0.3}]), rep)
        self.assertEqual(rep["spans_dropped"], 4)
        self.assertEqual(len(json.loads(r["extra"])["rspans"]), 1)
        self.assertEqual([c["iso3"] for c in json.loads(r["extra"])["countries"]], ["GBR"])

    def test_spans_are_capped_at_twelve_keeping_the_most_confident(self):
        spans = [[i * 3, i * 3 + 2, "Doubt", i / 100.0] for i in range(30)]
        _, r = rt_enrich.validate_record(real_row("x:1", spans=spans))
        kept = json.loads(r["extra"])["rspans"]
        self.assertEqual(len(kept), 12)
        self.assertEqual(r["n_spans"], 30)
        self.assertEqual(min(s[3] for s in kept), 0.18)
        self.assertEqual([s[0] for s in kept], sorted(s[0] for s in kept))   # text order

    def test_old_shape_still_validates(self):
        iid, r = rt_enrich.validate_record({"id": "reddit_posts:5", "p_prop": 0.4, "techniques": [
            {"technique": "x", "count": 2, "max_confidence": 0.5}], "p_fake": None, "verdict": None})
        self.assertEqual((r["p_prop"], r.get("too_short"), r.get("tp")), (0.4, None, None))


class Resolve(unittest.TestCase):
    def test_title_and_text_fields_and_bounds(self):
        title, body = "Alpha beta", "gamma delta epsilon"
        full = title + "\n" + body
        r = [[0, 5, 0, 0.9], [full.index("delta"), full.index("delta") + 5, "Doubt", 0.8], [4, 15, 4, 0.7],
             [0, 500, 0, 0.5], [3, 3, 0, 0.5], [-1, 4, 0, 0.5], [10, 11, 0, 0.5]]   # straddle, out of range, empty, negative, the "\n"
        out, dropped = rt_enrich.Enrich.resolve_spans(r, (title, body))
        self.assertEqual([(s["field"], s["text"], s["start"], s["end"]) for s in out],
                         [("title", "Alpha", 0, 5), ("text", "delta", 6, 11)])
        self.assertEqual(out[0]["technique"], CLASSES[0])
        self.assertEqual(dropped, 5)

    def test_no_text_means_no_spans(self):
        self.assertEqual(rt_enrich.Enrich.resolve_spans([[0, 2, 0, 0.5]], None), ([], 1))

    def test_unicode_offsets_are_codepoints(self):
        title, body = "t", "café नमस्ते x"
        a = (title + "\n" + body).index("x")
        out, _ = rt_enrich.Enrich.resolve_spans([[a, a + 1, 0, 0.5]], (title, body))
        self.assertEqual(out[0]["text"], "x")


class Loader(RealFixture):
    def test_loads_real_format_and_report_matches_meta(self):
        write_real(self.export, self.default_parts())
        st = self.enrich.status()
        rep = st["load_report"]
        self.assertEqual((rep["mode"], rep["rows_ok"], rep["rejected"], rep["rows"], rep["files_read"]), ("export_parts", 5, 0, 5, 2))
        self.assertTrue(rep["count_matches_meta"])
        self.assertFalse(st["partial"])
        self.assertEqual(rep["too_short"], 1)
        self.assertEqual(st["built_at"], BUILT)
        self.assertEqual(st["model_version"], "prop_span_test+kb_test")

    def test_bad_rows_are_counted_and_listed_never_crash(self):
        raw = {"social/part_0000.jsonl": ["{not json", '{"id": "x:1", "p_prop": NaN}', json.dumps(real_row("reddit_posts:98", p_prop=3)),
                                          json.dumps([1, 2]), json.dumps({"scored_at": "x", "technique_probs": {}}),
                                          "ÿþ not utf8".encode("latin-1").decode("latin-1")]}
        parts = self.default_parts()
        write_real(self.export, parts, raw=raw, n_total=5 + 6)
        rep = self.enrich.status()["load_report"]
        self.assertEqual((rep["rows_ok"], rep["rejected"]), (5, 6))
        self.assertTrue(rep["count_matches_meta"])      # rows + rejected == meta.n_items_total
        self.assertTrue(rep["reasons"])
        self.assertLessEqual(len(rep["reasons"]), rt_enrich.REASON_KEYS + 1)
        self.assertEqual(self.enrich.status()["records"], 5)

    def test_binary_garbage_line_is_rejected_not_fatal(self):
        parts = self.default_parts()
        write_real(self.export, parts)
        with open(os.path.join(self.export, "social/part_0000.jsonl"), "ab") as f:
            f.write(b"\xff\xfe\x00bad\n")
        os.utime(os.path.join(self.export, "meta.json"), None)
        self.enrich.refresh(force=True)
        self.assertEqual(self.enrich.status()["load_report"]["rejected"], 1)

    def test_many_small_part_files_and_tmp_and_prev_ignored(self):
        parts = {}
        n = 0
        for k in range(60):
            sub = "news" if k % 2 else "social"
            rows = []
            for j in range(3):
                n += 1
                rows.append(real_row(("news:scraped_articles:%d" if sub == "news" else "reddit_posts:%d") % (1000 + n)))
            parts["%s/part_%04d.jsonl" % (sub, k)] = rows
        v1 = os.path.join(self.tmp, "reports", "v1")
        write_real(v1, parts)
        # a half-written .tmp part and the previous export beside it must both be ignored
        with open(os.path.join(v1, "news", "part_9999.jsonl.tmp"), "w") as f:
            f.write(json.dumps(real_row("news:scraped_articles:555555")) + "\n")
        write_real(os.path.join(self.tmp, "reports", "v1.prev"), {"news/part_0000.jsonl": [real_row("news:scraped_articles:777777")]})
        e = rt_enrich.Enrich(export_dir=v1, triage_dir=self.triage, sidecar=os.path.join(self.tmp, "many.sqlite"), store_path=self.db)
        st = e.status()
        self.assertEqual((st["records"], st["load_report"]["files_read"], st["load_report"]["rejected"]), (180, 60, 0))
        self.assertIsNone(e.lookup("news:scraped_articles:555555"))
        self.assertIsNone(e.lookup("news:scraped_articles:777777"))
        self.assertIsNotNone(e.lookup("news:scraped_articles:1004", text=("t", "b")))

    def test_rows_beyond_one_flush_batch_stream_through(self):
        n = rt_enrich.BATCH + 500
        write_real(self.export, {"news/part_0000.jsonl": [real_row("news:scraped_articles:%d" % (i + 100)) for i in range(n)]})
        st = self.enrich.status()
        self.assertEqual((st["records"], st["load_report"]["rows_ok"]), (n, n))

    def test_reload_only_when_built_at_changes(self):
        write_real(self.export, self.default_parts())
        self.assertEqual(self.enrich.status()["records"], 5)
        loaded = self.enrich.status()["loaded_at"]
        time.sleep(1.1)
        os.utime(os.path.join(self.export, "meta.json"), None)          # touch: same built_at -> no rebuild
        self.assertEqual(self.enrich.status()["loaded_at"], loaded)
        parts = self.default_parts()
        parts["news/part_0001.jsonl"] = [real_row("news:scraped_articles:3")]
        write_real(self.export, parts, built_at="2026-11-12T13:00:00+00:00")
        st = self.enrich.status()
        self.assertEqual((st["records"], st["built_at"]), (6, "2026-11-12T13:00:00+00:00"))

    def test_part_vanishes_once_restarts_from_a_fresh_listing(self):
        write_real(self.export, self.default_parts())
        real = rt_enrich.Enrich._read_part
        calls = {"n": 0}

        def flaky(self_, con, path, rep, st, seen, h=None):
            calls["n"] += 1
            if calls["n"] == 2:
                raise FileNotFoundError(path)     # a part replaced by an atomic swap between listing and reading
            return real(self_, con, path, rep, st, seen, h)
        with mock.patch.object(rt_enrich.Enrich, "_read_part", flaky):
            self.enrich.refresh(force=True)
        rep = self.enrich.status()["load_report"]
        self.assertEqual((rep["restarts"], rep["partial"], rep["rows_ok"], self.enrich.status()["records"]), (1, False, 5, 5))

    def test_part_missing_twice_is_partial_and_flagged_and_retried(self):
        write_real(self.export, self.default_parts())
        real = rt_enrich.Enrich._read_part

        def gone(self_, con, path, rep, st, seen, h=None):
            if path.endswith(os.path.join("social", "part_0000.jsonl")):
                raise FileNotFoundError(path)
            return real(self_, con, path, rep, st, seen, h)
        with mock.patch.object(rt_enrich.Enrich, "_read_part", gone):
            self.enrich.refresh(force=True)
        st = self.enrich.status()
        self.assertTrue(st["partial"])
        self.assertEqual(st["load_report"]["parts_missing"], ["social/part_0000.jsonl"])
        self.assertEqual(st["records"], 2)       # what did load is served
        self.assertIsNone(st["error"])
        self.enrich._retry_at = time.monotonic() - 1     # the retry window has passed: the next request reloads, now complete
        st = self.enrich.status()
        self.assertEqual((st["partial"], st["records"]), (False, 5))

    def test_export_replaced_during_the_read_is_flagged_and_picked_up_next(self):
        write_real(self.export, self.default_parts())
        real = rt_enrich.Enrich._read_part
        swapped = {"done": False}

        def swap(self_, con, path, rep, st, seen, h=None):
            r = real(self_, con, path, rep, st, seen, h)
            if not swapped["done"]:
                swapped["done"] = True
                write_real(self.export, {"news/part_0000.jsonl": [real_row("news:scraped_articles:4")]}, built_at="2026-11-12T13:00:00+00:00")
            return r
        with mock.patch.object(rt_enrich.Enrich, "_read_part", swap):
            self.enrich.refresh(force=True)
        self.assertTrue(self.enrich._status["load_report"]["meta_changed_during_load"])
        self.assertEqual(self.enrich._status["built_at"], BUILT)             # labelled with the meta it started from
        self.enrich._sig = None
        st = self.enrich.status()                                             # the next request sees the new built_at
        self.assertEqual((st["built_at"], st["records"]), ("2026-11-12T13:00:00+00:00", 4))   # the new news part + the 3 untouched social rows

    def test_empty_export_dir_keeps_previous_and_reports(self):
        write_real(self.export, self.default_parts())
        self.assertEqual(self.enrich.status()["records"], 5)
        for sub in ("news", "social"):
            for f in os.listdir(os.path.join(self.export, sub)):
                os.unlink(os.path.join(self.export, sub, f))
        write_real(self.export, {}, built_at="2026-11-12T16:00:00+00:00")
        st = self.enrich.status()
        self.assertEqual(st["records"], 5)
        self.assertIn("no part files", st["error"])

    def test_old_sidecar_schema_is_migrated_in_place(self):
        side = os.path.join(self.tmp, "old.sqlite")
        con = sqlite3.connect(side)
        con.executescript("""CREATE TABLE enr (id TEXT PRIMARY KEY, kind TEXT NOT NULL, p_prop REAL, p_fake REAL, p_misleading REAL,
            verdict TEXT, model_version TEXT, built_at TEXT, src TEXT NOT NULL, extra TEXT) WITHOUT ROWID;
            CREATE TABLE enr_tech (id TEXT NOT NULL, technique TEXT NOT NULL, cnt INTEGER NOT NULL, max_conf REAL);
            CREATE TABLE enr_meta (k TEXT PRIMARY KEY, v TEXT);
            INSERT INTO enr VALUES ('reddit_posts:10','social',0.5,NULL,NULL,NULL,'m','b','realtime_export','{}');""")
        con.commit()
        con.close()
        e = rt_enrich.Enrich(export_dir=self.export, triage_dir=self.triage, sidecar=side, store_path=self.db)
        con = sqlite3.connect(self.db)
        e.attach(con)
        row = con.execute("SELECT e.too_short, e.n_spans, e.fc_id FROM enr.enr e WHERE e.id='reddit_posts:10'").fetchone()
        self.assertEqual(row, (None, None, None))

    def test_background_reload_serves_then_fills(self):
        write_real(self.export, self.default_parts())
        e = rt_enrich.Enrich(export_dir=self.export, triage_dir=self.triage, sidecar=os.path.join(self.tmp, "bg.sqlite"),
                             store_path=self.db, background=True)
        e.refresh()
        e.wait()
        self.assertEqual(e.status()["records"], 5)
        self.assertFalse(e.status()["loading"])


class Api(RealFixture):
    def setUp(self):
        super().setUp()
        write_real(self.export, self.default_parts())

    def batch(self, iid):
        d = self.api.item(iid)
        json.dumps(d, allow_nan=False)      # strict JSON: no NaN / Infinity anywhere in the payload
        return d["inferred"]["batch"], d["inferred"]

    def test_technique_probs_all_14_sorted_with_basis(self):
        b, inf = self.batch("reddit_posts:10")
        tp = b["technique_probs"]
        self.assertEqual((tp["basis"], tp["basis_note"]),
                         ("encoder", "encoder score (max token probability): uncalibrated, not validated on social posts"))
        self.assertEqual(len(tp["classes"]), 14)
        self.assertEqual({c["technique"] for c in tp["classes"]}, set(CLASSES))
        ps = [c["probability"] for c in tp["classes"]]
        self.assertEqual(ps, sorted(ps, reverse=True))
        self.assertEqual(tp["classes"][0]["technique"], "Loaded_Language")
        self.assertEqual(inf["technique_probs"], tp)
        self.assertEqual(b["strong_techniques"], ["Loaded_Language"])
        self.assertEqual((b["n_spans"], b["n_strong"], b["too_short"]), (4, 1, False))
        self.assertAlmostEqual(b["p_prop"], 0.9)

    def test_spans_are_sliced_from_title_and_text_and_bad_offsets_dropped(self):
        b, _ = self.batch("reddit_posts:10")
        spans = b["spans"]
        self.assertEqual(b["spans_dropped"], 2)          # out of range, and the one straddling title / text
        by = {s["technique"]: s for s in spans}
        self.assertEqual(by["Loaded_Language"]["text"], "vaccine")
        self.assertEqual(by["Loaded_Language"]["field"], "text")
        self.assertEqual(by["Doubt"]["field"], "title")
        self.assertEqual(by["Doubt"]["text"], "Rumour")
        text = "they say the vaccine is a plot"
        s = by["Loaded_Language"]
        self.assertEqual(text[s["start"]:s["end"]], "vaccine")      # start/end are relative to the excerpt for a text span

    def test_too_short_item(self):
        b, inf = self.batch("news:scraped_articles:2")
        self.assertTrue(b["too_short"])
        self.assertIsNone(b["p_prop"])
        self.assertIsNone(inf["p_prop"])
        row = self.api.items(self.q(kind="news", sort="recent"))["items"]
        short = [r for r in row if r["id"] == "news:scraped_articles:2"][0]
        self.assertTrue(short["inferred"]["batch"]["too_short"])
        self.assertIsNone(short["inferred"]["p_prop"])
        json.dumps(row, allow_nan=False)

    def test_countries_with_terms_and_names(self):
        b, _ = self.batch("news:scraped_articles:1")
        self.assertEqual(b["countries"], [{"iso3": "IND", "weight": 0.5, "methods": ["name"], "terms": ["invented place"], "name": "India"}])
        b, _ = self.batch("reddit_posts:11")
        self.assertEqual(b["countries"], [])

    def test_nearest_factcheck_with_and_without_rerank_and_desk_id(self):
        b, _ = self.batch("news:scraped_articles:1")
        nf = b["nearest_factcheck"]
        self.assertEqual((nf["similarity"], nf["rerank"], nf["label"], nf["id"]), (0.61, None, "FAKE", "kb:test-1"))
        self.assertTrue(nf["below_rerank_floor"])
        self.assertEqual(nf["desk_id"], "db:desk-777")           # normalised URL (trailing slash, scheme) matches the Desk index
        self.assertEqual(nf["url"], FC_URL)
        self.assertIn("<script>", nf["claim"])                    # stored as data; the UI escapes it
        b, _ = self.batch("reddit_posts:10")
        nf = b["nearest_factcheck"]
        self.assertEqual((nf["similarity"], nf["rerank"], nf["label"], nf["below_rerank_floor"]), (0.81, 0.66, "MISLEADING", False))
        b, _ = self.batch("reddit_posts:11")
        self.assertIsNone(b["nearest_factcheck"])

    def test_desk_id_is_null_when_url_is_not_in_the_index_or_no_mapper(self):
        self.desk.clear()
        b, _ = self.batch("news:scraped_articles:1")
        self.assertIsNone(b["nearest_factcheck"]["desk_id"])
        api = rt_api.RtApi(db_path=self.db, enrich=self.enrich, clock=lambda: NOW)
        self.assertIsNone(api.item("news:scraped_articles:1")["inferred"]["batch"]["nearest_factcheck"]["desk_id"])

    def test_desk_url_map_prefers_factchecks(self):
        items = [{"id": "db:cand", "kind": "candidate", "url": "https://www.example.org/a/"},
                 {"id": "db:fc", "kind": "factcheck", "url": "http://example.org/a"}, {"id": "db:none", "url": None}]
        import server
        m = server.desk_url_map(items)
        self.assertEqual(m[norm_url("https://example.org/a")], "db:fc")

    def test_list_payload_carries_flags_not_detail(self):
        rows = self.api.items(self.q(sort="p_prop"))["items"]
        r = [x for x in rows if x["id"] == "reddit_posts:10"][0]
        self.assertEqual(r["inferred"]["batch"]["too_short"], False)
        self.assertNotIn("spans", r["inferred"]["batch"])
        self.assertEqual([t["technique"] for t in r["inferred"]["techniques"]], ["Loaded_Language"])

    def test_stats_batch_block_from_meta(self):
        s = self.api.stats(self.q())
        b = s["coverage"]["batch"]
        self.assertEqual((b["scored"], b["n_in_window"], b["n_unscored_in_window"], b["pct_of_window"]), (5, 1000, 995, 0.5))
        self.assertEqual((b["built_at"], b["next_run_at"], b["interval_hours"], b["n_failed_this_run"]), (BUILT, NEXT, 3, 0))
        self.assertEqual((b["caveat"], b["model_version"]), (CAVEAT, "prop_span_test+kb_test"))
        self.assertIsNone(b["estimated_runs"])           # meta gives no per-run limit: no guess
        self.assertEqual(b["too_short"], 1)
        self.assertEqual(b["analysed"], 4 + 1 - 0)       # in-cache batch rows: 1,2 (news), 10,11,12 (social)
        json.dumps(s, allow_nan=False)

    def test_stats_separates_too_short_from_signal_and_technique_mix(self):
        s = self.api.stats(self.q())
        self.assertEqual(s["p_prop_distribution"]["too_short"], 1)
        self.assertEqual(s["p_prop_distribution"]["n"], 4)       # five analysed, one too short
        mix = {t["technique"]: t["items"] for t in s["technique_mix"]["top"]}
        self.assertEqual(mix["Loaded_Language"], 4)               # the too-short item's techniques are not counted

    def test_stats_estimated_runs_when_meta_gives_a_limit(self):
        write_real(self.export, self.default_parts(), built_at="2026-11-12T14:00:00+00:00",
                   meta_extra=dict(max_items_per_run=40000, n_scored_this_run=40000, n_in_window=628535, n_unscored_in_window=588535,
                                   n_failed_this_run=3))
        b = self.api.stats(self.q())["coverage"]["batch"]
        self.assertEqual((b["per_run_limit"], b["runs_rate"], b["estimated_runs"]), (40000, 40000, 15))
        self.assertEqual(b["n_failed_this_run"], 3)

    def test_meta_without_next_run_at(self):
        meta = write_real(self.export, self.default_parts(), built_at="2026-11-12T15:00:00+00:00")
        meta.pop("next_run_at")
        with open(os.path.join(self.export, "meta.json"), "w") as f:
            json.dump(meta, f)
        os.utime(os.path.join(self.export, "meta.json"), (time.time() + 5, time.time() + 5))
        b = self.api.stats(self.q())["coverage"]["batch"]
        self.assertIsNone(b["next_run_at"])
        self.assertEqual(b["scored"], 5)

    def test_batch_block_without_any_meta(self):
        b = rt_api.batch_block(None, 0, 7)
        self.assertFalse(b["present"])
        self.assertEqual((b["scored"], b["n_in_window"], b["pct_of_window"], b["estimated_runs"]), (7, None, None, None))

    def test_old_contract_export_has_no_real_fields(self):
        import test_rt_api as t
        shutil_dir = os.path.join(self.tmp, "old")
        t.default_export(shutil_dir)
        e = rt_enrich.Enrich(export_dir=shutil_dir, triage_dir=self.triage, sidecar=os.path.join(self.tmp, "o.sqlite"), store_path=self.db)
        api = rt_api.RtApi(db_path=self.db, enrich=e, clock=lambda: NOW)
        inf = api.item("news:scraped_articles:1")["inferred"]
        self.assertEqual(inf["spans"][0]["text"], "river valley")
        self.assertNotIn("technique_probs", inf["batch"])
        self.assertEqual(inf["nearest_factcheck"], {"id": "kb2:af-1", "similarity": 0.71})
        self.assertEqual(api.stats(self.q())["coverage"]["batch"]["scored"], 8)

    def test_country_names_missing_file_gives_null_name(self):
        api = rt_api.RtApi(db_path=self.db, enrich=self.enrich, clock=lambda: NOW, country_names=rt_api.CountryNames("/nonexistent/x.json"))
        c = api.item("news:scraped_articles:1")["inferred"]["batch"]["countries"][0]
        self.assertIsNone(c["name"])
        self.assertEqual(c["iso3"], "IND")


if __name__ == "__main__":
    unittest.main()
