"""Offline tests: normalisation, dedup keys, language groups, quotas, consensus, splits, agreement maths."""
import random
from datetime import datetime

import pytest

from fnp.annotate import export as E
from fnp.annotate import ingest as I
from fnp.annotate import preannotate as P
from fnp.annotate import sources as S


# ---------------------------------------------------------------- sources
def test_clean_text_removes_nul_and_normalises_newlines():
    assert S.clean_text("  a\x00b\r\nc\rd  ") == "ab\nc\nd"


def test_compose_puts_title_and_body_in_one_offset_space():
    text, title, tl = S.compose("Headline", "Body text")
    assert text == "Headline\n\nBody text" and title == "Headline" and tl == len("Headline") + 2
    assert text[tl:] == "Body text"
    assert S.compose("Same", "Same")[2] == 0               # no duplicated title
    assert S.compose(None, " only body ")[0] == "only body"


def test_dup_key_groups_reposts_but_separates_different_texts():
    core = "Breaking news the old bridge collapsed in the city centre this morning and rescue teams are at the scene"
    a = f"RT @bob: {core} http://t.co/x"
    b = f"{core} @alice please share https://t.co/y #breaking"        # repost with a different tail
    c = "Completely different statement about elections and the vote count in several districts across the state"
    assert S.dup_key(a) == S.dup_key(b) != S.dup_key(c)
    assert S.norm_hash(a) != S.norm_hash(b)                # exact-dup guard is stricter than the split key
    # documented limit: texts shorter than 12 words are compared whole, so a short repost with extra words is a new group
    assert S.dup_key("Bridge collapses in the city centre") != S.dup_key("Bridge collapses in the city centre look")


@pytest.mark.parametrize("lang,text,expect", [
    ("en", "x", ("en", "en")), ("English", "x", ("en", "en")), ("hi", "x", ("hi", "hi")), ("hin", "x", ("hi", "hi")),
    ("ta", "x", ("ta", "other")), (None, "यह एक परीक्षण वाक्य है", ("hi", "hi")),
    (None, "This is plainly English text", ("en", "en")), ("", "🙂🙂", ("unknown", "other"))])
def test_lang_group(lang, text, expect):
    assert S.lang_group(lang, text) == expect


def test_flags_are_deterministic_and_roughly_at_the_configured_fraction():
    hs = [S.norm_hash(f"text {i}") for i in range(4000)]
    assert S.flag_fraction(hs[0], "double") == S.flag_fraction(hs[0], "double")
    frac = sum(S.flag_fraction(h, "double") < 0.2 for h in hs) / len(hs)
    assert 0.17 < frac < 0.23
    assert S.flag_fraction(hs[1], "double") != S.flag_fraction(hs[1], "blind")


def test_source_registry_is_consistent():
    assert len({(s.kind, s.platform) for s in S.SOURCES}) == len(S.SOURCES)
    assert all(s.text and s.pk and s.ts and s.pub for s in S.SOURCES)
    assert {s.kind for s in S.SOURCES} == {"news", "social"}


# ---------------------------------------------------------------- ingest selection
def _cand(i, grp, period="2026-01", ts=None):
    return dict(source_pk=str(i), lang_group=grp, period=period, scraped_at=ts or datetime(2026, 1, 1, 0, 0, i % 60))


def test_select_applies_language_quotas_and_backfills_shortfalls():
    cands = [_cand(i, "en") for i in range(50)] + [_cand(100 + i, "hi") for i in range(50)] + [_cand(200 + i, "other") for i in range(50)]
    got = I._select(cands, 20, "newest", random.Random(1))
    from collections import Counter
    c = Counter(x["lang_group"] for x in got)
    assert c == {"en": 18, "hi": 2}                        # 90% / 10%; 'other' is excluded by policy (share 0)
    only_en = I._select([_cand(i, "en") for i in range(30)], 20, "backfill", random.Random(1))
    assert len(only_en) == 20                              # quota shortfall is back-filled from what exists


def test_backfill_is_time_balanced_across_months():
    cands = [_cand(i, "en", period=f"2026-0{1 + i % 3}") for i in range(90)]
    got = I._select(cands, 30, "backfill", random.Random(3))
    from collections import Counter
    assert set(Counter(x["period"] for x in got).values()) == {10}


def test_newest_mode_prefers_recent_rows():
    cands = [_cand(i, "en", ts=datetime(2026, 1, 1 + i)) for i in range(20)]
    got = I._select(cands, 5, "newest", random.Random(0))
    assert {int(x["source_pk"]) for x in got} == {15, 16, 17, 18, 19}


# ---------------------------------------------------------------- preannotate
def test_claim_suggestions_only_for_verbatim_quotes_with_correct_offsets():
    text = "Intro. The minister said taxes fall 5% 🙂 next year. Outro."
    claims = [{"claim": "Taxes will fall 5%", "evidence_quote": "taxes fall 5% 🙂 next year", "claim_type": "prediction",
               "quoted_or_reported": "reported", "check_worthy": 0.8, "speaker": "the minister"},
              {"claim": "x", "evidence_quote": "not in the text", "claim_type": "factual", "quoted_or_reported": "asserted", "check_worthy": 0.5}]
    out = P.claim_suggestions(text, claims)
    assert len(out) == 1
    s = out[0]
    assert text[s["start"]:s["end"]] == s["text"] == "taxes fall 5% 🙂 next year" and s["label"] == "PREDICTION_OR_FORECAST"


@pytest.mark.parametrize("c,exp", [
    ({"claim_type": "factual", "quoted_or_reported": "asserted"}, "DIRECT_STATEMENT"),
    ({"claim_type": "factual", "quoted_or_reported": "quoted"}, "REPORTED_OR_REQUOTED_STATEMENT"),
    ({"claim_type": "quote", "quoted_or_reported": "asserted"}, "REPORTED_OR_REQUOTED_STATEMENT"),
    ({"claim_type": "prediction"}, "PREDICTION_OR_FORECAST"), ({"claim_type": "opinion"}, "OTHER"), ({}, "UNCLEAR")])
def test_claim_type_mapping(c, exp):
    assert P.map_claim_type(c) == exp


# ---------------------------------------------------------------- agreement maths
def test_iou_and_one_to_one_matching():
    assert E.iou((0, 10), (0, 10)) == 1 and E.iou((0, 10), (10, 20)) == 0 and E.iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    A = [{"start": 0, "end": 10}, {"start": 0, "end": 11}]
    B = [{"start": 0, "end": 10}]
    assert E.match_spans(A, B) == [(0, 0)]                 # B can be matched only once, to the best A


def test_cohen_kappa_known_values():
    assert E.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert E.cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0
    # Wikipedia worked example: 20+15 agree, kappa = 0.4
    x = [1] * 25 + [0] * 25
    y = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert E.cohen_kappa(x, y) == pytest.approx(0.4)
    assert E.cohen_kappa([], []) is None


def test_wilson_interval():
    lo, hi = E.wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=2e-3) and hi == pytest.approx(0.943, abs=2e-3)
    assert E.wilson(0, 0) is None


def test_split_is_deterministic_grouped_and_roughly_proportional():
    keys = [S.dup_key(f"unique doc {i} " + "w" * (i % 7)) for i in range(3000)]
    s1 = [E.split_of(k, "v1", 0.1, 0.1) for k in keys]
    assert s1 == [E.split_of(k, "v1", 0.1, 0.1) for k in keys]
    assert E.split_of(keys[0], "v1", .1, .1) == E.split_of(keys[0], "v1", .1, .1)      # same dup_key -> same split
    n = len(keys)
    assert 0.07 < s1.count("test") / n < 0.13 and 0.07 < s1.count("dev") / n < 0.13
    assert s1 != [E.split_of(k, "v2", 0.1, 0.1) for k in keys]                         # new version reshuffles


# ---------------------------------------------------------------- consensus
def _sub(uid, name, spans, role="annotator", outcome=None):
    return dict(annotator_id=uid, username=name, role=role, outcome=outcome or ("annotated" if spans else "none"), spans=spans)


def _c(a, b, t="DIRECT_STATEMENT", cw="yes"):
    return dict(start=a, end=b, text="x" * (b - a), claim_type=t, check_worthy=cw, normalised_claim=None, speaker=None)


def _p(a, b, techs):
    return dict(start=a, end=b, text="x" * (b - a), techniques=list(techs))


def test_single_annotator_and_negative():
    r = E.resolve_item("claim", [_sub(1, "a", [_c(0, 10)])])
    assert r["status"] == "positive" and r["n_annotators"] == 1
    assert E.resolve_item("claim", [_sub(1, "a", [])])["status"] == "negative"


def test_double_agree_keeps_matched_spans_and_flags_type_disagreement():
    a = _sub(1, "a", [_c(0, 20, "DIRECT_STATEMENT", "yes"), _c(50, 60)])
    b = _sub(2, "b", [_c(2, 20, "ALLEGATION_OR_ACCUSATION", "yes"), _c(50, 61)])
    r = E.resolve_item("claim", [b, a])                    # order independent
    assert r["status"] == "positive" and len(r["spans"]) == 2
    s0 = r["spans"][0]
    assert (s0["start"], s0["end"]) == (0, 20) and s0["claim_type"] is None and s0["claim_type_agreed"] is False and s0["check_worthy"] == "yes"
    assert r["spans"][1]["claim_type"] == "DIRECT_STATEMENT"


def test_double_disagreement_is_disputed_not_silently_negative():
    r = E.resolve_item("claim", [_sub(1, "a", [_c(0, 10)]), _sub(2, "b", [])])
    assert r["status"] == "disputed" and r["spans"] == []
    assert E.resolve_item("claim", [_sub(1, "a", []), _sub(2, "b", [])])["status"] == "negative"


def test_union_and_adjudicator_policies():
    a, b = _sub(1, "a", [_c(0, 10)]), _sub(2, "b", [_c(30, 40)])
    assert len(E.resolve_item("claim", [a, b], "union")["spans"]) == 2
    adj = _sub(3, "boss", [_c(0, 10)], role="adjudicator")
    r = E.resolve_item("claim", [a, b, adj])
    assert r["status"] == "positive" and r["policy_used"] == "adjudicated" and r["annotators"] == ["boss"]


def test_propaganda_consensus_requires_same_technique_and_groups_multi_technique_spans():
    a = _sub(1, "a", [_p(0, 10, ["Doubt", "Loaded_Language"]), _p(20, 30, ["Slogans"])])
    b = _sub(2, "b", [_p(0, 10, ["Loaded_Language", "Doubt"]), _p(20, 30, ["Repetition"])])
    r = E.resolve_item("propaganda", [a, b])
    assert r["status"] == "disputed"                       # the Slogans/Repetition span has no same-technique partner
    assert r["spans"] == [{"start": 0, "end": 10, "techniques": ["Doubt", "Loaded_Language"], "annotators": ["a", "b"]}]


def test_validate_submission_catches_offset_text_and_outcome_mismatches():
    text = "héllo 🙂 world"
    ok = _sub(1, "a", [dict(start=0, end=5, text="héllo", claim_type="OTHER", check_worthy="yes")])
    assert E.validate_submission(text, ok) == []
    bad = _sub(1, "a", [dict(start=0, end=5, text="hello", claim_type="OTHER", check_worthy="yes")])
    assert E.validate_submission(text, bad)
    assert E.validate_submission(text, dict(outcome="none", spans=[ok["spans"][0]]))
    assert E.validate_submission(text, dict(outcome="annotated", spans=[]))
    emoji = dict(start=6, end=7, text="🙂", claim_type="OTHER", check_worthy="no")
    assert E.validate_submission(text, dict(outcome="annotated", spans=[emoji])) == []      # code-point offsets


def test_agreement_report_claims_and_propaganda():
    per = {1: [_sub(1, "a", [_c(0, 10)]), _sub(2, "b", [_c(0, 10)])], 2: [_sub(1, "a", []), _sub(2, "b", [])],
           3: [_sub(1, "a", [_c(0, 10, "OTHER")]), _sub(2, "b", [_c(0, 10, "UNCLEAR")])], 4: [_sub(1, "a", [_c(5, 9)]), _sub(2, "b", [])]}
    rep = E.agreement_report("claim", per, n_boot=50)
    assert rep["n_double_items"] == 4 and rep["item_level"]["has_label_agreement"] == 0.75
    assert rep["span_iou50"]["matched"] == 2 and rep["claim_type_on_matched"]["accuracy"] == 0.5
    pper = {1: [_sub(1, "a", [_p(0, 10, ["Doubt"])]), _sub(2, "b", [_p(0, 10, ["Doubt"])])]}
    prep = E.agreement_report("propaganda", pper, n_boot=10)
    assert prep["semeval_style_A_vs_B"]["TC_span_f1"] == pytest.approx(1.0)
    assert E.agreement_report("claim", {})["n_double_items"] == 0


# ------------------------------------------------------------------ auth, language policy, class-deficit scoring (offline)
def test_password_hashing_roundtrip_and_failures():
    from fnp.annotate import auth
    h = auth.hash_password("correct horse 7")
    assert h.startswith("scrypt$") and "correct horse" not in h
    assert auth.verify_password("correct horse 7", h) and not auth.verify_password("Correct horse 7", h)
    assert auth.hash_password("correct horse 7") != h                      # fresh salt every time
    for bad in (None, "", "plain", "scrypt$x$y$z$q$w", "md5$1$2$3$ab$cd"):
        assert auth.verify_password("anything", bad) is False              # malformed hashes never raise or pass


def test_generated_passwords_and_policy():
    from fnp.annotate import auth
    pws = {auth.gen_password() for _ in range(50)}
    assert len(pws) == 50 and all(len(p) == 12 and not set(p) & set("0O1lI") for p in pws)
    with pytest.raises(auth.AuthError):
        auth.check_password_policy("short")
    with pytest.raises(auth.AuthError):
        auth.check_password_policy("sameasname1", "SameAsName1")
    auth.check_password_policy("long enough pw", "someone")


def test_only_english_and_hindi_pass_the_ingest_filter():
    from fnp.annotate import ingest, sources as S
    src = S.BY_KEY[("social", "twitter")]
    row = lambda text, lang: dict(pk="1", ts=None, pub=None, title=None, body=text, url=None, author=None, outlet=None, lang=lang)
    en = ingest._prepare(src, row("The minister announced a new policy on rural roads today and critics were quick to respond.", "en"))
    hi = ingest._prepare(src, row("प्रधानमंत्री ने आज ग्रामीण सड़कों पर एक नई नीति की घोषणा की और आलोचकों ने तुरंत प्रतिक्रिया दी।", "hi"))
    fr = ingest._prepare(src, row("Le ministre a annoncé aujourd'hui une nouvelle politique pour les routes rurales en France.", "fr"))
    assert en and en["lang_group"] == "en" and hi and hi["lang_group"] == "hi"
    assert fr is None                                                       # ann_share_other = 0


def test_select_never_back_fills_from_other_languages():
    import random
    from fnp.annotate import ingest
    cands = [dict(lang_group="en", scraped_at=i, source_pk=str(i), period="2026-01") for i in range(3)] + \
            [dict(lang_group="other", scraped_at=i, source_pk="o%d" % i, period="2026-01") for i in range(50)]
    got = ingest._select(cands, 20, "newest", random.Random(0))
    assert len(got) == 3 and {c["lang_group"] for c in got} == {"en"}      # short rather than serving a language nobody reads


def test_class_deficit_scoring_prefers_rare_classes_and_updates_as_batch_fills():
    from fnp.annotate.batching import Deficit
    d = object.__new__(Deficit)
    d.target = {"claim": {"DIRECT_STATEMENT": 100}, "propaganda": {"Doubt": 100, "Slogans": 100}}
    d.have = {"claim": {"DIRECT_STATEMENT": 0.0}, "propaganda": {"Doubt": 100.0, "Slogans": 0.0}}
    d.neg_target = {"claim": 20, "propaganda": 20}; d.neg_have = {"claim": 0.0, "propaganda": 20.0}
    mk = lambda c, p, cr=True, pr=True: dict(claim_labels=c, prop_labels=p, claim_ran=cr, prop_ran=pr)
    full, rare, none_, unseen = mk([], ["Doubt"]), mk([], ["Slogans"]), mk([], []), mk([], [], False, False)
    assert d.score(rare) > d.score(full)                                    # saturated Doubt scores 0, Slogans still needs items
    assert d.score(unseen) == 1.0                                           # unseen by the models: neutral on both layers
    assert d.layer_score("propaganda", [], True) == 0.0                     # propaganda negatives already at target
    assert d.layer_score("claim", [], True) == 1.0                          # claim negatives still needed
    before = d.score(rare)
    for _ in range(50):
        d.take(rare)
    assert d.score(rare) < before                                           # each pick lowers the class's deficit
    d.take(unseen)                                                          # unseen items change nothing
    assert d.have["propaganda"]["Slogans"] == 50


# ---------------------------------------------------------------- YouTube sources and truncation
def test_truncate_prefers_sentence_end_and_is_a_verbatim_prefix():
    text = "First sentence here. Second sentence is a bit longer than the first. Third one runs on and on without end"
    out, how = S.truncate_at_boundary(text, 70, 0.5)
    assert how == "sentence" and out == "First sentence here. Second sentence is a bit longer than the first."
    out, how = S.truncate_at_boundary(text, 60, 0.5)          # 2nd sentence does not fit: keep the 1st only if past the floor
    assert how in ("word", "sentence") and text.startswith(out) and len(out) <= 60


def test_truncate_handles_hindi_danda_and_unpunctuated_transcripts():
    hi = "यह पहला वाक्य है। यह दूसरा वाक्य थोड़ा लंबा है। और यह तीसरा वाक्य बिना अंत के चलता रहता है"
    out, how = S.truncate_at_boundary(hi, 50, 0.4)
    assert how == "sentence" and out.endswith("है।") and hi.startswith(out) and len(out) <= 50
    raw = "so today we are going to talk about the budget and what it means for farmers " * 6   # auto-captions: no punctuation
    out, how = S.truncate_at_boundary(raw.strip(), 100, 0.6)
    assert how == "word" and len(out) <= 100 and raw.startswith(out) and not out.endswith(" ")


def test_truncate_leaves_short_text_and_does_not_split_numbers_or_words():
    assert S.truncate_at_boundary("short text.", 100) == ("short text.", None)
    out, how = S.truncate_at_boundary("Growth was 3.5 percent in the last quarter of the year", 12, 0.3)
    assert (out, how) == ("Growth was", "word")                # not "Growth was 3." and not mid-word


def test_truncate_does_not_stop_after_an_abbreviation():
    text = "We spoke earlier to Mr. Sharma about the budget. He said it was fine for everyone concerned here today"
    out, how = S.truncate_at_boundary(text, 28, 0.5)
    assert how != "sentence"                                  # "Mr." is not a sentence end
    out, how = S.truncate_at_boundary(text, 60, 0.3)
    assert how == "sentence" and out.endswith("budget.")


def test_youtube_sources_are_registered_with_distinct_platforms_and_truncation():
    yt = {s.platform: s for s in S.SOURCES if s.platform.startswith("youtube")}
    assert set(yt) == {"youtube_comment", "youtube_video", "youtube_transcript"}
    assert yt["youtube_comment"].url_expr and not yt["youtube_comment"].truncate
    assert yt["youtube_video"].truncate and yt["youtube_transcript"].truncate
    assert yt["youtube_transcript"].lang[0] == "transcript_language"     # spoken language, not the metadata language
    assert yt["youtube_video"].group_by_pk and yt["youtube_transcript"].group_by_pk


def test_prepare_truncates_instead_of_dropping_and_groups_a_videos_items_together():
    from fnp.annotate import ingest as I
    src = next(s for s in S.SOURCES if s.platform == "youtube_transcript")
    body = ("The minister said the policy would help farmers. " * 60).strip()            # ~2.9k chars, over the 1500 cap
    r = dict(pk="vid1", ts=datetime(2026, 1, 1), pub=datetime(2026, 1, 1), title="Budget talk", body=body, url="u", author="a",
             outlet="a", lang="en")
    p = I._prepare(src, r)
    assert p and len(p["text"]) - p["title_len"] <= 1500 and p["meta_extra"]["truncated"] == "sentence"
    assert p["text"].rstrip().endswith("farmers.")
    d = I._prepare(next(s for s in S.SOURCES if s.platform == "youtube_video"), {**r, "body": "A different description of the same video, long enough to pass the filter."})
    assert d and d["dup_key"] == p["dup_key"] and len(p["dup_key"]) == 40                                          # same video -> same split group
    assert I._prepare(next(s for s in S.SOURCES if s.platform == "youtube_comment"), {**r, "body": "x" * 2000}) is None  # comments keep the hard cap


# ---------------------------------------------------------------- duplicates, substance, class weights
def test_substance_and_body_key_ignore_links_and_titles():
    assert S.substance("https://t.co/abc @someone") == 0 and S.substance("Great speech today https://t.co/x") > 10
    assert S.substance("#MAGA #AmericaFirst") > 10                     # hashtags can carry slogans
    a = S.body_key("Title A\n\nPM says growth will reach five percent this year, officials said https://t.co/1", 9)
    b = S.body_key("A different title\n\nPM says growth will reach five percent this year, officials said https://t.co/2", 19)
    assert a == b and len(a) == 40


def test_prepare_drops_link_only_posts_and_sets_body_key():
    from fnp.annotate import ingest as I
    src = next(s for s in S.SOURCES if s.platform == "twitter")
    base = dict(pk="1", ts=datetime(2026, 1, 1), pub=datetime(2026, 1, 1), title=None, url="u", author="a", outlet=None, lang="en")
    assert I._prepare(src, {**base, "body": "https://t.co/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}) is None
    p = I._prepare(src, {**base, "body": "The finance minister announced new tariffs on steel imports today."})
    assert p and len(p["body_key"]) == 40


def test_effective_number_weights_favour_rare_classes_and_record_weight_is_rarest_class():
    from fnp.annotate import export as E
    w = E.effective_number_weights({"common": 10000, "rare": 10, "mid": 500}, 0.999)
    assert w["rare"] > w["mid"] > w["common"] and abs(sum(w.values()) / 3 - 1.0) < 1e-3
    assert E.sample_weight(["common", "rare"], w) == w["rare"]
    assert E.sample_weight([], {**w, "__none__": 0.7}) == 0.7 and E.sample_weight([], w) == 1.0
    rows = [{"split": "train", "status": "positive", "spans": [{"claim_type": "DIRECT_STATEMENT"}]}] * 5 + \
           [{"split": "train", "status": "negative", "spans": []}] + [{"split": "dev", "status": "positive", "spans": [{"claim_type": "X"}]}]
    cw = E.class_weights("claim", rows, 0.99)
    assert cw["counts"] == {"DIRECT_STATEMENT": 5, "__none__": 1} and "X" not in cw["weights"]       # train split only
