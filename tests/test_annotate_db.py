"""DB-backed tests for the annotation service + export. They create a throwaway schema in the annotation DB
(FNP_ANNOT_DB_*), apply the real migrations, and drop it afterwards; skipped when the DB is unreachable.
Source databases are never touched."""
import hashlib
import json
import os
import secrets

import psycopg2
import pytest

from fnp.annotate import batching, db, export, metrics, migrate, service as S


@pytest.fixture(scope="module")
def schema():
    try:
        params = db.annot_params()
        c = psycopg2.connect(**{**params, "connect_timeout": 4})
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"annotation DB not reachable: {type(e).__name__}")
    name = "annot_t_" + secrets.token_hex(4)
    cur = c.cursor()
    cur.execute(f"CREATE SCHEMA {name}")
    migrate.apply_all(cur, name)
    c.commit()
    old = os.environ.get("FNP_ANNOT_SCHEMA")
    os.environ["FNP_ANNOT_SCHEMA"] = name
    db._pool = None
    yield name
    if db._pool:
        db._pool.closeall()
    db._pool = None
    if old is None:
        os.environ.pop("FNP_ANNOT_SCHEMA", None)
    else:
        os.environ["FNP_ANNOT_SCHEMA"] = old
    c.rollback()
    cur.execute(f"DROP SCHEMA {name} CASCADE")
    c.commit()
    c.close()


_n = [0]


def mk_source(platform="twitter", kind="social"):
    with db.tx() as cur:
        cur.execute("INSERT INTO ingest_source (kind,platform,source_db,source_table) VALUES (%s,%s,'t',%s) RETURNING source_id",
                    (kind, platform, f"public.{platform}_{secrets.token_hex(3)}"))
        return cur.fetchone()["source_id"]


def mk_item(sid, text, lang="en", period="2026-01", double=False, blind=False, dup=None):
    _n[0] += 1
    h = hashlib.sha1(f"{text}{_n[0]}".encode()).hexdigest()
    with db.tx() as cur:
        cur.execute("""INSERT INTO item (source_id, source_pk, ingest_mode, lang_group, period, text, text_sha256, norm_hash, dup_key,
                       is_double, is_blind) VALUES (%s,%s,'newest',%s,%s,%s,%s,%s,%s,%s,%s) RETURNING item_id""",
                    (sid, str(_n[0]), lang, period, text, hashlib.sha256(text.encode()).hexdigest(), h, dup or h, double, blind))
        return cur.fetchone()["item_id"]


def quiet():
    """The queue is global: take every earlier test item out of it so queue tests see only their own items."""
    with db.tx() as cur:
        cur.execute("UPDATE item SET status='excluded' WHERE status <> 'excluded'")


def assign(item_id, ann_id):
    with db.tx() as cur:
        cur.execute("INSERT INTO assignment (item_id, annotator_id, status) VALUES (%s,%s,'in_progress') ON CONFLICT DO NOTHING", (item_id, ann_id))


def ann(name, batch_size=None):
    a = S.get_or_create_annotator(name)["annotator_id"]
    if batch_size:
        with db.tx() as cur:
            cur.execute("UPDATE annotator SET batch_size=%s WHERE annotator_id=%s", (batch_size, a))
    return a


def test_item_text_immutable_and_events_append_only(schema):
    i = mk_item(mk_source(), "some stable text")
    with pytest.raises(psycopg2.Error):
        with db.tx() as cur:
            cur.execute("UPDATE item SET text='tampered' WHERE item_id=%s", (i,))
    a = ann("evt_user"); assign(i, a)
    S.add_claim(i, a, 0, 4, "OTHER", "no")
    with pytest.raises(psycopg2.Error):
        with db.tx() as cur:
            cur.execute("DELETE FROM label_event WHERE item_id=%s", (i,))
    with pytest.raises(psycopg2.Error):
        with db.tx() as cur:
            cur.execute("UPDATE label_event SET action='delete' WHERE item_id=%s", (i,))


def test_username_validation(schema):
    for bad in ("", "a", "has space", "x" * 70, "semi;colon"):
        with pytest.raises(S.AnnotError):
            S.get_or_create_annotator(bad)
    assert S.get_or_create_annotator("ok.name-1")["role"] == "annotator"


def test_offsets_are_codepoints_and_text_is_server_derived(schema):
    text = "Ball-by-ball 🏏 अपडेट: PM says 5% growth 🙂 expected"
    i = mk_item(mk_source(), text)
    a = ann("cp_user"); assign(i, a)
    s = text.index("PM")                                   # python str index == code points
    sp = S.add_claim(i, a, s, len(text), "REPORTED_OR_REQUOTED_STATEMENT", "yes")
    assert sp["text"] == text[s:] and s != len(text[:s].encode("utf-16-le")) // 2   # UTF-16 index would differ after the emoji
    for st, en in [(5, 5), (-1, 4), (0, len(text) + 1), (3, 2), ("0", 4), (True, 4)]:
        with pytest.raises(S.AnnotError):
            S.add_claim(i, a, st, en, "OTHER", "no")
    with pytest.raises(S.AnnotError):                       # whitespace-only span
        j = mk_item(mk_source(), "a    b"); assign(j, a)
        S.add_claim(j, a, 1, 4, "OTHER", "no")
    with pytest.raises(S.AnnotError):
        S.add_claim(i, a, 0, 5, "NOT_A_TYPE", "yes")
    with pytest.raises(S.AnnotError):
        S.add_claim(i, a, 0, 5, "OTHER", "maybe")
    with pytest.raises(S.AnnotError):
        S.add_propaganda(i, a, 0, 5, [])
    with pytest.raises(S.AnnotError):
        S.add_propaganda(i, a, 0, 5, ["Made_Up"])


def test_layer_invariants_and_reopen_on_edit(schema):
    i = mk_item(mk_source(), "The mayor stole the money and everyone knows it is a disgrace")
    a = ann("inv_user"); assign(i, a)
    with pytest.raises(S.AnnotError):
        S.submit_layer(i, a, "claim", "annotated")           # no spans
    p = S.add_propaganda(i, a, 0, 20, ["Loaded_Language", "Doubt"])
    with pytest.raises(S.AnnotError):
        S.submit_layer(i, a, "propaganda", "none")           # spans exist
    assert S.submit_layer(i, a, "propaganda", "annotated")["item_complete"] is False
    assert S.submit_layer(i, a, "claim", "none")["item_complete"] is True
    assert S.get_item(i, a)["assignment"]["status"] == "done"
    S.update_propaganda(p["span_id"], a, techniques=["Slogans"])     # edit after submit -> layer re-opened
    d = S.get_item(i, a)
    assert "propaganda" not in d["results"] and d["results"]["claim"] == "none" and d["assignment"]["status"] == "in_progress"
    S.delete_propaganda(p["span_id"], a)
    S.submit_layer(i, a, "propaganda", "none")
    ev = [(e["entity_type"], e["action"]) for e in S.history(i, a)]
    assert ("propaganda_span", "create") in ev and ("propaganda_span", "edit") in ev and ("propaganda_span", "delete") in ev
    assert ("layer_result", "reopen") in ev and ("layer_result", "submit") in ev


def test_cannot_touch_another_annotators_span(schema):
    i = mk_item(mk_source(), "Statement about something that happened yesterday in town")
    a, b = ann("own_a"), ann("own_b"); assign(i, a); assign(i, b)
    sp = S.add_claim(i, a, 0, 9, "DIRECT_STATEMENT", "yes")
    with pytest.raises(S.AnnotError):
        S.update_claim(sp["span_id"], b, claim_type="OTHER")
    with pytest.raises(S.AnnotError):
        S.delete_claim(sp["span_id"], b)
    assert S.get_item(i, b)["claims"] == []                  # b does not see a's labels


def test_suggestions_origin_blind_and_reject(schema):
    sid = mk_source()
    text = "Officials said the bridge will reopen in March, which is absurd nonsense."
    i, blind = mk_item(sid, text), mk_item(sid, text + " x", blind=True)
    with db.tx() as cur:
        for item in (i, blind):
            cur.execute("INSERT INTO suggestion (item_id, layer, model_version, start_char, end_char, span_text, label) "
                        "VALUES (%s,'propaganda','enc_v1',%s,%s,%s,'Loaded_Language') RETURNING suggestion_id", (item, 42, 70, text[42:70]))
            cur.fetchone()
        cur.execute("SELECT suggestion_id FROM suggestion WHERE item_id=%s", (i,)); sug = cur.fetchone()["suggestion_id"]
    a = ann("sug_user"); assign(i, a); assign(blind, a)
    assert len(S.get_item(i, a)["suggestions"]) == 1 and S.get_item(blind, a)["suggestions"] == []
    sp = S.add_propaganda(i, a, 42, 70, ["Loaded_Language"], suggestion_id=sug)
    assert sp["origin"] == "suggestion_accepted"
    assert S.get_item(i, a)["suggestions"] == []              # handled suggestions disappear
    edited = S.update_propaganda(sp["span_id"], a, start=44, end=70)
    assert edited["origin"] == "suggestion_edited"
    b = ann("sug_user2"); assign(i, b)
    sp2 = S.add_propaganda(i, b, 42, 70, ["Doubt"], suggestion_id=sug)
    assert sp2["origin"] == "suggestion_edited"               # same span, different technique -> counts as an edit
    c = ann("sug_user3"); assign(i, c)
    S.reject_suggestion(i, c, sug)
    assert S.get_item(i, c)["suggestions"] == []


def test_not_assigned_items_are_off_limits(schema):
    i = mk_item(mk_source(), "An item that belongs to nobody yet, so nobody may read or label it")
    a = ann("lim_user")
    with pytest.raises(S.AnnotError):
        S.get_item(i, a)
    with pytest.raises(S.AnnotError):
        S.add_claim(i, a, 0, 4, "OTHER", "no")
    with pytest.raises(S.AnnotError):
        S.add_propaganda(i, a, 0, 4, ["Doubt"])
    assign(i, a)
    S.skip_item(i, a, "unreadable")
    with pytest.raises(S.AnnotError):
        S.add_claim(i, a, 0, 4, "OTHER", "no")               # skipped items must be re-opened first


def test_batch_of_n_then_next_n_added_when_finished(schema):
    quiet()
    sid = mk_source("batchsrc")
    for k in range(40):
        mk_item(sid, f"batch test item number {k} with plenty of words to count as a real item", "en" if k % 5 else "hi")
    a = ann("batch_user", batch_size=6)
    served = []
    for _ in range(6):
        i = S.next_item(a)
        assert S.next_item(a) == i                              # resumes the open item until it is finished
        served.append(i)
        S.submit_layer(i, a, "claim", "none"); S.submit_layer(i, a, "propaganda", "none")
    assert len(set(served)) == 6
    with db.tx() as cur:
        cur.execute("SELECT seq, size, completed_at FROM batch WHERE annotator_id=%s ORDER BY seq", (a,))
        rows = cur.fetchall()
    # batch 1 completed the moment its 6th item was submitted, and batch 2 was added right then
    assert [(r["seq"], r["size"]) for r in rows] == [(1, 6), (2, 6)] and rows[0]["completed_at"] is not None and rows[1]["completed_at"] is None
    nxt = S.next_item(a)
    assert nxt not in served
    with db.tx() as cur:
        cur.execute("SELECT count(*) n FROM assignment WHERE annotator_id=%s AND status='assigned'", (a,))
        assert cur.fetchone()["n"] == 5                         # the other 5 of batch 2 are reserved, not yet opened
    pr = metrics.my_progress(a)
    assert pr["batch"]["seq"] == 2 and pr["batch"]["size"] == 6 and pr["batch"]["finished"] == 0 and pr["done"] == 6


def test_skipped_counts_as_finished_for_the_batch(schema):
    quiet()
    sid = mk_source("skipsrc")
    for k in range(8):
        mk_item(sid, f"skip test item {k} with plenty of words to count as a real item here")
    a = ann("skip_user", batch_size=3)
    for _ in range(3):
        S.skip_item(S.next_item(a), a, "unreadable")
    with db.tx() as cur:
        cur.execute("SELECT seq, completed_at FROM batch WHERE annotator_id=%s ORDER BY seq", (a,))
        rows = cur.fetchall()
    assert rows[0]["completed_at"] is not None and len(rows) == 2          # skipped items finish the batch; the next one is added
    assert S.next_item(a) is not None


def test_no_two_annotators_share_a_single_item_and_double_items_reach_two(schema):
    quiet()
    sid = mk_source("dblsrc")
    single = [mk_item(sid, f"single item {k} with enough words to be annotated properly here") for k in range(6)]
    dbl = mk_item(sid, "double annotation item text for queue testing purposes here", double=True)
    a, b, c = ann("d_a", 10), ann("d_b", 10), ann("d_c", 10)
    ia, ib, ic = {S.next_item(a)}, {S.next_item(b)}, {S.next_item(c)}
    def mine_items(x):
        with db.tx() as cur:
            cur.execute("SELECT item_id FROM assignment WHERE annotator_id=%s", (x,))
            return {r["item_id"] for r in cur.fetchall()}
    A, B, Cc = mine_items(a), mine_items(b), mine_items(c)
    assert not (A & B & set(single)) and not (A & B - {dbl}) and not (A & Cc - {dbl}) and not (B & Cc - {dbl})   # singles never shared
    assert sum(dbl in s for s in (A, B, Cc)) == 2                                                               # the double item: exactly two annotators
    assert len(A | B | Cc) == 7 and len(A) + len(B) + len(Cc) == 8


def test_other_languages_are_never_served(schema):
    quiet()
    sid = mk_source("langsrc")
    for k in range(10):
        mk_item(sid, f"french item number {k} with enough words to be a real item here", "other")
    ok = mk_item(sid, "an english item with enough words to be a real item here", "en")
    a = ann("lang_user", 10)
    assert S.next_item(a) == ok
    S.skip_item(ok, a, "x")
    assert S.next_item(a) is None                                  # only 'other' items are left: nothing to serve


def test_stratified_batch_balances_sources_and_languages(schema):
    quiet()
    s1, s2 = mk_source("strat_a"), mk_source("strat_b")
    for k in range(24):
        mk_item(s1, f"source one item {k} with enough words to be a real item", "en" if k % 2 else "hi")
        mk_item(s2, f"source two item {k} with enough words to be a real item", "en" if k % 2 else "hi")
    a = ann("strat_user", 16)
    first = S.next_item(a)
    with db.tx() as cur:
        cur.execute("SELECT s.platform, i.lang_group, count(*) n FROM assignment x JOIN item i USING (item_id) JOIN ingest_source s USING (source_id) "
                    "WHERE x.annotator_id=%s GROUP BY 1,2", (a,))
        rows = {(r["platform"], r["lang_group"]): r["n"] for r in cur.fetchall()}
    assert first and sum(rows.values()) == 16
    per_src = {p: sum(n for (q, _), n in rows.items() if q == p) for p in ("strat_a", "strat_b")}
    assert abs(per_src["strat_a"] - per_src["strat_b"]) <= 1           # balanced across sources
    assert rows.get(("strat_a", "en"), 0) > rows.get(("strat_a", "hi"), 0) >= 1   # both languages, English favoured by share


def test_class_deficit_steers_which_items_are_served(schema, monkeypatch):
    quiet()
    sid = mk_source("steer")
    common, rare = [], []
    with db.tx() as cur:
        for k in range(20):
            for label, bucket in (("Loaded_Language", common), ("Doubt", rare)):
                i = mk_item(sid, f"steer item {label} {k} with enough words to be a real item here")
                bucket.append(i)
                cur.execute("INSERT INTO suggestion (item_id, layer, model_version, start_char, end_char, span_text, label) "
                            "VALUES (%s,'propaganda','enc_v1',0,5,'steer',%s)", (i, label))
                cur.execute("INSERT INTO suggestion_done (item_id, model_version) VALUES (%s,'enc_v1')", (i,))
                cur.execute("INSERT INTO suggestion_done (item_id, model_version) VALUES (%s,'gemma_claims_v1')", (i,))
    monkeypatch.setattr(metrics, "targets", lambda cur: {"items": 1000, "neg": 0, "claim": {}, "propaganda": {"Loaded_Language": 100, "Doubt": 100}})
    monkeypatch.setattr(metrics, "class_counts", lambda cur: {"claim": {}, "propaganda": {"Loaded_Language": 100, "Doubt": 0},
                                                              "none": {"claim": 0, "propaganda": 0}, "annotated": {"claim": 0, "propaganda": 0}})
    real_c = batching.C
    monkeypatch.setattr(batching, "C", lambda k: 0.0 if k == "ann_random_frac" else real_c(k))   # pure steering for this test
    a = ann("steer_user", 10)
    S.next_item(a)
    with db.tx() as cur:
        cur.execute("SELECT item_id, selected_by, sel_score, sel_meta FROM assignment WHERE annotator_id=%s", (a,))
        rows = [dict(r) for r in cur.fetchall()]
    got = {r["item_id"] for r in rows}
    assert got <= set(rare) and len(got) == 10                      # the saturated class is not served while the other has a deficit
    assert all(r["selected_by"] == "steered" and r["sel_score"] is not None and r["sel_meta"]["prop_labels"] == ["Doubt"] for r in rows)


def test_random_slice_is_unsteered_and_flagged(schema, monkeypatch):
    quiet()
    sid = mk_source("randslice")
    with db.tx() as cur:
        for k in range(60):
            i = mk_item(sid, f"random slice item {k} with enough words to be a real item here")
            cur.execute("INSERT INTO suggestion (item_id, layer, model_version, start_char, end_char, span_text, label) "
                        "VALUES (%s,'propaganda','enc_v1',0,5,'rando','Loaded_Language')", (i,))
            cur.execute("INSERT INTO suggestion_done (item_id, model_version) VALUES (%s,'enc_v1')", (i,))
    real_c = batching.C
    monkeypatch.setattr(batching, "C", lambda k: 0.25 if k == "ann_random_frac" else real_c(k))
    a = ann("rand_user", 20)
    S.next_item(a)
    with db.tx() as cur:
        cur.execute("SELECT selected_by, count(*) n FROM assignment WHERE annotator_id=%s GROUP BY 1", (a,))
        by = {r["selected_by"]: r["n"] for r in cur.fetchall()}
        cur.execute("SELECT meta FROM batch WHERE annotator_id=%s", (a,))
        meta = cur.fetchone()["meta"]
    assert by == {"random": 5, "steered": 15} and meta["n_random"] == 5


def test_class_targets_are_capped_at_what_the_pool_can_supply(schema, monkeypatch):
    quiet()
    sid = mk_source("supply")
    with db.tx() as cur:
        for k in range(40):
            i = mk_item(sid, f"supply item {k} with enough words to be a real item here")
            cur.execute("INSERT INTO suggestion_done (item_id, model_version) VALUES (%s,'enc_v1')", (i,))
            cur.execute("INSERT INTO suggestion_done (item_id, model_version) VALUES (%s,'gemma_claims_v1')", (i,))
            if k < 10:                                              # 10 of 40 items carry the (rare) technique
                cur.execute("INSERT INTO suggestion (item_id, layer, model_version, start_char, end_char, span_text, label) "
                            "VALUES (%s,'propaganda','enc_v1',0,5,'suppl','Slogans')", (i,))
    with db.tx() as cur:
        t = {"items": 1000, "neg": 0, "claim": {"ALLEGATION": 500}, "propaganda": {"Slogans": 500, "Doubt": 500}}
        cc = {"claim": {}, "propaganda": {"Slogans": 3}, "none": {"claim": 0, "propaganda": 0}, "annotated": {"claim": 0, "propaganda": 0}}
        ft = metrics.feasible_targets(cur, t, cc)
    assert ft["propaganda"]["Slogans"] == 3 + int(0.5 * 10)          # finished + frac x predicted supply (full model coverage here)
    assert ft["propaganda"]["Doubt"] == 0 and ft["claim"]["ALLEGATION"] == 0      # never predicted: no steering towards it


def test_release_unstarted_returns_items_to_the_pool(schema):
    quiet()
    sid = mk_source("relsrc")
    ids = [mk_item(sid, f"release item {k} with enough words to be a real item here") for k in range(6)]
    a = ann("rel_user", 6)
    first = S.next_item(a)
    S.add_claim(first, a, 0, 7, "OTHER", "no")                      # started: must stay with the annotator
    n = batching.release_unstarted(a)
    assert n == 5
    b = ann("rel_user2", 6)
    got = S.next_item(b)
    assert got in ids and got != first


# ------------------------------------------------------------------ auth + monitoring
def test_password_login_lockout_sessions_and_password_change(schema):
    from fnp.annotate import auth
    r = auth.create_user("login_user", "correct horse 7", must_change=False)
    assert r["password"] is None and r["annotator"]["username"] == "login_user"
    with pytest.raises(auth.AuthError):
        auth.create_user("login_user", "another pass 99")           # duplicate
    with pytest.raises(auth.AuthError):
        auth.create_user("weak_user", "short")                       # policy
    assert "error" in auth.login("login_user", "nope nope nope", "1.2.3.4")
    assert "error" in auth.login("nobody_here", "x" * 10, "1.2.3.4")
    ok = auth.login("LOGIN_USER", "correct horse 7", "1.2.3.4", "pytest")   # usernames are case-insensitive
    me = auth.resolve(ok["token"])
    assert me["username"] == "login_user" and me["session_id"] == ok["session_id"]
    with db.tx() as cur:
        cur.execute("SELECT token_hash FROM auth_session WHERE session_id=%s", (ok["session_id"],))
        assert cur.fetchone()["token_hash"] != ok["token"]            # only a hash is stored
        cur.execute("SELECT password_hash FROM annotator WHERE username='login_user'")
        assert "correct horse" not in cur.fetchone()["password_hash"]
    auth.logout(ok["token"])
    with pytest.raises(auth.Unauthorized):
        auth.resolve(ok["token"])
    for _ in range(int(auth.C("ann_max_failed_logins"))):
        auth.login("login_user", "wrong wrong 1")
    assert "locked" in auth.login("login_user", "correct horse 7")["error"]      # even the right password is refused while locked
    aid = me["annotator_id"]
    with db.tx() as cur:
        cur.execute("UPDATE annotator SET locked_until=NULL WHERE annotator_id=%s", (aid,))
    ok2 = auth.login("login_user", "correct horse 7")
    s2 = auth.resolve(ok2["token"])
    with pytest.raises(auth.AuthError):
        auth.change_password(aid, "bad old", "new password 5")
    auth.change_password(aid, "correct horse 7", "new password 5", keep_session=s2["session_id"])
    assert auth.resolve(ok2["token"])["annotator_id"] == aid                      # current session survives, others end
    assert "token" in auth.login("login_user", "new password 5")
    auth.set_active(aid, False)
    with pytest.raises(auth.Unauthorized):
        auth.resolve(ok2["token"])
    assert "disabled" in auth.login("login_user", "new password 5")["error"]
    with db.tx() as cur:
        cur.execute("SELECT outcome FROM login_event WHERE username ILIKE 'login_user'")
        outs = {r["outcome"] for r in cur.fetchall()}
    assert {"ok", "bad_password", "locked", "logout", "password_changed", "disabled", "user_created"} <= outs


def test_must_change_password_flag_and_generated_password(schema):
    from fnp.annotate import auth
    r = auth.create_user("gen_user")
    assert len(r["password"]) == 12 and r["annotator"]["must_change_password"] is True
    assert "token" in auth.login("gen_user", r["password"])
    new = auth.set_password(r["annotator"]["annotator_id"])
    assert new and "error" in auth.login("gen_user", r["password"]) and "token" in auth.login("gen_user", new)


def test_work_minutes_count_once_per_minute_and_feed_monitoring(schema):
    from fnp.annotate import auth
    quiet()
    sid = mk_source("mon")
    for k in range(4):
        mk_item(sid, f"monitoring item {k} with enough words to be a real item here for the test")
    u = auth.create_user("mon_user", "monitor pass 1", must_change=False)["annotator"]["annotator_id"]
    tok = auth.login("mon_user", "monitor pass 1", "9.9.9.9")
    me = auth.resolve(tok["token"])
    for _ in range(5):
        auth.heartbeat(me, True)                                   # same minute, five heartbeats, two sessions: one minute
    auth.heartbeat(me, False)                                      # idle heartbeat: nothing
    tok2 = auth.login("mon_user", "monitor pass 1"); auth.heartbeat(auth.resolve(tok2["token"]), True)
    assert auth.worked_minutes(u)["total"] == 1
    with db.tx() as cur:                                           # pretend an earlier active minute and a longer session
        cur.execute("INSERT INTO activity_minute (annotator_id, minute, session_id) VALUES (%s, date_trunc('minute', now()) - interval '10 minutes', %s)",
                    (u, me["session_id"]))
        cur.execute("UPDATE auth_session SET created_at = now() - interval '30 minutes' WHERE session_id=%s", (me["session_id"],))
    i = S.next_item(u)
    S.add_claim(i, u, 0, 7, "OTHER", "no")
    S.submit_layer(i, u, "claim", "annotated"); S.submit_layer(i, u, "propaganda", "none")
    t = metrics.team(30)
    row = next(r for r in t["annotators"] if r["username"] == "mon_user")
    assert row["active_minutes"] == 2 and row["items_done"] == 1 and row["claim_spans"] == 1 and row["none_labels"] == 1
    assert row["logins"] == 2 and row["logged_in_seconds"] >= 29 * 60 and row["online"] is True
    assert row["batch"]["seq"] == 1 and row["batch"]["finished"] == 1
    d = metrics.user_detail(u, 30)
    assert d["sessions"][-1]["active_minutes"] == 2 and d["daily"][0]["items_done"] == 1
    assert any(c["code"] == "OTHER" and c["n"] == 1 for c in d["claim_types"])


def test_coverage_report_counts_finished_items_per_class_and_balance(schema):
    with db.tx() as cur:
        for tbl in ("layer_result", "propaganda_span_technique", "claim_span", "propaganda_span"):
            cur.execute(f"DELETE FROM {tbl}")
        cur.execute("DELETE FROM assignment")
    quiet()
    sid = mk_source("covsrc")
    t = "Officials said the bridge opens in March and critics call this complete nonsense and a disgrace"
    i1, i2, i3 = (mk_item(sid, t + f" v{k}") for k in range(3))
    a, b = ann("cov_a"), ann("cov_b")
    for i in (i1, i2):
        assign(i, a)
    _label(i1, a, claims=[(0, 14, "REPORTED_OR_REQUOTED_STATEMENT", "yes")], props=[(60, 77, ["Loaded_Language", "Doubt"])])
    _label(i2, a)                                                   # explicit negatives on both layers
    assign(i3, a); assign(i3, b)
    cov = metrics.coverage()
    claim = {r["code"]: r for r in cov["tasks"]["claim"]["classes"]}
    prop = {r["code"]: r for r in cov["tasks"]["propaganda"]["classes"]}
    assert claim["REPORTED_OR_REQUOTED_STATEMENT"]["done"] == 1 and claim["OTHER"]["target"] is None
    assert prop["Loaded_Language"]["done"] == 1 and prop["Doubt"]["done"] == 1 and prop["Slogans"]["done"] == 0
    assert cov["tasks"]["claim"]["none_items"] == 1 and cov["tasks"]["propaganda"]["none_items"] == 1
    assert cov["target_items"] == 50000 and cov["done_items"] >= 2
    assert any(p["platform"] == "covsrc" and p["done"] == 2 for p in cov["by_platform"])


def _label(i, a, claims=(), props=()):
    assign(i, a)
    for c in claims:
        S.add_claim(i, a, c[0], c[1], c[2], c[3])
    for p in props:
        S.add_propaganda(i, a, p[0], p[1], p[2])
    S.submit_layer(i, a, "claim", "annotated" if claims else "none")
    S.submit_layer(i, a, "propaganda", "annotated" if props else "none")


def test_export_end_to_end_agreement_disputes_splits_and_versioning(schema, tmp_path):
    with db.tx() as cur:                                     # labels left by earlier tests must not leak into this export
        for t in ("layer_result", "claim_span", "propaganda_span"):
            cur.execute(f"DELETE FROM {t}")
    sid = mk_source("exp")
    t = "Officials said the new bridge will open in March 🙂 and critics call this complete nonsense, a total disgrace to the city"
    agree_i = mk_item(sid, t + " A", double=True)
    dispute_i = mk_item(sid, t + " B", double=True)
    neg_i = mk_item(sid, "Nice weather today in the capital and nothing else to report here at all", double=False)
    waiting_i = mk_item(sid, t + " C", double=True)          # second annotator never came: excluded
    a, b = ann("exp_a"), ann("exp_b")
    s = t.index("Officials")
    e = t.index("March") + 5
    p0 = t.index("complete nonsense")
    _label(agree_i, a, claims=[(s, e, "REPORTED_OR_REQUOTED_STATEMENT", "yes")], props=[(p0, p0 + 17, ["Loaded_Language"])])
    _label(agree_i, b, claims=[(s, e - 2, "REPORTED_OR_REQUOTED_STATEMENT", "yes")], props=[(p0, p0 + 17, ["Loaded_Language"])])
    _label(dispute_i, a, claims=[(s, e, "DIRECT_STATEMENT", "yes")])
    _label(dispute_i, b)
    _label(neg_i, a)
    _label(waiting_i, a, claims=[(s, e, "DIRECT_STATEMENT", "yes")])

    r = export.build("claims", "claims_t1", apply=True, out_root=tmp_path, created_by="test")
    assert r["n_items"] == 2 and r["n_disputed"] == 1 and r["by_status"] == {"positive": 1, "negative": 1}
    assert r["agreement"]["n_double_items"] == 2
    out = tmp_path / "claims_t1"
    canon = [json.loads(l) for l in (out / "canonical.jsonl").read_text().splitlines()]
    pos = next(x for x in canon if x["status"] == "positive")
    assert pos["spans"][0]["annotators"] == ["exp_a", "exp_b"] and pos["text"][pos["spans"][0]["start"]:pos["spans"][0]["end"]] == pos["spans"][0]["text"]
    assert [json.loads(l)["item_id"] for l in (out / "disputed.jsonl").read_text().splitlines()] == [dispute_i]
    sft = [json.loads(l) for sp in ("train", "dev", "test") for l in (out / f"claims_sft_{sp}.jsonl").read_text().splitlines()]
    assert len(sft) == 2
    row = next(x for x in sft if x["extra_info"]["n_claims"] == 1)
    assert row["prompt"][0]["role"] == "system" and row["prompt"][1]["content"].startswith("Document:\n")
    gold = json.loads(row["reward_model"]["ground_truth"])["claims"]
    assert gold[0]["claim"] in row["prompt"][1]["content"]               # verbatim span, like claimExtraction/claims-data
    neg = next(x for x in sft if x["extra_info"]["n_claims"] == 0)
    assert json.loads(neg["reward_model"]["ground_truth"]) == {"claims": []}
    man = json.loads((out / "manifest.json").read_text())
    assert man["sha256"]["canonical.jsonl"] == hashlib.sha256((out / "canonical.jsonl").read_bytes()).hexdigest()
    lm = json.loads((out / "label_map.json").read_text())
    assert len(lm["claim_types"]) == 9 and len(lm["techniques_14"]) == 14
    with db.tx() as cur:
        cur.execute("SELECT count(*) n FROM export_split JOIN export_run USING (export_id) WHERE version='claims_t1'")
        assert cur.fetchone()["n"] == 2
    with pytest.raises(FileExistsError):
        export.build("claims", "claims_t1", apply=True, out_root=tmp_path)   # versions are never overwritten

    rp = export.build("propaganda", "spans_t1", apply=True, out_root=tmp_path)
    assert rp["by_technique"] == {"Loaded_Language": 1} and rp["agreement"]["semeval_style_A_vs_B"]["TC_span_f1"] == pytest.approx(1.0)
    arts, spans = export.load_export_spans(next(p for p in (tmp_path / "spans_t1").glob("spans_ptc_style_*.jsonl") if json.loads((p.read_text().splitlines() or ["{}"])[0] or "{}").get("spans")))
    assert list(spans.columns) == ["article_id", "split", "start", "end", "technique"] and len(spans) == 1


# ---------------------------------------------------------------- skips, duplicates, adjudication, gold, export provenance
def test_item_is_excluded_after_n_content_skips_but_not_after_language_skips(schema):
    from fnp.annotate import service
    quiet()
    sid = mk_source("skipx")
    spam, lang = mk_item(sid, "spam item that nobody wants to label at all"), mk_item(sid, "item only some annotators can read well enough")
    a, b, c = ann("skip_a"), ann("skip_b"), ann("skip_c")
    for i in (spam, lang):
        assign(i, a); assign(i, b)
    with db.tx() as cur:                                            # a third annotator holds an unstarted copy of the spam item
        cur.execute("INSERT INTO assignment (item_id, annotator_id, status) VALUES (%s,%s,'assigned')", (spam, c))
    S.skip_item(spam, a, "Spam or non-content")
    assert _status(spam) != "excluded"                              # one skip is not enough
    S.skip_item(spam, b, "Other: just emojis")
    assert _status(spam) == "excluded" and _reason(spam) == "skipped_by_2_annotators"
    with db.tx() as cur:
        cur.execute("SELECT count(*) n FROM assignment WHERE item_id=%s AND annotator_id=%s", (spam, c))
        assert cur.fetchone()["n"] == 0                             # unstarted copy released
    S.skip_item(lang, a, "Language I cannot annotate"); S.skip_item(lang, b, "Language I cannot annotate")
    assert _status(lang) != "excluded"                              # annotator-specific reason never excludes the item


def _status(i):
    with db.tx() as cur:
        cur.execute("SELECT status FROM item WHERE item_id=%s", (i,)); return cur.fetchone()["status"]


def _reason(i):
    with db.tx() as cur:
        cur.execute("SELECT excluded_reason FROM item WHERE item_id=%s", (i,)); return cur.fetchone()["excluded_reason"]


def test_cleanup_keeps_one_per_duplicate_group_and_protects_assigned_items(schema):
    from fnp.annotate import cleanup
    quiet()
    sid = mk_source("dedupe")
    dup = [mk_item(sid, f"the very same repost body with a different tail number {k} and more words") for k in range(3)]
    link = mk_item(sid, "https://example.org/only-a-link")
    solo = mk_item(sid, "an item with a body that nobody else shares in this pool")
    with db.tx() as cur:
        cur.execute("UPDATE item SET body_key=%s WHERE item_id = ANY(%s)", ("k" * 40, dup))
        cur.execute("UPDATE item SET body_key=%s WHERE item_id=%s", ("l" * 40, link))
        cur.execute("UPDATE item SET body_key=%s WHERE item_id=%s", ("s" * 40, solo))
    a = ann("dedupe_user"); assign(dup[1], a)                        # the used copy must be the one kept
    p = cleanup.plan()
    assert sorted(d for d, _ in p["duplicates"]) == sorted([dup[0], dup[2]]) and {k for _, k in p["duplicates"]} == {dup[1]}
    assert link in p["low_substance"] and solo not in p["low_substance"]
    cleanup.apply(p)
    assert [_status(i) for i in dup] == ["excluded", "queued", "excluded"] or _status(dup[1]) != "excluded"
    assert _status(dup[1]) != "excluded" and _status(link) == "excluded" and _reason(link) == "low_substance"
    assert _reason(dup[0]) == f"near_duplicate_of:{dup[1]}" and _status(solo) == "queued"


def _lab(item, annotator, claim_span=None, techniques_span=None):
    """Annotate both layers as the given annotator and submit."""
    if claim_span:
        S.add_claim(item, annotator, claim_span[0], claim_span[1], "DIRECT_STATEMENT", "yes"); S.submit_layer(item, annotator, "claim", "annotated")
    else:
        S.submit_layer(item, annotator, "claim", "none")
    if techniques_span:
        S.add_propaganda(item, annotator, techniques_span[0], techniques_span[1], ["Loaded_Language"]); S.submit_layer(item, annotator, "propaganda", "annotated")
    else:
        S.submit_layer(item, annotator, "propaganda", "none")


def test_adjudication_resolves_a_dispute_and_gold_calibrates_without_entering_exports(schema, monkeypatch):
    from fnp.annotate import adjudicate, quality
    with db.tx() as cur:
        for tbl in ("layer_result", "propaganda_span_technique", "claim_span", "propaganda_span"):
            cur.execute(f"DELETE FROM {tbl}")
        cur.execute("DELETE FROM assignment")
    quiet()
    sid = mk_source("adj")
    text = "The minister said inflation will fall by five percent next year, officials confirmed."
    item = mk_item(sid, text, double=True, blind=True)
    a, b = ann("adj_a"), ann("adj_b")
    j = ann("adj_judge")
    with db.tx() as cur:
        cur.execute("UPDATE annotator SET role='adjudicator' WHERE annotator_id=%s", (j,))
    assign(item, a); assign(item, b)
    _lab(item, a, claim_span=(0, len(text)))
    _lab(item, b)                                                  # B sees no claim: a dispute in the claim layer
    res = export.build("claims", "adj_v0", "agree", False)
    assert res["n_disputed"] == 1 and res["n_items"] == 0
    d = adjudicate.disputes()
    assert d["counts"]["pending"] == 1 and d["items"][0]["item_id"] == item
    with pytest.raises(S.AnnotError):                                # an annotator cannot adjudicate
        adjudicate.start(item, a)
    adjudicate.start(item, j)
    got = S.get_item(item, j)
    assert {p["username"] for p in got["adjudication"]["annotators"]} == {"adj_a", "adj_b"} and got["assignment"].get("kind") is None
    with pytest.raises(S.AnnotError):                                # annotators never see each other's labels
        S.get_item(item, ann("adj_outsider"))
    adjudicate.copy_from(item, j, a, "claim")
    S.submit_layer(item, j, "claim", "annotated"); S.submit_layer(item, j, "propaganda", "none")
    res = export.build("claims", "adj_v1", "agree", False)
    assert res["n_items"] == 1 and res["n_disputed"] == 0 and res["n_adjudicated"] == 1
    assert adjudicate.disputes()["counts"]["adjudicated"] == 1

    adjudicate.mark_gold(item, j)
    real_c = batching.C
    monkeypatch.setattr(batching, "C", lambda k: 0.2 if k == "ann_gold_frac" else real_c(k))
    quiet_items = mk_source("adj_pool")
    for k in range(12):
        mk_item(quiet_items, f"ordinary pool item {k} with enough words to be a real item here")
    c = ann("adj_c", 10)
    S.next_item(c)
    with db.tx() as cur:
        cur.execute("SELECT kind, selected_by FROM assignment WHERE annotator_id=%s AND item_id=%s", (c, item))
        g = cur.fetchone()
        cur.execute("SELECT count(*) n FROM assignment WHERE annotator_id=%s AND kind='normal'", (c,))
        n_normal = cur.fetchone()["n"]
    assert g["kind"] == "gold" and g["selected_by"] == "gold" and n_normal == 9      # 10 slots: 1 gold + 9 ordinary
    seen = S.get_item(item, c)
    assert seen["item"]["is_blind"] is True and seen["suggestions"] == [] and "adjudication" not in seen and "kind" not in seen["assignment"]
    _lab(item, c, claim_span=(0, len(text)))                       # C reproduces the reference exactly
    gs = quality.gold_scores()[c]
    assert gs["n_items"] == 1 and gs["claim"]["f1"] == 1.0 and gs["claim"]["claim_type_accuracy"] == 1.0
    res = export.build("claims", "adj_v2", "agree", False)           # C's gold label changed nothing
    assert res["n_items"] == 1 and res["n_adjudicated"] == 1 and res["agreement"]["n_double_items"] == 1
    row = next(r for r in quality.team_quality()["rows"] if r["username"] == "adj_c")
    assert row["gold"]["claim"]["f1"] == 1.0


def test_export_records_selection_blindness_and_class_weights(schema):
    with db.tx() as cur:
        for tbl in ("layer_result", "propaganda_span_technique", "claim_span", "propaganda_span"):
            cur.execute(f"DELETE FROM {tbl}")
        cur.execute("DELETE FROM assignment")
    quiet()
    sid = mk_source("expw")
    ann_id = ann("expw_user")
    texts = {}
    with db.tx() as cur:
        for k in range(6):
            blind = k % 2 == 0
            i = mk_item(sid, f"export weight item number {k} with enough words to be annotated here", blind=blind)
            texts[i] = blind
            cur.execute("INSERT INTO assignment (item_id, annotator_id, status, selected_by) VALUES (%s,%s,'in_progress',%s)",
                        (i, ann_id, "random" if k < 2 else "steered"))
    for i, blind in texts.items():
        if i % 3 == 0:
            _lab(i, ann_id, claim_span=(0, 10))
        else:
            _lab(i, ann_id)
    res = export.build("claims", "expw_v1", "agree", False)
    assert res["n_items"] == 6 and res["n_blind"] == 3 and res["by_selection"] == {"random": 2, "steered": 4}
    assert "weights" in res["class_weights"] and res["assistance_effect"]["blind"]["items"] == 3
    only = export.build("claims", "expw_v2", "agree", False, subset="random")
    assert only["n_items"] == 2 and only["by_selection"] == {"random": 2}
