import threading

import pytest

from fnp import gemma
from fnp.llm import cache as C


@pytest.fixture
def cache(tmp_path):
    c = C.GemmaCache(tmp_path / "t.duckdb")
    yield c
    c.close()


def fake_factory(responses, calls):
    def fake(system, user, *, max_tokens=1200, validate=None, retries=1):
        calls.append(user)
        r = responses.pop(0) if isinstance(responses, list) else responses
        if r is None:
            return {"ok": False, "obj": None, "error": "net_x", "tokens": 5, "latency_s": 0.1, "attempts": 2}
        bad = validate(r) if validate else None
        if bad:
            return {"ok": False, "obj": None, "error": "invalid:" + bad, "tokens": 5, "latency_s": 0.1, "attempts": 2}
        return {"ok": True, "obj": r, "error": None, "tokens": 10, "latency_s": 0.2, "attempts": 1}
    return fake


def test_cache_hit_avoids_second_call(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(gemma, "chat_json", fake_factory({"a": 1}, calls))
    r1 = C.cached_call("t", "v1", "sys", "usr", cache=cache)
    r2 = C.cached_call("t", "v1", "sys", "usr", cache=cache)
    assert len(calls) == 1
    assert (r1["cached"], r2["cached"]) == (False, True)
    assert r2["obj"] == {"a": 1} and r2["ok"] and r2["tokens"] == 10


def test_version_and_input_change_key(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(gemma, "chat_json", fake_factory({"a": 1}, calls))
    C.cached_call("t", "v1", "s", "u", cache=cache)
    C.cached_call("t", "v2", "s", "u", cache=cache)
    C.cached_call("t", "v1", "s", "u2", cache=cache)
    assert len(calls) == 3


def test_failure_cached_but_retried(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(gemma, "chat_json", fake_factory([None, {"ok": 1}], calls))
    r1 = C.cached_call("t", "v", "s", "u", cache=cache)
    assert not r1["ok"]
    row = cache.get(r1["key"])
    assert row is not None and row["ok"] is False and row["error"] == "net_x"
    # retry_failed=False serves the cached failure without a call
    r2 = C.cached_call("t", "v", "s", "u", retry_failed=False, cache=cache)
    assert not r2["ok"] and r2["cached"] and len(calls) == 1
    # retry_failed=True re-asks and overwrites with the success
    r3 = C.cached_call("t", "v", "s", "u", retry_failed=True, cache=cache)
    assert r3["ok"] and not r3["cached"] and len(calls) == 2
    assert cache.get(r1["key"])["ok"] is True


def test_stale_cached_object_revalidated(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(gemma, "chat_json", fake_factory([{"a": 1}, {"a": 1, "b": 2}], calls))
    C.cached_call("t", "v", "s", "u", cache=cache)
    strict = lambda o: None if "b" in o else "need_b"
    r = C.cached_call("t", "v", "s", "u", validate=strict, cache=cache)
    assert len(calls) == 2 and r["obj"]["b"] == 2


def test_thread_safety_and_batch_insert(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(gemma, "chat_json", fake_factory({"a": 1}, calls))
    errs = []

    def work(i):
        try:
            for j in range(10):
                C.cached_call("t", "v", "s", f"u{i}-{j}", cache=cache)
        except Exception as e:  # pragma: no cover
            errs.append(e)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(calls) == 80
    cache.put_many([{"key": f"k{i}", "task": "x", "prompt_version": "v", "input_sha": "s",
                     "obj": {"i": i}, "ok": True} for i in range(50)])
    assert cache.get("k49")["obj"] == {"i": 49}
    assert dict((r[0], r[2]) for r in cache.stats())["x"] == 50


def test_default_cache_follows_config_path(tmp_path, monkeypatch):
    from fnp import config
    monkeypatch.setattr(config, "DUCKDB_PATH", tmp_path / "d.duckdb")
    C.reset_default_cache()
    try:
        c = C.get_cache()
        assert c.path == tmp_path / "d.duckdb"
    finally:
        C.reset_default_cache()
