"""Custom VERL reward for free-form IE JSON (entities + events).

Signature required by VERL:
  compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float | dict
"""

from __future__ import annotations

import json
import re
from typing import Any

NOISE_TYPES = {"HDL", "PR", "MEN", "CLM", "HTG"}

WEIGHTS = {
    "format": 0.15,
    "entity": 0.40,
    "subtype": 0.10,
    "topic": 0.25,
    "action": 0.10,
}


def _repair_truncated_json(s: str) -> str | None:
    """Salvage the longest valid prefix of a JSON object cut off mid-generation.

    The model is not constrained to valid JSON at decode time, and a long
    entity list frequently hits the response-token budget mid-object. Without
    this, such a response scores identically to complete garbage (0.0), so RL
    gets no gradient distinguishing "nearly right but cut off" from "nonsense".

    Walks the text tracking string/escape state and the open-bracket stack,
    rewinds to the last cleanly-closed nested element, then closes the still
    -open containers. Returns None when nothing salvageable precedes the cut.
    """
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]

    stack: list[str] = []
    in_str = esc = False
    cut: int | None = None  # exclusive index of last cleanly-closed nested element

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if stack:
                cut = i + 1
            else:
                return s[: i + 1]  # root closed: not actually truncated

    if cut is None:
        return None

    head = s[:cut].rstrip().rstrip(",")
    # Recompute which containers remain open across the retained prefix.
    reopen: list[str] = []
    in_str = esc = False
    for ch in head:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            reopen.append("}" if ch == "{" else "]")
        elif ch in "}]" and reopen:
            reopen.pop()
    return head + "".join(reversed(reopen))


def _extract_json_with_status(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Return (parsed_object, was_repaired). `was_repaired` marks output that
    was not valid JSON as emitted and only parsed after truncation repair."""
    if not text or not str(text).strip():
        return None, False
    s = str(text).strip()
    # strip markdown fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.I)
    if fence:
        s = fence.group(1).strip()
    # direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, False
    except json.JSONDecodeError:
        pass
    # outermost {...}
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict):
                return obj, False
        except json.JSONDecodeError:
            pass
    # last resort: repair a response truncated mid-object
    repaired = _repair_truncated_json(s)
    if repaired:
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj, True
        except json.JSONDecodeError:
            pass
    return None, False


def _extract_json(text: str) -> dict[str, Any] | None:
    return _extract_json_with_status(text)[0]


def _norm_text(t: Any) -> str:
    return re.sub(r"\s+", " ", str(t or "").strip().lower())


def _as_list(v: Any) -> list:
    """Coerce a generated field to a list without ever raising.

    The model is free-form: nothing stops it emitting `"actions": true`,
    `"entities": {...}`, or `"topics": 5` where a list belongs. The previous
    `x or []` idiom only guarded *falsy* values, so a truthy non-iterable
    (`True`, `5`) reached the `for` loop and raised `TypeError: 'bool' object is
    not iterable` — which propagated out of the reward worker and killed an
    entire training run on one malformed rollout out of ~480.

    A single scalar or dict is treated as a one-element list (the charitable
    reading of "wrote one item instead of a list of one"); anything else scores
    as empty. Format credit is unaffected — `score_prediction` still requires a
    real `list` for that, so proper output stays strictly better.
    """
    if v is None or isinstance(v, bool):
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, (str, int, float, dict)):
        return [v]
    return []


def _entity_keys(entities: list) -> set[tuple[str, str]]:
    keys = set()
    for e in _as_list(entities):
        if not isinstance(e, dict):
            continue
        et = str(e.get("type") or "").upper().strip()
        if not et or et in NOISE_TYPES:
            continue
        text = _norm_text(e.get("text") or e.get("normalized"))
        if not text:
            continue
        keys.add((text, et))
    return keys


def _subtype_map(entities: list) -> dict[tuple[str, str], str]:
    m = {}
    for e in _as_list(entities):
        if not isinstance(e, dict):
            continue
        et = str(e.get("type") or "").upper().strip()
        text = _norm_text(e.get("text") or e.get("normalized"))
        if not text or not et or et in NOISE_TYPES:
            continue
        st = str(e.get("subtype") or "").strip()
        if st:
            m[(text, et)] = st.lower()
    return m


def _topic_codes(events: list) -> set[str]:
    codes = set()
    for ev in _as_list(events):
        if not isinstance(ev, dict):
            continue
        for t in _as_list(ev.get("topics")):
            if isinstance(t, dict):
                c = str(t.get("code") or "").strip()
            else:
                c = str(t).strip()
            if re.match(r"^\d+\.\d+$", c):
                codes.add(c)
    return codes


def _action_codes(events: list) -> set[str]:
    codes = set()
    for ev in _as_list(events):
        if not isinstance(ev, dict):
            continue
        for a in _as_list(ev.get("actions")):
            if isinstance(a, dict):
                c = str(a.get("code") or "").strip()
            else:
                c = str(a).strip()
            if re.match(r"^\d{1,2}$", c) and 13 <= int(c) <= 20:
                codes.add(c)
    return codes


def _f1(pred: set, gold: set) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r)


def _parse_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            obj = _extract_json(ground_truth)
            return obj or {}
    return {}


def score_prediction(
    pred_obj: dict[str, Any] | None,
    gold_obj: dict[str, Any],
    repaired: bool = False,
) -> dict[str, float]:
    if pred_obj is None:
        return {
            "score": 0.0,
            "format": 0.0,
            "entity_f1": 0.0,
            "subtype": 0.0,
            "topic_f1": 0.0,
            "action_f1": 0.0,
            "repaired": 0.0,
        }

    has_ent = "entities" in pred_obj and isinstance(pred_obj.get("entities"), list)
    has_ev = "events" in pred_obj and isinstance(pred_obj.get("events"), list)
    fmt = 1.0 if (has_ent and has_ev) else (0.5 if (has_ent or has_ev) else 0.0)
    # Output that only parsed after truncation repair was not valid JSON as
    # emitted, so it must not earn full format credit — otherwise truncating
    # early would be free. Content sub-scores below are still awarded, and the
    # missing entities/events cost recall in the F1 terms on their own.
    if repaired:
        fmt = min(fmt, 0.5)

    pred_ent = _entity_keys(pred_obj.get("entities"))
    gold_ent = _entity_keys(gold_obj.get("entities"))
    ent_f1 = _f1(pred_ent, gold_ent)

    # subtype among intersection
    pred_st = _subtype_map(pred_obj.get("entities"))
    gold_st = _subtype_map(gold_obj.get("entities"))
    inter = set(pred_st) & set(gold_st) & pred_ent & gold_ent
    if inter:
        subtype = sum(1 for k in inter if pred_st.get(k) == gold_st.get(k)) / len(inter)
    elif not gold_st:
        subtype = 1.0 if fmt > 0 else 0.0
    else:
        subtype = 0.0

    topic_f1 = _f1(
        _topic_codes(pred_obj.get("events")),
        _topic_codes(gold_obj.get("events")),
    )
    action_f1 = _f1(
        _action_codes(pred_obj.get("events")),
        _action_codes(gold_obj.get("events")),
    )

    score = (
        WEIGHTS["format"] * fmt
        + WEIGHTS["entity"] * ent_f1
        + WEIGHTS["subtype"] * subtype
        + WEIGHTS["topic"] * topic_f1
        + WEIGHTS["action"] * action_f1
    )
    return {
        "score": float(score),
        "format": float(fmt),
        "entity_f1": float(ent_f1),
        "subtype": float(subtype),
        "topic_f1": float(topic_f1),
        "action_f1": float(action_f1),
        "repaired": 1.0 if repaired else 0.0,
    }


ZERO_METRICS = {
    "score": 0.0,
    "format": 0.0,
    "entity_f1": 0.0,
    "subtype": 0.0,
    "topic_f1": 0.0,
    "action_f1": 0.0,
    "repaired": 0.0,
}


def _score_safe(solution_str: Any, ground_truth: Any) -> dict[str, float]:
    """Score without ever propagating an exception.

    This function runs inside VERL's reward worker on model-generated text. An
    exception here does not degrade one sample's reward — it tears down the Ray
    worker and kills the entire training run (that is exactly how a 38-step run
    died at step 15 on a single `"actions": true`). The type coercion in
    `_as_list` fixes the known shapes; this guard bounds the damage from any
    shape nobody has thought of yet, since the space of strings a language model
    can emit is not enumerable in advance.
    """
    try:
        gold = _parse_ground_truth(ground_truth)
        pred, repaired = _extract_json_with_status(solution_str)
        return score_prediction(pred, gold, repaired=repaired)
    except Exception as e:  # noqa: BLE001 — deliberately total
        preview = str(solution_str)[:200].replace("\n", "\\n")
        print(f"WARN reward_ie: scoring failed ({type(e).__name__}: {e}); returning 0.0. Output: {preview!r}")
        return dict(ZERO_METRICS)


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """VERL entrypoint — returns scalar reward in ~[0, 1]."""
    return float(_score_safe(solution_str, ground_truth)["score"])


def compute_score_detailed(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, float]:
    return _score_safe(solution_str, ground_truth)


if __name__ == "__main__":
    gold = {
        "entities": [
            {"text": "Donald Trump", "type": "PERSON", "subtype": "head_of_government"},
            {"text": "Iran", "type": "LOCATION", "subtype": "country"},
        ],
        "events": [
            {
                "actor1": "the U.S.",
                "actor2": "Iran",
                "topics": [{"code": "06.1", "name": "Military"}],
                "actions": [{"code": "19", "name": "Fight"}],
                "evidence_text": "strikes",
            }
        ],
    }
    perfect = json.dumps(gold)
    m = compute_score_detailed(solution_str=perfect, ground_truth=json.dumps(gold))
    print("gold-self:", m)
    assert m["score"] >= 0.9, m

    bad = "not json at all"
    m2 = compute_score_detailed(solution_str=bad, ground_truth=json.dumps(gold))
    print("garbage:", m2)
    assert m2["score"] < 0.2, m2

    partial = json.dumps(
        {
            "entities": [{"text": "Iran", "type": "LOCATION"}],
            "events": [{"topics": [{"code": "06.1"}], "actions": []}],
        }
    )
    m3 = compute_score_detailed(solution_str=partial, ground_truth=json.dumps(gold))
    print("partial:", m3)
    assert 0.2 < m3["score"] < 0.95

    # Truncated mid-object: the dominant real failure mode. Must earn partial
    # credit for the entities it did emit, strictly above garbage and strictly
    # below the same content emitted as valid JSON.
    truncated = (
        '{\n "entities": [\n'
        '  {"text": "Donald Trump", "type": "PERSON", "subtype": "head_of_government"},\n'
        '  {"text": "Iran", "type": "LOCATION", "subtype": "country"},\n'
        '  {"text": "Tehr'
    )
    m4 = compute_score_detailed(solution_str=truncated, ground_truth=json.dumps(gold))
    print("truncated:", m4)
    assert m4["repaired"] == 1.0, m4
    assert m4["score"] > m2["score"], (m4, m2)  # beats garbage
    assert m4["entity_f1"] > 0.9, m4  # both gold entities recovered
    assert m4["format"] == 0.5, m4  # but not full format credit
    assert m4["score"] < m["score"], (m4, m)  # still below valid-JSON gold-self

    # A response with no salvageable prefix must stay at zero.
    m5 = compute_score_detailed(solution_str='{"entities": [{"te', ground_truth=json.dumps(gold))
    print("unsalvageable:", m5)
    assert m5["score"] == 0.0, m5

    # Repair must not corrupt already-valid JSON.
    assert compute_score_detailed(
        solution_str=perfect, ground_truth=json.dumps(gold)
    )["repaired"] == 0.0

    # Wrong-typed fields must score, never raise. A truthy non-iterable where a
    # list belongs (`"actions": true`) used to raise TypeError out of the reward
    # worker and kill the whole training run — this is that regression test.
    gt_s = json.dumps(gold)
    malformed = [
        '{"entities": [], "events": [{"topics": [{"code":"06.1"}], "actions": true}]}',
        '{"entities": [], "events": [{"topics": true, "actions": []}]}',
        '{"entities": true, "events": true}',
        '{"entities": [], "events": [{"actions": 5}]}',
        '{"entities": {"text": "Iran", "type": "LOCATION"}, "events": []}',
        '{"entities": [], "events": "none"}',
        '{"entities": [null, 5, "x"], "events": [null, true]}',
        '{"entities": [{"text": null, "type": null}], "events": [{"topics": [null]}]}',
        '[]',
        'null',
        '{"events": [{"actions": [{"code": {"nested": "dict"}}]}]}',
    ]
    for bad in malformed:
        r = compute_score(solution_str=bad, ground_truth=gt_s)
        assert isinstance(r, float) and 0.0 <= r <= 1.0, (bad, r)
    print(f"malformed-shape robustness: {len(malformed)} cases scored without raising")

    # The safety net itself must convert an unforeseen internal failure into 0.0
    # rather than an exception.
    assert _score_safe(object(), gt_s)["score"] == 0.0
    print("reward_ie self-tests OK")
