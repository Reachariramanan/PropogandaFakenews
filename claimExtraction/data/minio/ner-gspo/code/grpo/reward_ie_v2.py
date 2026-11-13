"""Custom VERL reward for the **v2** free-form IE JSON schema (entities + events).

Sibling of `grpo/reward_ie.py`, not a replacement. v1 is frozen because a live
training run depends on it byte-for-byte; every robustness pattern that file
earned the hard way is reproduced here rather than imported, so a future edit to
either file cannot silently change the other.

Signature required by VERL:
  compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

What v2 scores that v1 did not
-----------------------------
v1: format, entity F1 (text+type), subtype, topic-code F1, action-code F1.
v2 adds four dimensions, per the "score everything" decision:

* **topic_name** — for every topic code the prediction got *right*, did it also
  attach that code's name? The expected name comes from gold itself, falling back
  to `grpo/topic_taxonomy.py` (the source taxonomy's own names — **not** CAMEO;
  see that module's docstring for why). Name credit is strictly conditional on
  code credit: a wrong code earns no name penalty, since it is already fully
  penalized by `topic_f1` and double-charging it would make the gradient on codes
  noisier, not sharper.
* **action_f1 / action_name** — v2's gold event carries a *singular* `action`
  object `{code, name}` (or null) instead of v1's plural `actions` list. The
  reader here extracts **at most one** code per event and tolerates every shape
  a model actually emits: a plain dict, a list-of-one (old habit), the v1
  `"actions"` key, a bare string `"15"`, or garbage. `action_name` is
  conditional on `action_f1` credit exactly as topic names are.
* **evidence_text** — a *grounding* check, not a comparison against gold's
  wording. The question is "is this a real span of the document, or did you
  invent it?", so the predicted evidence is matched as a substring of the source
  document after normalization. See `_evidence_scores` for the exact rule and
  `_grounding_corpus` for what happens when the document is unavailable.
* **actor_f1** — normalized `actor1` and `actor2` matched against gold,
  scored separately then averaged. `actor2` is null in ~12.5% of real source
  events; null-vs-null is a **match**, never a penalty (this falls out of the
  empty-vs-empty convention in `_f1`/`_soft_f1`).

Everything v1 scored is carried over with its logic essentially unchanged
(`_entity_keys` / `_subtype_map` / `_topic_codes` / `_f1`), so entity and
subtype numbers remain comparable across the two schemas.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from grpo.topic_taxonomy import canonical_action_name, canonical_topic_name

# Type-level noise filter, copied from grpo/reward_ie.py (kept local so v1 stays
# untouchable). This is the `type` field filter; the *subtype*-level drop list
# lives in grpo/subtype_taxonomy.py and is applied at data-prep time, not here.
NOISE_TYPES = {"HDL", "PR", "MEN", "CLM", "HTG"}

# Sums to 1.0 (asserted in the self-tests below).
WEIGHTS = {
    "format": 0.10,
    "entity": 0.30,
    "subtype": 0.10,
    "topic": 0.20,
    "topic_name": 0.05,
    "action": 0.10,
    "action_name": 0.05,
    "evidence": 0.05,
    "actor": 0.05,
}

_TOPIC_RE = re.compile(r"^\d+\.\d+$")
_ACTION_RE = re.compile(r"^\d{1,2}$")
ACTION_CODE_MIN, ACTION_CODE_MAX = 13, 20

# Minimum characters an evidence fragment must have to be checked at all. A
# 1-3 character string is a substring of almost any document, so crediting it
# would hand out free grounding points for "a" / "the".
_MIN_EVIDENCE_CHARS = 4

# Minimum whitespace-separated words an *event* evidence span must have. The
# character floor alone was farmable: a single common 4-letter word ("near",
# "said") is a substring of nearly every document, and events had no anchor
# requirement, so such a span scored a full 1.0 — identical to quoting the
# correct sentence. Gold event evidence is a full sentence (mean 17.6 words,
# median 17), and only 0.24% of real gold spans fall below 3 words, so this
# floor costs gold essentially nothing while removing the free lunch.
_MIN_EVENT_EVIDENCE_WORDS = 3

# Similarity floor for counting a predicted actor as matching a gold actor.
_ACTOR_MATCH_FLOOR = 0.6

# Typographic folding applied to both the document and the predicted evidence
# before substring matching. The source corpus is full of curly quotes and
# en/em dashes; a model that emits the ASCII equivalent is not hallucinating.
_TYPO_FOLD = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote / apostrophe
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u201e": '"',
    "\u2032": "'",   # prime
    "\u2033": '"',   # double prime
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2212": "-",   # minus sign
    "\u2026": "...",  # horizontal ellipsis -> the fragment separator
    "\u00a0": " ",   # non-breaking space
    "\u200b": "",    # zero-width space
    "\ufeff": "",    # BOM
}


# --------------------------------------------------------------------------- #
# JSON extraction / truncation repair — copied verbatim from grpo/reward_ie.py
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Coercion / normalization
# --------------------------------------------------------------------------- #
def _norm_text(t: Any) -> str:
    return re.sub(r"\s+", " ", str(t if t is not None else "").strip().lower())


def _as_list(v: Any) -> list:
    """Coerce a generated field to a list without ever raising.

    The model is free-form: nothing stops it emitting `"actions": true`,
    `"entities": {...}`, `"topics": 5`, or (new in v2) `"action": [ {...} ]`
    where a single object belongs. The naive `x or []` idiom only guarded
    *falsy* values, so a truthy non-iterable (`True`, `5`) reached the `for`
    loop and raised `TypeError: 'bool' object is not iterable` — which
    propagated out of the reward worker and killed an entire training run on one
    malformed rollout out of ~480.

    A single scalar or dict is treated as a one-element list (the charitable
    reading of "wrote one item instead of a list of one", and the reading that
    makes v2's singular `action` object and a stray list-of-one both work);
    anything else scores as empty. Format credit is unaffected —
    `score_prediction` still requires a real `list` for that, so proper output
    stays strictly better.
    """
    if v is None or isinstance(v, bool):
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, (str, int, float, dict)):
        return [v]
    return []


def _fold_typography(s: str) -> str:
    for k, v in _TYPO_FOLD.items():
        if k in s:
            s = s.replace(k, v)
    return s


def _norm_span(t: Any) -> str:
    """Whitespace/case/typography-normalized text for substring grounding.

    NFKC comes first and matters more than it looks. Social-media source text
    routinely uses Unicode "mathematical alphanumeric" styling for emphasis
    (U+1D400.., e.g. "\U0001d5d4 \U0001d5cb\U0001d5c2\U0001d5cc\U0001d5ce\U0001d5ba\U0001d5c5" for "A visual"), and the labeller quotes such
    spans back in plain ASCII. Without NFKC those spans are not substrings of
    their own document, so every evidence item on the affected row scored 0.0 —
    a real gold row in the smoke set (doc_id 91) sat at evidence_text=0.0 for
    exactly this reason. NFKC also folds full-width forms, which the corpus uses
    for CJK-adjacent Latin, and leaves CJK and accented Latin untouched.
    """
    s = unicodedata.normalize("NFKC", str(t if t is not None else ""))
    return re.sub(r"\s+", " ", _fold_typography(s).strip().lower())


def _loose_span(t: Any) -> str:
    """`_norm_span` with punctuation stripped — the near-verbatim fallback."""
    # Keep alphanumerics, spaces, and everything non-ASCII (CJK et al); drop
    # ASCII punctuation, which is what "near-verbatim" is allowed to differ by.
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z\u0080-\uffff ]+", " ", _norm_span(t))).strip()


# --------------------------------------------------------------------------- #
# v1 content extractors (carried over unchanged in behaviour)
# --------------------------------------------------------------------------- #
def _entity_keys(entities: Any) -> set[tuple[str, str]]:
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


def _subtype_map(entities: Any) -> dict[tuple[str, str], str]:
    m: dict[tuple[str, str], str] = {}
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


# --------------------------------------------------------------------------- #
# v2 extractors: topics / action (singular) / actors
# --------------------------------------------------------------------------- #
def _code_and_name(item: Any) -> tuple[str, str | None]:
    """(code, name) from `{"code":..,"name":..}` or a bare `"NN.N"` scalar.

    `code` is stringified defensively: a model that nests a dict under "code"
    yields a string that no code regex matches, which is exactly the desired
    outcome (dropped, not raised).
    """
    if isinstance(item, dict):
        code, name = item.get("code"), item.get("name")
    else:
        code, name = item, None
    c = str(code if code is not None else "").strip()
    n = name.strip() if isinstance(name, str) and name.strip() else None
    return c, n


def _topic_pairs(events: Any) -> tuple[set[str], dict[str, set[str]]]:
    """All valid NN.N topic codes across events, plus code -> emitted names."""
    codes: set[str] = set()
    names: dict[str, set[str]] = {}
    for ev in _as_list(events):
        if not isinstance(ev, dict):
            continue
        for t in _as_list(ev.get("topics")):
            c, n = _code_and_name(t)
            if not _TOPIC_RE.match(c):
                continue
            codes.add(c)
            if n:
                names.setdefault(c, set()).add(_norm_text(n))
    return codes, names


def _action_pairs(events: Any) -> tuple[set[str], dict[str, set[str]]]:
    """At most **one** action code per event, from whatever shape was emitted.

    v2's gold `action` is a singular object, so only the first valid code in an
    event counts even if the model emitted a list. `"actions"` (v1's plural key)
    is accepted as a fallback for the same slot: reading it costs nothing and
    scoring an otherwise-correct answer 0.0 over a key name the model was
    weaned off would be a misleading gradient. Format credit still docks the
    drift (see `_schema_drift`), so the singular key remains strictly better.
    """
    codes: set[str] = set()
    names: dict[str, set[str]] = {}
    for ev in _as_list(events):
        if not isinstance(ev, dict):
            continue
        raw = ev.get("action")
        if raw is None:
            raw = ev.get("actions")
        for a in _as_list(raw):
            c, n = _code_and_name(a)
            if not _ACTION_RE.match(c):
                continue
            if not (ACTION_CODE_MIN <= int(c) <= ACTION_CODE_MAX):
                continue
            codes.add(c)
            if n:
                names.setdefault(c, set()).add(_norm_text(n))
            break  # singular: one code per event, first valid one wins
    return codes, names


def _actor_sets(events: Any) -> tuple[set[str], set[str]]:
    """(actor1 values, actor2 values), normalized, nulls excluded.

    A genuinely absent `actor2` contributes nothing to the set, so gold-null vs
    pred-null becomes empty-vs-empty, which `_soft_f1` scores 1.0. The literal
    strings "null"/"none"/"n/a" are treated as absent too — models write them.
    """
    a1: set[str] = set()
    a2: set[str] = set()
    for ev in _as_list(events):
        if not isinstance(ev, dict):
            continue
        v1 = _norm_text(ev.get("actor1"))
        if v1 and v1 not in ("null", "none", "n/a", "-"):
            a1.add(v1)
        v2 = _norm_text(ev.get("actor2"))
        if v2 and v2 not in ("null", "none", "n/a", "-"):
            a2.add(v2)
    return a1, a2


def _str_sim(a: str, b: str) -> float:
    """Similarity in [0, 1] for two normalized actor spans."""
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.9
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if not inter:
        return 0.0
    return 2 * inter / (len(ta) + len(tb))


def _soft_f1(pred: set[str], gold: set[str]) -> float:
    """F1 with graded, greedily-matched string pairs.

    Gold actors are long spans lifted from the document ("a senior official at
    the U.S. Department of Defense"), so exact set equality would score almost
    every near-correct answer 0.0 and flatten the gradient. Identical strings
    still score exactly 1.0, so gold-against-itself is unaffected. Inputs are
    capped so a pathological rollout with hundreds of events cannot turn this
    into a quadratic blowup inside the reward worker.
    """
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    p = sorted(pred)[:64]
    g = sorted(gold)[:64]
    used: set[int] = set()
    tp = 0.0
    for a in p:
        best_j, best_sc = -1, 0.0
        for j, b in enumerate(g):
            if j in used:
                continue
            sc = _str_sim(a, b)
            if sc > best_sc:
                best_j, best_sc = j, sc
        if best_j >= 0 and best_sc >= _ACTOR_MATCH_FLOOR:
            used.add(best_j)
            tp += best_sc
    if tp <= 0.0:
        return 0.0
    prec = tp / len(p)
    rec = tp / len(g)
    return 2 * prec * rec / (prec + rec)


def _name_score(
    matched: set[str],
    pred_names: dict[str, set[str]],
    gold_names: dict[str, set[str]],
    canon,
) -> float:
    """Fraction of correctly-coded items that also carry the expected name.

    Conditional on code credit by construction: `matched` is the intersection of
    predicted and gold codes.

    The expected name is taken from **gold's own emitted name** for that code,
    falling back to the lookup table only when gold supplies none. This ordering
    matters: it makes the name term self-consistent with the target the model is
    being trained on by definition, so gold-against-itself always scores 1.0 and
    a stale or wrong lookup table can never turn this into supervision that
    contradicts `topic_f1`/`action_f1`. (That is not hypothetical — the table
    this file previously trusted held CAMEO names, which disagree with the source
    taxonomy on 100% of topic codes.)

    A code that neither gold nor the table can name is unjudgeable, so it is
    credited rather than penalized — gold legitimately emits `name: null` for the
    three `NN.N` codes the source leaves unnamed and must not be punished for it.
    """
    if not matched:
        return 0.0
    hit = 0
    for c in sorted(matched):
        want = gold_names.get(c) or set()
        if not want:
            fallback = canon(c)
            if fallback is None:
                hit += 1  # unnameable code: not the model's fault
                continue
            want = {_norm_text(fallback)}
        if pred_names.get(c, set()) & want:
            hit += 1
    return hit / len(matched)


# --------------------------------------------------------------------------- #
# Evidence grounding
# --------------------------------------------------------------------------- #
def _grounding_corpus(gold_obj: dict[str, Any], doc_text: Any) -> tuple[str, str, bool]:
    """(normalized corpus, punctuation-stripped corpus, used_document).

    Primary channel is the source document, threaded through
    `extra_info["doc_text"]` by `grpo/prepare_parquet_v2.py` (the *truncated*
    text the model actually saw, so a span past the truncation point is
    correctly treated as ungrounded). When it is missing — an older parquet, or
    a caller that drops `extra_info` — the fallback corpus is gold's own
    evidence spans concatenated. That fallback is strictly weaker (it can only
    confirm evidence gold happened to quote) and is reported as
    `grounded_vs_doc=0.0` so it is visible in MLflow instead of quietly
    depressing the evidence term.
    """
    if isinstance(doc_text, str) and doc_text.strip():
        return _norm_span(doc_text), _loose_span(doc_text), True
    spans: list[str] = []
    for key in ("entities", "events"):
        for item in _as_list(gold_obj.get(key)):
            if isinstance(item, dict):
                ev = item.get("evidence_text")
                if isinstance(ev, str) and ev.strip():
                    spans.append(ev)
    joined = "\n".join(spans)
    return _norm_span(joined), _loose_span(joined), False


def _is_grounded(evidence: Any, corpus_norm: str, corpus_loose: str) -> bool:
    """Is `evidence` a verbatim-or-near-verbatim span of the corpus?

    Exact normalized substring first, then a punctuation-stripped retry (which
    forgives a dropped comma or a possessive apostrophe). `...` splits the
    string into fragments that must *each* be grounded — the source labeller
    elides mid-span with an ellipsis, and 2.5% of real gold evidence uses that
    convention, so honouring it lifts gold self-grounding from 95.3% to 97.8%
    on event evidence.
    """
    if not corpus_norm:
        return False
    n = _norm_span(evidence)
    if len(n) < _MIN_EVIDENCE_CHARS:
        return False
    frags = [f.strip() for f in n.split("...") if len(f.strip()) >= _MIN_EVIDENCE_CHARS]
    if not frags:
        frags = [n]
    for f in frags:
        if f in corpus_norm:
            continue
        lf = _loose_span(f)
        if lf and lf in corpus_loose:
            continue
        return False
    return True


def _event_evidence_score(
    ev: dict[str, Any], corpus_norm: str, corpus_loose: str
) -> float:
    """Grounding score for one event's `evidence_text`: 1.0 / 0.5 / 0.0.

    Mirrors the entity rule's anchor requirement. An event's anchor is its own
    actors: full credit needs a grounded span that actually contains `actor1` or
    `actor2`, which is what stops a single grounded common phrase from farming
    the whole term for every event in the document. 92.93% of real gold event
    spans contain one of their actors (81.30% contain `actor1`), so gold against
    itself lands at ~0.965 on this side — the same shape as the entity side's
    ~0.97, and worth ~0.002 of total score at this weight.
    """
    evidence = ev.get("evidence_text")
    if not _is_grounded(evidence, corpus_norm, corpus_loose):
        return 0.0
    n = _norm_span(evidence)
    if len(n.split()) < _MIN_EVENT_EVIDENCE_WORDS:
        return 0.0
    for key in ("actor1", "actor2"):
        a = _norm_span(ev.get(key))
        if a and a not in ("null", "none", "n/a", "-") and a in n:
            return 1.0
    return 0.5


def _evidence_scores(
    pred_obj: dict[str, Any],
    gold_obj: dict[str, Any],
    corpus_norm: str,
    corpus_loose: str,
    fmt: float,
) -> tuple[float, float, float]:
    """(combined, entity-side, event-side) evidence-grounding scores.

    Per item:
      * **entity** — 1.0 grounded *and* the span contains the entity's own
        `text`; 0.5 grounded but anchored on something else (a real span of the
        document, but not the one that supports this entity); 0.0 not grounded
        or missing. The containment clause is what stops "emit one common
        phrase for every entity" from farming the whole term; 96.6% of real
        gold entities satisfy both clauses and 97.9% satisfy grounding alone,
        so gold-against-itself lands at ~0.97 here rather than exactly 1.0 —
        worth ~0.001 of total score at this weight.
      * **event** — the same graded rule with the event's own actors as the
        anchor; see `_event_evidence_score`. Previously this side was a flat
        "1.0 if grounded", which a single 4-character common word satisfied for
        every event at once — `evidence_text: "near"` scored exactly what
        quoting the correct sentence did.

    An item with no `evidence_text` scores 0.0 rather than being skipped: the
    prompt requires the field and it is non-null in 100% of gold.

    The two sides are combined as an **unweighted mean of whichever sides made
    claims**. Entities outnumber events ~4:1, so pooling every item would let
    entity evidence drown out event evidence entirely; averaging the two rates
    keeps event grounding audible. A side the prediction left empty while gold
    had items scores 0.0 for that side; a side empty on both is skipped.
    """
    sides: list[float] = []
    per_side: list[float] = [-1.0, -1.0]

    for idx, key in enumerate(("entities", "events")):
        pred_items = [x for x in _as_list(pred_obj.get(key)) if isinstance(x, dict)]
        gold_items = [x for x in _as_list(gold_obj.get(key)) if isinstance(x, dict)]
        if not pred_items:
            if gold_items:
                sides.append(0.0)
                per_side[idx] = 0.0
            continue
        acc = 0.0
        for item in pred_items:
            if key == "events":
                acc += _event_evidence_score(item, corpus_norm, corpus_loose)
                continue
            if not _is_grounded(item.get("evidence_text"), corpus_norm, corpus_loose):
                continue
            text = _norm_span(item.get("text"))
            acc += 1.0 if (text and text in _norm_span(item.get("evidence_text"))) else 0.5
        val = acc / len(pred_items)
        sides.append(val)
        per_side[idx] = val

    if not sides:
        # Nothing claimed and nothing required.
        combined = 1.0 if fmt > 0 else 0.0
        return combined, combined, combined
    combined = sum(sides) / len(sides)
    ent = per_side[0] if per_side[0] >= 0.0 else combined
    evt = per_side[1] if per_side[1] >= 0.0 else combined
    return combined, ent, evt


# --------------------------------------------------------------------------- #
# Format
# --------------------------------------------------------------------------- #
def _schema_drift(events: Any) -> bool:
    """True when an event uses a v1/plural shape where v2 wants a singular one.

    Caught cases: the `"actions"` key instead of `"action"`, `action` emitted as
    a list, and `topics` emitted as a non-list. Content credit is still awarded
    (see `_action_pairs`); this only trims format credit, so correct shape wins.
    """
    for ev in _as_list(events):
        if not isinstance(ev, dict):
            continue
        if "action" not in ev and "actions" in ev:
            return True
        if isinstance(ev.get("action"), (list, tuple)):
            return True
        topics = ev.get("topics")
        if topics is not None and not isinstance(topics, list):
            return True
    return False


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
METRIC_KEYS = (
    "score",
    "format",
    "entity_f1",
    "subtype",
    "topic_f1",
    "topic_name",
    "action_f1",
    "action_name",
    "evidence_text",
    "evidence_entity",
    "evidence_event",
    "actor_f1",
    "actor1_f1",
    "actor2_f1",
    "repaired",
    "grounded_vs_doc",
)

ZERO_METRICS = {k: 0.0 for k in METRIC_KEYS}


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


def _doc_text_from_extra(extra_info: Any) -> Any:
    """Pull the source document out of `extra_info` without ever raising.

    `extra_info` arrives from VERL's non-tensor batch, so it may be a plain
    dict, a numpy-boxed mapping, or None depending on the reward manager and the
    parquet writer.
    """
    if extra_info is None:
        return None
    try:
        if isinstance(extra_info, dict):
            return extra_info.get("doc_text")
        getter = getattr(extra_info, "get", None)
        if callable(getter):
            return getter("doc_text")
        return getattr(extra_info, "doc_text", None)
    except Exception:  # noqa: BLE001 — extra_info shape is not ours to trust
        return None


def score_prediction(
    pred_obj: dict[str, Any] | None,
    gold_obj: dict[str, Any],
    repaired: bool = False,
    doc_text: Any = None,
) -> dict[str, float]:
    if pred_obj is None:
        return dict(ZERO_METRICS)

    has_ent = "entities" in pred_obj and isinstance(pred_obj.get("entities"), list)
    has_ev = "events" in pred_obj and isinstance(pred_obj.get("events"), list)
    fmt = 1.0 if (has_ent and has_ev) else (0.5 if (has_ent or has_ev) else 0.0)
    # Output that only parsed after truncation repair was not valid JSON as
    # emitted, so it must not earn full format credit — otherwise truncating
    # early would be free. Content sub-scores below are still awarded, and the
    # missing entities/events cost recall in the F1 terms on their own.
    if repaired:
        fmt = min(fmt, 0.5)
    if fmt > 0.0 and _schema_drift(pred_obj.get("events")):
        fmt = min(fmt, 0.75)

    # --- entities (v1 logic) ---
    pred_ent = _entity_keys(pred_obj.get("entities"))
    gold_ent = _entity_keys(gold_obj.get("entities"))
    ent_f1 = _f1(pred_ent, gold_ent)

    pred_st = _subtype_map(pred_obj.get("entities"))
    gold_st = _subtype_map(gold_obj.get("entities"))
    inter = set(pred_st) & set(gold_st) & pred_ent & gold_ent
    if inter:
        subtype = sum(1 for k in inter if pred_st.get(k) == gold_st.get(k)) / len(inter)
    elif not gold_st:
        subtype = 1.0 if fmt > 0 else 0.0
    else:
        subtype = 0.0

    # --- topics: code F1 + conditional name credit ---
    pred_tc, pred_tn = _topic_pairs(pred_obj.get("events"))
    gold_tc, gold_tn = _topic_pairs(gold_obj.get("events"))
    topic_f1 = _f1(pred_tc, gold_tc)
    matched_tc = pred_tc & gold_tc
    if matched_tc:
        topic_name = _name_score(matched_tc, pred_tn, gold_tn, canonical_topic_name)
    elif not gold_tc and not pred_tc:
        topic_name = 1.0 if fmt > 0 else 0.0
    else:
        topic_name = 0.0

    # --- action (singular) : code F1 + conditional name credit ---
    pred_ac, pred_an = _action_pairs(pred_obj.get("events"))
    gold_ac, gold_an = _action_pairs(gold_obj.get("events"))
    action_f1 = _f1(pred_ac, gold_ac)
    matched_ac = pred_ac & gold_ac
    if matched_ac:
        action_name = _name_score(matched_ac, pred_an, gold_an, canonical_action_name)
    elif not gold_ac and not pred_ac:
        action_name = 1.0 if fmt > 0 else 0.0
    else:
        action_name = 0.0

    # --- actors ---
    p_a1, p_a2 = _actor_sets(pred_obj.get("events"))
    g_a1, g_a2 = _actor_sets(gold_obj.get("events"))
    actor1_f1 = _soft_f1(p_a1, g_a1)
    actor2_f1 = _soft_f1(p_a2, g_a2)  # empty-vs-empty == 1.0: null actor2 is fine
    actor_f1 = 0.5 * (actor1_f1 + actor2_f1)

    # --- evidence grounding ---
    corpus_norm, corpus_loose, used_doc = _grounding_corpus(gold_obj, doc_text)
    evidence, ev_ent, ev_evt = _evidence_scores(
        pred_obj, gold_obj, corpus_norm, corpus_loose, fmt
    )

    score = (
        WEIGHTS["format"] * fmt
        + WEIGHTS["entity"] * ent_f1
        + WEIGHTS["subtype"] * subtype
        + WEIGHTS["topic"] * topic_f1
        + WEIGHTS["topic_name"] * topic_name
        + WEIGHTS["action"] * action_f1
        + WEIGHTS["action_name"] * action_name
        + WEIGHTS["evidence"] * evidence
        + WEIGHTS["actor"] * actor_f1
    )
    return {
        "score": float(score),
        "format": float(fmt),
        "entity_f1": float(ent_f1),
        "subtype": float(subtype),
        "topic_f1": float(topic_f1),
        "topic_name": float(topic_name),
        "action_f1": float(action_f1),
        "action_name": float(action_name),
        "evidence_text": float(evidence),
        "evidence_entity": float(ev_ent),
        "evidence_event": float(ev_evt),
        "actor_f1": float(actor_f1),
        "actor1_f1": float(actor1_f1),
        "actor2_f1": float(actor2_f1),
        "repaired": 1.0 if repaired else 0.0,
        "grounded_vs_doc": 1.0 if used_doc else 0.0,
    }


def _score_safe(
    solution_str: Any, ground_truth: Any, extra_info: Any = None
) -> dict[str, float]:
    """Score without ever propagating an exception.

    This function runs inside VERL's reward worker on model-generated text. An
    exception here does not degrade one sample's reward — it tears down the Ray
    worker and kills the entire training run (that is exactly how a 38-step run
    died at step 15 on a single `"actions": true`). The type coercion in
    `_as_list` fixes the known shapes; this guard bounds the damage from any
    shape nobody has thought of yet, since the space of strings a language model
    can emit is not enumerable in advance. v2 widens the attack surface — nested
    `action` objects, `topics` name fields, evidence substrings, `extra_info`
    plumbing — so the guard matters more here, not less.
    """
    try:
        gold = _parse_ground_truth(ground_truth)
        pred, repaired = _extract_json_with_status(solution_str)
        return score_prediction(
            pred, gold, repaired=repaired, doc_text=_doc_text_from_extra(extra_info)
        )
    except Exception as e:  # noqa: BLE001 — deliberately total
        preview = str(solution_str)[:200].replace("\n", "\\n")
        print(
            f"WARN reward_ie_v2: scoring failed ({type(e).__name__}: {e}); "
            f"returning 0.0. Output: {preview!r}"
        )
        return dict(ZERO_METRICS)


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """VERL entrypoint — returns scalar reward in ~[0, 1]."""
    return float(_score_safe(solution_str, ground_truth, extra_info)["score"])


def compute_score_detailed(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, float]:
    return _score_safe(solution_str, ground_truth, extra_info)


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, WEIGHTS

    DOC = (
        "China's increasingly aggressive actions near Taiwan serve not just as "
        "exercises but as rehearsals for potential forced unification, a senior "
        "official at the U.S. Department of Defense said on Wednesday, urging "
        "Taiwan's political parties to work together to approve a special defense "
        "budget. Taiwan President Lai Ching-te proposed an eight-year NT$1.25 "
        "trillion special defense budget, but the plan has been facing resistance "
        "from opposition parties."
    )
    GOLD = {
        "entities": [
            {
                "text": "Lai Ching-te",
                "type": "PERSON",
                "subtype": "political_leader",
                "evidence_text": "Taiwan President Lai Ching-te",
            },
            {
                "text": "Taiwan",
                "type": "LOCATION",
                "subtype": "country",
                "evidence_text": "aggressive actions near Taiwan",
            },
            {
                "text": "U.S. Department of Defense",
                "type": "ORGANIZATION",
                "subtype": "government_body",
                "evidence_text": "a senior official at the U.S. Department of Defense",
            },
        ],
        "events": [
            {
                "actor1": "China",
                "actor2": "Taiwan",
                # Names below are the SOURCE taxonomy's own names (see
                # grpo/topic_taxonomy.py), not CAMEO. The CAMEO names for these
                # same codes would be "Exhibit Military Posture" / "Defend
                # Verbally" / "Cooperate Economically" — asserted wrong in test 10.
                "action": {"code": "15", "name": "Exhibit Force Posture"},
                "topics": [
                    {"code": "05.2", "name": "Territorial & Maritime Disputes"},
                    {"code": "06.1", "name": "Military-to-Military Engagement"},
                ],
                "evidence_text": (
                    "China's increasingly aggressive actions near Taiwan serve not "
                    "just as exercises but as rehearsals for potential forced unification"
                ),
            },
            {
                # actor1 is contained in this event's evidence span, which is what
                # earns full event-evidence credit (92.93% of real gold events are
                # anchored on actor1 or actor2 this way).
                "actor1": "Lai Ching-te",
                "actor2": None,
                "action": None,
                "topics": [{"code": "03.2", "name": "Project Financing & Debt"}],
                "evidence_text": "Lai Ching-te proposed an eight-year NT$1.25 trillion special defense budget",
            },
        ],
    }
    GT = json.dumps(GOLD, ensure_ascii=False)
    XI = {"doc_text": DOC, "schema_version": "v2"}

    def show(label: str, m: dict[str, float]) -> None:
        keys = (
            "score", "format", "entity_f1", "subtype", "topic_f1", "topic_name",
            "action_f1", "action_name", "evidence_text", "actor_f1", "repaired",
        )
        print(f"  {label:<26} " + " ".join(f"{k}={m[k]:.3f}" for k in keys))

    print("== 1. gold scored against itself (synthetic doc) ==")
    m_gold = compute_score_detailed(solution_str=GT, ground_truth=GT, extra_info=XI)
    show("gold-self", m_gold)
    for k in ("format", "entity_f1", "subtype", "topic_f1", "topic_name",
              "action_f1", "action_name", "actor_f1", "evidence_text"):
        assert m_gold[k] == 1.0, (k, m_gold[k])
    assert m_gold["score"] > 0.999, m_gold
    assert m_gold["grounded_vs_doc"] == 1.0

    print("== 2. gold-self on real prepared rows (data/grpo_v2_smoke) ==")
    real_rows: list[tuple[str, dict]] = []
    try:
        import pandas as pd  # noqa: PLC0415 — optional, self-test only

        smoke = _ROOT / "data" / "grpo_v2_smoke"
        frames = [
            pd.read_parquet(smoke / f"{s}.parquet")
            for s in ("train", "val", "test")
            if (smoke / f"{s}.parquet").exists()
        ]
        if frames:
            df = pd.concat(frames)
            for _, row in df.iterrows():
                real_rows.append((row["reward_model"]["ground_truth"], dict(row["extra_info"])))
    except Exception as e:  # noqa: BLE001 — the self-test must not depend on data
        print(f"  (skipped real-row check: {type(e).__name__}: {e})")
    if real_rows:
        scores = []
        evs = []
        for gt, xi in real_rows:
            m = compute_score_detailed(solution_str=gt, ground_truth=gt, extra_info=xi)
            scores.append(m["score"])
            evs.append(m["evidence_text"])
            # Every hard-matched dimension must be exactly 1.0 on real gold.
            for k in ("format", "entity_f1", "subtype", "topic_f1", "topic_name",
                      "action_f1", "action_name", "actor_f1", "actor1_f1", "actor2_f1"):
                assert m[k] == 1.0, (k, m, gt[:200])
            # Evidence is the one soft dimension and lands below 1.0 on ~35% of
            # real rows, for three understood reasons — none of them a scoring
            # bug, all of them bounded by the 0.05 weight:
            #   1. the labeller paraphrases rather than quoting verbatim (~2-3%
            #      of spans), so the span is genuinely not in the document;
            #   2. an event's gold evidence does not contain either of its own
            #      actors (7.07% of gold events), which earns 0.5 under the
            #      anchor rule that stops evidence farming;
            #   3. the document was trimmed to fit the prompt token budget
            #      (grpo/prepare_parquet_v2.py::PromptBudget), putting some gold
            #      spans past the cut — counted there as
            #      `gold_evidence_outside_doc`.
            # Reason 3 is the one to watch: if it grows, lower the trim pressure
            # rather than loosening this bound.
            assert m["score"] >= 0.95, (m, gt[:200])
        mean = sum(scores) / len(scores)
        print(f"  {len(scores)} real gold rows: score min={min(scores):.4f} "
              f"mean={mean:.4f} max={max(scores):.4f} (all >= 0.95)")
        print(f"  {len(evs)} real gold rows: evidence min={min(evs):.4f} "
              f"mean={sum(evs) / len(evs):.4f} (soft dimension by design)")
        assert mean >= 0.99, mean
        # A real row's evidence must be grounded against the real document.
        gt0, xi0 = real_rows[0]
        m0 = compute_score_detailed(solution_str=gt0, ground_truth=gt0, extra_info=xi0)
        show("real-row-0", m0)
    else:
        print("  (no real rows available)")

    print("== 3. monotone quality ordering ==")
    garbage = compute_score_detailed(solution_str="not json at all", ground_truth=GT, extra_info=XI)
    empty = compute_score_detailed(solution_str='{"entities": [], "events": []}', ground_truth=GT, extra_info=XI)
    partial = compute_score_detailed(
        solution_str=json.dumps(
            {
                "entities": [
                    {"text": "Taiwan", "type": "LOCATION", "subtype": "country",
                     "evidence_text": "aggressive actions near Taiwan"}
                ],
                "events": [
                    {"actor1": "China", "actor2": "Taiwan", "action": {"code": "15", "name": "Wrong Name"},
                     "topics": [{"code": "05.2", "name": "Territorial & Maritime Disputes"}],
                     "evidence_text": "actions near Taiwan serve not just as exercises"}
                ],
            }
        ),
        ground_truth=GT,
        extra_info=XI,
    )
    show("garbage", garbage)
    show("empty-json", empty)
    show("partial", partial)
    assert garbage["score"] == 0.0, garbage
    assert garbage["score"] < empty["score"] < partial["score"] < m_gold["score"], (
        garbage["score"], empty["score"], partial["score"], m_gold["score"]
    )
    # Right code, wrong name: code credit kept, name credit lost.
    assert partial["action_f1"] == 1.0 and partial["action_name"] == 0.0, partial

    print("== 4. singular `action` shapes ==")
    def one_event(action_val, as_json=True):
        ev = {"actor1": "China", "actor2": "Taiwan", "topics": [{"code": "05.2", "name": "Territorial & Maritime Disputes"}],
              "evidence_text": "aggressive actions near Taiwan"}
        ev["action"] = action_val
        body = {"entities": [], "events": [ev]}
        return json.dumps(body) if as_json else body

    shapes = {
        "plain dict": one_event({"code": "15", "name": "Exhibit Force Posture"}),
        "list-of-one": one_event([{"code": "15", "name": "Exhibit Force Posture"}]),
        "bare string": one_event("15"),
        "v1 plural key": json.dumps(
            {"entities": [], "events": [{"actor1": "China", "actor2": "Taiwan",
                                         "actions": [{"code": "15", "name": "Exhibit Force Posture"}],
                                         "topics": [{"code": "05.2", "name": "Territorial & Maritime Disputes"}],
                                         "evidence_text": "aggressive actions near Taiwan"}]}),
        "nested-dict code": one_event({"code": {"nested": "dict"}, "name": "x"}),
        "bool": one_event(True),
        "int out of range": one_event(3),
        "null": one_event(None),
    }
    for label, payload in shapes.items():
        m = compute_score_detailed(solution_str=payload, ground_truth=GT, extra_info=XI)
        show(f"action:{label}", m)
    for label in ("plain dict", "list-of-one", "bare string", "v1 plural key"):
        m = compute_score_detailed(solution_str=shapes[label], ground_truth=GT, extra_info=XI)
        assert m["action_f1"] == 1.0, (label, m)
    assert compute_score_detailed(solution_str=shapes["plain dict"], ground_truth=GT, extra_info=XI)["format"] == 1.0
    # The old plural key still earns content credit but loses format credit, so
    # the singular shape stays strictly better.
    m_sing = compute_score_detailed(solution_str=shapes["plain dict"], ground_truth=GT, extra_info=XI)
    m_plur = compute_score_detailed(solution_str=shapes["v1 plural key"], ground_truth=GT, extra_info=XI)
    assert m_plur["format"] == 0.75 and m_plur["score"] < m_sing["score"], (m_plur, m_sing)
    for label in ("nested-dict code", "bool", "int out of range", "null"):
        m = compute_score_detailed(solution_str=shapes[label], ground_truth=GT, extra_info=XI)
        assert m["action_f1"] == 0.0 and m["action_name"] == 0.0, (label, m)  # gold has "15"

    print("== 5. actor2 null-vs-null is a match, not a penalty ==")
    g_null = {"entities": [], "events": [{"actor1": "China", "actor2": None, "action": None, "topics": [],
                                          "evidence_text": "aggressive actions near Taiwan"}]}
    gt_null = json.dumps(g_null)
    m_nn = compute_score_detailed(solution_str=gt_null, ground_truth=gt_null, extra_info=XI)
    show("actor2 null/null", m_nn)
    assert m_nn["actor2_f1"] == 1.0 and m_nn["actor_f1"] == 1.0, m_nn
    # Omitting the key entirely is equivalent to an explicit null.
    p_omit = json.dumps({"entities": [], "events": [{"actor1": "China", "action": None, "topics": [],
                                                     "evidence_text": "aggressive actions near Taiwan"}]})
    m_om = compute_score_detailed(solution_str=p_omit, ground_truth=gt_null, extra_info=XI)
    assert m_om["actor2_f1"] == 1.0, m_om
    # Hallucinating an actor2 where gold has none IS penalized.
    p_hall = json.dumps({"entities": [], "events": [{"actor1": "China", "actor2": "Japan", "action": None,
                                                     "topics": [], "evidence_text": "aggressive actions near Taiwan"}]})
    m_hl = compute_score_detailed(solution_str=p_hall, ground_truth=gt_null, extra_info=XI)
    show("actor2 hallucinated", m_hl)
    assert m_hl["actor2_f1"] == 0.0 and m_hl["actor1_f1"] == 1.0, m_hl
    # Missing a gold actor2 is penalized symmetrically.
    m_miss = compute_score_detailed(solution_str=p_omit, ground_truth=GT, extra_info=XI)
    assert m_miss["actor2_f1"] == 0.0, m_miss

    print("== 6. evidence grounding: verbatim vs hallucinated ==")
    def ev_payload(entity_ev: str, event_ev: str) -> str:
        return json.dumps(
            {
                "entities": [{"text": "Taiwan", "type": "LOCATION", "subtype": "country",
                              "evidence_text": entity_ev}],
                "events": [{"actor1": "China", "actor2": "Taiwan",
                            "action": {"code": "15", "name": "Exhibit Force Posture"},
                            "topics": [], "evidence_text": event_ev}],
            }
        )

    verbatim = compute_score_detailed(
        solution_str=ev_payload("aggressive actions near Taiwan",
                                "urging Taiwan's political parties to work together"),
        ground_truth=GT, extra_info=XI)
    fabricated = compute_score_detailed(
        solution_str=ev_payload("Taiwan declared independence on Tuesday",
                                "the Kremlin denied all involvement in the incident"),
        ground_truth=GT, extra_info=XI)
    caseshift = compute_score_detailed(
        solution_str=ev_payload("AGGRESSIVE   ACTIONS  NEAR TAIWAN",
                                "URGING TAIWAN'S POLITICAL PARTIES"),
        ground_truth=GT, extra_info=XI)
    trivial = compute_score_detailed(solution_str=ev_payload("the", "as"), ground_truth=GT, extra_info=XI)
    show("evidence verbatim", verbatim)
    show("evidence fabricated", fabricated)
    show("evidence case/space", caseshift)
    show("evidence trivial", trivial)
    assert verbatim["evidence_text"] == 1.0, verbatim
    assert fabricated["evidence_text"] == 0.0, fabricated
    assert caseshift["evidence_text"] == 1.0, caseshift  # normalization is forgiving
    assert trivial["evidence_text"] == 0.0, trivial      # too short to be evidence
    # An entity span that is real but does not contain the entity gets half credit.
    half = compute_score_detailed(
        solution_str=ev_payload("special defense budget", "urging Taiwan's political parties"),
        ground_truth=GT, extra_info=XI)
    assert half["evidence_entity"] == 0.5, half
    # Ellipsis-elided spans (the labeller's own convention) stay grounded.
    elided = compute_score_detailed(
        solution_str=ev_payload("aggressive actions near Taiwan",
                                "China's increasingly aggressive actions ... forced unification"),
        ground_truth=GT, extra_info=XI)
    assert elided["evidence_event"] == 1.0, elided
    # Unicode "mathematical alphanumeric" styling (social-media bold) must fold
    # to ASCII, or gold's own plain-text quote is not a substring of its document.
    styled_doc = "\U0001d5d4 \U0001d5cb\U0001d5c2\U0001d5cc\U0001d5ce\U0001d5ba\U0001d5c5 journey through the landscapes where the Navy stands guard"
    styled_gt = json.dumps({
        "entities": [{"text": "Navy", "type": "ORGANIZATION", "subtype": "military_branch_coalition",
                      "evidence_text": "the Navy stands guard"}],
        "events": [{"actor1": "the Navy", "actor2": None, "action": None, "topics": [],
                    "evidence_text": "the landscapes where the Navy stands guard"}],
    })
    m_styled = compute_score_detailed(
        solution_str=styled_gt, ground_truth=styled_gt, extra_info={"doc_text": styled_doc})
    show("unicode-styled doc", m_styled)
    assert m_styled["evidence_text"] == 1.0, m_styled
    # Full-width Latin folds the same way.
    assert compute_score_detailed(
        solution_str=json.dumps({"entities": [], "events": [
            {"actor1": "ABC", "action": None, "topics": [],
             "evidence_text": "ABC denied the report"}]}),
        ground_truth=GT,
        extra_info={"doc_text": "ＡＢＣ denied the report yesterday"},
    )["evidence_event"] == 1.0

    # No document available -> documented fallback to gold's own spans, flagged.
    nodoc = compute_score_detailed(solution_str=GT, ground_truth=GT, extra_info=None)
    show("gold-self (no doc_text)", nodoc)
    assert nodoc["grounded_vs_doc"] == 0.0 and nodoc["evidence_text"] > 0.9, nodoc

    print("== 7. truncation repair still works (v1 regression) ==")
    truncated = (
        '{\n "entities": [\n'
        '  {"text": "Lai Ching-te", "type": "PERSON", "subtype": "political_leader", '
        '"evidence_text": "Taiwan President Lai Ching-te"},\n'
        '  {"text": "Taiwan", "type": "LOCATION", "subtype": "country", '
        '"evidence_text": "aggressive actions near Taiwan"},\n'
        '  {"text": "U.S. Dep'
    )
    m_tr = compute_score_detailed(solution_str=truncated, ground_truth=GT, extra_info=XI)
    show("truncated", m_tr)
    assert m_tr["repaired"] == 1.0 and m_tr["format"] == 0.5, m_tr
    assert m_tr["score"] > garbage["score"] and m_tr["score"] < m_gold["score"], m_tr
    assert m_tr["entity_f1"] > 0.5, m_tr
    m_unsalv = compute_score_detailed(solution_str='{"entities": [{"te', ground_truth=GT, extra_info=XI)
    assert m_unsalv["score"] == 0.0, m_unsalv
    assert compute_score_detailed(solution_str=GT, ground_truth=GT, extra_info=XI)["repaired"] == 0.0
    # Markdown fences are tolerated.
    fenced = compute_score_detailed(solution_str=f"```json\n{GT}\n```", ground_truth=GT, extra_info=XI)
    assert fenced["score"] > 0.999, fenced

    print("== 8. malformed-shape robustness (must never raise) ==")
    malformed = [
        '{"entities": [], "events": [{"topics": [{"code":"05.2"}], "action": true}]}',
        '{"entities": [], "events": [{"topics": true, "action": []}]}',
        '{"entities": true, "events": true}',
        '{"entities": [], "events": [{"action": 5}]}',
        '{"entities": {"text": "Taiwan", "type": "LOCATION"}, "events": []}',
        '{"entities": [], "events": "none"}',
        '{"entities": [null, 5, "x"], "events": [null, true]}',
        '{"entities": [{"text": null, "type": null, "evidence_text": null}], "events": [{"topics": [null]}]}',
        "[]",
        "null",
        '{"events": [{"action": {"code": {"nested": "dict"}, "name": {"also": "dict"}}}]}',
        '{"entities": [{"text": 5, "type": 7, "subtype": [], "evidence_text": 9}], "events": []}',
        '{"entities": [], "events": [{"actor1": {"name": "China"}, "actor2": ["a", "b"], "action": "abc"}]}',
        '{"entities": [], "events": [{"topics": [{"code": "05.2", "name": 42}], "action": {"code": "15", "name": []}}]}',
        '{"entities": [], "events": [{"evidence_text": {"span": "x"}}]}',
        '{"entities": [{"evidence_text": ["a", "b"], "text": "Taiwan", "type": "LOCATION"}], "events": []}',
        '{"entities": [], "events": [[], {}, 0, ""]}',
        '   ',
        '{"entities": [{"text": "Taiwan", "type": "LOCATION"}], "events": [{"action": {"code": "15"}}]',
        "😀 not json 😀",
    ]
    for bad in malformed:
        r = compute_score(solution_str=bad, ground_truth=GT, extra_info=XI)
        assert isinstance(r, float) and 0.0 <= r <= 1.0, (bad, r)
        d = compute_score_detailed(solution_str=bad, ground_truth=GT, extra_info=XI)
        assert set(d) == set(METRIC_KEYS), (bad, sorted(set(d) ^ set(METRIC_KEYS)))
        for k, v in d.items():
            assert isinstance(v, float) and 0.0 <= v <= 1.0, (bad, k, v)
    print(f"  {len(malformed)} malformed payloads scored without raising, all metrics in [0,1]")

    # Pure-garbage inputs must be exactly 0.0, not merely finite.
    zero_cases = ["not json at all", "", "   ", "null", "[]", "😀 not json 😀",
                  '{"entities": [{"te', "```json\n{oops\n```", "0", "The answer is: entities and events."]
    for z in zero_cases:
        assert compute_score(solution_str=z, ground_truth=GT, extra_info=XI) == 0.0, z
    print(f"  {len(zero_cases)} garbage payloads scored exactly 0.0")

    print("== 9. safety net converts unforeseen internal failures to 0.0 ==")
    assert _score_safe(object(), GT, XI)["score"] == 0.0
    assert _score_safe(GT, object(), XI)["score"] >= 0.0     # unparseable gold -> empty gold
    assert _score_safe(GT, GT, object())["score"] > 0.0      # unusable extra_info -> fallback corpus
    assert _score_safe(GT, GT, {"doc_text": 12345})["score"] > 0.0
    assert _score_safe(GT, GT, [1, 2, 3])["score"] > 0.0
    assert compute_score() == 0.0                             # no arguments at all
    assert compute_score("ner_topic_ie", GT, GT, XI) > 0.99   # positional order: data_source first
    print("  positional signature (data_source, solution_str, ground_truth, extra_info) OK")

    print("== 10. event evidence is not farmable with one short common word ==")
    def farm(event_ev: str) -> str:
        return json.dumps({
            "entities": [],
            "events": [{"actor1": "China", "actor2": "Taiwan",
                        "action": {"code": "15", "name": "Exhibit Force Posture"},
                        "topics": [], "evidence_text": event_ev}],
        })

    # "near" and "said" are both real 4-char substrings of DOC. Before the event
    # side gained an anchor requirement they each scored evidence_event=1.00 —
    # exactly what quoting the correct sentence scored.
    for word in ("near", "said", "but", "the"):
        m = compute_score_detailed(solution_str=farm(word), ground_truth=GT, extra_info=XI)
        assert m["evidence_event"] == 0.0, (word, m["evidence_event"])
    # Grounded, substantive, but anchored on nothing this event names: half credit.
    m_unanchored = compute_score_detailed(
        solution_str=farm("serve not just as exercises"), ground_truth=GT, extra_info=XI)
    # Grounded and containing the event's own actor1: full credit.
    m_anchored = compute_score_detailed(
        solution_str=farm("China's increasingly aggressive actions near Taiwan"),
        ground_truth=GT, extra_info=XI)
    m_farmed = compute_score_detailed(solution_str=farm("near"), ground_truth=GT, extra_info=XI)
    show("event ev farmed(near)", m_farmed)
    show("event ev unanchored", m_unanchored)
    show("event ev anchored", m_anchored)
    assert m_unanchored["evidence_event"] == 0.5, m_unanchored
    assert m_anchored["evidence_event"] == 1.0, m_anchored
    assert m_farmed["score"] < m_unanchored["score"] < m_anchored["score"], (
        m_farmed["score"], m_unanchored["score"], m_anchored["score"])
    # A span too short to be evidence stays worthless even if it names the actor.
    assert compute_score_detailed(
        solution_str=farm("China"), ground_truth=GT, extra_info=XI)["evidence_event"] == 0.0

    print("== 11. topic/action names are the SOURCE taxonomy, not CAMEO ==")
    # The exact codes whose CAMEO names contradict the source labels. Emitting the
    # CAMEO name must earn no name credit; the source name must earn full credit.
    CAMEO_VS_SOURCE = {
        "05.2": ("Defend Verbally", "Territorial & Maritime Disputes"),
        "06.1": ("Cooperate Economically", "Military-to-Military Engagement"),
    }
    for code, (cameo, source) in CAMEO_VS_SOURCE.items():
        assert canonical_topic_name(code) == source, code
        assert canonical_topic_name(code) != cameo, code
        g = json.dumps({"entities": [], "events": [
            {"actor1": "China", "topics": [{"code": code, "name": source}],
             "action": None, "evidence_text": "aggressive actions near Taiwan"}]})
        def topic_pred(name):
            return json.dumps({"entities": [], "events": [
                {"actor1": "China", "topics": [{"code": code, "name": name}],
                 "action": None, "evidence_text": "aggressive actions near Taiwan"}]})
        m_src = compute_score_detailed(solution_str=topic_pred(source), ground_truth=g, extra_info=XI)
        m_cam = compute_score_detailed(solution_str=topic_pred(cameo), ground_truth=g, extra_info=XI)
        assert m_src["topic_f1"] == 1.0 and m_src["topic_name"] == 1.0, (code, m_src)
        # Same code credit, no name credit: the name term never fights the code term.
        assert m_cam["topic_f1"] == 1.0 and m_cam["topic_name"] == 0.0, (code, m_cam)
    assert canonical_action_name("15") == "Exhibit Force Posture"
    assert canonical_action_name("20") == "Engage in Unconventional Mass Violence"

    # Name credit follows GOLD, not the lookup table: if gold names a code
    # something the table does not know, matching gold is what scores.
    g_odd = json.dumps({"entities": [], "events": [
        {"actor1": "China", "topics": [{"code": "05.2", "name": "Some Other Label"}],
         "action": None, "evidence_text": "aggressive actions near Taiwan"}]})
    p_odd = g_odd
    p_tbl = json.dumps({"entities": [], "events": [
        {"actor1": "China", "topics": [{"code": "05.2", "name": "Territorial & Maritime Disputes"}],
         "action": None, "evidence_text": "aggressive actions near Taiwan"}]})
    assert compute_score_detailed(solution_str=p_odd, ground_truth=g_odd, extra_info=XI)["topic_name"] == 1.0
    assert compute_score_detailed(solution_str=p_tbl, ground_truth=g_odd, extra_info=XI)["topic_name"] == 0.0
    # A code neither gold nor the table can name is unjudgeable -> credited.
    g_unnamed = json.dumps({"entities": [], "events": [
        {"actor1": "China", "topics": [{"code": "01.6", "name": None}],
         "action": None, "evidence_text": "aggressive actions near Taiwan"}]})
    assert canonical_topic_name("01.6") is None
    assert compute_score_detailed(
        solution_str=g_unnamed, ground_truth=g_unnamed, extra_info=XI)["topic_name"] == 1.0

    print("reward_ie_v2 self-tests OK")
