"""Engine adapter: one internal shape for every FNP engine response (no fnp import, stdlib only).

Two upstream shapes are understood:

* `predict_v0`  (today, POST :8879/api/predict, an `analyse_text` dict):
      {doc:{label,probs,plausibly_fake,secondary,decider,rationale}, claims:[{verdict:{label,rationale,decider,..}}],
       spans:[{start,end,text,technique,confidence}], spans_summary:{n,techniques,one_sided}, secs}
  The same data nested as `propaganda:{n,techniques,one_sided}` (techniques = names or span objects) is also accepted.
* `check_v1`    (POST :8879/api/check_v1, docs/CHECK_V1_CONTRACT.md; detected by the keys fake_call / verdict /
  degraded): {verdict:{...raw 5-class...}, fake_call:{label,probs,fake_score,basis}, laya:{label,probs,confidence},
       propaganda:{is_propaganda_p,laya_is_propaganda_raw,techniques:[...],one_sided (a 0-1 float)}, claims, degraded,
       error, model_versions, secs}.
  Step 1 uses fake_call (basis 'none' -> INCONCLUSIVE); the raw verdict only supplies rationale/decider/plausibly_fake.
  A degraded run (55 s timeout) has verdict null: `verdict` is then the fake_call one or None.

Internal shape (what `normalise` returns):
      {verdict:{label,probs,plausibly_fake,rationale,decider[,fake_score,basis]} | None (degraded), fake_call, laya,
       degraded, error, raw_propaganda_p (the raw 5-class PROPAGANDA probability, secondary: a length artefact),
       propaganda:{is_propaganda_p|None, laya_is_propaganda_raw|None (0-1), techniques:[{technique,start,end,text,
                   confidence}], one_sided (bool|None, = one_sided_score >= 0.5), one_sided_score (0-1|None), text_words,
                   n, dropped, technique_probs:{14 buckets: 0-1}|None (only when the engine sent all 14)},
       n_claims (check_v1: len(claims), else None), degraded_why ('no_claims'|'timeout'|'unavailable'|None),
       model_versions, secs, engine:'predict_v0'|'check_v1'}

`propaganda_signal` (span-only summary: STRONG / WEAK / NONE) and `laya_view` (the Laya block of step 2) are derived
from these; see their docstrings.

Technique names are mapped onto the 14 merged buckets (PTC_MERGE below; the 14 names equal
reports/prop_lab/v1/heatmap.json['techniques'], enforced by a test). A name that maps to no bucket (the 18th PTC
technique, Obfuscation, has none) is dropped and counted in `propaganda.dropped`. Span offsets are repaired against
the submitted text: a span whose offsets do not select its own text is relocated by its first match, and a span that
cannot be placed is dropped. `is_propaganda_p` is None for predict_v0 (its PROPAGANDA probability is a length artefact).
"""
import json
import math
import re
import socket
import urllib.error
import urllib.request

BUCKETS = (
    "Loaded_Language", "Name_Calling,Labeling", "Repetition", "Exaggeration,Minimisation", "Doubt",
    "Appeal_to_fear-prejudice", "Flag-Waving", "Causal_Oversimplification", "Slogans", "Appeal_to_Authority",
    "Black-and-White_Fallacy", "Thought-terminating_Cliches", "Whataboutism,Straw_Men,Red_Herring",
    "Bandwagon,Reductio_ad_hitlerum",
)
# Gemma's 18 PTC names (fnp/schema.py PROPAGANDA_TECHNIQUES) and the 14 merged names -> merged bucket.
# None = no bucket (dropped).
PTC_MERGE = {
    "Loaded_Language": "Loaded_Language",
    "Name_Calling,Labeling": "Name_Calling,Labeling",
    "Repetition": "Repetition",
    "Exaggeration,Minimisation": "Exaggeration,Minimisation",
    "Doubt": "Doubt",
    "Appeal_to_fear-prejudice": "Appeal_to_fear-prejudice",
    "Flag-Waving": "Flag-Waving",
    "Causal_Oversimplification": "Causal_Oversimplification",
    "Slogans": "Slogans",
    "Appeal_to_Authority": "Appeal_to_Authority",
    "Black-and-White_Fallacy": "Black-and-White_Fallacy",
    "Thought-terminating_Cliches": "Thought-terminating_Cliches",
    "Whataboutism": "Whataboutism,Straw_Men,Red_Herring",
    "Straw_Men": "Whataboutism,Straw_Men,Red_Herring",
    "Red_Herring": "Whataboutism,Straw_Men,Red_Herring",
    "Whataboutism,Straw_Men,Red_Herring": "Whataboutism,Straw_Men,Red_Herring",
    "Bandwagon": "Bandwagon,Reductio_ad_hitlerum",
    "Reductio_ad_hitlerum": "Bandwagon,Reductio_ad_hitlerum",
    "Bandwagon,Reductio_ad_hitlerum": "Bandwagon,Reductio_ad_hitlerum",
    "Obfuscation,Intentional_Vagueness,Confusion": None,
}


def _key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_BY_KEY = {_key(k): v for k, v in PTC_MERGE.items()}
_BY_KEY["obfuscation"] = None
_BY_KEY["obfuscationintentionalvaguenessconfusion"] = None


def merge_technique(name):
    """Merged bucket for any spelling of a PTC technique name ('Name Calling/Labeling', 'Straw_Men', ...), else None."""
    return _BY_KEY.get(_key(name)) if isinstance(name, str) else None


class AdapterError(Exception):
    """The engine answered but the response is not a recognisable shape (message is short and safe to show)."""


def _num(x, default=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _probs(p):
    if not isinstance(p, dict):
        return {}
    return {str(k): v for k, x in p.items() if (v := _num(x)) is not None}



# Step 2 constants (assumed, listed in README "Assumed constants"; not tuned on social-media data)
SPAN_STRONG = 0.5   # a span at or above this is a "strong" span. Provenance: FNP's own STRONG_P in scripts/30_prop_lab_infer.py.
LAYA_YES = 0.5      # Laya propaganda-head raw value at or above this counts as "Laya flags". ASSUMED (0.5 is the natural cut
                    # of a probability; nobody validated it).
SHORT_WORDS = 60    # below this many words a text is "short": the span tagger (trained on long articles) and Laya's head read
                    # about 0 there. Provenance: the claim-extraction cutoff described in the README (texts over about 60
                    # words go through automatic claim extraction); the "short text" meaning here is ASSUMED.
BASIS_SPAN = "span_max"
BASIS_ENGINE = "engine"
BASIS_NOTES = {BASIS_SPAN: "highest span confidence, not a calibrated probability",
               BASIS_ENGINE: "encoder score = max token probability, uncalibrated, not validated on social posts"}
VERDICT_LABELS = ("FAKE", "MISLEADING", "LEGIT", "INCONCLUSIVE")  # the fake-or-not axis; PROPAGANDA is secondary
VERDICT_NOTE = ("engine estimate, uncalibrated; PROPAGANDA is the engine's raw class (a length artefact on short texts), "
                "shown separately and not part of the fake-or-not call")


def parse_technique_probs(raw):
    """The engine's document-level `propaganda.technique_probs` -> {14 bucket names: 0-1}, or None unless every one of
    the 14 buckets is present with a finite number between 0 and 1 (partial or malformed maps are not trusted)."""
    if not isinstance(raw, dict):
        return None
    out = {}
    for k, v in raw.items():
        b = merge_technique(k)
        x = _num(v) if not isinstance(v, bool) else None
        if b is None:
            continue
        if x is None or not 0.0 <= x <= 1.0:
            return None
        out[b] = max(out.get(b, 0.0), x)
    return {b: out[b] for b in BUCKETS} if all(b in out for b in BUCKETS) else None


def propaganda_classes(prop):
    """A propaganda block -> (classes, basis, basis_note). `classes` is [{technique, probability, basis}] for ALL 14
    merged buckets (explicit 0 when absent), sorted by probability descending (ties in BUCKETS order). With the
    engine's technique_probs the basis is 'engine'; otherwise the highest span confidence per bucket ('span_max')."""
    prop = prop if isinstance(prop, dict) else {}
    tp = parse_technique_probs(prop.get("technique_probs"))
    if tp is not None:
        basis, probs = BASIS_ENGINE, tp
    else:
        basis, probs = BASIS_SPAN, {b: 0.0 for b in BUCKETS}
        for sp in prop.get("techniques") or []:
            b = merge_technique(sp.get("technique")) if isinstance(sp, dict) else None
            c = _num(sp.get("confidence")) if isinstance(sp, dict) else None
            if b is not None and c is not None:
                probs[b] = max(probs[b], min(1.0, max(0.0, c)))
    order = {b: i for i, b in enumerate(BUCKETS)}
    classes = [{"technique": b, "probability": round(float(probs[b]), 4), "basis": basis}
               for b in sorted(BUCKETS, key=lambda b: (-probs[b], order[b]))]
    return classes, basis, BASIS_NOTES[basis]


def verdict_classes(probs, raw_propaganda_p=None):
    """Engine verdict probabilities -> step 1 bars [{label, probability, secondary}]: FAKE, MISLEADING, LEGIT,
    INCONCLUSIVE (primary) then the raw PROPAGANDA (secondary) when known. [] when there are no probabilities."""
    probs = probs if isinstance(probs, dict) else {}
    if not any(isinstance(probs.get(k), (int, float)) for k in VERDICT_LABELS):
        return []
    out = [{"label": k, "probability": round(min(1.0, max(0.0, float(probs.get(k) or 0.0))), 4), "secondary": False}
           for k in VERDICT_LABELS]
    pp = raw_propaganda_p if raw_propaganda_p is not None else probs.get("PROPAGANDA")
    pp = _num(pp)
    if pp is not None:
        out.append({"label": "PROPAGANDA", "probability": round(min(1.0, max(0.0, pp)), 4), "secondary": True})
    return out


def _locate(text, needle):
    """(start, end) of the first match of `needle` in `text` (exact, then case-insensitive), or None."""
    if not needle:
        return None
    i = text.find(needle)
    if i >= 0:
        return i, i + len(needle)
    low, nl = text.lower(), needle.lower()
    if len(low) == len(text):
        i = low.find(nl)
        if i >= 0:
            return i, i + len(nl)
    return None


def repair_span(sp, text):
    """One span object -> {technique,start,end,text,confidence} against `text`, or None when it cannot be placed
    or its technique has no bucket."""
    if not isinstance(sp, dict):
        return None
    tech = merge_technique(sp.get("technique"))
    if tech is None:
        return None
    s, e = sp.get("start"), sp.get("end")
    st = sp.get("text") if isinstance(sp.get("text"), str) else None
    ok_off = (isinstance(s, int) and isinstance(e, int) and not isinstance(s, bool) and not isinstance(e, bool)
              and 0 <= s < e <= len(text))
    if ok_off and (st is None or text[s:e] == st):
        out = (s, e)
    elif st is not None:
        out = _locate(text, st)  # offsets missing or wrong: first match of the span text
        if out is None:
            return None
    else:
        return None
    return {"technique": tech, "start": out[0], "end": out[1], "text": text[out[0]:out[1]],
            "confidence": _num(sp.get("confidence")), "strong": sp["strong"] if isinstance(sp.get("strong"), bool) else None}


def repair_spans(spans, text):
    """(kept, dropped): spans repaired, de-duplicated and sorted by position."""
    kept, seen, dropped = [], set(), 0
    for sp in spans if isinstance(spans, list) else []:
        r = repair_span(sp, text)
        if r is None:
            dropped += 1
            continue
        k = (r["technique"], r["start"], r["end"])
        if k in seen:
            continue
        seen.add(k)
        kept.append(r)
    kept.sort(key=lambda x: (x["start"], x["end"], x["technique"]))
    return kept, dropped


def _verdict(v, rationale=None, decider=None):
    v = v if isinstance(v, dict) else {}
    label = v.get("label")
    if not isinstance(label, str) or not label:
        raise AdapterError("engine response has no verdict label")
    rat = rationale if rationale is not None else v.get("rationale")
    dec = decider if decider is not None else v.get("decider")
    return {"label": label, "probs": _probs(v.get("probs")), "plausibly_fake": bool(v.get("plausibly_fake")),
            "rationale": rat.strip()[:600] if isinstance(rat, str) else "",
            "decider": dec if isinstance(dec, str) else None}


def _unit(x):
    """A finite number in [0, 1] (bool rejected), else None."""
    if isinstance(x, bool):
        return None
    v = _num(x)
    return v if v is not None and 0.0 <= v <= 1.0 else None


def _one_sided_score(x):
    """The engine's one_sided (a 0-1 float; a bool from older engines -> 1.0 / 0.0), else None."""
    return (1.0 if x else 0.0) if isinstance(x, bool) else _unit(x)


def propaganda_signal(prop):
    """Span-only summary of a propaganda block: {level: strong|weak|none, strong_n, weak_n, max_confidence, threshold,
    techniques, basis, p_prop}. A span is strong by the engine's own `strong` flag when every span carries one
    (basis 'engine': mean probability >= max(class threshold, 0.5)), otherwise by confidence >= SPAN_STRONG (basis
    'confidence', the fallback for older engines). STRONG when any span is strong, WEAK when spans exist but none is,
    NONE without spans. `techniques` = merged bucket names of the strong spans, highest confidence first. `p_prop` is
    the engine's uncalibrated score (None when absent or too short). Laya is deliberately NOT part of this (its
    input already contains the tagger's spans: no double counting)."""
    prop = prop if isinstance(prop, dict) else {}
    spans = [s for s in prop.get("techniques") or [] if isinstance(s, dict)]
    engine = (all(isinstance(s.get("strong"), bool) for s in spans) if spans
              else prop.get("p_prop") is not None or prop.get("technique_probs") is not None)  # engine sent the new fields

    def is_strong(s):
        return s["strong"] if engine else (s.get("confidence") is not None and s["confidence"] >= SPAN_STRONG)
    strong = [s for s in spans if is_strong(s)]
    confs = [s["confidence"] for s in spans if s.get("confidence") is not None]
    names = []
    for s in sorted(strong, key=lambda s: -(s.get("confidence") or 0)):
        if s.get("technique") not in names:
            names.append(s.get("technique"))
    return {"level": "strong" if strong else "weak" if spans else "none", "strong_n": len(strong),
            "weak_n": len(spans) - len(strong), "max_confidence": max(confs) if confs else None,
            "threshold": SPAN_STRONG, "techniques": names, "basis": "engine" if engine else "confidence",
            "p_prop": _unit(prop.get("p_prop"))}


def laya_view(laya, prop, n_claims, degraded_why, laya_reason=None):
    """The Laya block of step 2: {available, state: ok|not_run_no_claims|timeout|not_returned, label, confidence,
    probs:[{label,probability,secondary}], propaganda_head|None, head_in_range, short_text (the engine's `too_short`
    when sent, else under SHORT_WORDS words), text_words, n_tokens,
    compared: agree|differs|not_compared, compared_note, independent: False}.
    `laya` is the adapter's {label, probs, confidence} or None; `prop` the adapter's propaganda block.
    The head value is the engine's `laya_is_propaganda_raw`: a raw output, not a verdict."""
    prop = prop if isinstance(prop, dict) else {}
    head = _unit(prop.get("laya_is_propaganda_raw"))
    words = prop.get("text_words") if isinstance(prop.get("text_words"), int) and not isinstance(prop.get("text_words"), bool) else None
    too_short = prop.get("too_short") if isinstance(prop.get("too_short"), bool) else None
    short = too_short if too_short is not None else (words is not None and words < SHORT_WORDS)  # engine flag, else the word proxy
    if laya:
        state = "ok"
    elif laya_reason == "no_claims" or (laya_reason is None and (n_claims == 0 or degraded_why == "no_claims")):
        state = "not_run_no_claims"
    elif laya_reason == "timeout" or (laya_reason is None and degraded_why == "timeout"):
        state = "timeout"
    else:
        state = "not_returned"
    probs = []
    if laya:
        lp = laya.get("probs") or {}
        for k in VERDICT_LABELS + ("PROPAGANDA",):
            if isinstance(lp.get(k), (int, float)):
                probs.append({"label": k, "probability": round(min(1.0, max(0.0, float(lp[k]))), 4),
                              "secondary": k == "PROPAGANDA"})
    tagger = propaganda_signal(prop)["level"] if prop else None
    compared, note = "not_compared", "Not compared: one of the two readings is missing."
    if head is not None and tagger is not None:
        lay_yes, tag_yes = head >= LAYA_YES, tagger == "strong"
        if lay_yes == tag_yes:
            compared = "agree"
            note = "Agrees: both flag." if lay_yes else "Agrees: both low."
        else:
            compared = "differs"
            note = ("Differs: the tagger finds strong spans, Laya's head reads %.2f." % head if tag_yes
                    else "Differs: Laya flags (%.2f), the tagger finds no strong spans." % head)
            note += (" The head appears to track article-style training data, not wording; not evidence against the spans."
                     if tag_yes else " The head appears to track article-style training data, not wording; treat as unreliable.")
    return {"available": state == "ok", "state": state, "label": laya.get("label") if laya else None,
            "confidence": laya.get("confidence") if laya else None, "probs": probs, "propaganda_head": head,
            "head_in_range": head is not None, "short_text": short, "text_words": words,
            "n_tokens": prop.get("n_tokens") if isinstance(prop.get("n_tokens"), int) else None, "compared": compared,
            "compared_note": note, "independent": False}


def _check_v1(resp, text, secs):
    """check_v1 -> internal shape. Step 1 uses `fake_call` (the raw 5-class `verdict` has a PROPAGANDA length
    artefact); `fake_call.basis == 'none'` means INCONCLUSIVE. Degraded runs have no usable raw verdict."""
    has_prop = isinstance(resp.get("propaganda"), dict)  # absent/null: the engine did not run the span step
    prop = resp["propaganda"] if has_prop else {}
    spans, dropped = repair_spans(prop.get("techniques"), text)
    mv = resp.get("model_versions")
    raw = resp.get("verdict") if isinstance(resp.get("verdict"), dict) else None
    fc = resp.get("fake_call") if isinstance(resp.get("fake_call"), dict) else None
    degraded = resp.get("degraded") is True
    err = resp.get("error")
    err = err.strip()[:200] if isinstance(err, str) and err.strip() else None
    verdict, fake_call = None, None
    if fc is not None and isinstance(fc.get("label"), str) and fc["label"]:
        basis = fc.get("basis") if isinstance(fc.get("basis"), str) else None
        none = basis == "none"
        score = None if none else _num(fc.get("fake_score"))
        verdict = _verdict({"label": "INCONCLUSIVE" if none else fc["label"], "probs": fc.get("probs"),
                            "plausibly_fake": False if none else bool((raw or {}).get("plausibly_fake"))},
                           rationale=(raw or {}).get("rationale"), decider=(raw or {}).get("decider") or basis)
        verdict["fake_score"] = score
        verdict["basis"] = basis
        fake_call = {"label": verdict["label"], "probs": verdict["probs"], "fake_score": score, "basis": basis}
    elif raw is not None and not degraded:
        verdict = _verdict(raw)
    elif not degraded:
        raise AdapterError("engine response has no verdict label")
    lay = resp.get("laya") if isinstance(resp.get("laya"), dict) else None
    laya = None
    if lay is not None and isinstance(lay.get("label"), str) and lay["label"]:
        laya = {"label": lay["label"], "probs": _probs(lay.get("probs")), "confidence": _num(lay.get("confidence"))}
    raw_pp = _num(_probs((raw or {}).get("probs")).get("PROPAGANDA"))
    claims = resp.get("claims")
    n_claims = len(claims) if isinstance(claims, list) else None
    reason = resp.get("degraded_reason") if isinstance(resp.get("degraded_reason"), str) else None
    lreason = resp.get("laya_reason") if resp.get("laya_reason") in ("no_claims", "timeout") else None
    if reason in ("no_claims", "timeout"):  # the engine says why (it may now report no_claims with degraded false)
        why = reason
    elif reason == "gemma_error":
        why = "unavailable"
    elif not degraded:
        why = None
    elif n_claims == 0 and err is None:  # older engines: no reason field
        why = "no_claims"
    elif err and ("timed out" in err.lower() or "timeout" in err.lower()):
        why = "timeout"
    else:
        why = "unavailable"
    score = _one_sided_score(prop.get("one_sided"))
    return {"verdict": verdict, "fake_call": fake_call, "laya": laya, "degraded": degraded, "error": err,
            "raw_propaganda_p": raw_pp, "n_claims": n_claims, "degraded_why": why, "laya_reason": lreason,
            "propaganda": {"is_propaganda_p": _num(prop.get("is_propaganda_p")),
                           "laya_is_propaganda_raw": _unit(prop.get("laya_is_propaganda_raw")), "techniques": spans,
                           "one_sided": None if score is None else score >= 0.5, "one_sided_score": score,
                           "text_words": len(text.split()), "p_prop": _unit(prop.get("p_prop")),
                           "too_short": prop["too_short"] if isinstance(prop.get("too_short"), bool) else None,
                           "n_tokens": prop["n_tokens"] if isinstance(prop.get("n_tokens"), int) and not isinstance(prop.get("n_tokens"), bool) else None,
                           "n": len(spans), "dropped": dropped,
                           "technique_probs": parse_technique_probs(prop.get("technique_probs"))} if has_prop else None,
            "model_versions": mv if isinstance(mv, dict) else {}, "secs": secs, "engine": "check_v1"}


def normalise(resp, text):
    """Raw engine JSON + the exact text that was sent -> internal shape. Raises AdapterError."""
    if not isinstance(resp, dict):
        raise AdapterError("engine response is not a JSON object")
    secs = _num(resp.get("secs"))
    if isinstance(resp.get("fake_call"), dict) or isinstance(resp.get("verdict"), dict) or (
            "degraded" in resp and "doc" not in resp):  # check_v1 (detected by keys, not by URL)
        return _check_v1(resp, text, secs)
    if isinstance(resp.get("doc"), dict):  # predict_v0
        doc = resp["doc"]
        claims = [c for c in resp.get("claims") or [] if isinstance(c, dict) and isinstance(c.get("verdict"), dict)]
        same = [c["verdict"] for c in claims if c["verdict"].get("label") == doc.get("label")] or \
               [c["verdict"] for c in claims]
        pick = next((v for v in same if isinstance(v.get("rationale"), str) and v["rationale"].strip()), None)
        rat = pick["rationale"] if pick else None
        dec = (pick or (same[0] if same else {})).get("decider") or doc.get("decider")
        prop = resp.get("propaganda") if isinstance(resp.get("propaganda"), dict) else {}
        summ = resp.get("spans_summary") if isinstance(resp.get("spans_summary"), dict) else {}
        raw = resp.get("spans") if isinstance(resp.get("spans"), list) else prop.get("techniques")
        spans, dropped = repair_spans(raw, text)  # name-only technique lists carry no spans: nothing to show
        one = prop.get("one_sided", summ.get("one_sided"))
        return {"verdict": _verdict(doc, rationale=rat, decider=dec), "n_claims": None, "degraded_why": None, "laya_reason": None,
                "propaganda": {"is_propaganda_p": None, "laya_is_propaganda_raw": None, "techniques": spans,
                               "p_prop": None, "too_short": None, "n_tokens": None,
                               "one_sided": one if isinstance(one, bool) else None,
                               "one_sided_score": _one_sided_score(one) if isinstance(one, bool) else None,
                               "text_words": len(text.split()), "n": len(spans),
                               "dropped": dropped, "technique_probs": None},
                "model_versions": {}, "secs": secs, "engine": "predict_v0"}
    raise AdapterError("unrecognised engine response")


class EngineError(Exception):
    """Transport-level failure. `state` is unavailable | timeout | busy | error; `reason` is short and safe."""

    def __init__(self, state, reason, retry_after=None):
        super().__init__(reason)
        self.state, self.reason, self.retry_after = state, reason, retry_after


def _is_timeout(e):
    return isinstance(e, (socket.timeout, TimeoutError)) or (
        isinstance(e, urllib.error.URLError) and isinstance(e.reason, (socket.timeout, TimeoutError)))


def call_engine(url, text, lang, timeout):
    """POST {text, lang} to the engine and return the parsed JSON. Raises EngineError (never a stack trace)."""
    body = {"text": text}
    if lang:
        body["lang"] = lang
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(4_000_000)
    except urllib.error.HTTPError as e:
        if e.code in (429, 503):
            try:
                ra = int(e.headers.get("Retry-After")) if e.headers else None
            except (TypeError, ValueError):
                ra = None
            raise EngineError("busy", "the engine is busy", ra if ra and ra > 0 else None)
        raise EngineError("error", "the engine returned HTTP %d" % e.code)
    except Exception as e:  # noqa: BLE001
        if _is_timeout(e):
            raise EngineError("timeout", "the engine did not answer within %g s" % timeout)
        raise EngineError("unavailable", "the engine is not reachable")
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001
        raise EngineError("error", "the engine sent an unreadable response")
