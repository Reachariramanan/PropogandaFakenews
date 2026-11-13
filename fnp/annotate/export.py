"""Versioned training exports from the human labels (claim extraction; propaganda spans + techniques).

Gold resolution per item (never mixing the two layers):
  * an adjudicator's submission, if any, wins;
  * a single annotator's submission is used as is;
  * with two+ annotators (double items): policy `agree` keeps only spans both marked (IoU >= 0.5, and the same
    technique for propaganda); the kept span is the one from the lowest annotator_id (no invented boundaries).
    Items where one annotator marked something the other did not are DISPUTED: written to disputed.jsonl and
    left out of train/dev/test. Policy `union` keeps every span of every annotator (noisier, more recall).
Splits are by near-duplicate group (item.dup_key), deterministic from (version, dup_key): no document or
re-post shares a split with its copy. Outputs are never overwritten: a version exists once.

Imbalance is accepted, not hidden (the rare classes cannot be supplied at the targets): class_weights.json holds
class-balanced weights from the effective number of samples (Cui et al. 2019, beta = ann_cb_beta) computed on the TRAIN
split only, and every record carries sample_weight. Every record also says how it was produced: `assisted` (an annotator
saw model suggestions; False for blind items) and `selection` (steered | random | unknown). The random slice and the
blind items are written to their own files (canonical_random.jsonl, canonical_blind.jsonl) because they are the only
unbiased evaluation material; `--only blind|random` restricts the whole export to one of them. Gold (calibration)
labels and adjudicated resolutions never double count: gold submissions are dropped, an adjudication wins.

  .venv/bin/python scripts/30_annot_export.py --kind claims --version claims_v1            # dry run
  .venv/bin/python scripts/30_annot_export.py --kind propaganda --version spans_v1 --apply
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from ..config import C, DATA_DIR
from ..eval.ptc_scorer import CLASSES14, score as ptc_score
from .db import tx

EXPORT_DIR = DATA_DIR / "annotation" / "exports"
IOU_MATCH = 0.5
SFT_SYSTEM = ("You are a claim extraction model. Given the document, extract every check-worthy factual claim, copying each "
              "claim verbatim from the document. Reply with ONLY a JSON object with the key \"claims\", a list of objects with a "
              "\"claim\" key. If the document contains no claim, return {\"claims\": []}. Do not wrap in markdown.")
SFT_USER = "Document:\n{doc}\n\nExtract the claims. Return ONLY a JSON object with the key \"claims\"."


# ------------------------------------------------------------------ pure helpers (unit-tested offline)
def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (a[1] - a[0] + b[1] - b[0] - inter) if inter else 0.0


def match_spans(A: list[dict], B: list[dict], thr: float = IOU_MATCH, same_label=None) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by descending IoU; `same_label(a, b)` can restrict pairs (e.g. same technique)."""
    cand = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if same_label and not same_label(a, b):
                continue
            v = iou((a["start"], a["end"]), (b["start"], b["end"]))
            if v >= thr:
                cand.append((v, i, j))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    ua, ub, out = set(), set(), []
    for _, i, j in cand:
        if i not in ua and j not in ub:
            ua.add(i); ub.add(j); out.append((i, j))
    return sorted(out)


def cohen_kappa(x: list, y: list) -> Optional[float]:
    n = len(x)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_kappa(x: list, y: list, n_boot: int = 1000, seed: int = 7) -> Optional[tuple[float, float]]:
    if len(x) < 5:
        return None
    rng, ks = random.Random(seed), []
    for _ in range(n_boot):
        idx = [rng.randrange(len(x)) for _ in x]
        k = cohen_kappa([x[i] for i in idx], [y[i] for i in idx])
        if k is not None:
            ks.append(k)
    ks.sort()
    return (ks[int(0.025 * len(ks))], ks[min(len(ks) - 1, int(0.975 * len(ks)))]) if ks else None


def split_of(dup_key: str, version: str, dev: float, test: float) -> str:
    u = int(hashlib.sha1(f"{version}:{dup_key}".encode()).hexdigest()[:8], 16) / 0x100000000
    return "test" if u < test else "dev" if u < test + dev else "train"


def effective_number_weights(counts: dict, beta: float) -> dict:
    """Class-balanced weights (Cui et al. 2019): w_c = (1 - beta) / (1 - beta^n_c), normalised to mean 1 over the classes present."""
    w = {c: (1.0 - beta) / (1.0 - beta ** n) for c, n in counts.items() if n > 0}
    m = sum(w.values()) / len(w) if w else 1.0
    return {c: round(v / m, 6) for c, v in w.items()}


def sample_weight(labels: list, weights: dict, none_key: str = "__none__") -> float:
    """A record's weight: that of its rarest class (highest weight); a record with no span uses the weight of the 'none' class."""
    ws = [weights[l] for l in labels if l in weights]
    return max(ws) if ws else weights.get(none_key, 1.0)


def record_labels(layer: str, r: dict) -> list:
    return [s["claim_type"] for s in r["spans"] if s.get("claim_type")] if layer == "claim" else [t for s in r["spans"] for t in s["techniques"]]


def class_weights(layer: str, rows: list[dict], beta: float) -> dict:
    """Weights from the TRAIN split only. Counts are records (items) per class, plus the explicit-negative class."""
    cnt: Counter = Counter()
    for r in rows:
        if r.get("split") != "train":
            continue
        labs = set(record_labels(layer, r))
        if labs:
            cnt.update(labs)
        elif r["status"] == "negative":
            cnt["__none__"] += 1
    return {"method": "effective_number", "beta": beta, "counts": dict(cnt), "weights": effective_number_weights(dict(cnt), beta)}


def assistance_effect(layer: str, items: dict, subs: dict) -> dict:
    """Do model suggestions change what annotators produce? Blind items (no suggestions shown) vs assisted items, first normal
    submitter per item. The blind flag is assigned by hash independent of content, so the comparison is not confounded by selection."""
    from collections import defaultdict as dd
    agg = dd(lambda: {"items": 0, "none": 0, "spans": 0, "from_suggestion": 0})
    for iid, ss in subs.items():
        ss = sorted([x for x in ss if x["role"] != "adjudicator"], key=lambda x: (x["submitted_at"], x["annotator_id"]))
        if not ss or iid not in items:
            continue
        f = ss[0]
        g = agg["blind" if items[iid]["is_blind"] else "assisted"]
        g["items"] += 1; g["none"] += f["outcome"] == "none"; g["spans"] += len(f["spans"])
        g["from_suggestion"] += sum(1 for sp in f["spans"] if sp.get("origin") in ("suggestion_accepted", "suggestion_edited"))
    out = {}
    for k, g in agg.items():
        n = g["items"]
        out[k] = {"items": n, "none_rate": g["none"] / n if n else None, "none_rate_wilson95": wilson(g["none"], n),
                  "spans_per_item": g["spans"] / n if n else None,
                  "share_spans_from_suggestions": (g["from_suggestion"] / g["spans"]) if g["spans"] else None}
    out["note"] = ("blind = suggestions never shown. A higher none-rate or fewer spans on blind items means suggestions raise recall; "
                   "a large share of spans taken from suggestions on assisted items signals anchoring: evaluate on blind/random records.")
    return out


def _group_techniques(spans: list[dict]) -> list[dict]:
    """Spans with identical offsets and different techniques become one multi-technique span."""
    by = defaultdict(set)
    for s in spans:
        by[(s["start"], s["end"])].update(s["techniques"])
    return [{"start": a, "end": b, "techniques": sorted(t)} for (a, b), t in sorted(by.items())]


def resolve_item(layer: str, subs: list[dict], policy: str = "agree") -> dict:
    """subs: [{annotator_id, username, role, outcome, spans}] -> {status, spans, annotators, n_annotators, policy_used}.
    status: 'positive' | 'negative' | 'disputed'. Spans keep their per-annotator provenance."""
    subs = sorted(subs, key=lambda s: s["annotator_id"])
    adj = [s for s in subs if s["role"] == "adjudicator"]
    use = adj[:1] if adj else subs
    names = [s["username"] for s in use]
    expand = (lambda s: [{**sp, "technique": t} for sp in s["spans"] for t in sp["techniques"]]) if layer == "propaganda" else (lambda s: s["spans"])
    if len(use) == 1:
        sp = use[0]["spans"]
        return {"status": "positive" if sp else "negative", "spans": [dict(x, annotators=names) for x in sp], "annotators": names,
                "n_annotators": 1, "policy_used": "adjudicated" if adj else "single"}
    a, b = use[0], use[1]
    if policy == "union":
        allsp = [dict(x, annotators=[s["username"]]) for s in use for x in s["spans"]]
        return {"status": "positive" if allsp else "negative", "spans": allsp, "annotators": names, "n_annotators": len(use), "policy_used": "union"}
    A, B = expand(a), expand(b)
    same = (lambda p, q: p["technique"] == q["technique"]) if layer == "propaganda" else None
    pairs = match_spans(A, B, IOU_MATCH, same)
    agreed = []
    for i, j in pairs:
        s = dict(A[i]); s["annotators"] = [a["username"], b["username"]]; s["other_offsets"] = [B[j]["start"], B[j]["end"]]
        if layer == "claim":
            s["claim_type_agreed"] = A[i]["claim_type"] == B[j]["claim_type"]
            s["check_worthy_agreed"] = A[i]["check_worthy"] == B[j]["check_worthy"]
            if not s["claim_type_agreed"]:
                s["claim_type"] = None
            if not s["check_worthy_agreed"]:
                s["check_worthy"] = None
        agreed.append(s)
    if layer == "propaganda":
        agreed = [dict(g, annotators=[a["username"], b["username"]]) for g in _group_techniques(agreed)]
    unmatched = (len(A) - len(pairs)) + (len(B) - len(pairs))
    if not A and not B:
        return {"status": "negative", "spans": [], "annotators": names, "n_annotators": 2, "policy_used": "agree"}
    if unmatched:
        return {"status": "disputed", "spans": agreed, "annotators": names, "n_annotators": 2, "policy_used": "agree",
                "unmatched": unmatched}
    return {"status": "positive", "spans": agreed, "annotators": names, "n_annotators": 2, "policy_used": "agree"}


def validate_submission(text: str, sub: dict) -> list[str]:
    """Invariants re-checked at export time (the service enforces them too)."""
    bad = []
    if sub["outcome"] == "none" and sub["spans"]:
        bad.append("outcome none but spans exist")
    if sub["outcome"] == "annotated" and not sub["spans"]:
        bad.append("outcome annotated but no spans")
    for s in sub["spans"]:
        if not (0 <= s["start"] < s["end"] <= len(text)) or text[s["start"]:s["end"]] != s["text"]:
            bad.append(f"span {s['start']}-{s['end']} does not index item text")
    return bad


def agreement_report(layer: str, per_item: dict[int, list[dict]], n_boot: int = 1000) -> dict:
    """Pairwise agreement of the first two non-adjudicator annotators on items both completed."""
    A_all, B_all, rows = [], [], []
    for iid, subs in per_item.items():
        s = sorted([x for x in subs if x["role"] != "adjudicator"], key=lambda x: x["annotator_id"])
        if len(s) >= 2:
            rows.append((iid, s[0], s[1]))
    rep = {"n_double_items": len(rows), "annotators": sorted({x["username"] for _, a, b in rows for x in (a, b)})}
    if not rows:
        return rep
    ha = [bool(a["spans"]) for _, a, b in rows]; hb = [bool(b["spans"]) for _, a, b in rows]
    agree = sum(x == y for x, y in zip(ha, hb))
    rep["item_level"] = {"has_label_agreement": agree / len(rows), "wilson95": wilson(agree, len(rows)),
                         "kappa": cohen_kappa(ha, hb), "kappa_bootstrap95": bootstrap_kappa(ha, hb, n_boot)}
    if layer == "claim":
        tp = na = nb = 0
        t1, t2, c1, c2 = [], [], [], []
        for _, a, b in rows:
            pairs = match_spans(a["spans"], b["spans"])
            tp += len(pairs); na += len(a["spans"]); nb += len(b["spans"])
            for i, j in pairs:
                t1.append(a["spans"][i]["claim_type"]); t2.append(b["spans"][j]["claim_type"])
                c1.append(a["spans"][i]["check_worthy"]); c2.append(b["spans"][j]["check_worthy"])
        p, r = tp / nb if nb else 0.0, tp / na if na else 0.0
        rep["span_iou50"] = {"matched": tp, "n_a": na, "n_b": nb, "f1": 2 * p * r / (p + r) if p + r else 0.0}
        rep["claim_type_on_matched"] = {"n": len(t1), "accuracy": (sum(x == y for x, y in zip(t1, t2)) / len(t1)) if t1 else None,
                                        "kappa": cohen_kappa(t1, t2), "kappa_bootstrap95": bootstrap_kappa(t1, t2, n_boot)}
        rep["check_worthy_on_matched"] = {"n": len(c1), "kappa": cohen_kappa(c1, c2), "kappa_bootstrap95": bootstrap_kappa(c1, c2, n_boot)}
    else:
        g = {str(i): [(s["start"], s["end"], t) for s in a["spans"] for t in s["techniques"]] for i, a, b in rows}
        p_ = {str(i): [(s["start"], s["end"], t) for s in b["spans"] for t in s["techniques"]] for i, a, b in rows}
        sc = ptc_score(g, p_)
        rep["semeval_style_A_vs_B"] = {"SI_f1": sc["SI"]["f1"], "TC_span_f1": sc["TC_span"]["f1"],
                                       "per_technique_f1": {k: v["f1"] for k, v in sc["per_technique"].items() if v["n_gold"] or v["n_pred"]}}
    return rep


# ------------------------------------------------------------------ DB loading
def load_submissions(layer: str) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    """-> (items {item_id: {...}}, submissions {item_id: [sub,...]}) for every completed layer_result."""
    tbl = "claim_span" if layer == "claim" else "propaganda_span"
    with tx() as cur:
        cur.execute("""SELECT i.item_id, i.text, i.dup_key, i.lang_group, i.lang, i.period, i.url, i.author, i.outlet, i.published_at,
                              i.title_len, i.is_double, i.is_blind, s.kind, s.platform, s.source_table, i.source_pk
                       FROM item i JOIN ingest_source s USING (source_id)
                       WHERE EXISTS (SELECT 1 FROM layer_result r WHERE r.item_id=i.item_id AND r.layer=%s)""", (layer,))
        items = {r["item_id"]: dict(r) for r in cur.fetchall()}
        cur.execute("""SELECT r.item_id, r.annotator_id, a.username, a.role, r.outcome, r.revision, r.submitted_at,
                              coalesce(x.kind, 'normal') AS kind, x.selected_by
                       FROM layer_result r JOIN annotator a USING (annotator_id)
                       LEFT JOIN assignment x ON x.item_id=r.item_id AND x.annotator_id=r.annotator_id
                       WHERE r.layer=%s AND a.active""", (layer,))
        subs = defaultdict(list)
        keyed = {}
        for r in cur.fetchall():
            if r["kind"] == "gold":
                continue                                  # calibration labels are scored in quality.py, never exported
            d = dict(r, spans=[])
            if r["kind"] == "adjudication":
                d["role"] = "adjudicator"                 # an adjudication assignment wins, whatever the account's role
            subs[r["item_id"]].append(d); keyed[(r["item_id"], r["annotator_id"])] = d
        if layer == "claim":
            cur.execute("SELECT * FROM claim_span WHERE deleted_at IS NULL ORDER BY start_char, span_id")
            for r in cur.fetchall():
                d = keyed.get((r["item_id"], r["annotator_id"]))
                if d is not None:
                    d["spans"].append({"start": r["start_char"], "end": r["end_char"], "text": r["span_text"], "claim_type": r["claim_type"],
                                       "check_worthy": r["check_worthy"], "normalised_claim": r["normalised_claim"], "speaker": r["speaker"],
                                       "origin": r["origin"]})
        else:
            cur.execute("""SELECT p.*, array_agg(t.technique ORDER BY t.technique) techs FROM propaganda_span p
                           JOIN propaganda_span_technique t USING (span_id) WHERE p.deleted_at IS NULL GROUP BY p.span_id ORDER BY p.start_char, p.span_id""")
            for r in cur.fetchall():
                d = keyed.get((r["item_id"], r["annotator_id"]))
                if d is not None:
                    d["spans"].append({"start": r["start_char"], "end": r["end_char"], "text": r["span_text"], "techniques": list(r["techs"]),
                                       "origin": r["origin"]})
    return items, dict(subs)


# ------------------------------------------------------------------ build
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _jl(path: Path, rows) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n"); n += 1
    return n


def _source(it: dict) -> dict:
    return {"kind": it["kind"], "platform": it["platform"], "table": it["source_table"], "source_pk": it["source_pk"], "url": it["url"],
            "author": it["author"], "outlet": it["outlet"], "published_at": it["published_at"], "lang": it["lang"],
            "lang_group": it["lang_group"], "period": it["period"], "title_len": it["title_len"]}


def build(kind: str, version: str, policy: str = "agree", apply: bool = False, created_by: Optional[str] = None,
          sft_check_worthy: tuple[str, ...] = ("yes",), max_sft_chars: int = 4000, out_root: Path = EXPORT_DIR,
          subset: Optional[str] = None) -> dict:
    """subset: None (everything) | 'blind' (items shown without suggestions) | 'random' (the unsteered slice)."""
    if subset not in (None, "blind", "random"):
        raise ValueError("subset must be blind|random")
    layer = {"claims": "claim", "propaganda": "propaganda"}[kind]
    dev, test = C("ann_split_dev"), C("ann_split_test")
    items, subs = load_submissions(layer)
    rows, disputed, invalid = [], [], []
    for iid, it in sorted(items.items()):
        ss = subs.get(iid, [])
        bad = [f"{s['username']}: {m}" for s in ss for m in validate_submission(it["text"], s)]
        if bad:
            invalid.append({"item_id": iid, "problems": bad}); continue
        need = 2 if it["is_double"] else 1
        if not any(s["role"] == "adjudicator" for s in ss) and len(ss) < need:
            continue                                  # double item still waiting for its second annotator
        r = resolve_item(layer, ss, policy)
        first = sorted([x for x in ss if x["role"] != "adjudicator"] or ss, key=lambda x: (x["submitted_at"], x["annotator_id"]))[0]
        rec = {"doc_id": f"annot:{iid}", "item_id": iid, "source": _source(it), "text": it["text"], "status": r["status"],
               "spans": r["spans"], "provenance": {"annotators": r["annotators"], "n_annotators": r["n_annotators"],
                                                   "policy": r["policy_used"], "blind": it["is_blind"], "layer": layer,
                                                   "assisted": not it["is_blind"], "selection": first.get("selected_by") or "unknown",
                                                   "submitted_at": max(s["submitted_at"] for s in ss)},
               "dup_key": it["dup_key"]}
        (disputed if r["status"] == "disputed" else rows).append(rec)
    if subset == "blind":
        rows = [r for r in rows if r["provenance"]["blind"]]
    elif subset == "random":
        rows = [r for r in rows if r["provenance"]["selection"] == "random"]
    for r in rows:
        r["split"] = split_of(r["dup_key"], version, dev, test)
    cw = class_weights(layer, rows, C("ann_cb_beta"))
    for r in rows:
        r["sample_weight"] = round(sample_weight(record_labels(layer, r), cw["weights"]), 6)
    res = {"kind": kind, "version": version, "policy": policy, "subset": subset, "n_items": len(rows), "n_disputed": len(disputed),
           "n_invalid": len(invalid), "by_split": dict(Counter(r["split"] for r in rows)), "by_status": dict(Counter(r["status"] for r in rows)),
           "by_selection": dict(Counter(r["provenance"]["selection"] for r in rows)),
           "n_blind": sum(r["provenance"]["blind"] for r in rows), "n_adjudicated": sum(r["provenance"]["policy"] == "adjudicated" for r in rows),
           "class_weights": cw}
    if kind == "claims":
        res["by_claim_type"] = dict(Counter(s["claim_type"] for r in rows for s in r["spans"]))
        res["by_check_worthy"] = dict(Counter(s["check_worthy"] for r in rows for s in r["spans"]))
    else:
        res["by_technique"] = dict(Counter(t for r in rows for s in r["spans"] for t in s["techniques"]))
    agree = agreement_report(layer, {i: subs[i] for i in subs if i in items})
    res["agreement"] = agree
    res["assistance_effect"] = assistance_effect(layer, items, subs)
    if not apply:
        return res
    out = out_root / version
    with tx() as cur:
        cur.execute("SELECT 1 FROM export_run WHERE version=%s", (version,))
        if cur.fetchone() or out.exists():
            raise FileExistsError(f"export {version} already exists; versions are never overwritten (pick a new --version)")
    out.mkdir(parents=True)
    files = {}
    for r in rows:
        r["label_map_version"] = 1
    files["canonical.jsonl"] = _jl(out / "canonical.jsonl", rows)
    files["canonical_random.jsonl"] = _jl(out / "canonical_random.jsonl", (r for r in rows if r["provenance"]["selection"] == "random"))
    files["canonical_blind.jsonl"] = _jl(out / "canonical_blind.jsonl", (r for r in rows if r["provenance"]["blind"]))
    (out / "class_weights.json").write_text(json.dumps(cw, indent=1))
    (out / "assistance_effect.json").write_text(json.dumps(res["assistance_effect"], indent=1, default=str))
    files["disputed.jsonl"] = _jl(out / "disputed.jsonl", disputed)
    files["invalid.jsonl"] = _jl(out / "invalid.jsonl", invalid)
    (out / "agreement.json").write_text(json.dumps(agree, indent=1, default=str))
    if kind == "claims":
        for sp in ("train", "dev", "test"):
            sft = []
            for r in (x for x in rows if x["split"] == sp):
                if len(r["text"]) > max_sft_chars:
                    continue
                keep = [s for s in r["spans"] if s["check_worthy"] in sft_check_worthy]
                if r["status"] == "positive" and not keep and any(s["check_worthy"] in ("no", "unsure", None) for s in r["spans"]):
                    continue                          # only non-check-worthy / unsure claims: ambiguous for a check-worthy extractor
                gold = json.dumps({"claims": [{"claim": s["text"]} for s in sorted(keep, key=lambda s: s["start"])]}, ensure_ascii=False)
                sft.append({"data_source": "claim_extraction",
                            "prompt": [{"role": "system", "content": SFT_SYSTEM}, {"role": "user", "content": SFT_USER.format(doc=r["text"])}],
                            "ability": "claims", "reward_model": {"style": "rule", "ground_truth": gold},
                            "extra_info": {"doc_id": r["doc_id"], "chunk_idx": 0, "n_chunks": 1, "source_dataset": "fnp_annot",
                                           "language": r["source"]["lang"], "domain": r["source"]["kind"], "annotation_type": "human",
                                           "split": sp, "n_claims": len(keep), "answer": gold, "annotators": r["provenance"]["annotators"],
                                           "sample_weight": r["sample_weight"], "assisted": r["provenance"]["assisted"],
                                           "selection": r["provenance"]["selection"],
                                           "claim_types": [s["claim_type"] for s in keep], "normalised": [s.get("normalised_claim") for s in keep]}})
            files[f"claims_sft_{sp}.jsonl"] = _jl(out / f"claims_sft_{sp}.jsonl", sft)
    else:
        for sp in ("train", "dev", "test"):
            ptc = [{"doc_id": r["doc_id"], "text": r["text"], "split": sp, "sample_weight": r["sample_weight"],
                    "assisted": r["provenance"]["assisted"], "selection": r["provenance"]["selection"],
                    "spans": [[s["start"], s["end"], t] for s in r["spans"] for t in s["techniques"]]} for r in rows if r["split"] == sp]
            files[f"spans_ptc_style_{sp}.jsonl"] = _jl(out / f"spans_ptc_style_{sp}.jsonl", ptc)
    from . import taxonomy
    (out / "label_map.json").write_text(json.dumps({
        "claim_types": [c[0] for c in taxonomy.CLAIM_TYPES], "check_worthy": ["yes", "no", "unsure"],
        "techniques_14": list(CLASSES14), "ptc18_members": {c: m for c, _, _, m, _ in taxonomy.technique_rows()},
        "offsets": "Unicode code points, half-open [start,end) into canonical.jsonl:text"}, indent=1, ensure_ascii=False))
    manifest = {"version": version, "kind": kind, "policy": policy, "created_by": created_by, "counts": res, "files": files,
                "sha256": {p.name: _sha(p) for p in sorted(out.iterdir()) if p.name != "manifest.json"},
                "config": {"iou_match": IOU_MATCH, "split_dev": dev, "split_test": test, "sft_check_worthy": list(sft_check_worthy),
                           "max_sft_chars": max_sft_chars, "split_unit": "item.dup_key", "subset": subset,
                           "class_weight": {"method": "effective_number", "beta": C("ann_cb_beta"), "fit_on": "train split only"}}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    with tx() as cur:
        cur.execute("INSERT INTO export_run (version, kind, created_by, config, counts, manifest_sha256) VALUES (%s,%s,%s,%s,%s,%s) RETURNING export_id",
                    (version, kind, created_by, json.dumps(manifest["config"]), json.dumps(res, default=str), _sha(out / "manifest.json")))
        eid = cur.fetchone()["export_id"]
        for k, sp in {(r["dup_key"], r["split"]) for r in rows}:
            cur.execute("INSERT INTO export_split (export_id, dup_key, split) VALUES (%s,%s,%s)", (eid, k, sp))
    res["out_dir"] = str(out)
    return res


# ------------------------------------------------------------------ loaders for training code
def load_export_spans(path: str | Path):
    """spans_ptc_style_*.jsonl -> (articles DataFrame[article_id, split, text], spans DataFrame[article_id, split, start, end,
    technique]) in the same shape as fnp.datasets.load_ptc, so scripts/23 can concatenate them with PTC."""
    import pandas as pd
    arts, spans = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        arts.append({"article_id": r["doc_id"], "split": r["split"], "text": r["text"]})
        spans += [{"article_id": r["doc_id"], "split": r["split"], "start": s, "end": e, "technique": t} for s, e, t in r["spans"]]
    return pd.DataFrame(arts), pd.DataFrame(spans, columns=["article_id", "split", "start", "end", "technique"])
