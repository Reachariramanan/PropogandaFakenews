"""Stage 14: fit the stacker (train) + conformal (dev), evaluate every decider on test,
and write verdicts.

Deciders compared (ablation):
  rule      math rule head only
  gemma_raw Gemma silver label on the raw claim (no evidence, no maths)
  gemma_adj Gemma adjudicator on the structured state
  stacker   multinomial logit over features + rule + gemma_adj (+ laya if given)
  laya      fine-tuned Laya decider (if --laya)
  *_conf    the same with split-conformal abstention (INCONCLUSIVE when |set| > 1)

Metrics on test units with a decidable gold label: macro-F1 (4 classes), per-class P/R
with Wilson 95% CI, ECE, Brier; selective accuracy and coverage for abstaining deciders;
abstention rate on gold-INCONCLUSIVE units. Breakdowns by dataset and by kb_match.

  .venv/bin/python scripts/14_evaluate.py [--laya data/models/laya_fnp_vN] [--apply]
--apply writes data/fnp.duckdb:verdict_v1 (claim + doc rows) and reports/EVAL_v1.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fnp.config import DATA_DIR, DUCKDB_PATH, REPORTS_DIR  # noqa: E402
from fnp.math.calibration import ece, wilson_ci as wilson  # noqa: E402
from fnp.math.conformal import SplitConformal  # noqa: E402
from fnp.math.verdict import Stacker, final_verdict  # noqa: E402
from fnp.schema import DECIDABLE, LABELS  # noqa: E402

UNITS = DATA_DIR / "runs" / "units_v1.jsonl"
NAN = float("nan")


def load_units() -> list[dict]:
    us = [json.loads(l) for l in UNITS.open()]
    for u in us:
        u["features"] = {k: (NAN if v is None else v) for k, v in u["features"].items()}
    return us


def probs_or_none(d: dict) -> dict | None:
    return d.get("probs") if d and d.get("ok") and d.get("probs") else None


def p4(p: dict | None) -> np.ndarray:
    """5-label dict -> renormalised 4-vector over DECIDABLE (uniform if missing)."""
    if not p:
        return np.full(4, 0.25)
    a = np.array([max(float(p.get(l, 0.0)), 0.0) for l in DECIDABLE])
    return a / a.sum() if a.sum() > 0 else np.full(4, 0.25)


def metrics(P: np.ndarray, y: np.ndarray, abstain: np.ndarray | None = None) -> dict:
    """P: (n,4) probs over DECIDABLE, y: (n,) int gold. abstain: bool mask (excluded)."""
    keep = ~abstain if abstain is not None else np.ones(len(y), bool)
    pred = P.argmax(1)
    out = {"n": int(len(y)), "coverage": float(keep.mean()) if len(y) else NAN}
    yk, pk = y[keep], pred[keep]
    f1s, per = [], {}
    for c, lab in enumerate(DECIDABLE):
        tp = int(((pk == c) & (yk == c)).sum())
        npred, ngold = int((pk == c).sum()), int((yk == c).sum())
        prec = tp / npred if npred else NAN
        rec = tp / ngold if ngold else NAN
        f1 = 2 * prec * rec / (prec + rec) if npred and ngold and (prec + rec) > 0 else 0.0
        if ngold:
            f1s.append(f1)
        lo, hi = wilson(tp, npred) if npred else (NAN, NAN)
        per[lab] = {"P": prec, "P_ci": (lo, hi), "R": rec, "F1": f1, "n_gold": ngold}
    out["macro_f1"] = float(np.mean(f1s)) if f1s else NAN
    out["acc"] = float((pk == yk).mean()) if len(yk) else NAN
    out["ece"] = float(ece(P[keep], yk)) if len(yk) else NAN
    oh = np.eye(4)[y]
    out["brier"] = float(((P - oh) ** 2).sum(1).mean()) if len(y) else NAN
    out["per_class"] = per
    return out


def fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya", default=None, help="Laya checkpoint dir to include as a decider")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    us = load_units()
    lidx = {l: i for i, l in enumerate(DECIDABLE)}
    by = {s: [u for u in us if u["split"] == s] for s in ("train", "dev", "test")}
    print({s: len(v) for s, v in by.items()})

    laya = {}
    if a.laya:
        from fnp.laya.predict import LayaDecider
        dec = LayaDecider.load(a.laya)
        for s in ("train", "dev", "test"):
            outs = dec.predict([u["state_text"] for u in by[s]])
            for u, o in zip(by[s], outs):
                laya[u["unit_id"]] = o["probs"]

    def heads(u):
        return (u["features"], u["rule"], probs_or_none(u["gemma_adj"]), laya.get(u["unit_id"]))

    def fit_stacker(units, use_laya):
        dec_units = [u for u in units if u["label5"] in lidx]
        F, R, G, L = zip(*[heads(u) for u in dec_units])
        y = np.array([lidx[u["label5"]] for u in dec_units])
        G = [g or {l: 0.2 for l in LABELS} for g in G]
        Lp = [l or {x: 0.2 for x in LABELS} for l in L] if use_laya else None
        return Stacker().fit(list(F), y, list(R), G, Lp)

    def stack_probs(st, units, use_laya):
        F, R, G, L = zip(*[heads(u) for u in units]) if units else ([], [], [], [])
        G = [g or {l: 0.2 for l in LABELS} for g in G]
        Lp = [l or {x: 0.2 for x in LABELS} for l in L] if use_laya else None
        return st.predict_proba(list(F), list(R), G, Lp)

    stackers = {"stacker": fit_stacker(by["train"], False)}
    if laya:
        stackers["stacker+laya"] = fit_stacker(by["train"], True)

    def decider_probs(name, units):
        if name == "rule":
            return np.array([p4(u["rule"]) for u in units])
        if name == "gemma_raw":
            return np.array([p4(probs_or_none(u["gemma_silver"])) for u in units])
        if name == "gemma_adj":
            return np.array([p4(probs_or_none(u["gemma_adj"])) for u in units])
        if name == "laya":
            return np.array([p4(laya.get(u["unit_id"])) for u in units])
        return stack_probs(stackers[name], units, name.endswith("laya"))

    names = ["rule", "gemma_raw", "gemma_adj"] + (["laya"] if laya else []) + list(stackers)
    dev_d = [u for u in by["dev"] if u["label5"] in lidx]
    test_d = [u for u in by["test"] if u["label5"] in lidx]
    test_inc = [u for u in by["test"] if u["label5"] == "INCONCLUSIVE"]
    y_dev = np.array([lidx[u["label5"]] for u in dev_d])
    y_test = np.array([lidx[u["label5"]] for u in test_d])

    results, confs = {}, {}
    for nm in names:
        Pt = decider_probs(nm, test_d)
        results[nm] = metrics(Pt, y_test)
        cf = SplitConformal(method="lac").fit(decider_probs(nm, dev_d), y_dev)
        confs[nm] = cf
        ab = cf.predict_mask(Pt).sum(1) != 1
        results[nm + "_conf"] = metrics(Pt, y_test, ab)
        if test_inc:
            ab_inc = cf.predict_mask(decider_probs(nm, test_inc)).sum(1) != 1
            results[nm + "_conf"]["abstain_on_gold_inconclusive"] = float(ab_inc.mean())
        # rule head's own INCONCLUSIVE mass as abstention, for reference
        if nm == "rule":
            ab_r = np.array([u["rule"]["INCONCLUSIVE"] >= max(u["rule"][l] for l in DECIDABLE)
                             for u in test_d])
            results["rule_selfabstain"] = metrics(Pt, y_test, ab_r)

    # breakdowns for the best stacker
    best = "stacker+laya" if "stacker+laya" in stackers else "stacker"
    bd = {}
    for key, fn in (("dataset", lambda u: u["dataset"]),
                    ("kb_match", lambda u: "kb_match" if u["features"].get("kb_match") == 1.0 else "no_kb_match")):
        groups = sorted({fn(u) for u in test_d})
        for g in groups:
            sub = [u for u in test_d if fn(u) == g]
            ys = np.array([lidx[u["label5"]] for u in sub])
            bd[f"{key}={g}"] = {nm: metrics(decider_probs(nm, sub), ys) for nm in ("rule", "gemma_adj", best)}

    # ---- report ----
    L = ["# FNP evaluation v1", "",
         f"Units: train {len(by['train'])}, dev {len(by['dev'])}, test {len(by['test'])} "
         f"(test decidable {len(test_d)}, test gold-INCONCLUSIVE {len(test_inc)}).",
         "Stacker fit on train; conformal (LAC, alpha=0.10) fit on dev; all numbers on test.", "",
         "| decider | macro-F1 | acc | ECE | Brier | coverage | abstain on gold-INCONCL |",
         "|---|---|---|---|---|---|---|"]
    for nm, r in results.items():
        L.append(f"| {nm} | {fmt(r['macro_f1'])} | {fmt(r['acc'])} | {fmt(r['ece'])} | "
                 f"{fmt(r['brier'])} | {fmt(r['coverage'])} | {fmt(r.get('abstain_on_gold_inconclusive'))} |")
    L += ["", f"## Per-class ({best}, no abstention)", "",
          "| class | n gold | P [95% Wilson] | R | F1 |", "|---|---|---|---|---|"]
    for lab, pc in results[best]["per_class"].items():
        L.append(f"| {lab} | {pc['n_gold']} | {fmt(pc['P'])} [{fmt(pc['P_ci'][0])}, {fmt(pc['P_ci'][1])}] "
                 f"| {fmt(pc['R'])} | {fmt(pc['F1'])} |")
    L += ["", "## Breakdowns (macro-F1 / acc)", "", f"| slice | n | rule | gemma_adj | {best} |",
          "|---|---|---|---|---|"]
    for k, d in bd.items():
        n = d["rule"]["n"]
        L.append(f"| {k} | {n} | " + " | ".join(f"{fmt(d[m]['macro_f1'])} / {fmt(d[m]['acc'])}"
                                                for m in ("rule", "gemma_adj", best)) + " |")
    report = "\n".join(L)
    print(report)

    if not a.apply:
        return
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "EVAL_v1.md").write_text(report + "\n")
    (REPORTS_DIR / "eval_v1.json").write_text(json.dumps({"results": results, "breakdowns": bd},
                                                         default=str, indent=1))
    # verdicts for every unit with the best stacker + its conformal set
    rows = []
    st, cf = stackers[best], confs[best]
    import pickle
    from fnp.config import MODELS_DIR
    # 15_predict uses the non-Laya stacker unless a Laya checkpoint is passed there too.
    with open(MODELS_DIR / "deciders_v1.pkl", "wb") as fh:
        pickle.dump({"stacker": stackers["stacker"], "conformal": confs["stacker"],
                     "stacker_laya": stackers.get("stacker+laya"),
                     "conformal_laya": confs.get("stacker+laya"), "laya_path": a.laya}, fh)
    for u in us:
        v = final_verdict(u["unit_id"], u["features"], probs_or_none(u["gemma_adj"]),
                          laya.get(u["unit_id"]) if best.endswith("laya") else None,
                          stacker=st, conformal=cf,
                          rationale=(u["gemma_adj"].get("rationale") or "")[:500],
                          evidence_ids=[e["evidence_id"] for e in u["evidence"]])
        rows.append({"unit_id": u["unit_id"], "unit": "claim", "doc_id": u["item_id"],
                     **{k: v[k] for k in ("label", "plausibly_fake", "decider", "rationale")},
                     "probs": json.dumps(v["probs"]), "secondary": json.dumps(v["secondary"]),
                     "prediction_set": json.dumps(v["prediction_set"]),
                     "features": json.dumps(v["features"], default=str),
                     "evidence_ids": json.dumps(v["evidence_ids"]),
                     "p_fake": v["probs"]["FAKE"], "p_misleading": v["probs"]["MISLEADING"],
                     "gold_label5": u["label5"], "dataset": u["dataset"], "split": u["split"]})
    df = pd.DataFrame(rows)
    # doc rows: one claim per gold item here, so the doc verdict is the claim verdict
    # (multi-claim posts go through fnp.math.features.doc_rollup in 15_predict).
    docs = df.copy()
    docs["unit"], docs["unit_id"] = "doc", docs["doc_id"]
    out = pd.concat([df, docs], ignore_index=True)
    import duckdb
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("CREATE TABLE IF NOT EXISTS verdict_v1 AS SELECT * FROM out LIMIT 0")
    con.execute("DELETE FROM verdict_v1 WHERE unit_id IN (SELECT unit_id FROM out)")
    con.execute("INSERT INTO verdict_v1 BY NAME SELECT * FROM out")
    print("verdict_v1 rows:", con.execute("SELECT count(*) FROM verdict_v1").fetchone()[0])
    con.close()


if __name__ == "__main__":
    main()
