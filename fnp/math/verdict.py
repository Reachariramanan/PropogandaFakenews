"""L7 verdict: soft rule head, stacker (multinomial LR + temperature), final verdict.

Rule head. Each threshold test [x >= t] is replaced by sigmoid((x - t)/tau); AND = product,
OR = noisy-OR, NOT = 1 - x. Missing features are filled conservatively (cannot prove):
p_true_hi->1, p_true_lo->0, strengths->0, kb_match->0, p_prop->0, one_sidedness->0,
coordination->0, ds_uncertainty->1, n_eff->0.
  fake_core = [p_hi <= fake_p_true_upper] AND [contradiction >= strong_refute_min]
  FAKE      = fake_core OR [kb_match]
  MISL      = ([cm >= cm_min] AND [cm >= max(sup, con)]) OR
              ([min(sup,con)/max(sup,con) >= mixed_ratio_min] AND [min(sup,con) >= cm_min])
  PROP      = [p_prop >= prop_threshold] AND ([one_sided >= one_sided_min] OR [coord >= coord_min])
              AND NOT FAKE
  LEGIT     = [p_lo >= legit_p_true_lower] AND NOT [p_prop >= prop_threshold]
  gate      = [ds_unc >= max_uncertainty] OR [n_eff <= min_n_eff - 0.5]
  INCONC    = gate * (1 - max(MISL, PROP, kb_match)) OR NOT max(FAKE, MISL, PROP, LEGIT)
  probs     = (score + floor) / sum(score + floor)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fnp.config import C
from fnp.math._util import fnum, soft_ge, soft_le, soft_or
from fnp.math.calibration import apply_temperature, fit_temperature
from fnp.math.conformal import SplitConformal, decide_label
from fnp.math.constants import M
from fnp.schema import DECIDABLE, FEATURE_NAMES, LABELS, Verdict

_FILL = {"p_true_hi": 1.0, "p_true_lo": 0.0, "support": 0.0, "contradiction": 0.0,
         "context_mismatch": 0.0, "kb_match": 0.0, "p_prop": 0.0, "one_sidedness": 0.0,
         "coordination": 0.0, "ds_uncertainty": 1.0, "n_eff": 0.0}


def rule_scores(features: dict) -> dict[str, float]:
    g = {k: fnum(features.get(k), v) for k, v in _FILL.items()}
    tp, ts, tn = M("rule_tau_prob"), M("rule_tau_strength"), M("rule_tau_neff")
    sup, con, cm = g["support"], g["contradiction"], g["context_mismatch"]
    fake_core = soft_le(g["p_true_hi"], C("fake_p_true_upper"), tp) * \
        soft_ge(con, M("strong_refute_min"), ts)
    fake = soft_or(fake_core, min(max(g["kb_match"], 0.0), 1.0))
    cm_dom = soft_ge(cm, M("cm_min"), ts) * soft_ge(cm - max(sup, con), 0.0, ts)
    lo, hi = min(sup, con), max(sup, con)
    mixed = soft_ge(lo / hi if hi > 0 else 0.0, M("mixed_ratio_min"), 0.1) * soft_ge(lo, M("cm_min"), ts)
    misl = soft_or(cm_dom, mixed)
    prop_hi = soft_ge(g["p_prop"], C("prop_threshold"), tp)
    prop = prop_hi * soft_or(soft_ge(g["one_sidedness"], M("one_sided_min"), tp),
                             soft_ge(g["coordination"], M("coord_min"), tp)) * (1 - fake)
    legit = soft_ge(g["p_true_lo"], C("legit_p_true_lower"), tp) * (1 - prop_hi)
    # the evidence-insufficiency gate does not veto labels that need no truth evidence
    gate = soft_or(soft_ge(g["ds_uncertainty"], C("max_uncertainty"), tp),
                   soft_le(g["n_eff"], C("min_n_eff") - 0.5, tn))
    inc = soft_or(gate * (1 - max(misl, prop, min(max(g["kb_match"], 0.0), 1.0))),
                  1 - max(fake, misl, prop, legit))
    return {"FAKE": fake, "PROPAGANDA": prop, "LEGIT": legit, "MISLEADING": misl,
            "INCONCLUSIVE": inc}


def rule_head(features: dict) -> dict[str, float]:
    """Soft rule probabilities over schema.LABELS (sum to 1; not calibrated)."""
    s = rule_scores(features)
    fl = M("rule_floor")
    z = sum(s[k] + fl for k in LABELS)
    return {k: (s[k] + fl) / z for k in LABELS}


# ---------------------------------------------------------------------------
class Stacker:
    """Multinomial LR over [features (median-imputed) + missing indicators, rule probs,
    gemma probs, laya probs], target = schema.DECIDABLE, then temperature scaling on a
    held-out slice. Probability blocks enter as log(p + 1e-4); an absent block is filled
    with the uniform distribution and flagged by an indicator column.
    """

    BLOCKS = ("rule", "gemma", "laya")

    def __init__(self, C_reg: Optional[float] = None, cal_frac: Optional[float] = None, seed: int = 0):
        self.C_reg = M("stacker_C") if C_reg is None else C_reg
        self.cal_frac = M("stacker_cal_frac") if cal_frac is None else cal_frac
        self.seed = seed
        self.medians: Optional[np.ndarray] = None
        self.T = 1.0

    @staticmethod
    def _block(p: Optional[dict], labels=LABELS) -> tuple[list[float], float]:
        if not p:
            return [float(np.log(1.0 / len(labels) + 1e-4))] * len(labels), 1.0
        v = np.clip([fnum(p.get(k), 0.0) for k in labels], 0.0, None)
        v = v / v.sum() if v.sum() > 0 else np.full(len(labels), 1 / len(labels))
        return list(np.log(v + 1e-4)), 0.0

    def build_matrix(self, features: Sequence[dict], rule_probs=None, gemma_probs=None,
                     laya_probs=None) -> np.ndarray:
        n = len(features)
        F = np.array([[fnum(f.get(k)) for k in FEATURE_NAMES] for f in features], dtype=float)
        miss = ~np.isfinite(F)
        if self.medians is None:
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(miss, np.nan, F), axis=0)
            self.medians = np.where(np.isfinite(med), med, 0.0)
        F = np.where(miss, self.medians[None, :], F)
        cols = [F, miss.astype(float)]
        rule_probs = rule_probs if rule_probs is not None else [rule_head(f) for f in features]
        for blk in (rule_probs, gemma_probs, laya_probs):
            rows = [self._block(blk[i] if blk is not None else None) for i in range(n)]
            cols.append(np.array([r[0] for r in rows]))
            cols.append(np.array([[r[1]] for r in rows]))
        return np.hstack(cols)

    def fit(self, features, y, rule_probs=None, gemma_probs=None, laya_probs=None) -> "Stacker":
        self.medians = None
        X = self.build_matrix(features, rule_probs, gemma_probs, laya_probs)
        yi = np.array([DECIDABLE.index(v) if isinstance(v, str) else int(v) for v in y])
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(yi))
        n_cal = int(round(self.cal_frac * len(yi))) if len(yi) >= 20 else 0
        cal, tr = idx[:n_cal], idx[n_cal:]
        self.scaler = StandardScaler().fit(X[tr])
        self.lr = LogisticRegression(C=self.C_reg, max_iter=2000)
        self.lr.fit(self.scaler.transform(X[tr]), yi[tr])
        self.classes_ = self.lr.classes_
        if n_cal:
            self.T = fit_temperature(self._raw(X[cal]), yi[cal])
        return self

    def _raw(self, X) -> np.ndarray:
        P = np.full((len(X), len(DECIDABLE)), 1e-9)
        P[:, self.classes_] = self.lr.predict_proba(self.scaler.transform(X))
        return P / P.sum(axis=1, keepdims=True)

    def predict_proba(self, features, rule_probs=None, gemma_probs=None, laya_probs=None) -> np.ndarray:
        """(n, 4) calibrated probabilities over schema.DECIDABLE."""
        X = self.build_matrix(features, rule_probs, gemma_probs, laya_probs)
        return apply_temperature(self._raw(X), self.T)


# ---------------------------------------------------------------------------
def final_verdict(unit_id: str, features: dict, gemma_probs: Optional[dict] = None,
                  laya_probs: Optional[dict] = None, stacker: Optional[Stacker] = None,
                  conformal: Optional[SplitConformal] = None, unit: str = "claim",
                  rationale: str = "", evidence_ids: Optional[list] = None) -> Verdict:
    """Combine heads into a schema.Verdict.

    probs come from the stacker if given (INCONCLUSIVE = 0, abstention is the conformal
    set's job), else from the rule head. With a conformal predictor, the label follows
    conformal.decide_label; without one, argmax over all 5 labels (rule head) / 4 labels.
    """
    rp = rule_head(features)
    if stacker is not None:
        p4 = stacker.predict_proba([features], [rp], [gemma_probs] if gemma_probs else None,
                                   [laya_probs] if laya_probs else None)[0]
        probs = {**dict(zip(DECIDABLE, map(float, p4))), "INCONCLUSIVE": 0.0}
        decider = "stacker"
    else:
        probs, decider = dict(rp), "rule"
    dz = sum(probs[k] for k in DECIDABLE) or 1.0
    p4n = {k: probs[k] / dz for k in DECIDABLE}
    pset = conformal.predict_set([[p4n[k] for k in DECIDABLE]])[0] if conformal else None
    if pset is None and probs["INCONCLUSIVE"] > max(probs[k] for k in DECIDABLE):
        d = decide_label(p4n, ["INCONCLUSIVE"])
        d["label"] = "INCONCLUSIVE"
    else:
        d = decide_label(p4n, pset)
    return Verdict(unit_id=unit_id, unit=unit, label=d["label"], probs=probs,
                   plausibly_fake=d["plausibly_fake"], secondary=d["secondary"],
                   prediction_set=pset if pset is not None else [d["label"]],
                   features=dict(features), decider=decider,
                   rationale=rationale or (f"{d['fake_tier']} fake" if d["fake_tier"] else ""),
                   evidence_ids=list(evidence_ids or []))
