"""Constants used only by fnp.math that are not (yet) in fnp/config.py:CONSTANTS.

Same (value, provenance) format as CONSTANTS. `M(name)` looks the name up in
fnp.config.CONSTANTS first, so moving a constant into the main register overrides the
value here with no code change. Every entry is listed in
docs/review/02_ASSUMPTIONS_REGISTER.md with its re-fit plan.
"""
from __future__ import annotations

from fnp.config import CONSTANTS as _CONFIG_CONSTANTS

MATH_CONSTANTS: dict[str, tuple[float, str]] = {
    # --- reliability (Beta prior over evidence-type reliability) ---
    "rel_prior_n0": (4.0, "assumed: Beta prior strength = 4 pseudo-observations per evidence type"),
    "rel_clip": (0.02, "assumed: type prior mean clipped to [0.02, 0.98] so Beta(a,b) is proper"),
    "rel_cred_level": (0.90, "assumed: credible-interval mass for source reliability"),
    "bias_discount": (0.5, "assumed: reliability *= 1 - 0.5*|bias|, bias in [-1,1]"),
    # --- independence / copy detection (Dong et al. 2009, simplified) ---
    "independence_alpha": (1.0, "assumed: ind_i = 1/|cluster|^alpha; 1.0 = a cluster counts once"),
    "copy_prior": (0.2, "assumed: prior P(two sources are dependent), Dong 2009 uses alpha=0.2"),
    "copy_rate": (0.8, "assumed: P(copier copies a given value), Dong 2009 'c'"),
    "copy_n_false": (10.0, "assumed: number of uniformly likely false values per object, Dong 'n'"),
    # --- truth discovery ---
    "ds_laplace": (1.0, "assumed: Laplace pseudo-count in Dawid-Skene confusion rows"),
    "tf_gamma": (0.3, "Yin et al. 2008: dampening factor gamma"),
    "tf_rho": (0.5, "Yin et al. 2008: weight of implication between facts"),
    "tf_init_trust": (0.9, "Yin et al. 2008: initial source trustworthiness"),
    "tf_base_sim": (0.5, "assumed: imp(f'->f) = sim - base_sim, sim=0 for distinct values"),
    # --- meta-analysis precision proxy ---
    "meta_eps": (0.01, "assumed: v_i = 1/(eps + scale*w_i); eps keeps v finite at w=0"),
    "meta_precision_scale": (1.0, "assumed: one unit of reliability*independence*relevance = precision 1 on the log-odds scale"),
    "meta_z": (1.959963984540054, "standard normal 97.5% quantile (95% CI)"),
    # --- loopy BP ---
    "bp_damping": (0.5, "assumed: message damping (new = d*old + (1-d)*update)"),
    "bp_max_iter": (200.0, "assumed"),
    "bp_tol": (1e-8, "assumed: max abs message change for convergence"),
    "bp_coupling_scale": (1.0, "assumed: Ising coupling J = scale*sign*strength (log-potential units)"),
    # --- centrality ---
    "pagerank_alpha": (0.85, "Page & Brin 1998 damping"),
    # --- tensor CP ---
    "cp_rank": (4.0, "assumed: CP rank for coordination search"),
    "cp_n_iter": (300.0, "assumed"),
    # --- timeline noisy-OR ---
    "timeline_leak": (0.02, "assumed: P(inconsistent | no flag fires)"),
    # --- propaganda ---
    "technique_precision_default": (0.5, "assumed (task spec): per-technique tagger precision pi_t"),
    # --- features / roll-up ---
    "kb_match_min": (0.80, "assumed: KB (debunked-claim) similarity at/above which kb_match = 1"),
    "check_worthy_min": (0.50, "assumed: claims below this are ignored by doc roll-up if any pass"),
    "doc_fake_min": (0.50, "assumed: a FAKE claim needs P(FAKE) >= this to make the doc FAKE"),
    # --- rule head soft margins ---
    "rule_tau_prob": (0.05, "assumed: logistic margin width for probability thresholds"),
    "rule_tau_strength": (0.15, "assumed: logistic margin width for summed-strength thresholds"),
    "rule_tau_neff": (0.25, "assumed: logistic margin width for the n_eff threshold"),
    "strong_refute_min": (0.50, "assumed: summed independent refute strength counted as 'strong'"),
    "cm_min": (0.30, "assumed: minimum context-mismatch strength for MISLEADING"),
    "mixed_ratio_min": (0.50, "assumed: min(sup,con)/max(sup,con) above which evidence is 'mixed'"),
    "one_sided_min": (0.60, "assumed: one_sidedness counted as one-sided"),
    "coord_min": (0.50, "assumed: coordination score counted as coordinated"),
    "rule_floor": (0.01, "assumed: additive floor before normalising rule scores"),
    "secondary_min": (0.30, "assumed: a non-primary label with P >= this is a secondary tag"),
    # --- stacker ---
    "stacker_C": (1.0, "assumed: inverse L2 strength for multinomial LR"),
    "stacker_cal_frac": (0.20, "assumed: share of training rows held out for temperature scaling"),
}


def M(name: str) -> float:
    """Value of a math constant; fnp.config.CONSTANTS wins if it defines the same name."""
    if name in _CONFIG_CONSTANTS:
        return _CONFIG_CONSTANTS[name][0]
    return MATH_CONSTANTS[name][0]
