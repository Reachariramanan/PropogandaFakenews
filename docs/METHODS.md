# FNP methods: the mathematical core (`fnp/math`)

Every model reads structured records (`fnp/schema.py`) and never raw text. Constants come from `fnp/config.py:CONSTANTS` (`C(...)`) or `fnp/math/constants.py:MATH_CONSTANTS` (`M(...)`). `M` checks `CONSTANTS` first, so moving a constant into config needs no code change. Every assumed value is listed in `docs/review/02_ASSUMPTIONS_REGISTER.md`.
Source notes: **F#** = `methodologoes/hypothesis_free_probability_frameworks.md` Method #. **P-V** = `propaganda_and_forensic_evidence_notes.md` Part V.

## 1. Independence (`independence.py`): F3/F5/F6/F10 preconditions, P-V step 4
- **Purpose.** Stop copied evidence from being counted more than once.
- **Model.** Union-find links two evidence units when they share any of the following: `origin_key`, a normalised URL, an entry in `source_links`, or an embedding cosine ≥ `dup_cosine`. Each unit gets `ind_i = 1/|cluster|^alpha`, with alpha = 1. At alpha = 1 a cluster adds exactly one (mean) member's strength.
- **n_eff.** `n_eff = (ΣW_c)²/ΣW_c²`, a Kish count over cluster weights `W_c = Σ_{i∈c} rel·rlv·ind`. Plain Kish over the copies would count k copies as k.
- **AccuCopy (simplified Dong 2009).** Each shared object falls into one of three cases: k_t (both report the same true value), k_f (the same false value) or k_d (different values).
  - Independent sources: `Pt=ab`, `Pf=(1-a)(1-b)/n`, `Pd=1-Pt-Pf`.
  - Copying, with copy rate c and mean accuracy m: `Pt'=mc+Pt(1-c)`, `Pf'=(1-m)c+Pf(1-c)`, `Pd'=Pd(1-c)`.
  - The posterior P(dependent) comes from the Bayes factor and the prior `copy_prior`. Sharing a false (rare) value is the strong signal. `copy_adjusted_independence` merges source pairs with P ≥ 0.5.
- **Assumptions.**
  - Transitive linking: A~B and B~C imply A~C, so chaining can over-merge.
  - Every member of a cluster is a pure copy. Partial dependence is not modelled.
  - The copy model does not resolve copy direction. False values are uniform over n.
  - With stance data, n is really 2, which makes the copy signal weak.
- **Failure modes.** A missing `origin_key` on syndicated copies leaves them unmerged when there is no embedding. A low cosine threshold merges independent reports of the same event.
- **Re-fit.** Tune `dup_cosine` on hand-labelled same-origin pairs, then fit `copy_rate` and `copy_prior` by EM on source×claim data.

## 2. Reliability (`reliability.py`): P-V step 3
- **Model.** Prior `Beta(n0·m, n0·(1-m))`, where m = `TYPE_RELIABILITY[type]` clipped to [0.02, 0.98]. With a track record of k correct out of n, the posterior is `Beta(a0+k, b0+n-k)`. Outputs are the posterior mean and a 90% equal-tailed interval.
- **Bias hook.** `r·(1-λ|bias|)`.
- **Assumptions.**
  - The type table is a prior worth n0 = 4 observations.
  - Correctness is exchangeable across claims and topics.
  - The bias penalty is symmetric.
- **Re-fit.** Use k/n from sources that match BOOM/KB gold. Choose n0 by marginal likelihood.

## 3. Truth discovery (`truth_discovery.py`): F6, F8
- **Dawid–Skene EM.**
  - Model: `P(z_i=k) ∝ p_k Π_s π_s[k, obs_si]`.
  - M-step: `π_s[k,j] ∝ Σ_i T_ik·[obs_si=j] + λ` (Laplace), with majority-vote initialisation.
  - Outputs: claim posteriors T, confusion matrices, and per-source reliability. A stance like `context_mismatch` can be a reported label that differs from the truth classes via `label_to_class`.
- **TruthFinder (Yin 2008).**
  - `τ=-ln(1-t)`, `σ*(f)=σ(f)+ρΣ_{f'}σ(f')imp(f'→f)`, `s(f)=1/(1+e^{-γσ*})`, and t(w) is the mean s over the facts w provides.
  - imp is −base_sim between distinct values. Iteration stops when the trust vector's cosine change is below tol.
  - The paper does not normalise s across an item. Our "posterior" renormalises it.
- **Assumptions.**
  - Sources are conditionally independent given the truth. Run the independence step and AccuCopy first.
  - Confusion matrices do not depend on the item.
  - Neutral stance is treated as missing.
- **Failure modes.** EM finds a local optimum. Label switching appears when most sources are adversarial. Identifiability requires overlap between sources.
- **Re-fit.** None: both methods are unsupervised per batch.

## 4. Weight of evidence (`woe.py`): F3, P-V final formula
- **Model.**
  - Strength: `s_i = rel·ind·rlv·stance_conf`.
  - Log-odds: `L = logit(prior) + κ Σ dir_i s_i`, with dir = +1 for support and −1 for refute. Output `P = σ(L)`.
  - `context_mismatch` sums into a separate MISLEADING channel. Neutral stances are ignored.
- **Fitting.** `fit_kappa` fits κ ≥ 0 and the prior by maximum likelihood (L-BFGS-B with a tiny L2 penalty) on gold.
- **Assumptions.**
  - Conditional independence after the independence step.
  - Strength is linear in each factor.
  - Support and refute are symmetric.
  - Missing independence, relevance or stance_conf default to 1. `claim_features` always assigns independence first.
- **Failure modes.** κ is a single global scale, so heavy-tailed evidence leads to overconfidence. Many weak supports can outweigh one court finding.
- **Re-fit.** Run `09_fuse --fit` on BOOM/LIAR gold. Store κ and the prior in CONSTANTS.

## 5. Dempster–Shafer (`dempster_shafer.py`): F2
- **Masses.** On the frame {T, F}: support gives `m({T}) = min(s, cap)`, refute gives `m({F}) = min(s, cap)`, and the rest goes to Θ. Neutral and context-mismatch evidence are vacuous.
- **Combination.**
  - The n-ary unnormalised conjunctive combination is associative and gives K = m(∅).
  - Dempster then divides by 1−K. Yager (the default) adds K to Θ.
  - Outputs: Bel, Pl, uncertainty = Pl(T) − Bel(T), and conflict K.
  - Zadeh's paradox is tested: Dempster puts mass 1 on the minority option, while Yager puts 0.9999 on Θ.
- **Assumptions.**
  - Sources are independent. This is the same requirement as WoE, and independence has already scaled s.
  - The cap stops any single item from committing a full mass of 1.
- **Failure modes.**
  - Yager is not associative pairwise. We apply it once to the n-ary product, so the result does not depend on order.
  - K grows with the number of mixed weak items.
- **Re-fit.** `ds_strength_cap` is set by a sensitivity sweep, not fitted.

## 6. Meta-analysis (`meta.py`): F10
- **Model.**
  - Effect: `q_i = rel·rlv·sc` (without independence), capped. `y_i = dir·logit(0.5+0.5q_i)`.
  - Precision: `w_i = rel·ind·rlv`, with variance `v_i = 1/(ε + scale·w_i)`.
  - Pooling: fixed effect `Σw y/Σw`, with SE `1/√Σw`.
  - DerSimonian–Laird: `Q=Σw(y-ȳ)²`, `τ²=max(0,(Q-df)/(Σw-Σw²/Σw))`, `I²=max(0,(Q-df)/Q)`, and re-weights `1/(v+τ²)`.
  - `p_true_lo/hi = σ(RE ± 1.96·SE)`. With no evidence the interval is [0, 1].
- **Key design.** Independence enters only the precision. k copies at weight w/k therefore give the same estimate and (up to ε) the same SE as one copy.
- **Validation.** Hand-computed DL values, plus the BCG dataset (Colditz 1994): μ = −0.7141, SE = 0.1787, τ² = 0.3088.
- **Assumptions.**
  - The variance proxy is not a sampling variance.
  - Effects are on a per-unit log-odds scale.
  - Neutral and context-mismatch evidence are excluded.
- **Shaky.** The CI covers the consensus per-unit effect, not the cumulative WoE posterior, so `p_true` can lie outside [lo, hi].
- **Re-fit.** Set `meta_precision_scale` so that the 90% CI is calibrated on gold.

## 7. Information gain (`infogain.py`): F4
- **Model.** KL(post‖prior) in bits, for Bernoulli and categorical distributions. Leave-one-out contribution is `KL(P_all ‖ P_{−i})`.
- **Caveat.** This measures belief change, not truth.

## 8. Loopy BP (`bp.py`): F1
- **Model.** Pairwise binary MRF: `φ_i=[1-p_i, p_i]` and `ψ=exp(J s_i s_j)`, with `J = scale·sign·strength`. Messages are damped. The output includes a convergence flag. `exact_marginals` enumerates up to 12 nodes, and BP matches it exactly on a tree (tested).
- **Assumptions.**
  - Entailment is symmetrised: A⇒B is modelled as "agree".
  - Using p_true as the unary double-counts evidence shared between linked claims.
- **Failure modes.** Loopy BP may oscillate. If it does not converge, read p_true instead.

## 9. Centrality (`centrality.py`): F5
- **Graph.** PageRank (α = 0.85) runs on a graph with one node per origin cluster, one per claim c, and one per negation ¬c. Origin→c edges carry support strengths and origin→¬c edges carry refute strengths, with back-edges in both cases. The personalisation vector is reliability.
- **Output.** `centrality(c) = r(c)/(r(c)+r(¬c))`. It is 0.5 if neither node receives mass.
- **Assumption.** Collapsing to origins removes the amplification bias, which is the note's warning about centrality.

## 10. Tensor CP (`tensor_cp.py`): F9
- **Model.** Non-negative CP (tensorly HALS, SVD init, so deterministic) on counts of source × claim × time × type.
- **Per-component score:** `(1−H(a)/log S)·(1−H(c)/log T)·min(1, e^{H(a)}−1)`. The last factor gates out single-source components.
- **Coordination.** The coordination score is the maximum over components. A planted block of 4 sources × 2 claims × 1 time bin is recovered in the test.
- **Assumptions.** The rank is fixed at 4, and the score's components are heuristic. This is a pattern detector, not a probability.

## 11. Timeline noisy-OR (`event_bn.py`): F7, reduced
- **Model.** `P(inconsistent) = 1 − (1−leak)·Π(1 − w_i p_i)`, with assumed flag strengths w.
- **Assumption.** Detectors are independent causes.

## 12. Propaganda (`propaganda_score.py`): Part I (Da San Martino 2019)
- **Scores.**
  - `P_prop = 1 − Π(1 − c_i π_t)`, with π_t = 0.5 by default.
  - Density is spans per 100 tokens. Technique entropy is in bits.
  - `one_sidedness` is the mean of whichever are available: stance asymmetry `|S−C|/(S+C)` and the share of emotive techniques.
- **Span P/R/F1.** Exactly as in the paper: `C(s,t,h)=|s∩t|/h·δ`. Precision sums over S with h=|s|, and recall sums over T with h=|t|. Results are given with and without technique matching, with optional PTC 18→14 merging. Empty S gives P = 0, and empty T gives R = 0.
- **Shaky.**
  - P_prop saturates as the span count grows, because noisy-OR treats spans as independent.
  - Stance asymmetry measures evidence consensus, not one-sidedness of the document. It is kept as specified but is the weakest feature.
- **Re-fit.** Set π_t to the tagger's per-technique precision on held-out PTC data.

## 13. Propagation (`propagation.py`)
- The EDA dict maps to features with NaN for anything missing. Share-type features are clipped to [0, 1].
- `propagation_risk` is the mean of the available risk terms.

## 14. Features and roll-up (`features.py`)
`claim_features` follows these steps:
1. Assign reliability if missing.
2. Recompute independence within the claim.
3. Compute WoE, meta, DS, n_eff, info gain, centrality, KB, propaganda, timeline, propagation and the independence-weighted mean of source reliability.
4. Return exactly `FEATURE_NAMES` in order.

Conventions:
- `spans=None` gives NaN (the tagger did not run). `[]` gives 0.
- KB matching counts only debunk-labelled or unlabelled hits, with `kb_match = [best ≥ 0.80]`.
- Without a `graph`, `bp_marginal = p_true`.
- `story_graph` computes BP and centrality at document or story level.

`doc_rollup` applies these rules in order to check-worthy claims:
1. FAKE if any claim is FAKE with P(FAKE) ≥ 0.5.
2. Else MISLEADING if any claim is FAKE or MISLEADING.
3. Else PROPAGANDA if any claim is PROPAGANDA or the document's p_prop ≥ the threshold.
4. Else LEGIT if all claims are LEGIT.
5. Else INCONCLUSIVE.

Document probabilities are descriptive: noisy-OR for FAKE, MISLEADING and PROPAGANDA, a product for LEGIT, and a mean for INCONCLUSIVE.

## 15. Verdict (`verdict.py`, `conformal.py`)
- **Rule head (soft).** Each threshold becomes `σ((x−t)/τ)`. AND is a product, OR is noisy-OR. Missing features are filled conservatively.
  - FAKE = (p_hi ≤ 0.2 ∧ contradiction ≥ 0.5) ∨ kb_match.
  - MISLEADING = (cm ≥ 0.3 ∧ cm ≥ max(sup, con)) ∨ mixed.
  - PROPAGANDA = p_prop ≥ 0.6 ∧ (one-sided ∨ coordinated) ∧ ¬FAKE.
  - LEGIT = p_lo ≥ 0.8 ∧ ¬high p_prop.
  - INCONCLUSIVE = gate·(1 − max(MISL, PROP, kb)) ∨ ¬any. The gate is ds_unc ≥ 0.6 ∨ n_eff < 1, and it does not veto labels that need no truth evidence.
  - Scores get a floor and are normalised. They are not calibrated.
- **Stacker.**
  - Inputs: median-imputed features plus missing indicators, log rule/Gemma/Laya probabilities, and block-missing flags (an absent block is filled as uniform).
  - Model: standardised multinomial LR (C = 1) on DECIDABLE, then temperature scaling on a 20% held-out split.
- **Conformal.** Split LAC or APS (optionally randomised with a seed), with `qhat` at the `ceil((n+1)(1−α))/n` quantile. Coverage of about 0.9 is tested.
  - The set decides the label: one label is used directly, {FAKE, PROPAGANDA} gives FAKE plus a PROPAGANDA tag, and anything else, including an empty set, gives INCONCLUSIVE.
  - plausibly_fake = label ≠ FAKE ∧ P(F)+P(M) ≥ 0.5. The tier is "likely" when it is ≥ 0.8.
- **`final_verdict`.** Uses stacker probabilities when available, else the rule head. Without a conformal model it takes the argmax.
- **Assumptions.** Calibration and coverage assume exchangeability between the calibration data and deployment. Coverage is marginal, not per class.

## 16. Calibration (`calibration.py`)
- Temperature scaling on log p.
- ECE with 15 bins on top-label confidence. Multiclass Brier.
- Reliability-diagram arrays.
- Wilson CI, required by the repository's reporting rule.
- Risk–coverage curve with AURC.

## Judged impractical (not built)
- **Full event-level BN with CPTs (F7).** Each story needs a hand-built DAG and CPTs. That is exactly the human-assumption injection the note warns against, and there is no data to fit them. Only the noisy-OR timeline slice is kept.
- **H1–H7 hypothesis agents (Part III/IV, P-V step 1).** These are specific to one investigation (NEET), need per-case hypothesis authoring, and do not generalise to claim-level verdicts. The hypothesis-free claim posterior replaces them.
- **Confidence formula and red-team penalties of P-V steps 11–12.** These are replaced by n_eff, the DS uncertainty and the meta CI, which do the same job with explicit definitions.
- **Tucker decomposition and HITS variants.** These are redundant with CP and PageRank at this scale.
