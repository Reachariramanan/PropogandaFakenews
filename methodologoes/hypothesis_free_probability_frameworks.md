# Deriving Probabilities from a Document Corpus — Without Predefined Hypotheses

**A research-grounded review of ten mathematical frameworks for evidence-based belief estimation, with references and a recommended hybrid architecture.**

---

## Why move away from hypotheses

The working note this document expands on makes a sharp observation: the moment a human writes down hypotheses H1–H7, they inject assumptions, framing, and selection bias into the analysis before any evidence is touched. A purely mathematical alternative is to treat the corpus as a *system of claims, evidence, sources, and uncertainty*, and to let probabilities emerge from the **structure** of that system.

This is not a fringe idea. It is essentially what several mature research fields already do — truth discovery, evidence theory, probabilistic graphical models, scientific meta-analysis, and intelligence analysis each developed independent machinery for "given many noisy, overlapping, partially-dependent reports, what should I believe and how sure should I be?" The ten methods below are the real, citable foundations underneath the original note, with the mathematics made explicit and the canonical literature attached.

A recurring theme runs through all of them: **counting is the enemy.** Twenty-five articles that all trace to one CBI filing are one piece of evidence, not twenty-five. Every method that works does so largely because it has an explicit mechanism for *not* rewarding repetition. The methods that lack such a mechanism (naive centrality, naive voting) fail in exactly the way the original note warns about.

---

## Method 1 — Evidence Graph + Belief Propagation

**Core idea.** Represent the corpus as a graphical model: nodes are articles, claims, evidence units, entities, and events; edges encode `supports`, `contradicts`, `cites`, `causes`, `references`. Each node holds a belief, initialized (e.g.) at 0.5, and beliefs are refined by passing messages between neighbors until convergence.

**The actual algorithm.** This is **belief propagation (BP)**, also called the **sum–product algorithm**, introduced by Judea Pearl in 1982 and developed in *Probabilistic Reasoning in Intelligent Systems* (1988). BP computes the marginal distribution of each unobserved node conditional on the observed ones, by passing real-valued "messages" along edges. On a tree or polytree (singly connected graph) BP is **exact** and runs in linear time; the forward–backward algorithm for HMMs is a special case. On a graph **with cycles**, "loopy" BP becomes an approximate inference method — famously the engine behind near-Shannon-limit turbo-code and LDPC decoding — but it is not guaranteed to converge, and fixed points correspond to stationary points of the Bethe free energy rather than the exact marginals.

**Message form (sum–product on a factor graph).** A variable-to-factor message multiplies incoming messages; a factor-to-variable message sums the factor against incoming messages:

```text
μ_{x→f}(x) = ∏_{g ∈ N(x)\f} μ_{g→x}(x)
μ_{f→x}(x) = Σ_{~x} f(X) ∏_{y ∈ N(f)\x} μ_{y→f}(y)
belief(x) ∝ ∏_{f ∈ N(x)} μ_{f→x}(x)
```

**Why it fits.** No hypotheses are required — you read out a marginal belief per claim (`Claim A = 0.92`, `Claim B = 0.34`, …). It is closely related to Bayesian networks, Markov random fields, and (as a degenerate diffusion) PageRank.

**Caveats.** Exact marginalization and MAP inference are #P-complete and NP-hard respectively in general graphical models, so on a large loopy claim graph you are relying on an approximation that may oscillate. Edge weights (reliability, independence) must be set somehow — BP propagates beliefs, it does not invent the conditional probability tables.

---

## Method 2 — Dempster–Shafer (Evidence) Theory

**Core idea.** Instead of a single probability `P(claim) = 0.7`, store a triple that distinguishes *committed belief* from *room left for doubt*:

```text
Belief        = 0.65   (minimum support that is provably committed)
Plausibility  = 0.90   (maximum support not ruled out)
Uncertainty   = Pl − Bel = 0.25   (genuine ignorance)
```

**The formalism.** Introduced by Arthur Dempster (upper/lower probabilities, 1967) and developed by Glenn Shafer into *A Mathematical Theory of Evidence* (1976), DST assigns a **basic probability assignment (mass)** `m(A)` to *subsets* of the frame of discernment, not just singletons — this is what lets it represent "I don't know" explicitly. Belief and plausibility are derived as:

```text
Bel(A) = Σ_{B ⊆ A} m(B)
Pl(A)  = Σ_{B ∩ A ≠ ∅} m(B)
```

**Combining sources (Dempster's rule).** Two independent mass functions combine conjunctively, with the mass that lands on the empty set (the conflict `K`) stripped out by normalization:

```text
m₁₂(A) = (1 / (1 − K)) · Σ_{B ∩ C = A} m₁(B) m₂(C),   A ≠ ∅
K      = Σ_{B ∩ C = ∅} m₁(B) m₂(C)
```

**Why it fits journalism.** Many facts in a news corpus are genuinely *partially unknown*. DST refuses to manufacture a point probability where the evidence only supports an interval — much closer to how an honest analyst reasons. It is the standard tool for **sensor fusion** for the same reason.

**Caveats.** Dempster's rule is notoriously counterintuitive under **high conflict** (Zadeh's classic critique: two near-certain but disagreeing sources can produce a combined belief concentrated on a third option nobody supported). The normalization by `1 − K` "throws away" conflict. Numerous repair rules exist (Yager, Dubois–Prade, Smets' Transferable Belief Model), but choosing among them is itself a modeling decision, and computation is exponential in the size of the frame.

---

## Method 3 — Log-Odds (Weight-of-Evidence) Accumulation

**Core idea.** Work in log-odds, where independent evidence simply **adds**. Each evidence unit contributes a positive (support) or negative (contradiction) weight; sum them; squash back to a probability with the logistic function.

```text
L = log( P(H) / P(¬H) )  +  Σᵢ wᵢ          (wᵢ = log-likelihood ratio of evidence i)
P = 1 / (1 + e^(−L))                        (logistic / sigmoid)
```

**Provenance.** This is the **weight of evidence** of I. J. Good (building on Turing's wartime "ban"/"deciban" log-likelihood units), and it is the rigorous form of Bayes' rule for sequential, conditionally-independent evidence: the posterior log-odds equals the prior log-odds plus the summed log-likelihood ratios. In the intelligence community it underlies **Analysis of Competing Hypotheses (ACH)** — Richards Heuer, *Psychology of Intelligence Analysis* (1999) — and its Bayesian/CACHE variants, which score each evidence item by its **diagnosticity** (how strongly it discriminates between hypotheses), credibility, and relevance, and deliberately seek *disconfirming* evidence to fight confirmation bias.

**Why it fits.** It is exactly the logistic-of-net-evidence machinery sketched in the original methodology note (Document 5). It is stable, scales to thousands of items, and bounds output to (0, 1) so you never get "127%."

**Caveats.** Additivity of log-likelihood ratios is **only valid under conditional independence**. Correlated or copied evidence must be down-weighted first, or the posterior becomes wildly overconfident — which is precisely the failure mode the deduplication and independence steps are meant to prevent.

---

## Method 4 — Information Theory (Information Gain per Article)

**Core idea.** Stop asking "what is the probability" and ask "how much did this article *change* my belief." The change is measured in bits.

**The measure — Bayesian surprise.** The information gain of an observation is the **Kullback–Leibler divergence between the posterior and the prior** (Itti & Baldi's "Bayesian surprise"):

```text
InfoGain(E) = D_KL( P(·|E) ‖ P(·) ) = Σ P(x|E) · log( P(x|E) / P(x) )
```

It is non-negative, zero exactly when the evidence changes nothing, and — like entropy — **additive** over independent observations. Each flip of new information contributes a measurable number of bits of evidence (the log-likelihood-ratio interpretation connects this directly to Method 3).

**Why it fits.** Rank articles by *actual contribution*. A piece that restates yesterday's wire copy yields ≈ 0.01 bits; a newly surfaced court document yields several bits. In a typical news corpus, most articles end up contributing near-zero because they re-report earlier information — the metric makes redundancy visible and quantifiable, and gives a principled stopping/triage rule for collection.

**Caveats.** Information gain measures *belief change*, not *truth*. A confident, well-written falsehood from a fresh source can carry high surprise. Use it to prioritize and to detect redundancy, not as a truth signal on its own.

---

## Method 5 — Evidence Centrality (PageRank / HITS / Eigenvector)

**Core idea.** Build a graph of evidence → claims → articles and let importance flow through it: claims repeatedly supported by high-quality evidence accumulate high scores via **eigenvector centrality**, **PageRank** (Page & Brin, 1998), or **HITS hubs-and-authorities** (Kleinberg, 1999).

```text
score = principal eigenvector of the (weighted) support matrix
# PageRank form:  r = (1−d)/N · 1 + d · Mᵀ r
```

**The truth-discovery connection.** This is not a loose analogy. The **Sums** truth-discovery algorithm *is* hubs-and-authorities applied to a source–claim bipartite graph: a source is trustworthy if it asserts believable claims; a claim is believable if trustworthy sources assert it. That mutual-reinforcement fixed point is computed exactly as an eigenvector iteration.

**Why it fits.** It naturally elevates claims anchored to authoritative, well-connected evidence and demotes isolated ones, with no hypotheses required.

**Caveats.** Raw centrality rewards **connectivity and repetition**, which in a media corpus correlates with *amplification*, not truth — the single biggest trap the original note flags. Sums is known to overestimate sources with large coverage. Centrality is only safe when the edges already encode source independence (so that 25 copies of one filing collapse to one high-weight edge, not 25 votes).

---

## Method 6 — Probabilistic Truth Discovery

**Core idea.** Assume *both* the truth and each source's reliability are unknown, and estimate them **simultaneously** by iteration until convergence:

```text
1. Fix source reliabilities → estimate truth of each claim
2. Fix claim truths        → estimate reliability of each source
3. Repeat to convergence
```

**Provenance.** The seminal method is **TruthFinder** (Yin, Han & Yu, 2007/2008), the first to apply Bayesian analysis to co-estimate source trustworthiness and claim confidence, incorporating similarity between related claims. A rich family followed: **Sums, Average-Log, Investment, PooledInvestment** (Pasternack & Roth, 2010), and the **Accu/AccuCopy** models (Dong et al., 2009) which add explicit **copy detection** — directly modeling the "everyone copied one source" problem. Confidence-aware variants (Li et al., 2015) prevent tiny sources from dominating. The principle, in one line: *a source is trustworthy if it provides many true claims, and a claim is likely true if trustworthy sources provide it.*

**Why it fits.** This is arguably the closest off-the-shelf match to the NEET problem: heterogeneous sources, conflicting values, unknown reliabilities, heavy copying. It is the standard machinery for web data fusion, sensor fusion, and crowdsourced-label aggregation.

**Caveats.** Most variants assume sources are independent unless copy-detection is bolted on; results depend on the (often unsupervised) initialization and the similarity function; and "the truth" it converges to is really "the reliability-weighted consensus," which can still be wrong if the whole corpus shares a bias.

---

## Method 7 — Bayesian Evidence Network (event-level DAG)

**Core idea.** Drop hypotheses as labels and instead model **events** as nodes in a directed acyclic graph, with conditional probability tables on the edges:

```text
Question Paper Accessed → Question Distributed → Candidates Benefited → Arrests Made → Court Findings
```

Observing some nodes (arrests, court findings) yields posterior distributions over the unobserved ones (extent of distribution, number benefited) by standard Bayesian-network inference (exact junction-tree, or BP from Method 1).

**Provenance.** This is the principled successor to ACH: structuring competing explanations as a **Bayesian network** gives a clear probabilistic characterization of uncertainty, makes assumptions explicit as nodes, and supports "what-if" sensitivity analysis that the original tabular ACH matrix cannot (e.g., the Bayesian-network ACH formulation of Heuer's method in the intelligence-analysis literature).

**Why it fits.** Highly interpretable, causally structured, and supports counterfactual queries ("if the Hazaribagh link were false, what happens to the posterior on systemic breach?").

**Caveats.** You still have to specify the graph topology and the CPTs — a softer form of the same human-assumption injection the note wanted to avoid. Garbage CPTs produce confident garbage.

---

## Method 8 — Latent-Variable / Hidden-Truth Model

**Core idea.** Posit a hidden variable `T` = underlying reality; each article is a noisy observation `Article_i = T + Noise_i`. Estimate `T` and the noise/reliability parameters jointly.

**Provenance.** The estimation engine is the **EM algorithm** (Dempster, Laird & Rubin, 1977); relatives include factor analysis and variational inference. For exactly this "unknown truth + unknown annotator reliability" setting, the canonical model is **Dawid–Skene** (1979), which estimates latent true labels and per-source confusion matrices from multiple unreliable observers — the statistical backbone of modern crowdsourcing aggregation. Truth discovery (Method 6) can be seen as a member of this family.

**Why it fits.** It cleanly separates *signal* (T) from *source noise*, and reliability falls out as an estimated parameter rather than a hand-set constant. Statistically elegant and well-understood.

**Caveats.** EM converges only to a local optimum and is sensitive to initialization; identifiability requires enough overlapping observations per source; and the additive-noise framing is a strong assumption for categorical claims (mitigated by the confusion-matrix formulation).

---

## Method 9 — Evidence Tensor Decomposition

**Core idea.** For large investigations, arrange the evidence as a multi-way array and factor it:

```text
Tensor T  ∈  Source × Claim × Time × Evidence-Type
            (e.g. 100 × 200 × 50 × 10)
```

**The machinery.** Apply **CP/PARAFAC** (a sum of rank-one tensors) or **Tucker** decomposition (a small core tensor times a factor matrix per mode — higher-order PCA). The standard reference is Kolda & Bader, *Tensor Decompositions and Applications*, SIAM Review (2009). The latent factors expose hidden structure automatically: clusters of sources that move together, claims that co-activate, temporal bursts.

```text
CP:     T ≈ Σ_r  a_r ∘ b_r ∘ c_r ∘ d_r
Tucker: T ≈ G ×₁ A ×₂ B ×₃ C ×₄ D
```

**Why it fits.** At scale it surfaces **coordinated behavior** — e.g., a block of outlets publishing the same claim in the same 24-hour window — which is exactly the signature of amplification rather than independent corroboration. Widely used in fraud analytics and intelligence.

**Caveats.** Choosing the rank is hard and somewhat subjective; factors are correlational, not causal; CP can be numerically unstable and Tucker has rotational ambiguity. It is a *pattern-discovery* layer, not a probability estimator — pair it with Methods 3/6 to turn discovered structure into belief.

---

## Method 10 — Scientific Meta-Analysis Style

**Core idea.** Treat every *independent* evidence unit like a study in a systematic review. Each contributes an effect size, a reliability, and a variance; pool them with **inverse-variance weighting** so precise evidence dominates.

```text
weight_i  = 1 / variance_i           (Cochrane inverse-variance method)
pooled    = ( Σ wᵢ · yᵢ ) / ( Σ wᵢ )
SE(pooled)= sqrt( 1 / Σ wᵢ )
95% CI    = pooled ± 1.96 · SE(pooled)
```

**Fixed vs random effects.** A **fixed-effect** model assumes all evidence estimates a single common truth (weights = within-study precision only). A **random-effects** model (DerSimonian & Laird, 1986) allows the true effect to vary across sources, adds the between-source variance `τ²` to each weight, and yields wider, more conservative intervals — usually the honest choice when sources are heterogeneous. Heterogeneity is quantified with Cochran's `Q` and `I²`; publication/selection bias is checked with funnel plots. A practical substitution for a news corpus: use `weight ≈ reliability × independence` as the precision proxy when a formal variance is unavailable.

**Why it fits.** It is the most battle-tested framework in all of empirical science for "combine many imperfect studies into one calibrated estimate with an interval," and it produces a **confidence interval**, not a false point estimate.

**Caveats.** Meta-analysis assumes the pooled units are independent — so, once again, deduplication and independence scoring are doing the heavy lifting. "Garbage in, garbage out" applies: pooling many low-quality, correlated reports yields a narrow, confident, and wrong interval.

---

## A recommended hybrid for a NEET-style investigation engine

No single method is sufficient; each contributes a different organ. A defensible stack layers four complementary mechanisms and adds two cross-cutting filters:

```text
Articles
   ↓                                   ┌─ Information Gain (Method 4): rank by bits added,
Claim extraction                       │   discard near-zero-information duplicates
   ↓                                   │
Independent Evidence Units  ───────────┤   (deduplication + independence scoring is the
   ↓                                   │    load-bearing step for everything below)
Probabilistic Truth Discovery (M6) ────┘   co-estimate source reliability ↔ claim belief
   ↓
Evidence Graph + Centrality (M5/M1)        propagate belief; centrality ONLY on the
   ↓                                       independence-corrected graph
Dempster–Shafer (M2)                       carry Belief / Plausibility / Uncertainty
   ↓                                       intervals instead of point masses
Bayesian / Log-Odds Update (M3/M7)         fuse into a posterior in log-odds space
   ↓
Meta-analytic pooling (M10)                attach a calibrated confidence interval
   ↓
Final Probability Distribution
```

The decisive design constraint is unchanged from the original notes and is *more* important than the choice of method: **no quantity may be counted twice.** Truth discovery, log-odds additivity, centrality, and meta-analytic pooling all silently assume independence; the deduplication and independence layers are what make every downstream number trustworthy.

### What this produces instead of a single number

A hypothesis-free pipeline should report a **vector**, not a scalar — which is far more honest and more useful to a decision-maker:

```text
Support            = 0.92      (committed belief, Method 2/3)
Contradiction      = 0.08      (mass on the negation)
Uncertainty        = 0.14      (Pl − Bel, genuine ignorance, Method 2)
Information Gain    = 4.7 bits  (how much the corpus actually moved belief, Method 4)
Evidence Centrality = 0.88      (anchoring in authoritative evidence, Method 5)
Confidence          = 0.81      (precision / interval width, Method 10)
```

---

## What the mathematics can and cannot do

Honesty about limits matters more than the elegance of any single estimator:

- **None of these establish ground truth.** Every method returns a *reliability-weighted, dependence-corrected consensus*. If the entire reachable corpus shares a blind spot or a bias, the consensus inherits it, no matter how sophisticated the math.
- **Independence is the whole game.** The methods differ mostly in how they represent belief and uncertainty; they agree completely that double-counting correlated evidence is fatal. A mediocre estimator with excellent deduplication beats an elegant estimator with naive counting.
- **Extraction errors propagate.** If claim extraction or evidence classification mislabels inputs (very possible when an LLM does the reading), every probability downstream is confidently wrong. The structured layers reduce *reasoning* errors, not *perception* errors.
- **Calibration ≠ confidence.** A 90% output is only meaningful if, across many such judgments, ~90% turn out correct. These frameworks give *internally coherent* numbers; external calibration requires backtesting against known outcomes.
- **The parameters still encode assumptions.** Reliability weights, graph topology, CPTs, priors, and combination rules are all human choices. The note's goal of "no human assumptions" is unreachable in full — but these methods make the remaining assumptions *explicit, inspectable, and adjustable*, which is the real win.

---

## Comparison table

| #  | Method                 | Core mechanism                             |   Models uncertainty?   |     Handles dependence (natively)?     | Needs hypotheses? | Best for                                                  |
| -- | ---------------------- | ------------------------------------------ | :----------------------: | :------------------------------------: | :---------------: | --------------------------------------------------------- |
| 1  | Belief Propagation     | Message passing on a graphical model       |        Marginals        |          Via graph structure          |        No        | Reading per-claim beliefs from a claim/evidence graph     |
| 2  | Dempster–Shafer       | Mass → Belief/Plausibility intervals      | **Yes (explicit)** | Combination rule (assumes independent) |        No        | Representing genuine "unknown" in news facts              |
| 3  | Log-Odds / WoE         | Additive log-likelihood ratios → logistic |         Indirect         |  **No** (needs pre-correction)  |     Optional     | Stable, scalable fusion of independent evidence           |
| 4  | Information Gain       | KL(posterior ‖ prior) in bits             |        As entropy        |                   No                   |        No        | Ranking articles by real contribution; killing redundancy |
| 5  | Centrality / PageRank  | Eigenvector on support graph               |            No            |   **No** (rewards repetition)   |        No        | Surfacing well-anchored claims (only after independence)  |
| 6  | Truth Discovery        | Iterate source ⇄ claim reliability        |      Some variants      |        Copy-detection variants        |        No        | Heterogeneous, conflicting, copied web sources            |
| 7  | Bayesian Event Network | DAG of events + CPTs + inference           |      **Yes**      |             **Yes**             |   Soft (events)   | Interpretable causal structure + sensitivity analysis     |
| 8  | Latent-Variable / EM   | Hidden truth + estimated source noise      |      **Yes**      |         Via confusion matrices         |        No        | Separating signal from per-source noise (Dawid–Skene)    |
| 9  | Tensor Decomposition   | CP/Tucker on multi-way array               |            No            |        Surfaces it as structure        |        No        | Large corpora; detecting coordinated amplification        |
| 10 | Meta-Analysis          | Inverse-variance pooling + CI              |    **Yes (CI)**    |          Assumes independence          |        No        | Calibrated pooled estimate with an interval               |

---

## References

**Graphical models & belief propagation**

- Pearl, J. (1982). *Reverend Bayes on Inference Engines: A Distributed Hierarchical Approach.* AAAI.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems.* Morgan Kaufmann.
- Yedidia, J., Freeman, W., Weiss, Y. (2003). *Understanding Belief Propagation and Its Generalizations.*

**Evidence theory**

- Dempster, A. P. (1967). *Upper and Lower Probabilities Induced by a Multivalued Mapping.* Ann. Math. Stat.
- Shafer, G. (1976). *A Mathematical Theory of Evidence.* Princeton Univ. Press.
- Smets, P. (1994). *The Transferable Belief Model.* Artificial Intelligence.

**Weight of evidence & intelligence analysis**

- Good, I. J. (1950). *Probability and the Weighing of Evidence.*
- Heuer, R. J. (1999). *Psychology of Intelligence Analysis* (Analysis of Competing Hypotheses). CIA Center for the Study of Intelligence.
- Bayesian-network formulations of ACH, *EURO Journal on Decision Processes* (2013).

**Information theory**

- Shannon, C. E. (1948). *A Mathematical Theory of Communication.*
- Itti, L., Baldi, P. (2009). *Bayesian Surprise Attracts Human Attention.* (KL prior↔posterior as surprise.)

**Centrality**

- Page, L., Brin, S., Motwani, R., Winograd, T. (1998). *The PageRank Citation Ranking.*
- Kleinberg, J. (1999). *Authoritative Sources in a Hyperlinked Environment* (HITS).

**Truth discovery / data fusion**

- Yin, X., Han, J., Yu, P. S. (2007/2008). *Truth Discovery with Multiple Conflicting Information Providers on the Web* (TruthFinder).
- Dong, X. L., Berti-Équille, L., Srivastava, D. (2009). *Integrating Conflicting Data: The Role of Source Dependence* (Accu/AccuCopy).
- Pasternack, J., Roth, D. (2010). *Knowing What to Believe* (Investment, Sums, Average-Log, PooledInvestment).
- Li, Y. et al. (2016). *A Survey on Truth Discovery.* SIGKDD Explorations.

**Latent-variable estimation**

- Dempster, A., Laird, N., Rubin, D. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm.* JRSS-B.
- Dawid, A. P., Skene, A. M. (1979). *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.* JRSS-C.

**Tensor methods**

- Kolda, T. G., Bader, B. W. (2009). *Tensor Decompositions and Applications.* SIAM Review 51(3): 455–500.

**Meta-analysis**

- DerSimonian, R., Laird, N. (1986). *Meta-Analysis in Clinical Trials* (random-effects model).
- Borenstein, M., Hedges, L., Higgins, J., Rothstein, H. (2009). *Introduction to Meta-Analysis.*
- Cochrane Handbook, Ch. 10: *Analysing Data and Undertaking Meta-Analyses* (inverse-variance method).

*Reference list compiled from the canonical works underlying each method; consult the originals for exact formulations and proofs.*
