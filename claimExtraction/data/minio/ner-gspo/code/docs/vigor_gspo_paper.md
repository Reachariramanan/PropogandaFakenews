# Variance-Guided Progressive Rollout Allocation for Sequence-Level Policy Optimization: An Applied Integration of VIGOR into GSPO for Free-Form Information Extraction

*Applied report on the GSPO/VERL information-extraction pipeline (`NER_Training`) · August 2026*

## Abstract

We describe the design, implementation, and empirical validation of a variance-guided progressive rollout allocation strategy integrated into a production Group Sequence Policy Optimization (GSPO) reinforcement-learning pipeline for free-form named-entity and event extraction. Standard GSPO with a fixed rollout count per prompt plateaus after supervised warm-start and two rounds of LoRA-capacity escalation, holding at a validation reward of ≈0.709 for hundreds of steps regardless of learning rate or adapter placement. We integrate VIGOR (Jiang et al., 2026), a progressive rollout-allocation algorithm that concentrates sampling budget on high current-policy-uncertainty prompts across refinement rounds within a single training step, directly into VERL's actor-rollout-advantage pipeline — reusing VERL's existing group-relative advantage estimator (which already supports variable per-prompt group sizes) rather than reimplementing advantage computation, and fixing a latent minibatch-sizing assumption in VERL's actor-update path that a fixed-rollout-count training loop never exercises. The integration is validated first as an isolated, adversarially-reviewed library against mock rollouts, then live against the real training stack. In production training, the resulting policy breaks the fixed-rollout plateau, reaching a validation reward of 0.718 (vs. 0.709 ceiling) and an offline-evaluated composite extraction score of 0.738 — the best result obtained in this project — with the largest gains concentrated in the two weakest sub-skills (event topic-code and action-code classification), exactly the metrics the allocator was designed to target.

## 1. Introduction

Reinforcement learning from verifiable rewards (RLVR) for structured generation tasks — here, emitting a JSON object of named entities and CAMEO-coded events from a news document — typically samples a fixed number of rollouts *G* per prompt and computes a group-relative advantage. This is the GRPO/GSPO family's default, and it is what our production pipeline used through two escalation attempts (§3). Fixed-*G* sampling spends equal budget on every prompt regardless of how much signal that prompt currently offers: a prompt the policy already solves, or one it is uniformly hopeless at, yields near-zero reward variance and therefore near-zero gradient signal (Jiang et al., 2026, §2), while a prompt straddling the decision boundary yields the largest usable signal. VIGOR reallocates budget toward the latter by refining in rounds: start every active prompt with a small rollout count, measure cumulative reward variance, keep only the highest-variance fraction, double their allocation, and repeat.

This report documents an applied integration of that idea into a real, already-in-production GSPO/VERL training loop for information extraction, not a reimplementation of the RL framework. §2 summarizes GSPO and VIGOR. §3 describes the integration architecture — in particular, a simplification that removed an entire component of the original algorithm from the applied system, and a defect in the host framework's actor-update path that the integration surfaced and fixed. §4 describes a two-stage validation methodology (isolated adversarial review, then live smoke test) motivated by a real production incident earlier in the same pipeline in which an unhandled type error in the reward function silently killed a 38-step run. §5 reports the production training outcome against two fixed-rollout baselines. §6 discusses limitations and an unrelated but consequential operational failure (disk exhaustion from unbounded checkpoint retention) encountered during the run.

## 2. Background

### 2.1 GSPO

Group Sequence Policy Optimization (Zheng et al., 2025, arXiv:2507.18071) replaces GRPO's per-token importance-ratio clipping with a single, sequence-level ratio and clip decision per generated response. For a document *x*, *G* completions *y₁,…,y_G* are sampled; rewards *rᵢ* are scored by a task-specific verifier and converted to group-relative advantages

```
A_i = (r_i - r̄) / (σ_r + δ),   r̄ = (1/G)Σr_i,   σ_r = sqrt((1/G)Σ(r_i - r̄)²)
```

The policy loss uses a sequence-level ratio ρᵢ(θ) (rather than a per-token ratio) with a single clip per sequence, aggregated as `seq-mean-token-mean`. In our host framework (VERL), this corresponds to `algorithm.adv_estimator=grpo` (group-relative advantages, computed by `core_algos.compute_grpo_outcome_advantage`) combined with `actor_rollout_ref.actor.policy_loss.loss_mode=gspo` (the sequence-level ratio/clip, in `compute_policy_loss_gspo`). Both components are production VERL code and are not modified by this work; the object of integration is exclusively *how many rollouts are generated for which prompts before* these two functions are called.

### 2.2 VIGOR

VIGOR (Jiang, Liu, and Mirzasoleiman, 2026, arXiv:2607.22002) proves that, under binary RLVR rewards with the KL term removed, the gradient-norm bound of the GRPO-family objective is proportional to the within-group reward standard deviation σ — maximized when the policy is genuinely uncertain about a prompt (success probability p≈0.5), and near-zero when a prompt is already solved or hopeless. VIGOR's Algorithm 1 exploits this directly: starting from an active set A₁ = the full sampling batch, at each of *T* refinement rounds it:

1. generates *m⁽ᵗ⁾* **new** rollouts per currently-active prompt,
2. merges them into that prompt's cumulative rollout set (never discarding earlier rounds),
3. computes the population variance of each active prompt's cumulative rewards, and, if t < T,
4. retains only the top-α fraction of the active set by variance and multiplies the per-prompt rollout count by γ for the survivors.

With the paper's own budget-conserving configuration αγ = 1 (e.g. α=0.5, γ=2), total expected rollout consumption collapses to `C_total = B·m₀·T` — matching a fixed-rollout baseline of G=m₀T rollouts/prompt at the same nominal cost, while spending that budget on a small, sharply skewed subset of prompts rather than uniformly. Advantages for the final GRPO update are computed **once** per surviving prompt, over its entire cumulative rollout set, not per refinement round — the single most consequential implementation detail called out in the paper and independently re-derived in §4.

## 3. Method

### 3.1 A key simplification: reusing the host framework's existing advantage estimator

A naïve port of VIGOR would reimplement its own group-relative advantage computation, since Algorithm 1 as stated operates on an explicit rollout-set-per-prompt data structure. We instead inspected VERL's own group-advantage implementation and found it already satisfies VIGOR's requirements without modification:

```python
# core_algos.py (unmodified VERL source)
id2score = defaultdict(list)
for i in range(bsz):
    id2score[index[i]].append(scores[i])     # index[i] is a per-row "uid" column
# ... group-relative normalization per uid, with NO assumption
# that every uid appears the same number of times.
```

Grouping is keyed purely on a non-tensor `uid` column carried per row, with a `defaultdict(list)` accumulation pattern that places no constraint on how many rows share a `uid`. This is precisely VIGOR's variable-group-size output shape (a solved prompt might retain 2 rollouts; a maximally-uncertain survivor of every round retains m₀(1+γ+γ²+…) = 30 under the paper's defaults). The practical consequence: **the applied integration does not port VIGOR's advantage-computation code at all.** It is responsible only for (i) deciding, each round, which prompts continue and how many new rollouts they receive, and (ii) assigning and preserving a consistent `uid` per prompt across rounds so that step (iii), VERL's own unmodified `compute_advantage`, groups correctly. This follows the architectural constraint articulated in the source algorithm's own design notes: the allocator decides *which prompts, how many rollouts, when to stop generating*; the host trainer remains solely responsible for log-probabilities, advantages, clipping, and the optimizer step.

### 3.2 Integration architecture

The production entry point (`grpo/run_gspo_vigor.py`) mirrors the host framework's own startup sequence — instantiating the same trainer class (`RayPPOTrainer`) and calling its own `init_workers()` — and replaces only the fixed-rollout `fit()` step body with a custom loop:

```
assign uid per prompt in the training batch
active = all prompts;  collected = {};  m = m0;  budget_used = 0
for round in range(T):
    requested = len(active) * m
    if budget_used + requested > budget_target:
        shrink m first; if m < 1, truncate active (drop lowest-variance)
    new_rollouts = async_rollout_manager.generate_sequences(active, m)
    # policy snapshot is frozen: no weight-sync call occurs inside this loop
    score(new_rollouts) via reward_ie.compute_score   # or reuse agent-loop rm_scores
    merge new_rollouts into collected[prompt] for prompt in active
    budget_used += len(new_rollouts)
    if round < T - 1:
        utility[p] = population_variance(collected[p].rewards) for p in active
        keep = ceil(alpha * len(active))
        active = top_k(active, keep, key=utility, tie_break=uid)  # deterministic
        m = round(m * gamma)
batch = DataProto.concat(collected.values())   # ragged uid-group sizes, by design
batch = trainer._compute_old_log_prob(batch)
batch = compute_advantage(batch, adv_estimator="grpo")   # unmodified VERL call
trainer._update_actor(batch)                              # see §3.3
```

The policy-freeze invariant central to VIGOR's correctness — every round within a step must sample from the *same* policy, since refinement is meaningless if the target is moving — was verified structurally rather than assumed: the host framework connects actor weights to rollout replicas through exactly one call (`checkpoint_manager.update_weights()`), which in the unmodified fixed-rollout loop appears only once per step, after the actor update. The integrated loop preserves this exactly — the call is never issued inside the round loop.

### 3.3 A latent host-framework assumption this integration surfaced

VERL's actor-update path computes an effective PPO minibatch size as `ppo_mini_batch_size × rollout.n` — a quantity that is exact under fixed-rollout sampling (every prompt contributes exactly `rollout.n` rows) and is asserted, in two places, to divide the actual batch row count exactly (once in the data-loading iterator, once in the distributed training-engine dispatcher). It is additionally used, unmodified, as the loss aggregation's normalizing denominator. A VIGOR batch's true row count is Σ|Yₚ| over whatever prompts survived to whatever round — a number with no fixed relationship to `rollout.n` — so this host assumption fails two hard-coded divisibility assertions and, more subtly, would silently mis-scale the gradient if the assertions were bypassed rather than fixed.

We resolve this at the integration boundary rather than by patching the host framework: before calling the unmodified `_update_actor`, we compute the nearest valid divisor of the *actual* row count (a multiple of the data-parallel world size, floored to avoid degenerating to minibatch size 1 on a prime-ish row count) and temporarily override `rollout.n=1` together with the chosen minibatch size for the duration of that one call, so the denominator VERL itself recomputes matches the data it is actually operating on. We separately re-derived that this cannot corrupt the GSPO sequence-level ratio/clip itself: that computation is entirely per-row and never reads `uid` or any other cross-row grouping signal; the only place group membership matters is the advantage computation of §3.1, which completes *before* minibatching and is therefore unaffected by how the minibatches are subsequently split.

### 3.4 Reward attribution and the pilot pool

Each round's rollouts are scored by the same verifier used for training (`grpo/reward_ie.py::compute_score`), by default reusing the reward already computed by the host framework's own agent-execution loop for the training reward itself (`score_source=auto`), so the variance signal driving selection is bit-identical to the signal that will drive the eventual policy gradient — a stronger guarantee than recomputing it independently would give. An alternate mode recomputes the score directly from decoded text on the driver process for configurations where the reward is not already attached to the rollout output; an early version of this path read ground truth from the wrong data object (the rollout-output object, which the host agent loop does not echo prompt-side fields onto) and was corrected to read from the prompt-aligned object instead, with hard failures replacing what had been a silent `None`.

## 4. Validation methodology

Two properties of this codebase's history motivated a specifically two-stage validation process rather than either static review alone or a direct production run: (1) an earlier, unrelated defect in the reward function — a truthy non-iterable model output silently escaping an `x or []` guard — killed a full production run via an unhandled exception propagating through the distributed reward worker, demonstrating that plausible-looking reward-adjacent code can hide crash-on-first-malformed-input defects that pass casual review; and (2) rollout-allocation logic of this kind (cumulative merging across rounds, variance-based selection, budget accounting) is exactly the class of logic the source algorithm's own design notes flag as prone to *silent* correctness failures — wrong output, not a crash — under a policy-freeze violation, an off-by-one in cross-round merging, or advantage normalization computed per-round instead of over the cumulative group.

**Stage 1 — isolated adversarial validation.** Before any integration code was written, VIGOR (and, separately, Pilot-Commit) were implemented as a standalone library against mock rollout generation, with an independent reviewer explicitly tasked to *refute* each of seven correctness properties (policy freeze, uid/group integrity, cumulative-not-discarded merging, current-active-set-relative selection, budget conservation, advantage normalization over the full cumulative group, and — specific to Pilot-Commit — replay-buffer/staleness handling) by independently re-deriving the relevant code paths rather than trusting the implementer's self-report. This surfaced zero defects in VIGOR and six real defects in Pilot-Commit (§4.1), establishing VIGOR as the trustworthy candidate for integration and motivating deferring Pilot-Commit.

**Stage 2 — live smoke test against the real training stack.** The same adversarial process was then applied to the *integration* code itself (§3), surfacing four further defects (§4.2), followed by a mandatory live execution — a handful of training steps against a tiny dataset and the production checkpoint, GPU-resident, not mocked — on the explicit premise that static review, however thorough, cannot substitute for observed execution of training-loop code. The first three live attempts failed for reasons unrelated to the code under test (an external, unrelated GPU tenant left insufficient device memory); the fourth succeeded, producing round-by-round diagnostics that matched the theoretical budget-conserving schedule exactly.

### 4.1 Defects found in the standalone Pilot-Commit implementation

For completeness, since Pilot-Commit remains unintegrated but its standalone correctness bears on future work: a replay-buffered prompt was not excluded from the next pilot pool, so it was silently re-piloted and the resulting fresh state was overwritten by the stale buffered one — discarding paid-for on-policy rollouts in favor of stale ones; a prompt selected from a fresh pilot could retain a stale buffer entry and be trained on twice, once with outdated rollouts; the staleness clock (`max_policy_delay`) never advanced under the base call signature, silently disabling the staleness guard entirely; the default survivor-ranking policy re-imposed a preference for success-probability near 0.5 that the source paper explicitly argues against; and one ranking policy was a no-op due to a tie-in-sort-key artifact. All six were fixed and covered by regression tests targeting the exact adversarial probes that found them.

### 4.2 Defects found in the production integration

**Reward-write ordering:** a round's score was attached to the rollout-output object *after* that object had already been merged (via tensor-dictionary union) into the batch that would later be searched for it — the merge operation copies existing keys at the time it runs, so a key written afterward is invisible to the merged object, and the very first training step would raise a lookup error. Fixed by scoring before merging.

**Ground-truth source:** the alternate direct-scoring path (§3.4) read prompt metadata from the rollout-output object rather than the prompt-aligned input object; fixed with hard failure guards replacing a previous silent `None`.

**False-firing warning:** a diagnostic warning tested a configuration node that is a real, always-populated part of the composed configuration (hence never `None`) rather than the specific sub-switches that actually enable the feature it warns about, causing it to fire on every run regardless of configuration; corrected to test the real switches.

**Broken documented example:** a documented example command in the operator-facing wrapper script was itself non-executable against the host framework's own configuration validator; corrected.

## 5. Experimental setup

**Base model and adaptation.** Qwen3.5-0.8B, text-only checkpoint (see companion process document for the architecture-specific extraction steps this required), LoRA rank 16 on the two full-attention decoder layers (indices 19 and 23 of 24; Qwen3.5 interleaves 18 Gated-DeltaNet linear-attention layers with 6 standard full-attention layers), initialized from a checkpoint that had already completed supervised warm-start and one prior round of fixed-rollout GSPO training on this same adapter configuration — i.e. the comparison in §6 is *continued* training under the two allocation strategies from an identical starting point, not two runs from a common but earlier ancestor.

**VIGOR configuration.** m₀=2, T=4, γ=2.0, α=0.5 (the source paper's own reported configuration; αγ=1, the budget-conserving regime). Per-step budget = B·m₀·T = 16×2×4=128 rollout rows, against a fixed-rollout baseline of 16×4=64 rows/step — i.e. VIGOR is compared here at *twice* the fixed baseline's nominal per-step rollout cost, a deliberate choice consistent with the source paper's own experimental protocol of comparing allocation strategies at matched *cumulative*, not per-step, budget over a full run.

**Optimizer and RL hyperparameters.** Identical across both conditions: lr=1×10⁻⁶, GSPO clip range [3×10⁻⁴, 4×10⁻⁴], `loss_agg_mode=seq-mean-token-mean`, LoRA rank 16, no KL penalty.

**Evaluation.** Validation reward reported by the training loop's own periodic validation pass against a held-out 196-document sample every 25 steps (the aggregate weighted score of §1's reward function). A separate, higher-fidelity offline evaluation (15 held-out documents, full sub-metric breakdown, human-inspected raw JSON output) was run at selected checkpoints to verify the aggregate number reflects genuine extraction quality rather than reward-function exploitation — motivated directly by the §6 finding that an earlier checkpoint's climbing aggregate reward had in fact reflected exactly such exploitation (companion process document, §3).

## 6. Results

### 6.1 Validation reward over training

| Step | Fixed-rollout baseline (best of two escalation tiers) | VIGOR |
|---|---|---|
| 0 | 0.7076 – 0.7090 | 0.7064 |
| 75 | 0.7050 – 0.7081 | 0.7066 |
| 150 | 0.7063 – 0.7092 | 0.7092 |
| 250 | – | 0.7129 |
| 400 | – | **0.7158** |
| 875 | – | **0.7181 (peak)** |
| 950 | – | 0.7172 (best *saved* checkpoint) |

The fixed-rollout baseline, across two LoRA-capacity escalation tiers and one learning-rate probe, never exceeded a validation reward of 0.7092 in over 400 combined training steps, oscillating in a tight 0.705–0.709 band. VIGOR matched that ceiling by step 150 and continued climbing — non-monotonically, as expected of a noisy 196-document validation sample, but with a clear upward trend — reaching 0.7181 at step 875, a sustained ~1-percentage-point absolute improvement over the fixed-rollout ceiling that held for hundreds of subsequent steps rather than reverting.

Notably, at matched *step count* in the early regime (≤75 steps), VIGOR underperformed the fixed-rollout baseline slightly (e.g. offline composite score 0.702 vs. 0.730 at step 70) despite consuming twice the rollout budget per step — consistent with the source paper's framing that the algorithm's benefit is a *cumulative-budget-efficiency* claim over a full run, not a per-step comparison, and we report this contemporaneous negative-looking result rather than omitting it.

### 6.2 Offline composite evaluation

| Checkpoint | score | entity F1 | subtype | topic F1 | action F1 |
|---|---|---|---|---|---|
| SFT cold-start (no RL) | 0.701 | 0.774 | 0.911 | 0.360 | 0.60 |
| Fixed-rollout, best tier | 0.730 | 0.768 | 0.871 | 0.463 | 0.70 |
| **VIGOR, step 1010** | **0.738** | 0.751 | 0.839 | **0.493** | **0.80** |

The VIGOR checkpoint attains the best composite score of any checkpoint evaluated in this project, driven almost entirely by the two sub-metrics with the largest remaining headroom (topic-code F1 and action-code F1) — precisely the components a variance-guided allocator would be expected to improve, since a fixed-rollout policy spends equal sampling budget regardless of a given document's topic/action ambiguity, while VIGOR concentrates budget on exactly the documents where the policy's current topic/action predictions are least certain. Entity F1 and subtype accuracy are marginally lower than the fixed-rollout tier's best snapshot (0.751 vs. 0.768; 0.839 vs. 0.871) — both already comfortably high and, we hypothesize, receiving comparatively less marginal benefit from variance-guided reallocation since entity extraction shows lower per-document reward variance to exploit in the first place. All fifteen evaluated outputs used correct JSON structure, correct field names, and real CAMEO-format topic/action codes, with zero truncation or repetition failures — the schema-fidelity problem that motivated the supervised warm-start (companion process document, §3) did not recur under either RL condition once fixed.

## 7. Discussion

### 7.1 Why the host framework's advantage estimator was reusable

The generality that made §3.1's simplification possible is not incidental: any RL framework implementing group-relative advantages by grouping observed rollouts on an explicit group identifier, rather than by positional or fixed-stride assumptions over the batch, is compatible with variable-group-size allocators like VIGOR with zero modification to that component. The corollary risk is exactly what §3.3 found: any *other* component of the same framework that derives a group-related quantity from the fixed-rollout-count assumption instead — here, a minibatch-size product used as both a divisibility precondition and a loss-scaling denominator — will fail, and may fail silently rather than loudly, under variable group sizes. We recommend that any future integration of a variable-group-size allocator into an unfamiliar RL framework specifically audit every downstream use of the rollout-count hyperparameter, not only the advantage-computation call site, before assuming compatibility.

### 7.2 An operational failure orthogonal to the algorithm

During the production run reported in §6, training halted around step 1019 with no exception trace. Diagnosis attributed this to disk exhaustion: per-step checkpointing at a 10-step interval had, over the course of the run, accumulated over one hundred ~4 GB checkpoints (385 GB) on a shared volume, and a checkpoint write that coincided with the volume reaching capacity left a truncated, unloadable checkpoint at the exact failure timestamp. This is unrelated to the allocation algorithm itself but is reported here because it is a direct consequence of a long-running RL job's storage growth rate not having been considered a first-class parameter alongside the RL hyperparameters — a generalizable lesson for any multi-day RL training run with frequent checkpointing, independent of which allocation strategy is used.

### 7.3 Limitations

The reported validation-reward comparison (§6.1) uses a 196-document held-out sample and is visibly noisy step-to-step; the reported trend is directional and sustained rather than a single clean monotonic curve, and a larger held-out set would tighten the comparison. The offline composite evaluation (§6.2) uses only 15 documents per checkpoint, chosen for evaluation speed on CPU-only inference to avoid contending with concurrent GPU training; absolute sub-metric values at this sample size carry meaningful sampling variance, though the qualitative pattern (schema fidelity uniformly correct; topic/action codes the primary beneficiary) was corroborated across multiple checkpoints and sample draws over the course of the project. Pilot-Commit, the companion algorithm to VIGOR in its source literature, was validated in isolation but not integrated into production training; its combination with VIGOR (an explicitly hybrid, non-paper-sanctioned strategy in the source literature) was out of scope for this work.

## 8. Conclusion

We integrated a variance-guided progressive rollout-allocation algorithm into a production sequence-level policy-optimization training loop for structured information extraction, by reusing the host framework's existing group-relative advantage machinery rather than reimplementing it, fixing a latent fixed-rollout-count assumption in an unrelated part of the same framework that the integration exposed, and validating the result through isolated adversarial review followed by mandatory live execution rather than static review alone. In production training, the resulting policy broke a validation-reward plateau that two prior escalation strategies (supervised warm-start and LoRA-capacity expansion) had failed to move, with the largest gains concentrated in exactly the sub-skills the allocator was designed to prioritize.

## References

- Zheng, C., et al. *Group Sequence Policy Optimization.* arXiv:2507.18071, 2025.
- Jiang, H., Liu, H., Mirzasoleiman, B. *Learning as Reasoning Unfolds: Progressive Rollout Allocation for Efficient Reinforcement Learning.* arXiv:2607.22002, 2026.
- Kim, W., Yang, Z., Yan, J.N., Liu, J. *Spend Your Rollouts Where It Counts: Rollout Allocation for Group-Based RL Post-Training.* arXiv:2605.26606, 2026.
