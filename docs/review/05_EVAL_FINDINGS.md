# 05: Evaluation v1 findings (for review)

Source: `reports/EVAL_v1.md`, 2026-10-28.
- **Splits:** 1,842 train / 822 dev / 1,045 test units.
- **Stacker:** fit on train.
- **Conformal:** LAC with α=0.10, fit on dev.
- **Reported numbers:** all on test.

## Headline numbers
| decider | macro-F1 | ECE | notes |
|---|---|---|---|
| rule head (maths only) | 0.382 | 0.462 | uncalibrated margins |
| Gemma, raw claim | 0.254 | 0.329 | over-predicts PROPAGANDA/INCONCLUSIVE |
| Gemma adjudicator (structured state) | 0.379 | 0.346 | |
| **stacker** (features + rule + Gemma-adj) | **0.554** | **0.055** | best; well calibrated |
| stacker + conformal | 0.618 | 0.056 | acc 0.805 at only 16% coverage |

The hybrid (maths features plus LLM) clearly beats either part alone, which is the core thesis of the system.

## Problems needing a decision

1. **The PROPAGANDA F1 = 1.000 is not real.** Every PROPAGANDA example is a long PTC news article and every other class is a short claim, so length and format give the label away (the classic shortcut). This can't be fixed without data. Options:
   - (a) Label propaganda on short social posts: Gemma silver labels plus your hand-check on `05_LABELLING_SHEET`.
   - (b) Add article-level non-propaganda news, e.g. the 48 low-span PTC articles, plus LEGIT news articles.
   - (c) Evaluate PROPAGANDA only within articles.

   Until then, treat the PROPAGANDA score as untested on social posts.
2. **Conformal abstention is too strict.** At α=0.10, prediction sets are usually larger than one label, so 84% of units become INCONCLUSIVE. It does abstain on 92% of the gold-INCONCLUSIVE items, which is good. Options: α=0.2 or 0.3, or abstain only when the top-2 mass is split (margin < m). I suggest α=0.2, with the trade-off curve shown in the dashboard. **Your choice.**
3. **MISLEADING recall is 0.14.** LIAR "half-true"/"barely-true" and X-Fact "partly true" are noisy MISLEADING targets, and the label mapping is ours (`01_DATA_GAPS.md`). An option is to keep MISLEADING only for BOOM/Snopes-style "out of context / missing context" verdicts.
4. **LIAR and X-Fact are hard claim-only benchmarks.** Accuracy of 0.38–0.43 is roughly the published claim-only level. This is a property of those datasets, not a pipeline bug.
5. **KB matches don't help as much as expected.** On the kb_match slice the stacker is 0.544 macro-F1, about the same as without a match. One cause is that a fact-check of a *similar but different* item is judged to refute or context-mismatch the claim (seen in the Sialkot example). Proposed fix: add a Gemma `same_claim` check (is this the same claim/media?) before a hit counts as a KB match, and use relevance as a multiplier.
6. **The Gemma raw-text baseline is the weakest decider.** This supports keeping Gemma as a reader and adjudicator rather than a direct classifier.

### Evidence for problem 5 (measured on units_v1)
- **Refute stance is only weakly informative.** On FAKE vs LEGIT units, KB evidence judged "refute" belongs to a FAKE item 72% of the time, against a 58% base rate.
- **The reranker score does not help.** Restricting to rerank ≥ 0.9 gives 76%, so it doesn't separate same-claim matches from look-alikes.
- **Relevance does not help either.** Gemma gives "refute" a median relevance of 1.0 for LEGIT items too.
- **"context_mismatch" is assigned evenly across all gold labels**, at a median relevance of 0.5. It currently adds noise to the MISLEADING channel.
- **Proposed v2:**
  - Add a `same_claim` field (0–1) to the stance prompt: "is the evidence about the same specific event/media/quote?". Treat stance as neutral unless same_claim ≥ 0.7.
  - Count context_mismatch only when same_claim is high and the fact-check verdict itself is misleading or out-of-context.
  - Cost: re-judge about 18.5k stance pairs, roughly 4–5 h on the shared Gemma. Run when EDA is idle.

7. **Social posts are over-labelled PROPAGANDA (external run, 2026-10-28).** `16_score_external` labelled 454 of 628 social posts PROPAGANDA, plus 18 FAKE, 2 MISLEADING, 6 LEGIT and 148 INCONCLUSIVE. Two things in the doc rollup (`fnp/math/features.py:doc_rollup`) cause this:
   - **Rule 3** fires when doc P_prop ≥ 0.60. Noisy-OR P_prop saturates: the median post has 3 spans and P_prop 0.70, and ordinary Loaded_Language / Name_Calling on Reddit is enough to trip it.
   - **Any claim labelled PROPAGANDA** also makes the doc PROPAGANDA. That claim label comes from the PTC length shortcut (problem 1).

   The gallery ranking by P(FAKE)+P(MISLEADING) is not affected. Only the PROPAGANDA *label* is.

   Options:
   - (a) Require P_prop ≥ 0.6 *and* prop_density / n_techniques above a threshold fitted on hand-checked posts.
   - (b) Show PROPAGANDA only as a secondary tag on social posts until problem 1 has data.
   - (c) Leave it as is.

   I suggest (b). It is a re-rollup from the saved claim verdicts, with no Gemma calls. **Your choice.**

## Laya result (decision for you)
- **Laya beats the Gemma adjudicator.** It reaches macro-F1 ≈ 0.61 against 0.38, with ECE 0.08–0.10 against 0.35. It passes the adoption gate, so Laya can take over the deciding step while Gemma stays the reader (claims, spans, stance).
- **Default recommendation:** keep the **stacker** (features + rule + Gemma-adj, ECE 0.055) as the calibrated production decider, and use **Laya RL** as the fast path.
- **Routing:** escalate to the full Gemma adjudicator only when Laya's conformal set has more than one label. The routing bandit (`fnp/rl/routing_bandit.py`) is built but not yet fit on logged outcomes.
- **Before stacking Laya:** refit the stacker on out-of-fold Laya probabilities. The current stacker+laya over-trusts Laya (ECE 0.22).
