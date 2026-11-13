# Laya limits (model card) and what we observed

Model-card limits (convaiinnovations/laya, pip `laya` 0.4.15):
- **noul follows option labels (#156).** The noul answer is driven by the rendered option text. We pass explicit criteria for `false`/`true` on every noul question; changing wording changes answers, so wording is frozen in `fnp/laya/state_builder.py:QUESTIONS`.
- **act/escalate head carries no signal (#185).** `answers[q]["action"]["act_probability"]` is returned but not used for routing; routing uses confidence/margin/entropy and the bandit (`fnp/rl/routing_bandit.py`).
- **More than ~20 options is weak.** We use 5 (verdict) and 2 (noul).
- **Shipped checkpoints are over-confident.** Temperatures must be refit on a held-out calib split (done in `calibrate_save_card`; clamp is [0.5, 5]).
- **Near-chance zero-shot on new typed decisions.** Observed: base multilingual checkpoint on the 5-class verdict, 26 synthetic eval rows: accuracy 0.15, macro-F1 0.09, log-loss 2.90, ECE 0.51 (chance = 0.20 / 1.61 nats). Fine-tuning is mandatory.
- State is cut from the END at `max_len - head`; the 5 option descriptions cost ~150-250 tokens of head, so at max_len 384-512 only ~1000-1200 chars of state are read. Put the decisive text first (state builder does).

Observed in our runs (synthetic labels generated from the displayed features, so accuracy only proves plumbing):
- Stage A (soft-CE, 1 epoch, 164 train rows -> 656 items): eval acc 1.00, log-loss 0.027, ECE 0.024 after temperature refit (verdict T=0.5 on the clamp: the fit wanted to sharpen, an artefact of tiny calib n=10, which the package flags).
- Stage B (RLCD + KL anchor, 1 epoch): eval log-loss 0.014; KL to anchor stays ~0.01.
- `laya.train.rlcd_loss` has no KL/anchor, no custom reward and no per-item reward target, so `fnp/laya/rlcd.py` is a custom loop (same Gaussian-logit REINFORCE, group-mean baseline, INCONCLUSIVE penalty, KL to precomputed stage-A logits). Soft targets are native (`gold.probabilities`).
- Assumed constants: beta=0.3 (gold/Gemma blend), lam=0.5, kl=0.1, ce=0.25, routing costs 0.001/0.05/1.0.
- Real-data accuracy is NOT yet measured; no claim of SOUND without held-out precision with Wilson CI.
