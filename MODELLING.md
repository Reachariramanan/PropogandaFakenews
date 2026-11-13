# Modelling branch: training code

This branch holds only the code that trains or evaluates models. The application (verdict engine, annotation tool,
Evidence Desk) is on the `dev` branch. Datasets, model checkpoints and databases are not in the repository: the
scripts read them from `data/` (see `fnp/config.py`) and write checkpoints there.

All numbered scripts are dry-run by default; `--apply` trains or writes files. They need the Laya package
(see `requirements.txt`) and a GPU for the training stages.

## Verdict model (Laya)

| Stage | File | What it does |
|---|---|---|
| 10 | `scripts/10_laya_rows.py` | Builds Laya training rows from the train split of the unit file. Target = (1 - beta) * gold + beta * the Gemma adjudicator's probabilities. dev/test units are never written. |
| 11 | `scripts/11_train_laya.py` | Stage A: supervised distillation of Laya (multilingual 322M) on the verdict rows. |
| 12 | `scripts/12_rl_laya.py` | Stage B: RLCD fine-tune of the stage-A checkpoint with a KL anchor, then refits temperatures and abstention thresholds on the calibration split. |
| | `fnp/laya/train.py`, `fnp/laya/rlcd.py` | The training loops used by stages 11 and 12. |
| | `fnp/laya/dataset.py`, `fnp/laya/state_builder.py` | Row format, deterministic splits, and the compact state text Laya reads. |
| | `fnp/laya/synth.py` | Synthetic verdict records for smoke tests only; not evidence of accuracy. |

## Propaganda span tagger

| Stage | File | What it does |
|---|---|---|
| 20 | `scripts/20_make_splits.py` | Frozen article-level splits for the PTC corpus (70/10/20 by hash of the article id). |
| 21 | `scripts/21_baseline_prop.py` | Baseline of the Gemma tagger on the frozen splits. |
| 22 | `scripts/22_prop_variants.py` | Scores Gemma tagger variants (chunking, recall-oriented prompt). |
| 23 | `scripts/23_train_span_encoder.py` | Fine-tunes the Laya mmBERT-base encoder as a per-token multi-label span tagger over the 14 PTC classes. |
| 24 | `scripts/24_ensemble_prop.py` | Combines the span encoder with the Gemma tagger; the source per technique is chosen on dev only, test is scored once. |
| | `fnp/eval/ptc_scorer.py` | PTC-style span scorer used by stages 21 to 24. |

Evaluation outputs from these stages are in `reports/EVAL_*.md`. Limits and open issues are in
`docs/review/04_LAYA_LIMITS.md`; the maths is in `docs/METHODS.md`.

## Claim extraction (GSPO)

`claimExtraction/` contains the GSPO / SFT training scripts, configs, reward code and handover notes for a small
Qwen model that extracts structured claims, plus evaluation summaries under `claimExtraction/results/`.
Checkpoints and training data are excluded. This is a separate effort from the verdict pipeline on `dev`.

## Tests

`tests/test_laya_state_dataset.py` and `tests/test_ptc_scorer.py` cover the Laya state/dataset code and the span scorer.
