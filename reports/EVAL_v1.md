# FNP evaluation v1

Units: train 1842, dev 822, test 1045 (test decidable 945, test gold-INCONCLUSIVE 100).
Stacker fit on train; conformal (LAC, alpha=0.10) fit on dev; all numbers on test.

| decider | macro-F1 | acc | ECE | Brier | coverage | abstain on gold-INCONCL |
|---|---|---|---|---|---|---|
| rule | 0.382 | 0.381 | 0.462 | 1.045 | 1.000 | — |
| rule_conf | 0.184 | 0.381 | 0.590 | 1.045 | 0.044 | 0.930 |
| rule_selfabstain | 0.431 | 0.431 | 0.470 | 1.045 | 0.695 | — |
| gemma_raw | 0.254 | 0.267 | 0.329 | 0.939 | 1.000 | — |
| gemma_raw_conf | 0.444 | 0.500 | 0.493 | 0.939 | 0.008 | 0.990 |
| gemma_adj | 0.379 | 0.389 | 0.346 | 0.909 | 1.000 | — |
| gemma_adj_conf | 0.499 | 0.519 | 0.457 | 0.909 | 0.029 | 0.890 |
| stacker | 0.554 | 0.506 | 0.055 | 0.584 | 1.000 | — |
| stacker_conf | 0.618 | 0.805 | 0.056 | 0.584 | 0.163 | 0.920 |

## Per-class (stacker, no abstention)

| class | n gold | P [95% Wilson] | R | F1 |
|---|---|---|---|---|
| FAKE | 370 | 0.493 [0.450, 0.535] | 0.714 | 0.583 |
| PROPAGANDA | 69 | 1.000 [0.947, 1.000] | 1.000 | 1.000 |
| LEGIT | 266 | 0.446 [0.386, 0.508] | 0.421 | 0.433 |
| MISLEADING | 240 | 0.371 [0.278, 0.475] | 0.138 | 0.201 |

## Breakdowns (macro-F1 / acc)

| slice | n | rule | gemma_adj | stacker |
|---|---|---|---|---|
| dataset=boomlive | 105 | 0.361 / 0.524 | 0.375 / 0.552 | 0.498 / 0.762 |
| dataset=fakenewsnet | 71 | 0.308 / 0.239 | 0.386 / 0.324 | 0.709 / 0.718 |
| dataset=liar | 400 | 0.296 / 0.347 | 0.305 / 0.343 | 0.350 / 0.375 |
| dataset=ptc | 69 | 0.831 / 0.710 | 0.840 / 0.725 | 1.000 / 1.000 |
| dataset=xfact | 300 | 0.302 / 0.333 | 0.312 / 0.333 | 0.387 / 0.427 |
| kb_match=kb_match | 139 | 0.171 / 0.518 | 0.181 / 0.511 | 0.544 / 0.511 |
| kb_match=no_kb_match | 806 | 0.366 / 0.357 | 0.370 / 0.368 | 0.553 / 0.505 |

## Laya deciders (added 2026-10-28; same test set)

Checkpoints: `laya_fnp_sl_v1` (stage A, gold + 0.3·Gemma soft targets) and `laya_fnp_rl_v1` (stage B, RLCD with KL anchor). Both are multilingual 322M; peak GPU memory 3.36 GB.

| decider | macro-F1 | acc | ECE | Brier | coverage |
|---|---|---|---|---|---|
| laya (SL) | 0.608 | 0.537 | 0.082 | 0.587 | 1.000 |
| laya (SL) + conformal | 0.487 | 0.932 | 0.297 | 0.587 | 0.078 |
| laya (RL) | 0.607 | 0.534 | 0.098 | 0.587 | 1.000 |
| laya (RL) + conformal | 0.678 | 0.909 | 0.300 | 0.587 | 0.081 |
| stacker + laya (RL) | 0.627 | 0.549 | 0.220 | 0.635 | 1.000 |
| stacker + laya (RL) + conformal | 0.726 | 0.818 | 0.158 | 0.635 | 0.261 |

**Adoption gate** (macro-F1 ≥ Gemma-adj − 0.03 and ECE ≤ Gemma-adj): **PASS** for both checkpoints. Laya replaces the Gemma adjudicator as the decider.
- RL keeps macro-F1 equal to SL and improves selective macro-F1 under conformal abstention (0.678 vs 0.487), but full-coverage ECE is slightly worse (0.098 vs 0.082).
- Caveats:
  - Every macro-F1 here includes the PROPAGANDA class, which is length-confounded (see `docs/review/05`).
  - `stacker + laya` is mis-calibrated (ECE 0.22) because Laya's probabilities on the train units are in-sample. A fix is to refit the stacker on out-of-fold Laya predictions, or on dev with cross-fitting.
