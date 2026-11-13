# Assumptions register (constants)

Sources: `fnp/config.py:CONSTANTS` (config) and `fnp/math/constants.py:MATH_CONSTANTS` (math). Constants in the math register are read through `M()`, so a same-name entry in config overrides them.
Sensitivity: H = changes labels on a material share of claims, M = moves probabilities, L = minor.

| name | value | where | why / provenance | how & when to re-fit | sensitivity |
|---|---|---|---|---|---|
| dup_cosine | 0.9 | config | assumed; evidence texts above this bge-m3 cosine share an origin | hand-labelled same-origin pairs (PR curve), before 09_fuse | H |
| prior_true | 0.5 | config | assumed; uninformative claim prior, re-fit as gold base rate | gold base rate (fit_kappa) | M |
| woe_kappa | 2.0 | config | assumed; scale from strength units to log-odds, fit on gold (09_fuse --fit) | fit_kappa on gold (09_fuse --fit) | H |
| ds_strength_cap | 0.95 | config | assumed; no single evidence commits more than this mass | sensitivity sweep | M |
| fake_p_true_upper | 0.2 | config | from notes: logistic(-1.4); upper CI bound for FAKE | dev sweep after kappa fit | H |
| legit_p_true_lower | 0.8 | config | assumed; lower CI bound for LEGIT | dev sweep after kappa fit | H |
| prop_threshold | 0.6 | config | assumed; P_prop above this with one-sidedness gives PROPAGANDA | PTC/QProp dev | H |
| max_uncertainty | 0.6 | config | assumed; DS Pl-Bel above this forces INCONCLUSIVE | risk-coverage on dev | H |
| min_n_eff | 1.0 | config | assumed; fewer independent evidence units -> INCONCLUSIVE | risk-coverage on dev | M |
| plausible_low | 0.5 | config | user spec: 'plausibly fake' tier lower bound on P(FAKE)+P(MISLEADING) | user spec | M |
| plausible_high | 0.8 | config | user spec: tier upper bound | user spec | M |
| conformal_alpha | 0.1 | config | assumed; 90% coverage prediction sets | product decision | M |
| rel_prior_n0 | 4.0 | math | assumed: Beta prior strength = 4 pseudo-observations per evidence type | marginal likelihood on source track records | M |
| rel_clip | 0.02 | math | assumed: type prior mean clipped to [0.02, 0.98] so Beta(a,b) is proper | keep | L |
| rel_cred_level | 0.9 | math | assumed: credible-interval mass for source reliability | reporting choice | L |
| bias_discount | 0.5 | math | assumed: reliability *= 1 - 0.5*|bias|, bias in [-1,1] | fit on gold once outlet-bias labels exist | M |
| independence_alpha | 1.0 | math | assumed: ind_i = 1/|cluster|^alpha; 1.0 = a cluster counts once | keep 1.0; sweep 0.5-1 on dev | H |
| copy_prior | 0.2 | math | assumed: prior P(two sources are dependent), Dong 2009 uses alpha=0.2 | EM over source x claim matrix | L |
| copy_rate | 0.8 | math | assumed: P(copier copies a given value), Dong 2009 'c' | EM over source x claim matrix | L |
| copy_n_false | 10.0 | math | assumed: number of uniformly likely false values per object, Dong 'n' | set to the value-space size of the data | H |
| ds_laplace | 1.0 | math | assumed: Laplace pseudo-count in Dawid-Skene confusion rows | keep | L |
| tf_gamma | 0.3 | math | Yin et al. 2008: dampening factor gamma | paper default | L |
| tf_rho | 0.5 | math | Yin et al. 2008: weight of implication between facts | paper default | L |
| tf_init_trust | 0.9 | math | Yin et al. 2008: initial source trustworthiness | paper default | L |
| tf_base_sim | 0.5 | math | assumed: imp(f'->f) = sim - base_sim, sim=0 for distinct values | dev sweep | L |
| meta_eps | 0.01 | math | assumed: v_i = 1/(eps + scale*w_i); eps keeps v finite at w=0 | keep | L |
| meta_precision_scale | 1.0 | math | assumed: one unit of reliability*independence*relevance = precision 1 on the log-odds scale | calibrate the CI coverage of p_true_lo/hi on gold | H |
| meta_z | 1.959963984540054 | math | standard normal 97.5% quantile (95% CI) | fixed | L |
| bp_damping | 0.5 | math | assumed: message damping (new = d*old + (1-d)*update) | raise if non-convergence | L |
| bp_max_iter | 200.0 | math | assumed | keep | L |
| bp_tol | 1e-08 | math | assumed: max abs message change for convergence | keep | L |
| bp_coupling_scale | 1.0 | math | assumed: Ising coupling J = scale*sign*strength (log-potential units) | fit on story graphs with gold | M |
| pagerank_alpha | 0.85 | math | Page & Brin 1998 damping | standard | L |
| cp_rank | 4.0 | math | assumed: CP rank for coordination search | choose by core-consistency or fit elbow | M |
| cp_n_iter | 300.0 | math | assumed | keep | L |
| timeline_leak | 0.02 | math | assumed: P(inconsistent | no flag fires) | fit on timeline-labelled gold | L |
| technique_precision_default | 0.5 | math | assumed (task spec): per-technique tagger precision pi_t | tagger per-technique precision on PTC dev | H |
| kb_match_min | 0.8 | math | assumed: KB (debunked-claim) similarity at/above which kb_match = 1 | PR curve of KB retrieval on BOOM leave-one-out | H |
| check_worthy_min | 0.5 | math | assumed: claims below this are ignored by doc roll-up if any pass | dev sweep | L |
| doc_fake_min | 0.5 | math | assumed: a FAKE claim needs P(FAKE) >= this to make the doc FAKE | dev sweep | M |
| rule_tau_prob | 0.05 | math | assumed: logistic margin width for probability thresholds | dev sweep (rule head only) | M |
| rule_tau_strength | 0.15 | math | assumed: logistic margin width for summed-strength thresholds | dev sweep | L |
| rule_tau_neff | 0.25 | math | assumed: logistic margin width for the n_eff threshold | dev sweep | L |
| strong_refute_min | 0.5 | math | assumed: summed independent refute strength counted as 'strong' | dev sweep | H |
| cm_min | 0.3 | math | assumed: minimum context-mismatch strength for MISLEADING | dev sweep | M |
| mixed_ratio_min | 0.5 | math | assumed: min(sup,con)/max(sup,con) above which evidence is 'mixed' | dev sweep | L |
| one_sided_min | 0.6 | math | assumed: one_sidedness counted as one-sided | dev sweep | M |
| coord_min | 0.5 | math | assumed: coordination score counted as coordinated | dev sweep | L |
| rule_floor | 0.01 | math | assumed: additive floor before normalising rule scores | keep | L |
| secondary_min | 0.3 | math | assumed: a non-primary label with P >= this is a secondary tag | dev sweep | L |
| stacker_C | 1.0 | math | assumed: inverse L2 strength for multinomial LR | CV | L |
| stacker_cal_frac | 0.2 | math | assumed: share of training rows held out for temperature scaling | keep | L |
| FLAG_STRENGTH (dict) | 0.6-0.95 | event_bn.py | assumed: per-flag strengths for the timeline noisy-OR | fit on timeline gold | M |
| DEFAULT_FLAG_STRENGTH | 0.5 | event_bn.py | assumed: strength for unknown flag names | - | L |
| mixed-evidence margin tau | 0.1 | verdict.py | assumed | dev sweep | L |
| ann_double_frac | 0.2 | config | scripts/25 design | agreement stats need enough double-annotated items; check kappa CI width after first 200 | M |
| ann_blind_frac | 0.1 | config | scripts/25 design | enough blind items to compare accept-rate vs non-blind; raise if CI too wide | L |
| ann_lookback_hours | 48 | config | assumed: late-arriving source rows | compare rows inserted by lookback vs by watermark after a week of runs | L |
| ann_lease_minutes | 30 | config | legacy; no longer reserves items (batches do) | remove once nothing reads leased_until | L |
| ann_social_min_chars | 40 | config | assumed | fraction of short posts annotators mark no-claim/no-propaganda | M |
| ann_social_max_chars | 1500 | config | scripts/25 | median annotation time vs length | L |
| ann_news_min_chars | 300 | config | assumed | fraction of short news items with any span | M |
| ann_news_max_chars | 5000 | config | assumed | annotator feedback on long items | M |
| ann_split_dev | 0.1 | config | assumed | once enough items: choose per-class minimum counts instead | M |
| ann_split_test | 0.1 | config | assumed | once enough items: choose per-class minimum counts instead | M |
| ann_share_en | 0.9 | config | owner request (mostly English) | compare with annotator-judged language mix after 500 items | L |
| ann_share_hi | 0.1 | config | owner request (a little Hindi) | Hindi supply in the sources is limited (~10-15k rows); lower if the pool runs dry | M |
| ann_share_other | 0 | config | owner request (annotators may not read other languages) | revisit only if multilingual annotators are recruited | L |
| ann_batch_size | 250 | config | owner request | per-user throughput | L |
| ann_target_items | 50000 | config | owner request; assumed to mean fully annotated items per task | confirm with owner whether spans or items | M |
| ann_target_neg_frac | 0.2 | config | assumed | observed none-rate after the first 2k items | M |
| ann_balance_sample | 30 | config | assumed | class-count skew after 5k items | L |
| ann_idle_seconds | 120 | config | assumed; time tracking counts a minute as active if input occurred within this window | compare tracked minutes with a self-timed sample | M |
| ann_session_hours | 12 | config | assumed | support feedback | L |
| ann_max_failed_logins / ann_lockout_minutes | 5 / 10 | config | assumed | login_event failures | L |
| ann_min_password_len | 8 | config | assumed | security review | L |
| ann_trunc_min_frac | 0.6 | config | assumed; YouTube descriptions and transcripts over `ann_social_max_chars` are cut at the last sentence end at or after 60% of the cap (else last line break, else word boundary) | share of cut items that annotators skip or mark "no claim"; check cut points on 50 transcripts | M |
| ann_min_substance_chars | 10 | config | assumed; items whose body is only links/mentions are dropped | sample 50 dropped items | L |
| ann_skip_exclude_n | 2 | config | assumed; non-language skips by 2 different annotators exclude the item | share of excluded items that a third annotator would have labelled | M |
| ann_random_frac | 0.10 | config | assumed; unbiased slice of every batch, flagged `selected_by=random` | width of the CI of a metric measured on the random slice only | M |
| ann_gold_frac | 0.03 | config | assumed; calibration items per batch (only once gold items exist) | annotator gold F1 stability after 30 gold items | M |
| ann_target_supply_frac | 0.5 | config | assumed; steer only towards what the models say the pool can supply (precision of predictions unknown) | measure encoder / Gemma precision on a held-out sample, then replace 0.5 by it | H |
| ann_cb_beta | 0.999 | config | Cui et al. 2019 | dev-set macro-F1 across beta in {0.99, 0.999, 0.9999} | M |
| ann_gold_min_items / ann_gold_min_f1 | 10 / 0.5 | config | assumed | compare flagged annotators with adjudicator judgement | M |
| ann_q_min_items / ann_q_none_dev / ann_q_fast_sec / ann_q_min_agree_f1 | 30 / 0.30 / 8 / 0.40 | config | assumed; behaviour flags on the Quality tab | compare flagged annotators with adjudicator judgement after 3 weeks | M |
