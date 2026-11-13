# FNP architecture (approved plan, 2026-10-27)

## Context
`NTRO/FakeNewsPropoganda/methodologoes/` has two design notes:
- 10 probability frameworks: belief propagation (BP), Dempster–Shafer (DS), log-odds, information gain, centrality, truth discovery, Bayesian network (BN), EM/Dawid–Skene, tensor, meta-analysis.
- Propaganda span detection (Da San Martino 2019) plus a forensic multi-agent design with evidence weighting.

Goal: turn the practical subset into a working system. Each claim, post or article gets one of 5 classes, **FAKE / PROPAGANDA / LEGIT / MISLEADING / INCONCLUSIVE**, plus a "plausibly fake" tier. LLM perception is fused with explicit mathematical models.
- **Gemma** (`gemma-4-26b-a4b-it`, sglang at `localhost:11650`, accessed the same way as EDA) does the reading and adjudication.
- **Laya** (convaiinnovations/laya: a non-autoregressive decision model, mmBERT-base 322M multilingual or ModernBERT 421M; RLCD = REINFORCE on proper scoring rules) is trained as a cheap, calibrated decider, distilled from Gemma plus gold labels.
- RL is used where the problem really is sequential or decision-theoretic.

This is built **in parallel** with the running EDA "soft-topic-hmm-events" session, and reuses its outputs read-only.

**User decisions:**
- **Unit:** claim-centric. Both social posts (shaktidb via EDA) and articles are decomposed into claims, and verdicts roll up to post/article.
- **Data:** public datasets and web retrieval are allowed.
- **GPU:** fit Laya training into the ~7 GB free on the L40S; Gemma keeps its ~40 GB.

## Practicality triage of the methodology notes
| Concept | Verdict | Where |
|---|---|---|
| Claim extraction, evidence typing, timeline (LLM-only readers) | Practical (Gemma) | L1 |
| Propaganda span + technique (18-technique PTC taxonomy) | Practical (Gemma + span-F1 eval on SemEval-20 T11) | L2 |
| Corroboration / contradiction / dedup / independence | Practical (embeddings + Gemma stance + clustering) | L3–L4 |
| Reliability tiers, bias | Practical as **priors**; updated by a Beta posterior | L5 |
| Truth discovery (TruthFinder, Dawid–Skene EM, AccuCopy) | Practical on the source × claim stance matrix | L5 |
| Log-odds weight of evidence (WoE), DS (Yager rule), DerSimonian–Laird meta-analysis, information gain | Practical, closed form | L6 |
| Loopy BP on the claim graph | Practical per story (small graphs) | L6 |
| Centrality | Only on the independence-collapsed graph; a feature, not a verdict | L6 |
| Tensor CP (source × claim × time × type) | Practical for coordination detection | L6 |
| Event BN with conditional probability tables | Restricted: noisy-OR timeline-consistency node only (full CPTs would be invented numbers) | L6 |
| HMM bursts / event chains / coordination | Reuse EDA libs (propagation features) | L6 |
| H1–H7 hypothesis agents, leader/planner agents | Replaced by generic per-claim hypotheses ("claim true") plus a code orchestrator | — |

## Architecture (new project `NTRO/FakeNewsPropoganda/`)
```
fnp/                      package
  config.py               env (.env: GEMMA_*, optional SEARCH_*), paths, constants register
  eda_bridge.py           sys.path import of EDA lib (gemma_client.chat_json/run_batch,
                          entity_routing.evidence_in_text, hmm_v1, event_chains, stats_v2);
                          reads EDA parquet / DuckDB read_only (copies if locked)
  llm/                    prompts.py + validators; claims, propaganda_spans, stance, timeline,
                          adjudicate (structured state only), redteam; sha1 cache table
  retrieval/              kb.py (BOOM + public fact-check claims; bge-m3 embeddings +
                          bge-reranker-v2-m3, both in HF cache), corpus.py, web.py (pluggable;
                          DuckDuckGo-HTML + trafilatura fetch, off by default in tests)
  math/                   woe.py, dempster_shafer.py, meta.py, infogain.py, independence.py,
                          truth_discovery.py, bp.py, tensor_cp.py, centrality.py, event_bn.py,
                          propaganda_score.py, propagation.py (EDA HMM/chain features),
                          calibration.py, conformal.py, verdict.py
  laya/                   state_builder.py, dataset.py, train_sl.py, train_rl.py (RLCD), calibrate.py
  rl/                     routing_bandit.py, voi_stop.py
scripts/ 00_fetch_datasets … 15_predict (numbered stages, dry-run default, --apply writes)
tests/                    offline pytest (mock Gemma, synthetic math checks)
docs/METHODS.md           every model: formula + ASSUMPTIONS stated explicitly
docs/ARCHITECTURE.md, README.md, CLAUDE.md
docs/review/              problems for the user to review (see below)
data/ (gitignored)        fnp.duckdb (versioned tables *_v1), datasets/, models/
```
Conventions are copied from EDA `CLAUDE.md`:
- No secrets in code.
- Never write to Postgres or the EDA DuckDB files.
- Versioned tables.
- Hand-judged precision with a Wilson CI before any classifier is called SOUND.

Venv: `FakeNewsPropoganda/.venv --system-site-packages` (reuses the torch 2.12 / transformers in `~/.local`), plus `pip install laya duckdb pandas scikit-learn tensorly datasets`.

## Pipeline and mathematical models (assumptions recorded in METHODS.md)
**L0 Ingest.**
- Inputs: shaktidb posts from EDA `post_facts` parquet, with embeddings, `post_topic_v4`, `post_scores.lang_resolved`, `event_codes`, and burst/chain tables when present.
- Arbitrary text or a URL via `15_predict`.
- Datasets:
  - BOOM 507 (the `claim` field only; the body leaks the label).
  - SemEval-2020 T11 PTC (propaganda spans).
  - LIAR (6 labels mapped to 5 classes).
  - X-Fact (multilingual, includes Hindi).
  - IFND (Indian).
  - FakeNewsNet-PolitiFact.

**L1 Claims (Gemma).**
- Output per claim: `{claim, speaker, claim_type, quoted_or_reported, check_worthy p, date, entities, evidence_quote}`.
- Quotes are verified verbatim with `evidence_in_text`. Only L1, L2 and the stance step see raw text.

**L2 Propaganda.**
- Gemma tags spans and techniques; spans must be verbatim.
- Score: `P_prop = 1 − Π_i(1 − c_i·π_t)`, where `π_t` is the per-technique precision measured on the PTC dev set.
- Also reports density per 100 tokens, technique entropy and a one-sidedness term.
- Evaluated with the paper's partial-overlap span F1.

**L3 Evidence.**
- Retrieve top-k from: the fact-check KB (cosine, then reranker), the corpus and the web.
- Gemma stance per pair: support / refute / context-mismatch / neutral, plus evidence type (the notes' 10 categories) and relevance.
- Matching a debunked KB claim is a strong refute.

**L4 Independence.**
- Cluster evidence by shared origin URL, `source_links`, and embedding cosine ≥ 0.9.
- `ind_i = 1/|cluster|`; `n_eff = (Σw)²/Σw²` (Kish).
- AccuCopy copy probability between sources.

**L5 Reliability.**
- Tier prior (table from the notes) gives `Beta(α0, β0)`.
- Posterior is updated by track record (debunked shares and verified claims).
- Truth discovery: Dawid–Skene EM and TruthFinder over the source × claim stance matrix, with per-source confusion matrices.

**L6 Fusion per claim.**
- WoE: `L = logit(p0) + κ·Σ r_i·ind_i·rel_i·dir_i`, `P_true = σ(L)`; κ is fitted on gold data.
- DS: masses on {T, F, Θ}, combined with the **Yager** rule (conflict goes to Θ) to give Bel, Pl and `U = Pl − Bel`; record K.
- DerSimonian–Laird random-effects meta-analysis on per-evidence log-odds gives a 95% CI and I².
- Information gain = KL(post ‖ prior) in bits per article.
- Loopy BP on the story's claim graph (entail/contradict edges from Gemma or mDeBERTa NLI; Ising potentials) gives consistent marginals.
- Tensor CP (ALS) gives a coordination factor score.
- Noisy-OR timeline BN: P(impossible chronology).
- Propagation features from EDA HMM bursts and chains: burst intensity at emergence, platform spread, author diversity, `dup_rate`, `coordination_flag`.
- Centrality on the collapsed graph.

**L7 Verdict.** Three stacked deciders give `p ∈ Δ⁵`:
- (a) Math rule head, using these definitions:
  - FAKE: upper CI of `P_true` ≤ 0.2 and a strong independent refute.
  - MISLEADING: context-mismatch, or mixed true/false claims.
  - PROPAGANDA: `P_prop` high and one-sided or coordinated, while the claims are not refuted.
  - LEGIT: lower CI of `P_true` ≥ 0.8 and low `P_prop`.
  - Otherwise uncertain.
- (b) Gemma adjudicator: sees only structured state, not the raw article; red-team pass.
- (c) Multinomial-logit stacker, then temperature scaling.
- **INCONCLUSIVE** comes from split-conformal prediction sets: an abstention set gives INCONCLUSIVE, as do high DS `U` and low `n_eff`. FAKE overrides PROPAGANDA as the primary label, and all scores are kept as secondary tags.
- "Plausibly fake" = `P(FAKE) + P(MISLEADING) ∈ [0.5, 0.8)`.
- Output vector, per the notes: `{support, contradiction, uncertainty, info_gain, centrality, confidence, CI}`.

## Laya and RL
1. **State builder.** Claim + top evidence summary + math feature vector, rendered as compact JSON text (≤1024 tokens, multilingual checkpoint). Questions:
   - verdict `choice`(5)
   - `is_propaganda` noul
   - `context_mismatch` noul
   - `check_worthy` noul
2. **SL distillation** (`train_sl.py`): KL to Gemma soft labels plus cross-entropy on gold. bf16, gradient checkpointing, batch 4–8 with accumulation, fits in ~6 GB. Temperature refit per question type (the card says the checkpoints ship over-confident).
3. **RLCD fine-tune** (`train_rl.py`, own implementation if the `laya` package has no training API):
   - Policy = Laya's distribution with Gaussian logit noise.
   - Reward = log + spherical proper scoring rule against gold or Gemma-consensus labels, plus a −λ penalty for INCONCLUSIVE when evidence is strong.
   - REINFORCE with a group-mean baseline (GRPO-style), KL to the SL checkpoint.
4. **Routing bandit** (`rl/routing_bandit.py`): contextual Thompson sampling / LinUCB over the actions {Laya decides, escalate to Gemma, escalate to human}. Reward = correctness (log score) − cost(tokens, latency, analyst time). This is how Laya takes over deciding without losing accuracy.
5. **Evidence-acquisition stopping** (`rl/voi_stop.py`): myopic value-of-information. Keep retrieving while the expected entropy drop exceeds the retrieval cost. The REINFORCE stopping policy is stretch/optional.
6. **Adoption gate:** Laya replaces Gemma for a decision type only if, on the held-out set, macro-F1 ≥ Gemma − 0.03 and ECE ≤ Gemma's ECE.

## Evaluation
**Gold sets:**
- BOOM claims (FAKE/MISLEADING; KB evaluated leave-one-out).
- LIAR test.
- X-Fact test.
- PTC dev (propaganda).
- A stratified shaktidb sample pre-labelled by Gemma, for the user to hand-judge. The sheet goes in `docs/review/`; reuse EDA `scripts/review/noise_gates.py` `sampling_plan` / `judge_precision`.

**Metrics:** macro-F1, per-class P/R with Wilson CI, ECE, Brier, the risk–coverage curve for abstention, span-F1.

**Ablations:** math-only vs Gemma-only vs hybrid vs Laya vs Laya + router.

## docs/review/ (problems surfaced for the user)
- `01_DATA_GAPS.md`: no LEGIT or PROPAGANDA in BOOM; label mapping from LIAR/X-Fact; dataset licences and access (PTC may need a form).
- `02_ASSUMPTIONS_REGISTER.md`: every assumed constant (tiers, κ, thresholds, priors), with how and when each is re-fit.
- `03_RESOURCE_CONTENTION.md`: GPU at ~7 GB free; shared Gemma concurrency (cap FNP at 12 so EDA isn't starved); DuckDB locks while EDA writes; read parquet or snapshot copies.
- `04_LAYA_LIMITS.md`: card-reported issues (noul follows labels #156, act head dead #185, >20 options weak, over-confidence).
- `05_LABELLING_SHEET.csv`: plus results and verdicts (SOUND / PARTIAL / UNSOUND per component).

## Agent execution (token-efficient, model routed by difficulty)
I scaffold the package, config, eda_bridge and docs skeleton myself. Then:
- **Wave A, parallel background agents:**
  - **A1 sonnet, lower effort:** dataset fetch, loaders, label maps, BOOM KB + reranker.
  - **A2 opus:** `fnp/math/*` + synthetic tests + METHODS.md. Hardest: DS/Yager, DerSimonian–Laird, Dawid–Skene, BP exact-on-tree check, CP planted rank, conformal.
  - **A3 sonnet:** LLM layer: prompts, validators, cache, mocked tests.
  - **A4 opus:** Laya: install and inspect the package API; state builder, SL, RLCD trainer, calibration, routing bandit, VOI.
- **Wave B:**
  - Me: integrate the numbered scripts, then run pilots: BOOM 507, LIAR/X-Fact samples, PTC dev, 2k shaktidb posts; Gemma concurrency ≤ 12.
  - Sonnet: evaluation report.
  - Fresh opus reviewer: leakage, double-counting and calibration audit.
- Then a train → evaluate → gate loop for Laya (SL, then RL) until the adoption gate passes or the run is reported as failing honestly.

## Verification
- `pytest tests/` passes offline.
- The math self-checks reproduce known results:
  - Zadeh example under Dempster vs Yager.
  - DerSimonian–Laird on a textbook dataset.
  - Dawid–Skene recovers synthetic confusion matrices.
  - BP equals brute-force marginals on a tree.
- `scripts/15_predict.py --text "<claim>"` / `--url` returns the 5-class JSON with evidence trail.
- `scripts/14_evaluate.py` writes `reports/EVAL_v1.md` with the metrics and ablations above.
- Laya checkpoint passes or fails the adoption gate, and the result is stated in `docs/review/`.

## Interface (fact-check database, gallery and detector; added 2026-10-28)

`Interface/` is a separate site built after this plan was approved. It is not part of the FNP pipeline and FNP never writes to it.
- It keeps one JSONL file per fact-check site in `Interface/factchecks/`, scraped from each article's ClaimReview markup, plus two exports from the shaktidb news and social databases in `Interface/data/`.
- `Interface/build_index.py` merges them into a TF-IDF index and a plausibility score per item. Where `verdict_v1` has a doc row for an item, the score blends in the engine's P(FAKE)+P(MISLEADING).
- `Interface/server.py` serves the gallery (ranked by plausibility), a claim detector and a sources dashboard at http://<SITE_HOST>:8766/.
- FNP reads the fact-check files as extra knowledge-base and gold inputs, and the site reads `reports/verdict_v1_docs.jsonl`.
- Rows, scoring details and known gaps are in `current-arch.md`, under "Integration with other sessions".
