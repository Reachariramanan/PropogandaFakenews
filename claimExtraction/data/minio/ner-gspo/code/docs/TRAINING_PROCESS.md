# GSPO Information Extraction Training — Full Process Record

**Model:** Qwen3.5-0.8B (text-only, LoRA) · **Task:** free-form entity + event JSON extraction from geopolitical news/social text · **RL framework:** VERL (vendored) · **Algorithm lineage:** GSPO → GSPO+SFT → GSPO+SFT+expanded LoRA → GSPO+SFT+VIGOR (custom rollout allocation)

This document is a chronological, technical record of everything done to take this pipeline from "trains without crashing" to "produces correct, improving extractions," including every dead end, root-cause diagnosis, and the custom RL algorithm ultimately built and integrated. It is written to be read start to finish, once, by someone with no prior context.

---

## 1. Task and pipeline recap

Given a document, the model must emit **only** a JSON object:

```json
{
  "entities": [{"text": "...", "type": "PERSON|ORGANIZATION|LOCATION|PRODUCT|EVENT", "subtype": "..."}],
  "events": [{"topics": [{"code": "NN.N"}], "actions": [{"code": "NN"}]}]
}
```

- `type` ∈ {PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT}; `subtype` is a free-text granular label (e.g. `civilian`, `government_body`, `media_outlet`).
- `topics` are CAMEO-style codes (`"01.1"`); `actions` are CAMEO action codes (`"13"`–`"20"`).
- This is the **minimal schema** (`grpo/prepare_parquet.py::SYSTEM_IE_MINIMAL`) — it emits only the fields the reward function scores, unlike the earlier **full schema** which also requested `score`, `normalized`, `evidence_text`, `actor1/actor2`, etc. Full-schema targets averaged 1257 tokens vs. 332 for minimal, and the overflow routinely truncated the JSON before it closed — a truncated response scored 0.0 under strict JSON parsing, identical to garbage. Minimal schema was adopted project-wide once this was diagnosed.

**Source data:** `../ner_topic_llm_dataset.jsonl`, 80,642 documents, entities/events extracted from news + social text. Prepared into VERL parquet via `grpo/prepare_parquet.py --schema minimal`: `data/grpo_full/{train,val,test}.parquet` = 72,574 / 4,031 / 4,033 rows, plus `val_small.parquet` (196-row sample used for fast validation during RL).

**Reward function** (`grpo/reward_ie.py::compute_score`), weighted sum:

| Component | Weight |
|---|---|
| Valid JSON format | 0.15 |
| Entity F1 (text+type exact match) | 0.40 |
| Subtype accuracy | 0.10 |
| Topic-code F1 | 0.25 |
| Action-code F1 | 0.10 |

`_extract_json` tolerates markdown fences and picks the outermost `{...}` on strict-parse failure; `_repair_truncated_json` recovers partial credit (capped at `format=0.5`) from mid-object truncation rather than scoring 0.0, since 9/9 early validation generations were truncated and this recovery alone lifted their mean score from 0.0 to 0.24.

**Base model prep:** `Qwen/Qwen3.5-0.8B` is a multimodal hybrid (`Qwen3_5ForConditionalGeneration`) — 24 decoder layers, 18 Gated DeltaNet (linear attention) + 6 standard full-attention at indices **3, 7, 11, 15, 19, 23** (`full_attention_interval=4`), plus an unused vision tower and MTP head. `grpo/build_qwen3_5_text_only.py` produces a text-only checkpoint (`outputs/qwen3_5_0.8b_text_only/`) by (a) stamping `architectures=["Qwen3_5ForCausalLM"]` so VERL's auto-class resolver picks the text-only class instead of the full VLM, and (b) stripping the `language_model.` weight-key prefix that transformers uses but vLLM's native class doesn't expect (mismatch here loads fine in the FSDP actor but crashes the vLLM rollout server).

**Two silent-failure traps fixed early and left permanently in place:**
- `use_remove_padding=False` is mandatory unless `flash-linear-attention` is installed (it isn't). VERL's Qwen3.5 monkey-patch mis-detects the torch-fallback GatedDeltaNet kernel as `cu_seqlens`-aware; it isn't, so padding-free packing silently leaks recurrent state across packed sequences with no error.
- `rollout.layered_summon=False` is mandatory on a single GPU. With `NO_SHARD` FSDP (single GPU, nothing to shard), the alternative path calls `summon_full_params(offload_to_cpu=True)`, which FSDP does not support under `NO_SHARD`, and raises during LoRA weight sync.

---

## 2. Infrastructure hardening (done before any real training)

Before training was trusted to run unattended, three defects were found and fixed:

1. **Reward function was not exception-total.** `grpo/reward_ie.py` used `x or []` to guard iteration over model-generated fields (`entities`, `events`, `topics`, `actions`). This guards only *falsy* values — a truthy non-iterable (the model once emitted `"actions": true`) reached a `for` loop and raised `TypeError`, which propagated through the reward worker, through Ray, and killed an entire 38-step run at step 15 on one malformed rollout out of ~480. Fixed with a total `_as_list()` coercion (scalars/dicts → 1-element list, everything else → `[]`) and a `_score_safe()` wrapper that catches *any* exception and returns an all-zero metrics dict rather than propagating. Verified 0/4000 fuzz failures afterward.
2. **SFT prompt masking was missing.** The cold-start dataset builder computed loss over the *entire* input sequence, including the system prompt and source document — up to 91% of tokens on long documents. Fixed by masking (`labels[i] = -100`) everything up to the token-level common prefix between the full sequence and the prompt-only rendering (not assumed via string length, since chat templates don't guarantee the assistant turn is a clean suffix).
3. **No on-demand checkpointing.** VERL's stock trainer only checkpoints on `save_freq` multiples, the last step, or cloud-instance-expiration signals — killing a run mid-flight discarded all progress since the last periodic save. A local patch to `verl/verl/trainer/ppo/ray_trainer.py` (grep `LOCAL PATCH`) adds a control-file channel (`SAVE_NOW`/`STOP_NOW` sentinels in the checkpoint dir, since `fit()` runs inside a Ray actor — a different process from the launcher, so POSIX signals to the launcher never reach it) polled once per step, plus `SIGUSR1`/`SIGUSR2` as an equivalent signal-based channel. `scripts/train_ctl.sh {status|save|stop|kill}` drives it. This was later exercised for real under an actual GPU-contention crash (§6) and worked as designed.

---

## 3. First real GSPO run — and the failure it revealed

**Config:** LoRA rank 16 on decoder layer 23 only (a full-attention layer — matches standard `q/k/v/o_proj` naming, unlike the Gated DeltaNet layers), `lr=1e-6` (paper-scale, deliberately conservative), GSPO clip `3e-4/4e-4`, `rollout_n=4`, starting from the **raw text-only base checkpoint** (no supervised warm-start).

Reward climbed from 0.193 → 0.28 over ~300 steps — looked like progress. It was not.

**Diagnosis (triggered by a direct qualitative request — "check the quality of generation"):** merged the checkpoint and inspected raw output. The model was inventing its own schema:

```json
{"entities": [{"code": "Nottingham", "text": "Nottingham"}]}   // should be {"text","type","subtype"}
{"events": [{"topics": ["Azul", "American Airlines"], "actions": [...]}]}  // should be CAMEO codes, not entity names
```

Measured over 3 samples: `entity_f1 = 0.0` and `topic_f1 = 0.0` on **all three**, 2/3 hit the token cap while stuck in repetition loops (the same ~15 entities repeated verbatim). `action_f1` showing 0.6–1.0 was a **vacuous match** — the reward's F1 function returns 1.0 when both gold and prediction have zero valid codes, not real success; on the one sample with real gold action codes, `action_f1 = 0.0`.

**Root cause:** the climbing reward (0.19→0.28) was coming almost entirely from the `format` component (0.15 weight, valid-JSON-container credit) and lucky vacuous matches — not from the 65%-weighted entity/topic components, which contributed essentially nothing. A raw pretrained model has never seen this exact field-naming convention or CAMEO code vocabulary; a sparse RL reward with one LoRA layer at `lr=1e-6` is an extremely weak signal for teaching brand-new output vocabulary from scratch. This needed supervised grounding, not more RL.

---

## 4. SFT cold-start — the fix

Added checkpoint-safety (periodic save + resume, since a bare training loop had none) to `grpo/run_sft_coldstart.py`, then ran full-parameter fine-tuning (no LoRA — the goal is teaching a new output convention broadly, not adapting one subspace) on the exact minimal-schema gold targets:

- `--batch_size 1 --grad_accum 16 --lr 2e-5 --max_seq_length 3072`, gradient checkpointing (mandatory — first attempt OOM'd without it: the torch-fallback GatedDeltaNet kernel is memory-hungry, and a crowded shared GPU left little headroom), 300 optimizer steps on the full 72,574-row train set.
- Loss: 0.372 → 0.147 monotonically, no instability.

**Result — measured on 5 held-out samples, before any RL:**

| Metric | Raw base + RL (300 steps) | SFT cold-start (no RL yet) |
|---|---|---|
| score | 0.167 | **0.701** |
| format | 0.67 | 1.0 |
| entity_f1 | **0.0** | **0.774** |
| subtype | 0.0 | 0.911 |
| topic_f1 | 0.0 | 0.36 |
| repetition / token-cap hits | 2/3 samples | 0/5 samples |

Visually confirmed correct schema throughout — real `type`/`subtype` values, real CAMEO codes, no repetition. This is the single highest-leverage fix in the whole project: RL cannot teach output *convention*; supervised warm-start can, cheaply.

---

## 5. GSPO on top of SFT — plateau, and the escalation ladder

With the schema fixed, GSPO was re-run from the SFT checkpoint, same LoRA-layer-23 config. Validation reward: **0.7065 → 0.7089 → 0.7077 → 0.7050 → 0.7054 → 0.7078 → …** — a tight, noisy plateau in the 0.705–0.709 band that held for 130+ steps. Real gains had been captured by SFT; RL at this scale (`lr=1e-6`, one LoRA layer) had nowhere further to go.

**User-directed escalation ladder** (agreed explicitly, RL always remains the final training step regardless of tier):

1. ~~SFT cold-start~~ — done, fixed schema.
2. **Expand LoRA to layers 19 + 23** (both full-attention layers), initialized from the layer-23 run's merged weights (so RL gains already made aren't discarded) — `LORA_TARGET_MODULES='.*\.layers\.(19|23)\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'`.
3. **Last resort:** bring in a fundamentally different rollout-allocation algorithm (VIGOR / Pilot-Commit) rather than another standard-GSPO variant.

**Tier 2 result:** also plateaued, same 0.70–0.71 band — but an offline sub-metric check (10-sample eval, since the aggregate reward dilutes a 25%-weighted sub-metric) showed topic_f1 moving from 0.36 → 0.463 at one checkpoint, though it didn't hold on a later recheck (0.413). Real signal, but not a reliable win. A brief attempt to raise `lr` to `5e-6` was explicitly rejected by the user in favor of tier 3 — "not the standard gspo."

---

## 6. Tier 3 — building and integrating VIGOR

### 6.1 Standalone validation first (separate repo, `/home/hariramanan/NTRO/rollout_allocators/`)

Per the user's explicit instruction to keep this decoupled from the live pipeline until proven, a standalone Python library was built and adversarially validated **before** touching any real training code — implementing VIGOR (arXiv:2607.22002, *"Learning as Reasoning Unfolds"*) and Pilot-Commit (arXiv:2605.26606) behind a common `RolloutAllocator` interface, using a tiered multi-agent workflow (Sonnet for the interface/GRPO-baseline/tests, Opus for the two correctness-critical allocators and adversarial review, Haiku for docs).

- **VIGOR: zero bugs** found under 3,000 randomized adversarial-review configs — cumulative-group advantage normalization, budget conservation, monotonic active-set narrowing all verified correct against the paper.
- **Pilot-Commit: 6 real bugs** found and fixed (a buffered replay entry silently overriding a freshly-piloted state and discarding paid-for fresh rollouts; a stale duplicate causing one prompt to train twice; `max_policy_delay` silently inert because the age clock never advanced on the default call signature; a default ranking policy that quietly re-imposed the p≈0.5 preference the paper explicitly argues against; two minor dead-config issues).

This confirmed VIGOR was the trustworthy piece to integrate first.

### 6.2 Real integration into VERL (`grpo/run_gspo_vigor.py`)

**The key simplification that made this tractable:** VERL's own `compute_advantage(adv_estimator="grpo")` already groups purely by a `uid` non-tensor column via `defaultdict(list)`, with **zero assumption of uniform count per uid**. This meant the standalone library's own `compute_advantages()` (built for the isolated validation, operating on `Rollout` dataclasses) was *not* reused — instead, only VIGOR's **round/select/expand algorithm** was reimplemented, operating directly on VERL's `DataProto` + `uid`, and VERL's existing advantage/loss/optimizer machinery was left untouched.

**Architecture** (`VigorTaskRunner.run()` mirrors VERL's own `main_ppo_v0.py::TaskRunner.run()` through `RayPPOTrainer(...)` → `init_workers()`, then calls a custom `vigor_fit()` instead of `.fit()`):

Per training step:
1. Assign a real VERL `uid` per prompt.
2. Run up to `T` rounds of `async_rollout_manager.generate_sequences()` against a **frozen policy snapshot** — verified structurally that weight sync (`checkpoint_manager.update_weights()`) is the *only* path connecting actor weights to rollout replicas, and it is called only pre-loop and after `_update_actor`, never between rounds. This is VIGOR's core correctness invariant.
3. Score each round with `reward_ie.compute_score` (reusing the agent-loop's own `rm_scores` by default — `vigor.score_source=auto` — so the selection reward is bit-identical to the training reward).
4. Top-alpha variance selection each round on **cumulative** per-prompt reward history (population variance, `ceil(alpha·|active|)` survivors, `m ← round(m·γ)` for survivors, prior rounds never discarded).
5. Merge all rounds via `DataProto.concat`, then the standard `_compute_old_log_prob` → `compute_advantage` → `_update_actor` path, unmodified.

**A real bug this surfaced in VERL's own assumptions (not a VIGOR bug):** `_update_actor` computes minibatch size as `ppo_mini_batch_size × rollout.n`, which asserts exact divisibility against the actual row count (`tensordict_utils.py:588`, `engine_workers.py:266`) and is also the loss's `seq-mean-token-mean` denominator. A ragged VIGOR batch generally isn't a multiple of that stale product — this needed `choose_effective_mini_batch_size()` (snaps to the nearest valid divisor of the real row count) plus a temporary config override (`rollout.n=1`, pinned `ppo_mini_batch_size`) around the `_update_actor` call so VERL's own recomputed product matches reality. Separately confirmed the sequence-level GSPO loss itself (`compute_policy_loss_gspo`) never reads `uid` and is entirely per-row — so minibatch splitting cannot corrupt the ratio/clip computation regardless.

**Adversarial review of the integration** (Opus, before any GPU time was spent) found and fixed 4 real defects:
- `rm_scores` was written to the rollout output *after* the round's `DataProto.union()` — the score never reached the merged training batch, `KeyError` on step 1. Fixed by scoring before the union.
- The `direct` scoring path read `ground_truth` from the wrong `DataProto` (the agent loop doesn't echo prompt-side non-tensors back onto its output) — every direct-mode reward would have silently scored against `ground_truth=None`. Fixed to read from the prompt-side batch, with hard `RuntimeError` guards replacing the previous silent `None`.
- A rollout-correction warning tested a config node that is *never* `None` (so it false-fired on every run regardless of actual settings) — fixed to test the real enabling switches.
- The documented smoke-test example command in the wrapper script was itself un-runnable (missing an override the actor's own validator requires) — fixed.

**Live smoke test — required, and initially blocked by an external tenant.** Static review is not suficient for training-loop code; a real, tiny, live run was mandated before trusting this on real data. First 3 attempts failed — not from a code defect, but because a root-owned Docker container (`vllm_gemma_4_26b_a4b_it`, serving `google/gemma-4-26B-A4B-it`) was occupying 42 of 49 GB of the shared GPU. Stopped it (`docker stop vllm_gemma_4_26b_a4b_it`, with explicit user confirmation given the shared/production nature of that container), freed the GPU, and the 4th attempt succeeded — real `[vigor]` diagnostics matched the theoretical budget-conserving schedule exactly (`budget_used=8/8`, round-by-round active-set shrinkage `4→2`, group sizes `1–3`), real checkpoints written, real finite loss/grad-norm values.

---

## 7. The real VIGOR training run — results

Launched from the tier-2 best checkpoint (`gspo_qwen3_5_0.8b_sft_l19l23/global_step_100_merged`, val reward 0.7090), same LoRA layers 19+23, `VIGOR_M0=2 VIGOR_T=4 VIGOR_GAMMA=2.0 VIGOR_ALPHA=0.5` (the paper's own budget-conserving configuration: `α·γ = 1`), per-step budget `= train_batch_size × m0 × T = 16×2×4 = 128` rollout rows (vs. the plain-GSPO baseline's `16×4=64`/step).

**Early steps (≤75) looked *worse* than the plain-GSPO baseline at the same step count** — this was reported honestly at the time, not spun: score 0.702 vs. 0.730, topic_f1 0.423 vs. 0.463, despite VIGOR spending 2× the rollout budget to get there. This matched VIGOR's own efficiency framing (benefit compounds over a longer horizon, not step-for-step early) — but that claim hadn't been verified yet either.

**It held up.** Validation reward over the full run:

| Step | val reward | | Step | val reward |
|---|---|---|---|---|
| 0 | 0.7064 | | 300 | 0.7122 |
| 75 | 0.7066 | | 400 | **0.7158** |
| 150 | 0.7092 | | 875 | **0.7181** (peak) |
| 200 | 0.7103 | | 950 | 0.7172 (best *saved* checkpoint) |
| 250 | **0.7129** | | 1010 | 0.7108 |

Both prior tiers topped out at ~0.7090 and stayed flat for hundreds of steps; VIGOR broke that ceiling from step ~200 onward and kept climbing, non-monotonically but with a clear upward trend, past **0.716–0.718** by step 875–950 — a genuine ~1% absolute gain sustained over hundreds of steps, not noise.

**Qualitative confirmation at step 1010** (15-sample offline eval, the best result of the entire project):

| Metric | Value |
|---|---|
| score | **0.738** |
| format | 1.0 |
| entity_f1 | 0.751 |
| subtype | 0.839 |
| **topic_f1** | **0.493** (best of any checkpoint tested) |
| **action_f1** | **0.80** (best of any checkpoint tested) |
| unparseable / truncated / hit-token-cap | 0 / 0 / 0 |

Visually confirmed correct schema throughout — proper `type`/`subtype`, real CAMEO topic and action codes, zero repetition.

---

## 8. Operational incident — disk exhaustion, diagnosis, and fix

Around step ~1019 the training process died with **no Python traceback** — the log simply cut off mid-line. Circumstantial evidence pointed to disk exhaustion, not a code defect:

- `df -h /home` showed 94% full (111 GB free) at the time.
- `global_step_1020`'s checkpoint was corrupt — only a 282 MB partial model file where a complete checkpoint is ~4 GB (missing optimizer state, `extra_state`, config) — timestamped at the exact crash moment.
- The active VIGOR run alone had silently accumulated **103 checkpoints at `SAVE_FREQ=10`, totaling 385 GB** — the actual root cause: unbounded checkpoint retention on a shared, finite disk.

**Fix, in order:**
1. Verified `latest_checkpointed_iteration.txt` still pointed to the last *good* checkpoint (1010, not the corrupt 1020) — no resume-safety risk.
2. Pruned to best (`global_step_950`, val 0.7172) + last-good (`global_step_1010`) + the two merged eval snapshots already in use — reclaimed 374 GB (94%→71% disk usage, 111 GB→485 GB free).
3. Killed orphaned Ray worker actors left by the crash (the driver died but several `ray::VigorTaskRunner.run` workers and a stray `VLLM::Worker` survived, holding GPU memory).
4. Relaunched, resuming cleanly from step 1010, with `SAVE_FREQ` raised **10 → 50** to prevent recurrence, and lowered rollout GPU-memory-utilization to accommodate the (unrelated) external gemma container's fluctuating usage.
5. The relaunch itself initially failed too — the external container's memory usage spiked to leave only 1.8 GB free at that exact moment — resolved by stopping the container again with explicit user confirmation.

**Standing operational lesson:** any long RL run with per-step checkpointing needs either an explicit retention policy (best+last, pruned periodically) or a much larger `SAVE_FREQ`; `SAVE_FREQ=10` on ~4 GB/checkpoint silently becomes hundreds of GB over a few thousand steps.

---

## 9. Current status and what's still open

- **Best validated checkpoint:** `outputs/gspo_vigor_qwen3_5_0.8b_run1/global_step_1010_merged` — score 0.738, hosted live via vLLM (`scripts/06_serve_vllm.sh`) for interim use while training resumes.
- **Training:** paused at checkpoint 1010 pending stable GPU availability (shared-GPU contention from an unrelated, intermittently-restarting external container is the main operational friction at this scale, not anything in this pipeline).
- **Not yet done, explicitly out of scope so far:**
  - Pilot-Commit integration into the real training loop (only the standalone, mock-rollout version exists; deferred since VIGOR alone already broke the plateau).
  - A finalized 32-type Defense/Security NER subtype taxonomy was drafted and audited against the actual training data (see `handover/` or ask for the audit) but **not yet wired into `prepare_parquet.py`/`reward_ie.py`** — the gold data and reward function still score against the old, more granular subtype labels. Explicitly deferred by the user ("not needed now").
  - Topic-code diversity was separately audited and found healthy (55 distinct CAMEO codes, moderately skewed but not degenerate, negligible long tail) — the persistent topic_f1 weakness is attributed to genuine semantic difficulty of a 55-way skewed classification, not a data or formatting artifact.

## 10. Two-line environment fixes worth remembering (unrelated to the ML pipeline, encountered while hosting)

- `VLLM_USE_FLASHINFER_SAMPLER=0` — the installed FlashInfer build's bundled CCCL headers reject both available `nvcc` versions on this machine (system one too old, venv-bundled one too new by FlashInfer's own compatibility check). This is a genuine version gap, not a PATH issue; disabling the FlashInfer sampler is vLLM's documented escape hatch, with no output-quality cost.
- vLLM's default `--max-num-seqs=256` reserves one Mamba cache block per concurrent sequence for this hybrid architecture; at low `--gpu-memory-utilization` (necessary when sharing a GPU), the available block count can fall below 256 and CUDA graph capture fails. Lower `--max-num-seqs` (64 has worked reliably here) rather than raising memory utilization if the GPU is contended.
