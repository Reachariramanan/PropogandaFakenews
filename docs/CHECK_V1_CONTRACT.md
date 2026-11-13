# Engine service: `POST /api/check_v1` (FNP dashboard :8879)

Used by the Evidence Desk "Claim Check" screen. Step 1 asks whether the text is fake; step 2 shows the propaganda techniques.

**Request**: `{"text": "...", "lang": "en"?}`. Limits: text ≤ 20,000 chars, body ≤ 256 KB.

**Status codes**
| Code | When | Body / headers |
|---|---|---|
| 200 | OK, including degraded runs | see below |
| 400 | empty text or bad JSON | |
| 413 | text or body too large | |
| 503 | busy (2 checks already running) | `{"error": "busy"}` + `Retry-After: 30` |
| 500 | unexpected error | |

The hard timeout is 55 s. When it hits, the response is still HTTP 200, with `degraded: true`, `verdict: null`, encoder techniques only, and an `error` message. Typical latency is 10–25 s warm and depends on load on the shared Gemma. The models preload when the service starts.

**Response (200)**
```json
{ "verdict": {"label": "FAKE|PROPAGANDA|LEGIT|MISLEADING|INCONCLUSIVE", "probs": {...5 classes...},
              "plausibly_fake": false, "prediction_set": ["FAKE"], "rationale": "...", "decider": "stacker"},
  "fake_call": {"label": "FAKE|MISLEADING|LEGIT|INCONCLUSIVE", "probs": {...4 classes...},
                "fake_score": 0.98, "basis": "stacker|gemma|none"},
  "laya": {"label": "...", "probs": {...5 classes...}, "confidence": 0.52} | null,
  "laya_reason": null | "no_claims" | "timeout",
  "claims": [{"claim": "...", "label": "...", "probs": {...}, "laya_probs": {...}|null,
              "evidence": [{"evidence_id": "...", "stance": "refute", "source": "...", "url": "...", "kb_claim": "..."}]}],
  "propaganda": {"is_propaganda_p": null, "laya_is_propaganda_raw": 0.0,
                 "technique_probs": {"Loaded_Language": 0.99, "Name_Calling,Labeling": 0.56, "...": "all 14 merged classes"},
                 "p_prop": 0.74 | 0.0 | null, "too_short": false, "n_tokens": 32,
                 "techniques": [{"technique": "Name_Calling,Labeling", "start": 82, "end": 98,
                                 "text": "traitorous media", "confidence": 0.56, "strong": true}],
                 "one_sided": 1.0},
  "degraded": false, "degraded_reason": null | "no_claims" | "timeout" | "gemma_error", "error": null,
  "model_versions": {"laya": "laya_fnp_rl_v1", "deciders": "deciders_v1", "span_encoder": "prop_span_v1",
                     "kb": "kb_v2", "fake_call": "fake_call_v1"},
  "pipeline_s": 14.9, "secs": 15.5, "queue_wait_s": 0.0 }
```

## How to use it
- **Step 1 (fake or not):** use `fake_call`. The `verdict` block is the raw 5-class stacker output, and its PROPAGANDA probability is a known length artefact.
  - `fake_call.label` ∈ FAKE / MISLEADING means go on to step 2.
  - `fake_call.basis == "none"` means no factual call was made, so show INCONCLUSIVE.
- **Step 2 (techniques):** use `propaganda.techniques`, the 14 merged PTC buckets with character offsets into the text you sent. They come from the span encoder, which isn't validated on social media, so present them as signals.
- **`is_propaganda_p` is always null for now.** Laya's is_propaganda head was trained on article-only propaganda labels and reads about 0 on short texts. Its raw value is in `laya_is_propaganda_raw`, for transparency only.
- **Deciders:**
  - The verdict uses the calibrated stacker (ECE 0.055).
  - `stacker_laya` isn't fitted yet: an in-sample fit over-trusted Laya (ECE 0.22). P1 refits it on out-of-fold Laya probabilities, and `verdict.decider` will then read `stacker_laya`.
  - Laya's own verdict is reported in `laya`.
- **`degraded` / `degraded_reason`:**
  - `"gemma_error"`: claims were extracted but every Gemma call failed. `degraded` is true; the verdict is unreliable, so offer a retry.
  - `"timeout"`: the 55 s cap hit. `degraded` is true, `verdict` is null, and only the encoder fields are filled.
  - `"no_claims"`: no checkable claim was extracted (typical for long opinion text). `degraded` is **false**: nothing failed. `verdict` is the empty-claims rollup, `fake_call.basis` is `"none"`, `laya` is null with `laya_reason: "no_claims"`.
  - Otherwise `degraded_reason` is null.
- **`laya` when there are no claims:** Laya scores one claim-state at a time, so with no claims there is nothing to run. There is no document-level fallback. `laya: null` with `laya_reason: "no_claims"` (or `"timeout"`).

## Propaganda fields (all from the span encoder `prop_span_v1`, not validated on social media)
- **`technique_probs`:** all 14 merged PTC classes. Each value is the MAX token probability over the text, in 0..1, whether or not it passed the class threshold. Use it for the bars. On short texts the values are small but real, not forced to 0. Also present on timeout.
- **`techniques[].strong`:** true when the span's mean token probability is ≥ max(class threshold, 0.5). Same rule as the Propaganda Lab.
- **`p_prop`:** the highest mean probability among strong spans. 0.0 when there are no strong spans, **null when `too_short`**. It is an uncalibrated signal score, **not** a probability that the text is propaganda, so don't caption it as one.
- **`too_short` / `n_tokens`:** `n_tokens` is the number of encoder tokens (the 6,000-character cap applies first). `too_short` is true below 12 tokens, where the encoder over-fires on titles. Show `technique_probs` as weak signals only.
- **`one_sided`:** a float in 0..1, not a bool. It is the Gemma span tagger's own judgement of how one-sided the framing is (0 = balanced or neutral facts, 1 = one side only, opponents caricatured). Not validated. 0.0 can also mean "tagger returned nothing or failed", so do not read 0.0 as proof of balance. Null on timeout.
- **`laya_is_propaganda_raw`: unreliable everywhere, caption it "not validated".**
  - What it is: the Laya `is_propaganda` yes/no head's P(true) in [0,1], taken as the maximum over claims. Null when there are no claims.
  - What it reads (measured 2026-11-12, `laya_fnp_rl_v1`, 40 states per group sampled from `units_v1`): **1.0 on all 40 PTC article states and 0.0 on all 40 other states** (liar / x-fact / boom / fakenewsnet).
  - Also observed: a loaded 30-token sentence ("These traitorous media elites are lying…") read **0.876**, so the head is not always 0 on short text. It does respond to loaded wording or to the p_prop features inside the state.
  - Reading: a perfect 1.0 / 0.0 split on dataset groups is a sign of a dataset shortcut (PTC news articles are the only propaganda labels). That is a hypothesis, not proven. There is no validation on social posts, so caption it "not validated" and don't use it for decisions. Retrain the head on short-post labels (docs/review/05, problem 1) before trusting it.
