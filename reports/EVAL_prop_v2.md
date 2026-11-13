# Propaganda span/technique tagger: final comparison (PTC held-out)

Scorer `fnp/eval/ptc_scorer.py`; splits `data/splits/ptc_v1.json`. Dev was used for thresholds / per-class source choice; **test scored once**.

| system | split | SI P | SI R | SI F1 | span+tech P | R | F1 |
|---|---|---|---|---|---|---|---|
| v0 Gemma (orig. prompt, 8k trunc) | dev | 0.545 | 0.227 | 0.320 | 0.295 | 0.122 | 0.173 |
| v0 Gemma (orig. prompt, 8k trunc) | test | 0.435 | 0.206 | 0.280 | 0.233 | 0.112 | 0.152 |
| enc_only | dev | 0.487 | 0.413 | 0.447 | 0.316 | 0.297 | 0.306 |
| enc_only | test | 0.383 | 0.434 | 0.407 | 0.228 | 0.327 | 0.268 |
| gem_only | dev | 0.403 | 0.645 | 0.496 | 0.174 | 0.376 | 0.238 |
| gem_only | test | 0.314 | 0.656 | 0.424 | 0.154 | 0.417 | 0.225 |
| union_all | dev | 0.380 | 0.731 | 0.500 | 0.192 | 0.488 | 0.275 |
| union_all | test | 0.290 | 0.743 | 0.417 | 0.147 | 0.529 | 0.230 |
| per_class_best(dev-selected) | dev | 0.474 | 0.478 | 0.476 | 0.301 | 0.323 | 0.312 |
| per_class_best(dev-selected) | test | 0.369 | 0.480 | 0.417 | 0.216 | 0.342 | 0.264 |

## Per technique, test F1 (gold spans)

| technique | gold | v0 | enc | gem | union_all | per-class best (src) |
|---|---|---|---|---|---|---|
| Loaded_Language | 608 | 0.223 | 0.359 | 0.253 | 0.251 | 0.359 (enc) |
| Name_Calling,Labeling | 305 | 0.189 | 0.340 | 0.285 | 0.280 | 0.340 (enc) |
| Repetition | 223 | 0.000 | 0.145 | 0.033 | 0.145 | 0.145 (enc) |
| Exaggeration,Minimisation | 127 | 0.118 | 0.185 | 0.156 | 0.161 | 0.185 (enc) |
| Doubt | 122 | 0.071 | 0.197 | 0.198 | 0.228 | 0.197 (enc) |
| Appeal_to_fear-prejudice | 54 | 0.063 | 0.220 | 0.166 | 0.208 | 0.208 (union) |
| Flag-Waving | 58 | 0.062 | 0.242 | 0.178 | 0.251 | 0.251 (union) |
| Causal_Oversimplification | 53 | 0.084 | 0.080 | 0.137 | 0.140 | 0.140 (union) |
| Slogans | 38 | 0.083 | 0.264 | 0.326 | 0.305 | 0.264 (enc) |
| Appeal_to_Authority | 39 | 0.000 | 0.031 | 0.125 | 0.128 | 0.031 (enc) |
| Black-and-White_Fallacy | 33 | 0.051 | 0.108 | 0.068 | 0.132 | 0.068 (gem) |
| Thought-terminating_Cliches | 13 | 0.000 | 0.094 | 0.097 | 0.119 | 0.094 (enc) |
| Whataboutism,Straw_Men,Red_Herring | 23 | 0.088 | 0.097 | 0.132 | 0.095 | 0.097 (enc) |
| Bandwagon,Reductio_ad_hitlerum | 14 | 0.085 | 0.002 | 0.169 | 0.133 | 0.169 (gem) |

## Decision and caveats

- **Primary tagger: fine-tuned encoder** (`fnp/llm/propaganda_enc.py`, weights `data/models/prop_span_v1/model.pt`): best span+technique F1 on test (0.268 vs 0.152 for v0, 0.225 for Gemma chunk_v2), runs locally with no Gemma calls (about 0.2 s/post on CPU).
- **Gemma chunk_v2** (`fnp/llm/propaganda_v2.py`) is the high-recall option: best span-only F1 (0.42-0.50) and better on Appeal_to_Authority, Causal_Oversimplification, Bandwagon, Slogans.
- The dev-selected per-class ensemble does **not** beat the encoder alone on test (0.264 vs 0.268): per-class choices on classes with <15 dev spans are noise, so it is not adopted. `union_all` raises recall to 0.74 but precision drops to 0.29.
- All numbers are PTC (English news, 2017-19) with the fractional-overlap scorer in this repo, not the official SemEval scorer. Test = 85 articles.
- **Not validated on social media.** A 150-post sanity run flagged 83% of posts and over-fires on short neutral titles ("Fake news", "factchecked ... as true"); the doc-level PROPAGANDA rollup must not use these spans until a hand-labelled social gold set exists (plan Phase 0, item 4).
- Still weak: Appeal_to_Authority (0.03 enc), Causal_Oversimplification (0.08 enc), Whataboutism group, Thought-terminating Cliché, Repetition (0.145).
