# How FNP decides: fake, propaganda, misleading, legit or inconclusive

*One section per slide. No formulas: for each step, what goes in, what it does and what comes out.*

---

## Slide 1: The question

- **Input:** a social post, a news article or a single claim.
- **Output:** one of 5 labels.
  - **FAKE:** the claim is false.
  - **MISLEADING:** real content in a false context, or a mix of true and false.
  - **PROPAGANDA:** persuasion techniques that push one side, even when nothing is outright false.
  - **LEGIT:** supported by evidence.
  - **INCONCLUSIVE:** not enough evidence to decide. The system abstains rather than guesses.
- **Plus a "plausibly fake" flag:** fake or misleading is likely, but not certain enough to say so outright.
- **Core idea:** the LLM *reads* the text, and the maths *weighs* the evidence. Neither decides alone.

---

## Slide 2: The pipeline at a glance

```
Text → 1 Claims → 2 Propaganda spans → 3 Evidence → 4 De-duplicate → 5 Trust sources
     → 6 Combine evidence → 7 Decide (+ abstain) → Post/article verdict
```

- Steps 1–3 use **Gemma**, the LLM. It reads and understands the text.
- Steps 4–7 use **maths**: counting, weighing and checking consistency.
- Step 7 adds two **learned deciders**: a stacker and **Laya**, a small trained model.

---

## Slide 3: Step 1, break the text into claims

- **Input:** the post or article text.
- **What it does:** Gemma pulls out each checkable statement ("Video shows X in city Y on date Z"). The system checks that every claim is quoted word-for-word from the text, so it can't be invented.
- **Output:** a list of claims. Each has who said it, its type, and how worth checking it is.
- **Why:** an article can mix true and false statements. We judge each claim, then combine.

---

## Slide 4: Step 2, find propaganda techniques

- **Input:** the text.
- **What it does:** Gemma marks exact phrases that use any of 18 known techniques, such as loaded language, name-calling, fear appeals, flag-waving, whataboutism and exaggeration. The 18 come from the standard research taxonomy (SemEval PTC).
- **Maths on top:**
  - **Propaganda score:** "how likely is at least one of these marks real?" Each technique is weighted by how often the model gets it right.
  - **Density:** techniques per 100 words.
  - **Variety:** how many different techniques appear.
  - **One-sidedness:** does it only ever push one side?
- **Output:** a propaganda probability plus the highlighted spans as evidence.

---

## Slide 5: Step 3, gather evidence for each claim

- **Input:** one claim.
- **What it does:**
  - Searches a knowledge base of about 32,000 already fact-checked claims, from BOOM, Vishvas News, Quint WebQoof, Snopes, Full Fact, PolitiFact and others.
  - An embedding model finds similar claims, and a reranker orders them.
  - Gemma then reads each match and says whether it **supports**, **refutes**, shows a **context mismatch** (real media, wrong story) or is **neutral**. It also rates relevance and evidence type: official record, photo forensics, eyewitness and so on.
- **Output:** for each claim, a list of evidence items, each with a stance and a strength.

---

## Slide 6: Step 4, don't count the same evidence twice

- **Input:** the evidence list.
- **What it does:** groups evidence that comes from the same origin (same link, same source, or near-identical text) and gives each group a single vote. It also estimates which sources copy each other.
- **Output:** the **effective number of independent sources**. Ten reposts of one tweet count as about 1, not 10.
- **Why:** coordinated reposting should not look like broad confirmation.

---

## Slide 7: Step 5, how much to trust each source

- **Input:** each source's type and its track record across many claims.
- **What it does:**
  - Starts from a prior by type: an official record or forensic analysis is trusted more than an anonymous post.
  - Updates that prior with the source's history: how often it was right, how often it was debunked.
  - **Truth discovery** (Dawid–Skene, TruthFinder) learns source accuracy and claim truth together. Reliable sources are the ones that agree with what turns out to be true.
- **Output:** a reliability score for each source.

---

## Slide 8: Step 6, combine the evidence (the core maths)

Each method answers a different question about the same evidence:

| Method | Question it answers | Output |
|---|---|---|
| **Weight of evidence** (log-odds) | Adding up every for/against item, scaled by trust and independence, how likely is the claim true? | P(true) |
| **Dempster–Shafer** | How much is believed, how much is merely *possible*, and how much is unknown? Conflicting evidence becomes "unknown" instead of being forced into a side. | belief, plausibility, uncertainty |
| **Meta-analysis** (random effects) | Do the sources agree? How wide is the honest error band? | confidence interval, disagreement level |
| **Information gain** | Did the evidence actually change our mind, or are we still at the starting guess? | bits learned |
| **Belief propagation** | Are the claims in one story consistent with each other? If A implies B and B is false, A suffers. | adjusted P(true) per claim |
| **Centrality** (PageRank) | Which claims are central to the story? | importance per claim |
| **Timeline check** | Is the chronology possible, e.g. a "new" video that existed years earlier? | P(timeline is inconsistent) |
| **Tensor decomposition** | Do groups of accounts post the same claims at the same time? | coordination score |
| **Spread signals** (HMM bursts, from EDA) | Did it spike suddenly, cross platforms, come from a few authors, get duplicated? | burst, spread and duplication features |

**Output:** 30 numbers per claim, the claim's "feature vector".

---

## Slide 9: Step 7, decide

- **Input:** the 30 numbers from step 6, plus a short summary of the evidence.
- **Three deciders:**
  1. **Rule head:** transparent rules. Very low P(true) plus a strong independent refutation gives FAKE. High P(true) plus low propaganda gives LEGIT, and so on.
  2. **Gemma adjudicator:** reads the *structured* evidence only, never the raw text, so it can't be swayed by the writing style. It explains its reasoning.
  3. **Stacker:** a simple learned model that weighs the 30 numbers, the rule and Gemma's vote against thousands of gold-labelled examples.
- **Laya**, a small 322M-parameter decision model, is trained on the same inputs:
  - first by imitation (supervised),
  - then by **reinforcement learning**, rewarded for honest, well-calibrated probabilities and penalised for lazy "inconclusive" answers.
- **Output:** a probability for each of the 5 labels.

---

## Slide 10: Knowing when not to answer

- **Calibration:** makes sure "80% sure" really is right about 80% of the time.
- **Conformal prediction:** builds the set of labels the system can't rule out at a chosen error rate.
  - One label left: it answers.
  - Several left: it says **INCONCLUSIVE**.
- Also INCONCLUSIVE when uncertainty is high or there is less than one independent source.
- **Routing (RL bandit):** the system learns when the cheap Laya answer is enough, and when to escalate to Gemma or a human analyst.

---

## Slide 11: From claims to the whole post or article

- **Input:** the verdicts for each claim, plus the post's propaganda score.
- **Rules, in order:**
  1. Any important claim is clearly FAKE → the post is **FAKE**.
  2. Else any claim is false or out of context → **MISLEADING**.
  3. Else strong propaganda → **PROPAGANDA**.
  4. Else all claims are supported → **LEGIT**.
  5. Otherwise → **INCONCLUSIVE**.
- **Output:** a post verdict, secondary tags (for example "also propaganda") and the full evidence trail.

---

## Slide 12: Results so far (945 test claims)

| Decider | Macro-F1 | Calibration error |
|---|---|---|
| Gemma reading raw text | 0.25 | high |
| Gemma on structured evidence | 0.38 | high |
| Maths + Gemma stacker | 0.55 | **lowest (0.055)** |
| **Laya (trained)** | **0.61** | low |
| Stacker + Laya, abstaining when unsure | 0.73 on 26% of claims | — |

- **The takeaway:** LLM plus maths beats the LLM alone by a wide margin.
- **Open issues:**
  - The propaganda label is only tested on long articles, and over-fires on short social posts.
  - MISLEADING recall is low.
  - Evidence about *similar but different* events sometimes counts as a refutation.
