# Propaganda Span Detection and Forensic Evidence Probability Systems

**Working Notes & System Designs — a consolidated record of five working documents**

Source paper: [arXiv:1910.02517](https://arxiv.org/pdf/1910.02517) (Da San Martino et al., EMNLP 2019)

## Contents

- [Part I — Understanding: Fine-Grained Analysis of Propaganda in News Articles](#part-i--understanding-fine-grained-analysis-of-propaganda-in-news-articles)
- [Part II — NEET Leak: Preliminary Evidence Assessment](#part-ii--neet-leak-preliminary-evidence-assessment)
- [Part III — Forensic Evidence Intelligence System (Architecture)](#part-iii--forensic-evidence-intelligence-system-architecture)
- [Part IV — Probabilistic Evidence Analysis System (Agents &amp; Prompts)](#part-iv--probabilistic-evidence-analysis-system-agents--prompts)
- [Part V — Evidence-Weighted Probability Methodology](#part-v--evidence-weighted-probability-methodology)

---

# Part I — Understanding: Fine-Grained Analysis of Propaganda in News Articles

**Paper.** *Fine-Grained Analysis of Propaganda in News Articles* by Giovanni Da San Martino et al. (EMNLP 2019). The paper introduces a new NLP task: detecting *exact spans* of propaganda in news articles and identifying the *specific propaganda technique* being used.

## Why this paper matters

Before this work, most propaganda detection systems operated at:

- News outlet level ("this site is propaganda")
- Article level ("this article is propaganda")

The authors argue this is too coarse and often noisy. Instead of asking:

> "Is this article propaganda?"

they ask:

> "Which exact words are propaganda, and what technique is being used?"

This makes the system more explainable, more actionable, and better for fact-checking and media literacy.

## Core problem

Given a news article:

```text
The corrupt elite are destroying our country.
```

The model should:

1. Detect the propagandistic span

```text
corrupt elite
```

2. Classify its propaganda technique

```text
Name Calling / Labeling
```

This becomes a sequence labeling + span classification problem.

## The 18 propaganda techniques

The paper defines 18 fine-grained propaganda techniques. Some important ones:

| Technique                   | Example                     |
| --------------------------- | --------------------------- |
| Loaded Language             | emotionally charged words   |
| Name Calling / Labeling     | "traitor", "radical"        |
| Flag-Waving                 | appeal to patriotism        |
| Appeal to Fear              | creating fear               |
| Doubt                       | casting suspicion           |
| Exaggeration / Minimization | overstating or understating |
| Repetition                  | repeating claims            |
| Slogans                     | short memorable phrases     |
| Whataboutism                | redirecting criticism       |
| Black-and-White Fallacy     | only two options presented  |
| Causal Oversimplification   | complex issue → one cause  |
| Bandwagon                   | "everyone believes this"    |

These techniques can overlap and occur at different text lengths.

## Dataset

The authors built a manually annotated corpus. Characteristics:

- News articles
- Expert annotators
- Fragment-level annotations
- About 350K tokens
- Every propaganda span labeled with one of the 18 techniques

This was one of the first high-quality propaganda datasets at span level.

## Main technical challenge

The techniques operate at different granularities. Example:

### Loaded Language

```text
evil regime
```

Very short span.

### Repetition

May require multiple occurrences across sentences.

### Flag-Waving

May require sentence-level context.

So a model needs token-level, phrase-level, and sentence-level understanding simultaneously.

## Proposed model

The paper proposes a **Multi-Granularity Neural Network (MGN)**. High-level architecture:

```text
Article
   ↓
Sentence Encoder
   ↓
Word Encoder
   ↓
Context Features
   ↓
Technique Classifier
```

The model combines:

### 1. Word-level information

Individual tokens.

### 2. Sentence-level information

Meaning of the whole sentence.

### 3. Document-level context

Broader article information.

The intuition:

```text
word meaning
+ sentence meaning
+ article context
```

helps identify propaganda more accurately.

## Evaluation

The authors point out a subtle issue. Normal classification metrics don't work well because:

```text
Predicted span:
"The corrupt elite"

Gold span:
"corrupt elite"
```

The prediction is mostly correct but not an exact match. So they design an evaluation measure that gives partial credit for overlapping spans instead of treating them as completely wrong.

## Results

Their Multi-Granularity Network outperformed several strong BERT-based baselines available at the time. The key takeaway:

> Simply applying BERT was not enough; explicitly modeling multiple context levels improved propaganda span detection.

## What is most relevant today (2026)?

The paper's biggest contribution is **the task definition and dataset**, not necessarily the model. Modern approaches would likely replace the architecture with:

- BERT
- RoBERTa
- DeBERTa
- Longformer
- Llama-style models
- Instruction-tuned LLMs

but still use span detection, the propaganda taxonomy, and fine-grained annotations. The dataset from this paper became a foundation for later propaganda and persuasion-technique detection research.

## The paper in one sentence

> Instead of labeling an entire article as propaganda, this work teaches NLP systems to **highlight the exact text fragments that are propagandistic and identify which persuasion technique is being used**, making propaganda detection far more explainable and precise.

## References (Part I)

- [1] [Fine-Grained Analysis of Propaganda in News Articles — arXiv:1910.02517](https://arxiv.org/abs/1910.02517)
- [2] [arXiv:1910.02517v1 [cs.CL] 6 Oct 2019 (PDF)](https://arxiv.org/pdf/1910.02517)

---

# Part II — NEET Leak: Preliminary Evidence Assessment

This can be done rigorously, but not honestly with the current evidence set. The request was for:

- 30 articles
- 3 collection rounds
- Hypothesis probabilities
- Article-level probabilities
- Consolidated probabilities
- Final report

The issue is that a large fraction of the search results available right now are either:

- 2026 NEET leak stories
- secondary summaries
- social posts
- commentary
- articles reporting on the same court filing

If probabilities were assigned now, that would be manufacturing precision rather than performing evidence analysis.

## What can already be concluded from the evidence collected

The strongest evidence currently found comes from:

- Supreme Court judgment (July 23, 2024)
- CBI submissions
- SC Observer reporting of court proceedings
- Arrest records
- Hazaribagh source tracing

These establish:

### H1: Paper leak occurred before exam

- Strongly supported.
- Supreme Court explicitly treated the leak as established fact.

### H2: Leak reached candidates

- Strongly supported.
- CBI identified beneficiaries and traced transmission from Hazaribagh to Patna.

### H3: Organized criminal activity

- Strongly supported.
- Multiple arrests, solver modules, paper theft allegations, distribution chain.

### H4: Localized impact

- Supported.
- Court repeatedly references Hazaribagh and Patna concentration.

### H5: National ranking materially altered

- Not established.
- Court found insufficient evidence.

### H6: Entire exam compromised

- Not established.
- Court rejected systemic breach argument.

### H7: Nationwide re-exam necessary

- Court rejected.

## Preliminary Bayesian-style estimate

These are not final numbers; they are provisional priors from the strongest sources only.

| Hypothesis                              | Probability |
| --------------------------------------- | ----------- |
| H1 Leak occurred                        | 99%         |
| H2 Leak reached candidates              | 97%         |
| H3 Organized criminal network           | 95%         |
| H4 Localized candidate impact           | 90%         |
| H5 National rankings materially altered | 30%         |
| H6 Entire exam compromised              | 15%         |
| H7 Nationwide re-exam justified         | 20%         |

The logic follows Bayesian evidence accumulation.

## What a proper 30-article investigation requires

For each article:

```text
Article A
├── Claims
├── Evidence type
├── Source quality
├── Independence score
├── Contradictions
├── Bayesian update
└── Posterior probabilities H1-H7
```

Then:

```text
30 articles
        ↓
Claim graph
        ↓
Deduplicate shared sources
        ↓
Weight evidence
        ↓
Posterior probability matrix
        ↓
Final report
```

The critical step is **deduplication**. If 15 articles are all quoting the same CBI filing, that is one piece of evidence, not 15.

## Recommendation

To produce the full report described, first gather:

- 10 court/legal sources
- 10 investigative sources
- 10 mainstream news sources

and then build a full evidence matrix with article-by-article posterior probabilities. That will be a fairly large research report (roughly 8–12 pages of analysis). The current search session has not yet collected a clean, independent set of 30 sources, so any article-level probability table right now would be unreliable.

## References (Part II)

- [1] [Supreme Court Judgment, WPC 335/2024 (23 Jul 2024)](https://api.sci.gov.in/supremecourt/2024/22797/22797_2024_1_21_53928_Judgement_23-Jul-2024.pdf)
- [2] [Supreme Court refuses to cancel 2024 NEET UG exam — SC Observer](https://www.scobserver.in/journal/supreme-court-refuses-to-cancel-2024-neet-ug-exam-due-to-insufficient-evidence-of-systemic-leak/)
- [3] [Man who &#39;stole&#39; NEET paper from NTA trunk arrested — cbi.gov.in](https://cbi.gov.in/assets/files/media/217553944Clippings%20dt.%2018.07.2024.pdf)
- [4] [Supreme Court verdict on NEET UG paper leak — Times of India](https://timesofindia.indiatimes.com/education/news/supreme-court-verdict-on-neet-ug-paper-leak-localised-breach-govt-committee-to-issue-exam-sops-and-other-key-takeaways-here/articleshow/112214662.cms)

---

# Part III — Forensic Evidence Intelligence System (Architecture)

For this kind of analysis, the right design is not a simple "research agent + summarizer." It is a **forensic evidence intelligence system** inspired by intelligence agencies, legal discovery systems, and scientific review pipelines. The key principle:

> No article should directly influence the final probability estimate.

Articles are merely evidence containers. The system should reason over **claims**, **evidence**, and **sources**, not articles.

## Level 0 — Executive Layer

```text
                    ┌─────────────────┐
                    │ LEADER AGENT    │
                    │ (Chief Analyst) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ PLANNER AGENT   │
                    │ Investigation   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
 Evidence Division    Probabilistic Div.   Report Division
```

## 1. Leader Agent

### Purpose

Acts as the Chief Intelligence Officer. Never reads articles directly.

Responsibilities:

- Defines hypotheses
- Sets investigation goals
- Resolves agent conflicts
- Requests additional searches
- Approves final report

Input:

```text
Analyze NEET leak
```

Output:

```json
{
  "hypotheses": [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7"
  ]
}
```

## 2. Planner Agent

### Purpose

Creates execution strategy. Example:

```text
Need:
10 legal sources
10 investigative sources
10 media sources

Then:
claim extraction
evidence graph
probability update
report generation
```

Produces work DAG.

## Level 1 — Acquisition Layer

```text
              Planner
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
 Search A     Search B     Search C
```

## 3. Search Agents (3–10 parallel)

### Purpose

Collect articles.

### Search Agent A — Legal

```text
Supreme Court
SC Observer
LiveLaw
Bar & Bench
```

### Search Agent B — Investigative

```text
CBI
Police
Chargesheets
Arrests
```

### Search Agent C — Media

```text
Indian Express
TOI
Hindu
HT
NDTV
```

Output:

```json
{
  "article_id":"A17",
  "url":"...",
  "source":"Indian Express"
}
```

## Level 2 — Understanding Layer

```text
                Articles
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
Claim Agent   Evidence Agent  Timeline Agent
```

## 4. Claim Extraction Agent

Reads article. Converts:

```text
CBI arrested 3 people.
```

Into:

```json
{
  "claim_id":"C41",
  "claim":"CBI arrested 3 suspects"
}
```

This is crucial. Articles disappear after this stage. Only claims remain.

## 5. Evidence Classification Agent

Determines evidence type. Example:

```json
{
  "claim":"Paper leaked",
  "evidence":"Supreme Court finding",
  "tier":"A"
}
```

Categories:

```text
Court Finding
CBI Finding
Police Evidence
Witness Testimony
Anonymous Source
Political Statement
Opinion
```

## 6. Timeline Agent

Builds chronology.

```text
May 5
    |
Paper printed
    |
Hazaribagh
    |
Patna
    |
Beneficiaries
    |
Arrests
```

Used later for consistency checks.

## Level 3 — Verification Layer

Most important layer.

```text
                    Claims
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 Corroboration   Contradiction   Duplication
    Agent           Agent          Agent
```

## 7. Corroboration Agent

Finds supporting claims. Example — Articles 3, 12, and 21 each state "Leak happened". Creates:

```json
{
  "claim":"Leak happened",
  "support_count":3
}
```

## 8. Contradiction Agent

Looks for conflicts. Example:

```text
Leak affected 150
versus
Leak affected 15000
```

Flags conflict.

## 9. Deduplication Agent

Critical. Suppose TOI, NDTV, Indian Express, and HT all cite the same CBI filing.

```text
Naive system:   4 evidences
Correct system: 1 evidence
```

This agent prevents evidence inflation.

## Level 4 — Source Reliability Layer

```text
                    Claims
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
 Reliability     Bias Agent     Independence
    Agent                         Agent
```

## 10. Reliability Agent

Assigns evidence weights.

| Source               | Weight |
| -------------------- | ------ |
| Supreme Court        | 1.00   |
| CBI                  | 0.95   |
| Chargesheet          | 0.90   |
| Investigative Report | 0.80   |
| News Summary         | 0.60   |
| Opinion              | 0.40   |

## 11. Bias Agent

Detects motivated actors. Examples:

```text
Political spokesperson
Exam coaching center
Government official
Opposition leader
```

Bias ≠ false. Bias simply modifies confidence.

## 12. Independence Agent

Determines whether evidence is truly independent. Court, CBI, and Police may not be independent — many may derive from the same underlying record. Creates a dependency graph.

## Level 5 — Hypothesis Layer

This is where the paper's ideas become useful.

```text
                 Evidence Graph
                        │
                        ▼
             Hypothesis Evaluator
```

## 13. Hypothesis Agents (one per hypothesis)

Run independently (H1 Agent … H7 Agent). Each agent sees:

```json
{
  "evidence": [...],
  "contradictions":[...],
  "weights":[...]
}
```

and computes:

```json
{
  "hypothesis":"H1",
  "probability":0.992
}
```

## 14. Bayesian Update Agent

Combines evidence. Instead of "10 articles agree", it computes:

```text
Prior
+
Independent Evidence
-
Contradictions
=
Posterior
```

This is the mathematical core.

## Level 6 — Adversarial Layer

Most systems miss this.

```text
               Posterior Results
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Red Team Agent          Skeptic Agent
```

## 15. Red Team Agent

Attempts to destroy conclusions. Example:

```text
What if all evidence
comes from one CBI filing?
```

## 16. Skeptic Agent

Looks for missing evidence, alternative explanations, and statistical errors. Output:

```text
Confidence overstated.
Reduce H5 by 12%.
```

## Level 7 — Report Layer

```text
                    Results
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
 Report Agent   Visualization    Executive
                    Agent         Summary
```

## 17. Report Agent

Produces: Findings, Evidence, Reasoning, Probabilities, Limitations.

## 18. Visualization Agent

Builds evidence graph.

```text
Article
   |
Claim
   |
Evidence
   |
Hypothesis
```

## 19. Executive Summary Agent

Produces a one-page summary for decision makers.

## Complete Hierarchy

```text
LEADER
│
├── PLANNER
│
├── SEARCH DIVISION
│   ├── Legal Search
│   ├── Investigative Search
│   └── Media Search
│
├── UNDERSTANDING DIVISION
│   ├── Claim Extractor
│   ├── Evidence Classifier
│   └── Timeline Builder
│
├── VERIFICATION DIVISION
│   ├── Corroboration
│   ├── Contradiction
│   └── Deduplication
│
├── RELIABILITY DIVISION
│   ├── Reliability
│   ├── Bias
│   └── Independence
│
├── HYPOTHESIS DIVISION
│   ├── H1 Agent
│   ├── H2 Agent
│   ├── H3 Agent
│   ├── H4 Agent
│   ├── H5 Agent
│   ├── H6 Agent
│   └── H7 Agent
│
├── BAYESIAN AGENT
│
├── ADVERSARIAL DIVISION
│   ├── Red Team
│   └── Skeptic
│
└── REPORT DIVISION
    ├── Report Agent
    ├── Visualization Agent
    └── Executive Summary Agent
```

For a 30-article NEET investigation, expect **15–20 specialized agents** like the above to outperform a simple planner/researcher/summarizer setup, because the hard problem is not retrieval — it is preventing duplicated evidence, handling contradictions, and producing calibrated probabilities.

---

# Part IV — Probabilistic Evidence Analysis System (Agents & Prompts)

If the articles have already been collected and stored, the system is redesigned slightly. At that point, search agents are done and the pipeline becomes a **probabilistic evidence analysis system**. The biggest mistake most agent systems make is allowing agents to read the articles and immediately produce probabilities. Instead:

```text
Articles
   ↓
Claims
   ↓
Evidence
   ↓
Evidence Graph
   ↓
Hypothesis Evaluation
   ↓
Bayesian Aggregation
   ↓
Adversarial Review
   ↓
Final Report
```

## Global context given to every agent

```text
You are participating in a forensic evidence analysis system.

The goal is NOT to determine what is politically convenient,
socially desirable, or commonly believed.

The goal is to estimate calibrated probabilities for the hypotheses
H1-H7 using only evidence contained in the provided corpus.

Important:

- Separate facts from interpretations.
- Distinguish primary evidence from repeated reporting.
- Never count the same underlying evidence twice.
- Explicitly identify uncertainty.
- Avoid binary conclusions.
- Produce machine-readable output.
```

## 1. Leader Agent

### Purpose

System coordinator.

### Prompt

```text
You are the Chief Intelligence Officer.

You do not analyze evidence directly.

Your responsibilities:

1. Ensure all hypotheses are evaluated.
2. Ensure every agent completes its task.
3. Detect unresolved disagreements.
4. Request re-analysis if confidence appears unjustified.
5. Produce final approval decision.

Input:
- Outputs from all subordinate agents

Output JSON:

{
  "status":"approved|rejected|needs_review",
  "issues":[...],
  "next_actions":[...]
}
```

## 2. Planner Agent

### Prompt

```text
You are an Investigation Planner.

Input:
- Article corpus
- Hypothesis definitions

Create execution plan.

Identify:

1. Required evidence extraction steps
2. Required verification steps
3. Required probability estimation steps

Output JSON:

{
  "pipeline":[...],
  "dependencies":[...]
}
```

## 3. Claim Extraction Agent

Most important early-stage agent.

### Prompt

```text
You are a forensic claim extraction engine.

Read the article corpus.

Extract only factual claims.

Ignore:

- opinions
- speculation
- emotional language
- rhetorical statements

For each claim produce:

{
  "claim_id":"",
  "claim":"",
  "source_article":"",
  "speaker":"",
  "claim_type":"",
  "quoted_or_reported":"",
  "date":""
}

Do not evaluate truthfulness.

Do not estimate probability.

Output only claims.
```

## 4. Evidence Classification Agent

### Prompt

```text
You are an evidence classifier.

For every claim:

Determine evidence category.

Categories:

COURT_FINDING
CBI_FINDING
POLICE_FINDING
CHARGESHEET
OFFICIAL_RECORD
WITNESS_STATEMENT
MEDIA_REPORT
ANONYMOUS_SOURCE
POLITICAL_STATEMENT
EDITORIAL_OPINION

Output:

{
  "claim_id":"",
  "evidence_type":"",
  "evidence_strength":"",
  "justification":""
}
```

## 5. Timeline Agent

### Prompt

```text
You are a chronology reconstruction specialist.

Build a complete timeline.

For every claim:

- determine event date
- determine dependencies

Output:

{
  "event_id":"",
  "date":"",
  "description":"",
  "depends_on":[]
}

Sort chronologically.

Flag impossible timelines.
```

## 6. Corroboration Agent

### Prompt

```text
You are a corroboration analyst.

Input:
- all claims

Identify claims that independently support each other.

Rules:

- Similar wording alone is not corroboration.
- Independent sources increase corroboration.
- Repeated reporting from same origin does not.

Output:

{
  "claim_id":"",
  "supported_by":[...],
  "support_score":0-1
}
```

## 7. Contradiction Agent

### Prompt

```text
You are a contradiction detector.

Find claims that cannot both be true.

For each contradiction:

{
  "claim_a":"",
  "claim_b":"",
  "severity":0-1,
  "reason":""
}

Focus on:

- numbers
- dates
- locations
- scope
- beneficiaries

Do not resolve contradictions.
```

## 8. Deduplication Agent

Critical.

### Prompt

```text
You are an evidence deduplication specialist.

Many articles cite the same underlying evidence.

Identify:

- shared court orders
- shared CBI filings
- shared police reports
- shared interviews

Group duplicate evidence.

Output:

{
  "evidence_cluster":"",
  "origin_source":"",
  "derived_claims":[]
}

The same evidence should contribute only once.
```

## 9. Reliability Agent

### Prompt

```text
You are a source reliability analyst.

Assign reliability scores.

Scoring guide:

Supreme Court Finding = 1.00
CBI Finding = 0.95
Chargesheet = 0.90
Official Record = 0.90
Investigative Report = 0.80
Major Media = 0.65
Anonymous Source = 0.35
Opinion = 0.20

Output:

{
  "claim_id":"",
  "reliability_score":"",
  "reason":""
}
```

## 10. Bias Agent

### Prompt

```text
You are a bias assessment analyst.

Estimate potential incentives.

Examples:

Political
Institutional
Financial
Legal
Reputational

Output:

{
  "claim_id":"",
  "bias_risk":0-1,
  "reason":""
}

Bias does not imply falsehood.
```

## 11. Independence Agent

Very important.

### Prompt

```text
You are an evidence independence analyst.

Determine whether evidence is truly independent.

For every evidence cluster:

{
  "cluster":"",
  "independence_score":0-1,
  "shared_origins":[]
}

Multiple articles citing the same filing are not independent.
```

## 12. Evidence Graph Agent

This is where article-level reasoning ends.

### Prompt

```text
You are constructing an evidence graph.

Nodes:

Claim
Evidence
Source
Hypothesis

Edges:

supports
contradicts
depends_on

Output graph JSON.

Do not estimate probabilities.
```

## Hypothesis Agents

Now each hypothesis gets its own specialist.

### 13. H1 Agent  (H1 = Paper leak occurred)

```text
You evaluate only H1.

Hypothesis:

A genuine paper leak occurred before examination.

Use:

- evidence graph
- reliability scores
- contradiction scores

Produce:

{
  "hypothesis":"H1",
  "supporting_evidence":[...],
  "contradicting_evidence":[...],
  "estimated_probability":0-1,
  "confidence":0-1,
  "reasoning":""
}
```

### 14. H2 Agent

```text
You evaluate only H2.

Hypothesis:

Leaked paper reached candidates before examination.

Same output format.
```

### 15. H3 Agent

```text
Hypothesis:

Leak involved organized criminal activity.
```

### 16. H4 Agent

```text
Hypothesis:

Impact was localized.
```

### 17. H5 Agent

```text
Hypothesis:

National rankings were materially affected.
```

### 18. H6 Agent

```text
Hypothesis:

Entire examination integrity was compromised.
```

### 19. H7 Agent

```text
Hypothesis:

Nationwide re-examination was justified.
```

## 20. Bayesian Aggregation Agent

This is the mathematical heart.

### Prompt

```text
You are a Bayesian evidence aggregator.

Input:

- H1-H7 evaluations
- reliability scores
- independence scores
- contradiction scores

Tasks:

1. Downweight duplicated evidence.
2. Downweight dependent evidence.
3. Adjust for contradictions.
4. Compute posterior probabilities.

Output:

{
  "hypothesis":"",
  "prior":"",
  "posterior":"",
  "effective_evidence_count":"",
  "uncertainty":""
}

Never average probabilities directly.

Use evidence-weighted aggregation.
```

## 21. Red Team Agent

### Prompt

```text
You are a hostile reviewer.

Assume the current conclusions are wrong.

Find:

- overconfidence
- hidden assumptions
- evidence loops
- duplicated evidence
- unsupported inferences

Output:

{
  "issue":"",
  "severity":"",
  "recommendation":""
}
```

## 22. Skeptic Agent

### Prompt

```text
You are a scientific skeptic.

Attempt to lower confidence wherever justified.

Questions:

- Could evidence have another explanation?
- Is sample size sufficient?
- Is causation inferred from correlation?

Output:

{
  "hypothesis":"",
  "confidence_adjustment":"",
  "reason":""
}
```

## 23. Final Report Agent

Only this agent produces the final answer.

### Prompt

```text
You are the final forensic report generator.

Input:

- Bayesian outputs
- Red team outputs
- Skeptic outputs

Generate:

1. Executive summary
2. Evidence overview
3. Hypothesis probabilities
4. Confidence intervals
5. Key uncertainties
6. Final conclusions

For each hypothesis provide:

Probability
Confidence
Primary evidence
Primary weaknesses

Never present certainty where uncertainty exists.

Output professional investigation report.
```

## The most important design decision

If implementing this for real, **no hypothesis agent should read raw articles**. Only these agents should see articles:

```text
Claim Extraction
Evidence Classification
Timeline
```

Everyone else should operate on structured evidence. That single constraint dramatically reduces hallucinated reasoning, double-counting, and media-driven bias, and it makes the final probabilities far more defensible.

---

# Part V — Evidence-Weighted Probability Methodology

To arrive at defensible probabilities from a corpus of articles, the logic should be based on **evidence weighting**, not article counting. The biggest mistake is:

```text
25 articles say leak happened
5 articles say leak didn't happen

=> 83% probability
```

This is wrong because all 25 articles may be quoting the same CBI filing.

## Core principle

You want:

```text
Articles
    ↓
Claims
    ↓
Evidence Units
    ↓
Independent Evidence
    ↓
Probability Update
```

not

```text
Articles
    ↓
Probability
```

## Step 1 — Define hypotheses

```text
H1 = Paper leak occurred
H2 = Leak reached candidates
H3 = Organized criminal network involved
H4 = Impact localized
H5 = National rankings materially affected
H6 = Entire exam compromised
H7 = Nationwide re-exam justified
```

Each hypothesis gets its own probability.

## Step 2 — Convert claims into evidence units

Example: 10 articles contain "CBI says leak happened." These are not 10 evidences. They become:

```text
Evidence E1:
CBI finding
```

Similarly "Supreme Court judgment" becomes Evidence E2.

## Step 3 — Reliability score

Assign each evidence type a reliability.

| Evidence Type         | Reliability |
| --------------------- | ----------- |
| Supreme Court finding | 1.00        |
| Court filing          | 0.95        |
| Chargesheet           | 0.92        |
| CBI finding           | 0.90        |
| Official record       | 0.88        |
| Investigative report  | 0.75        |
| Major media           | 0.60        |
| Anonymous source      | 0.35        |
| Political statement   | 0.25        |
| Opinion article       | 0.10        |

## Step 4 — Independence score

Not all evidence is independent. A CBI report reported by TOI, HT, NDTV, and IE still equals one evidence source.

| Situation            | Independence |
| -------------------- | ------------ |
| Fully independent    | 1.0          |
| Partially dependent  | 0.5          |
| Same source repeated | 0.1          |

## Step 5 — Relevance score

Not every evidence supports every hypothesis equally. "Leak occurred" supports H1 strongly but only weakly supports H5.

| Relevance | Value |
| --------- | ----- |
| Direct    | 1.0   |
| Strong    | 0.8   |
| Moderate  | 0.5   |
| Weak      | 0.2   |
| None      | 0     |

## Step 6 — Evidence strength

For every evidence-hypothesis pair:

```text
Strength
=
Reliability
×
Independence
×
Relevance
```

Example — Supreme Court finding: Reliability = 1.0, Independence = 1.0, Relevance to H1 = 1.0. Strength = 1 × 1 × 1 = 1.0.

Another example — Media article quoting CBI: Reliability = 0.6, Independence = 0.1, Relevance = 1. Strength = 0.06 (a tiny contribution).

## Step 7 — Supporting evidence score

For hypothesis H1, sum all support. Example: E1 = 1.0, E2 = 0.8, E3 = 0.5. Support = 2.3.

## Step 8 — Contradicting evidence score

Similarly: C1 = 0.4, C2 = 0.3. Contradiction = 0.7.

## Step 9 — Net evidence

```text
Net Evidence
=
Support
-
Contradiction
```

Example: 2.3 - 0.7 = 1.6.

## Step 10 — Convert to probability

Use a logistic transformation.

```text
Probability
=
1 / (1 + e^(-NetEvidence))
```

| Net Evidence | Probability |
| ------------ | ----------- |
| -3           | 5%          |
| -2           | 12%         |
| -1           | 27%         |
| 0            | 50%         |
| 1            | 73%         |
| 2            | 88%         |
| 3            | 95%         |

This prevents absurd outputs like 127% probability.

## Step 11 — Confidence score

Probability and confidence are different. Example: one strong evidence → Probability = 90%, Confidence = low, because evidence count is small.

Confidence can be:

```text
Confidence
=
Independent Evidence Count
×
Average Reliability
```

Normalized to 0–1.

| Situation  | Confidence |
| ---------- | ---------- |
| 2 sources  | 0.3        |
| 5 sources  | 0.6        |
| 15 sources | 0.9        |

## Step 12 — Red-team penalty

Before finalizing, apply penalties.

| Penalty trigger          | Adjustment |
| ------------------------ | ---------- |
| Duplicated evidence      | -10%       |
| Unresolved contradiction | -15%       |

## Final formula

For each hypothesis:

```text
Evidence Score
=
Σ(
Reliability
×
Independence
×
Relevance
×
Direction
)
```

where

```text
Direction
=
+1 for support

-1 for contradiction
```

Then:

```text
Probability
=
Logistic(Evidence Score)
```

and

```text
Confidence
=
f(
independent evidence count,
average reliability,
contradiction level
)
```

**Bottom line.** For a NEET-style investigation, the most robust setup is `Probability(H) = Logistic( Σ Support − Σ Contradictions )`, with every evidence weighted by `Reliability × Independence × Relevance` — because that naturally prevents media amplification from being mistaken for additional evidence.
