# Fake News & Propaganda Detection

A hybrid LLM and metrics system that classifies claims, posts and articles into **FAKE / PROPAGANDA / LEGIT /
MISLEADING / INCONCLUSIVE**, with a "plausibly fake" tier, and a web app (the **Evidence Desk**) for working with the
results.

## What is in this repository

| Path | What it is |
|---|---|
| `fnp/` | The verdict engine: claim extraction, evidence retrieval, stance, propaganda span tagging, evidence maths, and the Laya decider. |
| `scripts/` | Numbered pipeline stages in run order (dry run by default, `--apply` writes), plus ops scripts. |
| `tests/` | Offline tests for the engine (the LLM is mocked, the maths runs on synthetic data). |
| `UI/` | The engine's dashboard and the annotation tool front end. |
| `Interface/` | The Evidence Desk: fact-check database, **Claim Check** (fake or not, then propaganda techniques), **Propaganda Lab** (map of techniques by country), **Evidence Base** with an analysis workbench for a connected news and social database, and a **Triage** queue. Has its own README and tests. |
| `claimExtraction/` | Training code for a small claim-extraction model (GSPO / SFT). Checkpoints and data are not included. |
| `docs/`, `methodologoes/`, `current-arch.md` | Architecture, methods, data contracts, review notes and design notes. |
| `reports/*.md` | Evaluation reports for the propaganda tagger and verdict engine. |

## Branches

- `main`: the initial commit.
- `dev`: the whole codebase (this branch).
- `modelling`: only the training code; see `MODELLING.md` on that branch.

## Getting started

Python 3.12. Install the packages in `requirements.txt` (`requirements.lock.txt` has the versions that were used). The
engine also needs a Gemma-compatible endpoint, the Laya package and the datasets; paths and settings are in
`fnp/config.py` and `.env.example`.

```
python -m pytest tests -q                              # engine tests (offline)
python3 -m unittest discover -s Interface/tests -q     # Evidence Desk tests (offline)
```

A few Evidence Desk tests are skipped on a fresh clone because they need the generated search index, which is built
from fact-check data that is not part of the repository. To run the Evidence Desk itself, see `Interface/README.md`
(data layout, `build_index.py`, `server.py`, settings and the API).

## What is not included

Datasets, scraped articles, databases, model checkpoints, logs, credentials and any customer data. Internal addresses
in the documents are replaced by placeholders such as `<DB_HOST>`.

## Status and limits

This is working research code, not a validated product.

- Verdict quality is modest and has been measured on mixed public datasets only (see `reports/` and
  `docs/review/05_EVAL_FINDINGS.md`). No classifier is called validated without held-out or hand-judged precision with
  a confidence interval, and none has that for social media yet.
- Several constants and the Triage tier rules are assumptions, listed in `docs/review/02_ASSUMPTIONS_REGISTER.md` and
  the Interface README.
- The propaganda scores are uncalibrated signals from an encoder trained on long news articles, not probabilities, and
  are not validated on short or social-media text.
- Laya's raw propaganda output appears to separate article-style training data rather than propaganda wording, so it is
  shown as "not validated" and never used for the verdict.

See `docs/review/` for the open problems, and `current-arch.md` for the current state.
