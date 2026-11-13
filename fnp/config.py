"""Paths, env-backed settings and the constants register for the FNP verdict system.

Every tunable number the system uses lives in `CONSTANTS` with a note on where it came
from. Values marked "assumed" are priors chosen before any data was seen; the
re-fitting stage for each one is listed in docs/review/02_ASSUMPTIONS_REGISTER.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

FNP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = FNP_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = FNP_ROOT / "reports"
DOCS_DIR = FNP_ROOT / "docs"
REVIEW_DIR = DOCS_DIR / "review"
DUCKDB_PATH = DATA_DIR / "fnp.duckdb"
BOOM_DIR = FNP_ROOT / "Interface" / "data"

# Sibling EDA project: read-only source of social posts, embeddings, topics, HMM/chain libs.
EDA_ROOT = FNP_ROOT.parent / "EDA"
EDA_PARQUET_DIR = EDA_ROOT / "data" / "parquet"
EDA_ANALYTICS_DB = EDA_ROOT / "data" / "analytics.duckdb"
EDA_GEO_DB = EDA_ROOT / "data" / "geo_ontology.duckdb"
EDA_METRICS_DB = EDA_ROOT / "data" / "metrics.duckdb"

EMBEDDING_MODEL = "BAAI/bge-m3"          # same model as EDA, so post vectors are reusable
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
LAYA_REPO = "convaiinnovations/laya"
LAYA_SUBFOLDER = "multilingual"          # mmBERT-base 322M: fits the ~7 GB GPU headroom

load_dotenv(FNP_ROOT / ".env", override=False)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return val


def gemma_config() -> dict:
    """Settings for the shared Gemma endpoint (same deployment EDA uses).

    Resolved lazily. Concurrency defaults to 12, lower than EDA's 32, because both
    projects share one sglang server and EDA's pipeline must not be starved.
    """
    return {
        "url": _require_env("GEMMA_URL").rstrip("/"),
        "model": _require_env("GEMMA_MODEL"),
        "api_key": _require_env("GEMMA_API_KEY"),
        "timeout": int(os.environ.get("GEMMA_TIMEOUT", "240")),
        "concurrency": int(os.environ.get("GEMMA_MAX_CONCURRENCY", "12")),
    }


def ensure_dirs() -> None:
    for d in (DATA_DIR, DATASETS_DIR, MODELS_DIR, CACHE_DIR, REPORTS_DIR, REVIEW_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Constants register. (value, provenance). "assumed" = prior, to be re-fit.
# ---------------------------------------------------------------------------
CONSTANTS: dict[str, tuple[float, str]] = {
    # L4 independence
    "dup_cosine": (0.90, "assumed; evidence texts above this bge-m3 cosine share an origin"),
    # L6 fusion
    "prior_true": (0.50, "assumed; uninformative claim prior, re-fit as gold base rate"),
    "woe_kappa": (2.0, "assumed; scale from strength units to log-odds, fit on gold (09_fuse --fit)"),
    "ds_strength_cap": (0.95, "assumed; no single evidence commits more than this mass"),
    # L7 verdict rule head
    "fake_p_true_upper": (0.20, "from notes: logistic(-1.4); upper CI bound for FAKE"),
    "legit_p_true_lower": (0.80, "assumed; lower CI bound for LEGIT"),
    "prop_threshold": (0.60, "assumed; P_prop above this with one-sidedness gives PROPAGANDA"),
    "max_uncertainty": (0.60, "assumed; DS Pl-Bel above this forces INCONCLUSIVE"),
    "min_n_eff": (1.0, "assumed; fewer independent evidence units -> INCONCLUSIVE"),
    "plausible_low": (0.50, "user spec: 'plausibly fake' tier lower bound on P(FAKE)+P(MISLEADING)"),
    "plausible_high": (0.80, "user spec: tier upper bound"),
    "conformal_alpha": (0.10, "assumed; 90% coverage prediction sets"),
    # annotation tool (fnp/annotate)
    "ann_double_frac": (0.20, "from scripts/25 (social_v1 pack): every 5th item is double-annotated for agreement"),
    "ann_blind_frac": (0.10, "from scripts/25: 10% of items get no model suggestions, to measure anchoring bias"),
    "ann_lookback_hours": (48.0, "assumed; re-read this far behind the ingest watermark to catch late-arriving rows"),
    "ann_lease_minutes": (30.0, "legacy: stamped on open items; reservation is now done by per-annotator batches (ann_batch_size), not by lease expiry"),
    "ann_social_min_chars": (40.0, "assumed; shorter posts rarely hold a claim or technique (scripts/25 used 80)"),
    "ann_social_max_chars": (1500.0, "from scripts/25: longer posts are too heavy to annotate in one sitting"),
    "ann_trunc_min_frac": (0.6, "assumed; YouTube descriptions/transcripts over the cap are cut at the last sentence end at or after this fraction of the cap, else at a word boundary"),
    "ann_min_substance_chars": (10.0, "assumed; letters left after removing URLs and @mentions: link-only posts carry nothing to annotate"),
    "ann_skip_exclude_n": (2.0, "assumed; an item is excluded once this many different annotators skipped it as unreadable / spam / offensive / other (language skips do not count)"),
    "ann_random_frac": (0.10, "assumed; share of every batch drawn uniformly at random (not steered): the unbiased slice for evaluation and for checking the steering"),
    "ann_gold_frac": (0.03, "assumed; share of every batch that is gold (adjudicated) items served blind for calibration, when enough gold exists"),
    "ann_target_supply_frac": (0.5, "assumed; a class target is capped at finished + this fraction of the pool items the models predict for it (precision unknown): imbalance is accepted and reweighted in training"),
    "ann_q_min_items": (30.0, "assumed; finished items an annotator needs before behaviour flags (none-rate, speed) are shown"),
    "ann_q_none_dev": (0.30, "assumed; flag an annotator whose 'none' share differs from the team median by more than this"),
    "ann_q_fast_sec": (8.0, "assumed; flag a median active time per item below this many seconds (a 150-word item cannot be read in less)"),
    "ann_q_min_agree_f1": (0.40, "assumed; flag span agreement with partners (double items) below this F1"),
    "ann_cb_beta": (0.999, "Cui et al. 2019 class-balanced loss, effective number of samples; default value from the paper"),
    "ann_gold_min_items": (10.0, "assumed; gold items an annotator needs before their gold score is flagged"),
    "ann_gold_min_f1": (0.5, "assumed; gold span-F1 below which an annotator is flagged for review"),
    "ann_news_min_chars": (300.0, "assumed; shorter items are headlines/stubs"),
    "ann_news_max_chars": (5000.0, "assumed; ~800 words, annotatable in one sitting; longer articles are excluded, not truncated"),
    "ann_share_en": (0.90, "owner request: annotators are mostly English-reading; target share of English items"),
    "ann_share_hi": (0.10, "owner request: 'a little Hindi'; target share of Hindi items"),
    "ann_share_other": (0.0, "owner request: no guarantee annotators read other languages, so they are never ingested or served (0 = excluded)"),
    "ann_batch_size": (250.0, "owner request: each annotator holds 250 items; when all are finished the next 250 are added"),
    "ann_target_items": (50000.0, "owner request: at least 50k fully annotated items per task (claims and propaganda)"),
    "ann_target_neg_frac": (0.20, "assumed; share of target items that carry an explicit no-claim / no-propaganda label"),
    "ann_balance_sample": (30.0, "assumed; candidates drawn per picked item inside a stratum, best class-deficit score wins"),
    "ann_idle_seconds": (120.0, "assumed; input within this window counts the minute as active work (reading a long text is work)"),
    "ann_session_hours": (12.0, "assumed; login session lifetime (rolling)"),
    "ann_max_failed_logins": (5.0, "assumed; consecutive bad passwords before the account is locked"),
    "ann_lockout_minutes": (10.0, "assumed; lock duration after too many bad passwords"),
    "ann_min_password_len": (8.0, "assumed; minimum password length"),
    "ann_split_dev": (0.10, "assumed; dev share of the annotation export, split by near-duplicate group"),
    "ann_split_test": (0.10, "assumed; test share of the annotation export, split by near-duplicate group"),
}


def C(name: str) -> float:
    return CONSTANTS[name][0]
