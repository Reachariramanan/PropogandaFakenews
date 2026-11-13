"""Shared contracts between layers. Every module exchanges plain dicts shaped like these
TypedDicts, so records round-trip through JSON, DuckDB and the Gemma cache unchanged.
"""
from __future__ import annotations

from typing import Optional, TypedDict

# Final classes. Order is fixed: it is the column order of every probability vector.
LABELS: tuple[str, ...] = ("FAKE", "PROPAGANDA", "LEGIT", "MISLEADING", "INCONCLUSIVE")
LABEL_INDEX = {l: i for i, l in enumerate(LABELS)}
DECIDABLE = LABELS[:4]          # INCONCLUSIVE is an abstention, never a training target for the stacker

CLAIM_TYPES = ("factual", "statistical", "quote", "media_attribution", "prediction",
               "opinion", "other")
STANCES = ("support", "refute", "context_mismatch", "neutral")

# Evidence categories from methodologoes/ Part IV, extended with the social/fact-check cases.
EVIDENCE_TYPES = ("COURT_FINDING", "AGENCY_FINDING", "POLICE_FINDING", "CHARGESHEET",
                  "OFFICIAL_RECORD", "FACT_CHECK", "WITNESS_STATEMENT", "INVESTIGATIVE_REPORT",
                  "MEDIA_REPORT", "ANONYMOUS_SOURCE", "POLITICAL_STATEMENT",
                  "EDITORIAL_OPINION", "SOCIAL_POST", "UNKNOWN")

# Prior reliability per evidence type (methodologoes Part V table, extended; "assumed").
# Used as the Beta prior mean in fnp.math.reliability; posterior updates override it.
TYPE_RELIABILITY = {
    "COURT_FINDING": 1.00, "AGENCY_FINDING": 0.90, "POLICE_FINDING": 0.85,
    "CHARGESHEET": 0.92, "OFFICIAL_RECORD": 0.88, "FACT_CHECK": 0.90,
    "INVESTIGATIVE_REPORT": 0.75, "WITNESS_STATEMENT": 0.55, "MEDIA_REPORT": 0.60,
    "ANONYMOUS_SOURCE": 0.35, "POLITICAL_STATEMENT": 0.25, "EDITORIAL_OPINION": 0.10,
    "SOCIAL_POST": 0.20, "UNKNOWN": 0.30,
}

# 18 techniques of Da San Martino et al. 2019 (PTC corpus); SemEval-2020 T11 merges some
# into 14 -- PTC_MERGE maps the 18 onto the 14 for span-F1 evaluation.
PROPAGANDA_TECHNIQUES = (
    "Loaded_Language", "Name_Calling,Labeling", "Repetition", "Exaggeration,Minimisation",
    "Doubt", "Appeal_to_fear-prejudice", "Flag-Waving", "Causal_Oversimplification",
    "Slogans", "Appeal_to_Authority", "Black-and-White_Fallacy", "Thought-terminating_Cliches",
    "Whataboutism", "Reductio_ad_hitlerum", "Red_Herring", "Bandwagon",
    "Obfuscation,Intentional_Vagueness,Confusion", "Straw_Men",
)
PTC_MERGE = {
    "Whataboutism": "Whataboutism,Straw_Men,Red_Herring",
    "Straw_Men": "Whataboutism,Straw_Men,Red_Herring",
    "Red_Herring": "Whataboutism,Straw_Men,Red_Herring",
    "Bandwagon": "Bandwagon,Reductio_ad_hitlerum",
    "Reductio_ad_hitlerum": "Bandwagon,Reductio_ad_hitlerum",
}


class Claim(TypedDict, total=False):
    claim_id: str            # sha1(doc_id + normalised claim)[:16]
    doc_id: str              # post_uid / article id / dataset row id
    claim: str               # self-contained restatement
    speaker: Optional[str]
    claim_type: str          # CLAIM_TYPES
    quoted_or_reported: str  # "asserted" | "quoted" | "reported"
    check_worthy: float      # [0,1]
    date: Optional[str]
    entities: list[str]
    evidence_quote: str      # verbatim span of the source text (validated)
    lang: Optional[str]


class Span(TypedDict, total=False):
    doc_id: str
    start: int               # char offsets into the source text
    end: int
    text: str
    technique: str           # PROPAGANDA_TECHNIQUES
    confidence: float


class Evidence(TypedDict, total=False):
    evidence_id: str
    claim_id: str
    text: str                # evidence passage (never the claim's own document)
    source: str              # outlet / author / dataset name
    url: Optional[str]
    origin_key: Optional[str]  # shared-origin key for dedup (cited filing, source link, KB id)
    evidence_type: str       # EVIDENCE_TYPES
    stance: str              # STANCES
    stance_conf: float       # [0,1]
    relevance: float         # [0,1]
    reliability: float       # [0,1], from fnp.math.reliability
    independence: float      # [0,1], from fnp.math.independence
    published: Optional[str]
    retrieval: str           # "kb" | "corpus" | "web"
    retrieval_score: float


# Numeric features produced by fnp.math (L4-L6) per claim, consumed by the rule head,
# the stacker, the Laya state builder and the routing bandit. Order is the vector order.
FEATURE_NAMES: tuple[str, ...] = (
    "p_true", "p_true_lo", "p_true_hi",          # WoE posterior + meta-analytic CI
    "support", "contradiction", "context_mismatch",  # summed independent strengths
    "ds_bel_true", "ds_bel_false", "ds_uncertainty", "ds_conflict",
    "n_evidence", "n_eff", "i2",                 # raw count, Kish effective n, heterogeneity
    "info_gain_bits", "centrality",
    "kb_match", "kb_match_score",                # best debunked-claim match
    "p_prop", "prop_density", "prop_n_techniques", "one_sidedness",
    "check_worthy", "bp_marginal",               # loopy-BP marginal on the story claim graph
    "timeline_inconsistency",
    "burst_intensity", "platform_spread", "author_diversity", "dup_rate",
    "coordination", "source_reliability",
)


class Verdict(TypedDict, total=False):
    unit_id: str             # claim_id, or doc_id for rolled-up verdicts
    unit: str                # "claim" | "doc"
    doc_id: str              # always set; fact-check items use "<site>:<id>" (e.g. "boomlive:32413")
    label: str               # LABELS
    probs: dict[str, float]  # over LABELS
    plausibly_fake: bool
    secondary: list[str]     # other labels whose score passed their own threshold
    prediction_set: list[str]  # conformal set
    features: dict[str, float]
    decider: str             # "rule" | "gemma" | "stacker" | "laya" | "router:<arm>"
    rationale: str
    evidence_ids: list[str]
