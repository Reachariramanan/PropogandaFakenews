"""Topic / action code -> name lookups for this project's **source taxonomy**.

Provenance — read this before changing a name
---------------------------------------------
These tables are **derived from the source labels themselves**
(`data/ner_topic_llm_dataset.jsonl`, all 80,642 documents), not from an external
codebook. Every entry below is the name the labeller actually attached to that
code, and the counts in the trailing comments are the observed occurrences.

This module replaces an earlier `grpo/cameo_lookup.py`, which asserted that the
source `topics` codes were CAMEO (Conflict and Mediation Event Observations)
sub-categories and supplied canonical CAMEO names for them. **They are not
CAMEO.** The source's `NN.N` codes are a bespoke *subject* taxonomy (what the
document is about) with a superficially CAMEO-like shape, and the two disagree
on every single code. Measured over the full corpus:

    code    source name (this table)          CAMEO name (what was there before)
    01.1    Bilateral Political Relations     Decline Comment
    08.3    Domestic Stability Narratives     Ease Economic Sanctions/Boycott
    01.3    Diplomatic Statements &           Make Optimistic Comment
            Positioning
    09.1    Digital Infrastructure &          Investigate Crime or Corruption
            Connectivity
    06.1    Military-to-Military Engagement   Cooperate Economically

Emitting the CAMEO name as gold trained the model to quote an evidence span and
then attach a label contradicting it — adversarial supervision against the
`topic_f1` term that rides on the same code. Do not reintroduce an external
codebook here without first checking it against the `name` field in the source.

Coverage and stability (full-corpus scan, 80,642 docs)
-----------------------------------------------------
* 44 distinct `NN.N` topic codes occur; 41 carry a name. Every one of the 41 is
  **single-valued** — no code ever appears with two different names — so a
  code -> name table is a faithful representation, not a lossy summary.
* `01.6` (n=4), `14.0` (n=1) and `20.0` (n=1) match the `NN.N` shape but carry a
  null name in the source. They are deliberately absent here: `canonical_topic_name`
  returns None, callers emit `name: null`, and `grpo/reward_ie_v2.py` treats an
  unnameable code as unjudgeable rather than penalising it.
* Codes that do not match `NN.N` at all (`01`, `02`, `06`, `13`..`20`, `EVT`)
  are rejected upstream by the `TOPIC_CODE_RE` filter in
  `grpo/prepare_parquet_v2.py` and never reach this table.
* Action codes: only the 8 top-level codes 13-20 are scored (per
  `grpo/reward_ie.py::_action_codes`), and all 8 are single-valued in the source.
  Note that two of them differ from the CAMEO top-level names that used to be
  here: `15` is "Exhibit Force Posture" (CAMEO: "Exhibit Military Posture") and
  `20` is "Engage in Unconventional Mass Violence" (CAMEO: "Use Unconventional
  Mass Violence").

Both lookups are total and never raise.
"""

from __future__ import annotations

# Where every name below came from, so a future reader can re-derive the tables.
TAXONOMY_SOURCE = "ner_topic_llm_dataset.jsonl (80,642 docs; per-code name field)"

# Source subject-taxonomy topic sub-codes ("NN.N") -> the labeller's own name.
# Trailing counts are full-corpus occurrences. All single-valued (see docstring).
TOPIC_CODE_NAMES: dict[str, str] = {
    # 01 political / diplomatic relations
    "01.1": "Bilateral Political Relations",                                  # n=35469
    "01.2": "Multilateral & International Relations",                         # n=10203
    "01.3": "Diplomatic Statements & Positioning",                            # n=21612
    "01.4": "Treaties & Agreements",                                          # n=4007
    # 02 economic relations
    "02.1": "Trade Relations",                                                # n=11614
    "02.2": "Investment & FDI",                                               # n=9602
    "02.3": "Financial & Monetary Cooperation",                               # n=3599
    "02.4": "Economic Sanctions & Restrictions",                              # n=5388
    # 03 infrastructure / connectivity projects
    "03.1": "Infrastructure Projects",                                        # n=11199
    "03.2": "Project Financing & Debt",                                       # n=787
    "03.3": "Named Corridors & Programs",                                     # n=643
    "03.4": "Initiative Diplomacy & Project Status",                          # n=1523
    # 04 energy
    "04.1": "Oil & Gas Access",                                               # n=4577
    "04.2": "Nuclear & Renewable Energy Cooperation",                         # n=2336
    "04.3": "Energy Infrastructure Security",                                 # n=1804
    "04.4": "Energy-related Strategic Competition",                           # n=1218
    # 05 maritime
    "05.1": "Naval Presence & Deployments",                                   # n=1220
    "05.2": "Territorial & Maritime Disputes",                                # n=2237
    "05.3": "Sea Lines of Communication (SLOC)",                              # n=4038
    "05.4": "Port Access & Dual-use Infrastructure",                          # n=949
    "05.5": "Maritime Law & Governance",                                      # n=597
    # 06 security & military relations
    "06.1": "Military-to-Military Engagement",                                # n=6697
    "06.2": "Arms Sales & Defense Trade",                                     # n=1623
    "06.3": "Military Modernization & Capability Development",                # n=5590
    "06.4": "Alliance & Security Partnerships",                               # n=3881
    # 07 borders
    "07.1": "Territorial / Border Disputes",                                  # n=2931
    "07.2": "Border Infrastructure",                                          # n=496
    "07.3": "Border Talks & CBMs",                                            # n=155
    "07.4": "Border Incidents",                                               # n=3511
    # 08 conflict / stability
    "08.1": "Mediation & Conflict Resolution",                                # n=2877
    "08.2": "External Support to State / Non-state Actors",                   # n=3547
    "08.3": "Domestic Stability Narratives",                                  # n=32152
    "08.4": "Refugee & Cross-border Spillover",                               # n=5666
    # 09 technology / cyber
    "09.1": "Digital Infrastructure & Connectivity",                          # n=8280
    "09.2": "Surveillance & Security Tech Exports",                           # n=358
    "09.3": "Cyber Activity",                                                 # n=3459
    "09.4": "Tech Standards & Export Controls",                               # n=1299
    # 10 humanitarian
    "10.1": "Disaster Relief & Emergency Aid",                                # n=7174
    "10.2": "Health & Medical Diplomacy",                                     # n=7003
    "10.3": "Humanitarian Assistance in Conflict Zones",                      # n=3219
    "10.4": "Refugee & Migration-related Assistance",                         # n=775
}

# Top-level action codes, restricted to the 13-20 range this project scores.
# These are the source's own names; `15` and `20` differ from CAMEO (see docstring).
ACTION_CODE_NAMES: dict[str, str] = {
    "13": "Threaten",                               # n=5377
    "14": "Protest",                                # n=13958
    "15": "Exhibit Force Posture",                  # n=2300
    "16": "Reduce Relations",                       # n=1146
    "17": "Coerce",                                 # n=23653
    "18": "Assault",                                # n=11257
    "19": "Fight",                                  # n=13959
    "20": "Engage in Unconventional Mass Violence", # n=1129
}


def canonical_topic_name(code: str) -> str | None:
    """The source taxonomy's name for a topic code, or None. Never raises."""
    if code is None:
        return None
    return TOPIC_CODE_NAMES.get(str(code).strip())


def canonical_action_name(code: str) -> str | None:
    """The source taxonomy's name for an action code, or None. Never raises."""
    if code is None:
        return None
    return ACTION_CODE_NAMES.get(str(code).strip())


if __name__ == "__main__":
    import re

    # Shape invariants: every key is a well-formed code of the expected kind.
    topic_re, action_re = re.compile(r"^\d+\.\d+$"), re.compile(r"^\d{1,2}$")
    for c in TOPIC_CODE_NAMES:
        assert topic_re.match(c), c
    for c in ACTION_CODE_NAMES:
        assert action_re.match(c), c
        assert 13 <= int(c) <= 20, c
    assert len(ACTION_CODE_NAMES) == 8, len(ACTION_CODE_NAMES)
    assert len(TOPIC_CODE_NAMES) == 41, len(TOPIC_CODE_NAMES)

    # Names must be non-empty and distinct (a duplicate name would make the
    # reward's name term ambiguous across two codes).
    assert all(isinstance(v, str) and v.strip() for v in TOPIC_CODE_NAMES.values())
    assert len(set(TOPIC_CODE_NAMES.values())) == len(TOPIC_CODE_NAMES)
    assert len(set(ACTION_CODE_NAMES.values())) == len(ACTION_CODE_NAMES)

    # Spot-check the codes that the removed CAMEO table got wrong — these are
    # the regression guards against reintroducing an external codebook.
    assert canonical_topic_name("01.1") == "Bilateral Political Relations"
    assert canonical_topic_name("08.3") == "Domestic Stability Narratives"
    assert canonical_topic_name("09.1") == "Digital Infrastructure & Connectivity"
    assert canonical_topic_name("06.1") == "Military-to-Military Engagement"
    assert canonical_action_name("15") == "Exhibit Force Posture"
    assert canonical_action_name("20") == "Engage in Unconventional Mass Violence"

    # Codes the source leaves unnamed stay unnamed (callers emit `name: null`).
    for unnamed in ("01.6", "14.0", "20.0"):
        assert canonical_topic_name(unnamed) is None, unnamed

    # Totality.
    assert canonical_topic_name("99.9") is None
    assert canonical_topic_name(None) is None
    assert canonical_topic_name("  01.1  ") == "Bilateral Political Relations"
    assert canonical_action_name("1") is None
    assert canonical_action_name("12") is None
    assert canonical_action_name(None) is None

    print(f"TOPIC_CODE_NAMES: {len(TOPIC_CODE_NAMES)} entries (source taxonomy, not CAMEO)")
    print(f"ACTION_CODE_NAMES: {len(ACTION_CODE_NAMES)} entries")
    print(f"source: {TAXONOMY_SOURCE}")
    print("topic_taxonomy self-checks OK")
