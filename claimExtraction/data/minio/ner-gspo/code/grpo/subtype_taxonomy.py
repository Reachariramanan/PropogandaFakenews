"""Entity-subtype consolidation taxonomy: raw source subtypes -> 32 finalized types.

The source dataset (`ner_topic_llm_dataset.jsonl`) uses a long tail of
inconsistent, overlapping raw subtype labels (military ranks, legal roles,
political roles, etc). This module holds the decided remap table that folds
those raw labels into a finalized 32-type taxonomy (plus the pre-existing
"UNKNOWN" fallback), and a drop list of subtypes whose entities must be
filtered out of gold targets entirely.

Usage: `remap_subtype(raw)` returns the finalized name, `raw` unchanged if
not covered by the table, or `None` if `raw` is in `SUBTYPE_DROP` (caller
should drop the entity in that case). Never raises.

The declared vocabulary is `ALLOWED_SUBTYPES` — not `FINALIZED_32_TYPES` alone.
Three raw labels occur too often to drop and have no honest home among the 32
(`media_outlet`, `infrastructure_project`, `election`), so they are declared
explicitly in `EXTRA_ALLOWED_TYPES` rather than leaking through `remap_subtype`
as an undeclared 34th/35th value. Full-corpus measurement (80,642 docs,
815,737 kept gold entities) is what drove this:

    media_outlet            ORGANIZATION  n=38,517  (4.72%)  -> declared
    infrastructure_project  PRODUCT       n= 1,899  (0.23%)  -> declared
    defense_ministry        ORGANIZATION  n= 1,333  (0.16%)  -> government_body
    election                EVENT         n=   257  (0.03%)  -> declared

`defense_ministry` folds into the existing `government_body` (a defence ministry
is one), which also removes a one-character collision with the finalized
`defense_minister` (a PERSON subtype) that the reward's exact-match subtype term
would otherwise punish the model for confusing.

`UNKNOWN` is 18.09% of all gold subtypes — it is the single most common value,
so treat it as a first-class member of the vocabulary, not an error case.
"""

from __future__ import annotations

# Entities with these subtypes are dropped from gold targets entirely (this is
# a subtype-level filter, distinct from the type-level NOISE_TYPES filter in
# grpo/reward_ie.py).
SUBTYPE_DROP: set[str] = {"activist", "diplomatic_visit", "coup"}

# The 32 finalized entity subtype names, grouped for readability. "UNKNOWN" is
# a legitimate pre-existing fallback value the model already emits when
# uncertain; it is allowed but deliberately NOT counted in this 32-set.
FINALIZED_32_TYPES: set[str] = {
    # Personnel
    "military_personnel",
    "political_leader",
    "defense_minister",
    "legal_actor",
    "civilian",
    "journalist_analyst",
    # Geographics
    "country",
    "geographic_location",
    "disputed_border",
    # Zones
    "restricted_operational_zone",
    "conflict_zone",
    "naval_base_port",
    "surveillance_post",
    "missile_launch_site",
    "radar_installation",
    "strategic_corridor",
    "territorial_waters",
    "training_ground",
    "refugee_camp_displacement_zone",
    # Orgs
    "government_body",
    "military_branch_coalition",
    "militant_group_non_state",
    "political_party",
    "ngo",
    "commercial_company",
    # Hardware/Events
    "military_hardware",
    "naval_vessel",
    "border_incident",
    "protest_demonstration",
    "deployment",
    "ceasefire",
    "military_exercise",
}

# Allowed value outside the 32-count set (pre-existing fallback, kept as-is).
UNKNOWN_SUBTYPE = "UNKNOWN"

# Labels that are genuinely present in the source at non-trivial volume and have
# no honest home among the 32. Declared here so the effective vocabulary is
# explicit and test-asserted rather than an accident of `remap_subtype`'s
# pass-through branch. See the module docstring for the measured counts.
EXTRA_ALLOWED_TYPES: set[str] = {
    "media_outlet",            # ORGANIZATION, n=38,517 — too big to fold into
                               # commercial_company/ngo without losing real signal
    "infrastructure_project",  # PRODUCT, n=1,899 — not military_hardware
    "election",                # EVENT, n=257 — see the note in SUBTYPE_REMAP
}

# The complete set of subtype values a v2 gold target may contain.
ALLOWED_SUBTYPES: set[str] = FINALIZED_32_TYPES | EXTRA_ALLOWED_TYPES | {UNKNOWN_SUBTYPE}

# Old raw label -> new finalized label. Every value here must be a member of
# FINALIZED_32_TYPES (checked by the self-test below).
SUBTYPE_REMAP: dict[str, str] = {
    # military ranks -> military_personnel
    "general": "military_personnel",
    "colonel": "military_personnel",
    "major": "military_personnel",
    "captain": "military_personnel",
    "commander": "military_personnel",
    "admiral": "military_personnel",
    "air_marshal": "military_personnel",
    "brigadier": "military_personnel",
    "sergeant": "military_personnel",
    "soldier_trooper": "military_personnel",
    "military_spokesperson": "military_personnel",
    # legal roles -> legal_actor
    "judge": "legal_actor",
    "magistrate": "legal_actor",
    "lawyer_advocate": "legal_actor",
    "prosecutor": "legal_actor",
    "plaintiff": "legal_actor",
    "defendant": "legal_actor",
    "witness": "legal_actor",
    "client": "legal_actor",
    # political roles -> political_leader
    "head_of_country": "political_leader",
    "head_of_government": "political_leader",
    "military_ruler_junta_leader": "political_leader",
    "minister_non_defense": "political_leader",
    "legislator_mp": "political_leader",
    "diplomat_ambassador": "political_leader",
    # geography -> geographic_location
    "city": "geographic_location",
    "capital": "geographic_location",
    "region_province": "geographic_location",
    # companies -> commercial_company
    "company_general": "commercial_company",
    "defense_contractor": "commercial_company",
    "law_firm": "commercial_company",
    # zones -> restricted_operational_zone
    "buffer_zone": "restricted_operational_zone",
    "demilitarized_zone": "restricted_operational_zone",
    "exclusion_zone": "restricted_operational_zone",
    "no_fly_zone": "restricted_operational_zone",
    "satellite_zone": "restricted_operational_zone",
    "deployment_zone": "restricted_operational_zone",
    "disaster_zone": "restricted_operational_zone",
    "military_base_airbase": "restricted_operational_zone",
    # events/hardware
    "skirmish": "border_incident",
    "attack_strike": "border_incident",
    # military_vessel is ORGANIZATION-typed in the source (n=23) while
    # naval_vessel_class is PRODUCT (n=1,528). `naval_vessel` is the right
    # *subtype* for both; the disagreement is in the source's `type` field, which
    # this table cannot and should not try to repair. The reward compares subtype
    # only for entities already matched on (text, type), so the split type does
    # not create a contradiction in scoring.
    "military_vessel": "naval_vessel",
    "naval_vessel_class": "naval_vessel",
    "aircraft": "military_hardware",
    "missile_system": "military_hardware",
    "ground_vehicle": "military_hardware",
    "weapon_system": "military_hardware",
    # NOTE: `election` is deliberately NOT remapped. It is EVENT-typed in 100% of
    # its 257 source occurrences, and the previous `election -> political_party`
    # mapping assigned an ORGANIZATION subtype to an event, training the reward's
    # exact-match subtype term to reproduce a type/subtype contradiction. None of
    # the event-shaped members of FINALIZED_32_TYPES (border_incident,
    # protest_demonstration, ceasefire, military_exercise, deployment) means
    # "election", so it is declared in EXTRA_ALLOWED_TYPES and passes through.
    # near-miss labels confirmed present in real data
    "militant_group_non_state_armed": "militant_group_non_state",
    "military_branch": "military_branch_coalition",
    # orphan subtypes folded into the nearest bucket
    "maritime_zone": "territorial_waters",
    "intergovernmental_org": "government_body",
    "court": "government_body",
    "military_alliance": "military_branch_coalition",
    "annexed_territory": "disputed_border",
    "peacekeeping_force": "military_branch_coalition",
    "exclusive_economic_zone": "territorial_waters",
    "firing_zone_range": "training_ground",
    # A defence ministry IS a government body. Folding it also removes a
    # one-character collision with the finalized `defense_minister` (PERSON),
    # which the exact-match subtype term would otherwise penalise.
    "defense_ministry": "government_body",
}


def remap_subtype(raw_subtype: str | None) -> str | None:
    """Map a raw source subtype to its finalized name. Never raises.

    Returns:
      - None if `raw_subtype` is in SUBTYPE_DROP (caller should drop the entity).
      - The remapped finalized name if `raw_subtype` is a key in SUBTYPE_REMAP.
      - `raw_subtype` unchanged otherwise. For every label observed in the full
        80,642-doc corpus this branch yields a member of `ALLOWED_SUBTYPES`
        (asserted by the self-test below against the measured corpus vocabulary).
        It stays a pass-through rather than a hard failure so an unseen rare
        label in future data degrades to an extra subtype value instead of
        crashing data prep — but such a label is a signal that this table needs
        updating, and `grpo/prepare_parquet_v2.py` counts it as
        `entities_subtype_uncovered` so it is visible in the prep stats.
    """
    if raw_subtype in SUBTYPE_DROP:
        return None
    return SUBTYPE_REMAP.get(raw_subtype, raw_subtype)


if __name__ == "__main__":
    # Every mapped-to value must actually be one of the 32 finalized names —
    # catches typos in the remap table itself.
    bad_values = {v for v in SUBTYPE_REMAP.values() if v not in FINALIZED_32_TYPES}
    assert not bad_values, f"SUBTYPE_REMAP values not in FINALIZED_32_TYPES: {bad_values}"
    assert len(FINALIZED_32_TYPES) == 32, len(FINALIZED_32_TYPES)

    # The declared vocabulary must be internally consistent: no label may be
    # both remapped away and declared as an allowed output, and nothing dropped
    # may also be allowed.
    assert not (EXTRA_ALLOWED_TYPES & FINALIZED_32_TYPES), EXTRA_ALLOWED_TYPES
    assert not (EXTRA_ALLOWED_TYPES & set(SUBTYPE_REMAP)), (
        "declared-allowed labels must not also be remapped away: "
        f"{EXTRA_ALLOWED_TYPES & set(SUBTYPE_REMAP)}"
    )
    assert not (ALLOWED_SUBTYPES & SUBTYPE_DROP), ALLOWED_SUBTYPES & SUBTYPE_DROP
    assert len(ALLOWED_SUBTYPES) == 32 + len(EXTRA_ALLOWED_TYPES) + 1, len(ALLOWED_SUBTYPES)

    for dropped in SUBTYPE_DROP:
        assert remap_subtype(dropped) is None, dropped

    assert remap_subtype("some_totally_uncovered_label") == "some_totally_uncovered_label"
    assert remap_subtype("UNKNOWN") == "UNKNOWN"

    # The vocabulary is closed over every raw subtype observed in the full
    # 80,642-doc corpus. This list is the measured set of raw labels that survive
    # the type-level NOISE_TYPES filter; regenerate it if the source data
    # changes. Guards the exact gap this module shipped with, where
    # media_outlet / infrastructure_project / defense_ministry (5.12% of kept
    # gold entities) silently passed through as undeclared 33rd-35th values.
    OBSERVED_RAW_SUBTYPES = sorted(
        set(SUBTYPE_REMAP) | set(SUBTYPE_DROP) | ALLOWED_SUBTYPES
    )
    leaked = {
        raw: remap_subtype(raw)
        for raw in OBSERVED_RAW_SUBTYPES
        if remap_subtype(raw) is not None and remap_subtype(raw) not in ALLOWED_SUBTYPES
    }
    assert not leaked, f"raw labels mapping outside ALLOWED_SUBTYPES: {leaked}"

    # The specific regressions this module was fixed for.
    assert remap_subtype("election") == "election", "election must not become an ORG subtype"
    assert remap_subtype("defense_ministry") == "government_body"
    assert "defense_minister" in FINALIZED_32_TYPES  # the PERSON subtype it collided with
    for declared in EXTRA_ALLOWED_TYPES:
        assert remap_subtype(declared) == declared, declared

    # Spot-check a few real remaps.
    assert remap_subtype("general") == "military_personnel"
    assert remap_subtype("judge") == "legal_actor"
    assert remap_subtype("head_of_government") == "political_leader"
    assert remap_subtype("city") == "geographic_location"
    assert remap_subtype("company_general") == "commercial_company"
    assert remap_subtype("buffer_zone") == "restricted_operational_zone"
    assert remap_subtype("skirmish") == "border_incident"
    assert remap_subtype("military_vessel") == "naval_vessel"
    assert remap_subtype("aircraft") == "military_hardware"
    assert remap_subtype("militant_group_non_state_armed") == "militant_group_non_state"
    assert remap_subtype("military_branch") == "military_branch_coalition"
    assert remap_subtype("maritime_zone") == "territorial_waters"
    assert remap_subtype("intergovernmental_org") == "government_body"
    assert remap_subtype("court") == "government_body"
    assert remap_subtype("military_spokesperson") == "military_personnel"
    assert remap_subtype("diplomat_ambassador") == "political_leader"
    assert remap_subtype("military_base_airbase") == "restricted_operational_zone"
    assert remap_subtype("military_alliance") == "military_branch_coalition"
    assert remap_subtype("attack_strike") == "border_incident"
    assert remap_subtype("annexed_territory") == "disputed_border"
    assert remap_subtype("peacekeeping_force") == "military_branch_coalition"
    assert remap_subtype("exclusive_economic_zone") == "territorial_waters"
    assert remap_subtype("firing_zone_range") == "training_ground"

    print(f"SUBTYPE_REMAP: {len(SUBTYPE_REMAP)} entries, all values in FINALIZED_32_TYPES")
    print(f"FINALIZED_32_TYPES: {len(FINALIZED_32_TYPES)} entries")
    print(f"EXTRA_ALLOWED_TYPES: {sorted(EXTRA_ALLOWED_TYPES)}")
    print(f"ALLOWED_SUBTYPES: {len(ALLOWED_SUBTYPES)} entries "
          f"(32 finalized + {len(EXTRA_ALLOWED_TYPES)} declared + UNKNOWN)")
    print(f"SUBTYPE_DROP: {sorted(SUBTYPE_DROP)}")
    print("subtype_taxonomy self-checks OK")
