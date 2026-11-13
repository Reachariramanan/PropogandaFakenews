#!/usr/bin/env python3
"""Convert ner_topic_llm_dataset.jsonl → VERL parquet for GSPO — **v2 schema**.

This is a sibling of `grpo/prepare_parquet.py`, not a replacement. v1 (full /
minimal schemas) is left completely untouched because a live training run
depends on the parquet files it produced.

What v2 changes relative to v1's `minimal` schema:

* **Entities** carry `text` / `type` / `subtype` / `evidence_text`. `subtype`
  is run through `grpo/subtype_taxonomy.py::remap_subtype`, which folds the
  long tail of raw source labels into the finalized 32-type taxonomy and
  drops entities whose subtype is in `SUBTYPE_DROP`
  (activist / diplomatic_visit / coup). `evidence_text` comes back because v2's
  reward grounds it against the source document (is it a real span of the doc,
  not a hallucination), which v1 never scored.
* **Events** carry `actor1` / `actor2` / a **singular** `action` object
  `{code, name}` / a **list** of `topics` `[{code, name}, …]` /
  `evidence_text`. `actions` (plural list) from the source is collapsed to the
  first code in the scored 13-20 CAMEO range; `action` is `null` when the
  source event has no such code (the event is still kept — its topics, actors
  and evidence are all scored, so dropping it would throw away real signal;
  most source events have an empty `actions` list).
* Topic/action **names** are the source taxonomy's own names, served by
  `grpo/topic_taxonomy.py`. An earlier version of this file used canonical CAMEO
  codebook names instead, on the theory that the source's `NN.N` codes were
  CAMEO sub-categories. They are not — they are a bespoke *subject* taxonomy that
  merely looks CAMEO-shaped, and the two disagree on **every** code (source
  `01.1` = "Bilateral Political Relations"; CAMEO `01.1` = "Decline Comment").
  Emitting the CAMEO name made gold self-contradictory: the model was trained to
  quote an evidence span and then label it with something unrelated, while
  `topic_f1` rode on the code that contradicted it. The lookup tables are now
  derived from the source `name` field itself (single-valued for all 41 named
  topic codes and all 8 action codes across the full corpus). A code the source
  never names emits `name: null` rather than dropping the code.
* Fields that caused the historical full-schema token blowup are **not** here:
  no `other_actors`, no `normalized`, no `score`, no `resolution_method`, no
  `char_start`/`char_end`, no `confidence`, no `code_verified`. Measured with the
  real `outputs/qwen3_5_0.8b_text_only` tokenizer over 4,029 documents sampled at
  stride 20 across the whole corpus, v2 targets are mean 572 / median 546 /
  p90 1046 / p95 1219 / p99 1591 / max 3050 tokens, with **0.10%** over a
  2048-token response budget, so `max_response_length=2048` holds comfortably.
  (An earlier revision of this docstring claimed mean 740 / p90 1150 / p99 1779
  and 0.3% over budget. Those figures came from a contiguous block near the head
  of the file, which is not representative — document length drifts through the
  corpus. Sample with a stride if you re-measure.) Do not add fields.
* **Prompt length is measured in tokens and enforced**, not left to chance — see
  `--max_prompt_tokens` and `_fit_prompt_to_budget` below.

Output goes to `data/grpo_v2/` (override with `--out_dir`). MLflow stats are
logged under run name `prepare_parquet_v2` with `schema_version=v2` tags so
they are distinguishable from the v1 stats already on the server.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpo.topic_taxonomy import canonical_action_name, canonical_topic_name
from grpo.mlflow_utils import log_artifact, log_dict_as_json, log_metrics, mlflow_run
from grpo.reward_ie import NOISE_TYPES
from grpo.subtype_taxonomy import ALLOWED_SUBTYPES, EXTRA_ALLOWED_TYPES

SCHEMA_VERSION = "v2"

# Same opening as grpo/prepare_parquet.py::_SYSTEM_PREFIX so the two prompt
# families stay recognisably one lineage.
_SYSTEM_PREFIX = (
    "You are an information extraction model for geopolitical news and social text. "
    "Given the document, extract named entities and events. Reply with ONLY a JSON "
    "object with keys \"entities\" and \"events\". Do not wrap in markdown. "
)

# Kept deliberately terse: every token here is charged against
# data.max_prompt_length for all 80k rows and pushes documents into the
# truncation path (see PromptBudget). This wording is 165 tokens against the 221
# of the original phrasing while specifying exactly the same fields — do not
# re-expand it into prose without re-measuring.
SYSTEM_IE_V2 = _SYSTEM_PREFIX + (
    "Emit ONLY these fields. "
    "Entity: \"text\", \"type\", \"subtype\", \"evidence_text\". "
    "Event: \"actor1\", \"actor2\", \"action\", \"topics\", \"evidence_text\". "
    "\"action\" is ONE object {\"code\": \"NN\", \"name\": \"...\"}, not a list, or "
    "null if no action applies. "
    "\"topics\" is a list [{\"code\": \"NN.N\", \"name\": \"...\"}, ...]. "
    "\"actor2\" may be null. "
    "Every \"evidence_text\" must be a span copied verbatim from the document. "
    "Omit all other fields."
)

# Fields the v2 reward function reads, kept next to the prompt so prompt and
# reward cannot drift apart silently (the failure mode v1 hit).
SCORED_ENTITY_FIELDS = ("text", "type", "subtype", "evidence_text")
SCORED_EVENT_FIELDS = ("actor1", "actor2", "action", "topics", "evidence_text")

TOPIC_CODE_RE = re.compile(r"^\d+\.\d+$")
ACTION_CODE_RE = re.compile(r"^\d{1,2}$")

# Marker appended to a document whose text had to be cut short.
TRUNCATION_MARK = "\n…[truncated]"

# Default tokenizer for the prompt-length guard: the same on-disk checkpoint the
# actor and the vLLM rollout both load, so the measurement matches training.
DEFAULT_TOKENIZER = str(ROOT / "outputs" / "qwen3_5_0.8b_text_only")

# Must match `data.max_prompt_length` in grpo/run_gspo_vigor.sh. VERL is
# configured there with `data.filter_overlong_prompts=True`, which **drops** rows
# whose tokenized prompt exceeds the limit (see
# verl/verl/utils/dataset/rl_dataset.py::maybe_filter_out_long_prompts — it
# filters, it does not truncate) and prints nothing per-row. Any row this file
# emits above the budget is therefore silently discarded at load.
DEFAULT_MAX_PROMPT_TOKENS = 1536

# CAMEO top-level action codes this project scores (matches
# grpo/reward_ie.py::_action_codes).
ACTION_CODE_MIN = 13
ACTION_CODE_MAX = 20


class V2Stats:
    """Mutable counters describing what the v2 remap/filter actually did."""

    def __init__(self) -> None:
        self.entities_in = 0
        self.entities_out = 0
        self.entities_dropped_noise_type = 0
        self.entities_dropped_subtype = 0  # SUBTYPE_DROP
        self.entities_dropped_empty_text = 0
        self.entities_dropped_dup = 0
        self.entities_subtype_remapped = 0
        self.entities_subtype_unchanged = 0
        self.entities_subtype_extra_allowed = 0  # declared, outside the 32
        self.entities_subtype_uncovered = 0  # outside ALLOWED_SUBTYPES entirely
        self.entities_evidence_missing = 0
        self.events_in = 0
        self.events_out = 0
        self.events_dropped_no_content = 0
        self.events_with_action = 0
        self.events_no_valid_action = 0  # action emitted as null
        self.events_multi_action_truncated = 0
        self.events_actor2_null = 0
        self.events_evidence_missing = 0
        self.topics_out = 0
        self.topics_dropped_bad_code = 0
        self.topic_name_null = 0
        self.topic_name_unverified_source = 0  # source code_verified is False
        self.action_name_null = 0
        self.dropped_subtype_counts: Counter[str] = Counter()
        self.extra_allowed_subtype_counts: Counter[str] = Counter()
        self.uncovered_subtype_counts: Counter[str] = Counter()
        self.unknown_topic_codes: Counter[str] = Counter()
        self.unknown_action_codes: Counter[str] = Counter()
        # Prompt-length guard (see PromptBudget).
        self.prompt_guard_active = 0
        self.prompt_over_budget = 0        # docs whose prompt exceeded the budget
        self.prompt_truncated = 0          # docs whose text was cut to fit
        self.prompt_chars_trimmed = 0
        self.prompt_max_tokens_seen = 0
        self.prompt_unfittable = 0         # could not fit even with empty body
        self.gold_evidence_outside_doc = 0  # gold spans past the truncation point

    def as_dict(self) -> dict[str, Any]:
        out = {
            k: v
            for k, v in vars(self).items()
            if not isinstance(v, Counter)
        }
        out["dropped_subtype_counts"] = dict(self.dropped_subtype_counts)
        out["extra_allowed_subtype_counts"] = dict(
            self.extra_allowed_subtype_counts.most_common(50)
        )
        out["uncovered_subtype_counts"] = dict(self.uncovered_subtype_counts.most_common(50))
        out["unknown_topic_codes"] = dict(self.unknown_topic_codes.most_common(50))
        out["unknown_action_codes"] = dict(self.unknown_action_codes.most_common(50))
        return out


def _remap_subtype_tracked(raw: Any, st: V2Stats) -> tuple[str | None, bool]:
    """`remap_subtype` plus bookkeeping. Returns (subtype, keep_entity)."""
    from grpo.subtype_taxonomy import remap_subtype

    raw_s = raw if raw is None else str(raw).strip()
    mapped = remap_subtype(raw_s)
    if mapped is None:
        st.entities_dropped_subtype += 1
        st.dropped_subtype_counts[str(raw_s)] += 1
        return None, False
    if mapped != raw_s:
        st.entities_subtype_remapped += 1
    else:
        st.entities_subtype_unchanged += 1
    if mapped and mapped in EXTRA_ALLOWED_TYPES:
        # Declared members of the vocabulary that sit outside the 32 (see
        # grpo/subtype_taxonomy.py). Counted separately so they cannot mask a
        # genuine gap: media_outlet alone is ~4.7% of gold entities, and lumping
        # it in with real misses made the "uncovered" alarm useless.
        st.entities_subtype_extra_allowed += 1
        st.extra_allowed_subtype_counts[mapped] += 1
    elif mapped and mapped not in ALLOWED_SUBTYPES:
        # A label outside the declared vocabulary entirely. Not a crash — data
        # prep degrades rather than dying on unseen rare labels — but this is a
        # real signal that grpo/subtype_taxonomy.py needs updating, and the
        # reward's exact-match subtype term will be scoring an undeclared value.
        st.entities_subtype_uncovered += 1
        st.uncovered_subtype_counts[mapped] += 1
    return mapped, True


def clean_entity_v2(e: dict[str, Any], st: V2Stats) -> dict[str, Any] | None:
    st.entities_in += 1
    et = (e.get("type") or "").upper().strip()
    if et in NOISE_TYPES or not et:
        st.entities_dropped_noise_type += 1
        return None
    text = (e.get("text") or "").strip()
    if not text:
        st.entities_dropped_empty_text += 1
        return None
    subtype, keep = _remap_subtype_tracked(e.get("subtype"), st)
    if not keep:
        return None
    evidence = e.get("evidence_text")
    if isinstance(evidence, str):
        evidence = evidence.strip() or None
    if not evidence:
        st.entities_evidence_missing += 1
    return {
        "text": text,
        "type": et,
        "subtype": subtype,
        "evidence_text": evidence,
    }


def clean_topics_v2(raw_topics: Any, st: V2Stats) -> list[dict[str, Any]]:
    """All source topics whose code matches NN.N, with canonical CAMEO names."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for t in raw_topics or []:
        if not isinstance(t, dict):
            st.topics_dropped_bad_code += 1
            continue
        code = str(t.get("code") or "").strip()
        if not TOPIC_CODE_RE.match(code):
            st.topics_dropped_bad_code += 1
            continue
        if code in seen:
            continue
        seen.add(code)
        if t.get("code_verified") is False:
            # Kept on purpose (the v2 decision is "all NN.N codes"), but counted
            # so the noise level stays visible.
            st.topic_name_unverified_source += 1
        name = canonical_topic_name(code)
        if name is None:
            st.topic_name_null += 1
            st.unknown_topic_codes[code] += 1
        out.append({"code": code, "name": name})
        st.topics_out += 1
    return out


def pick_action_v2(raw_actions: Any, st: V2Stats) -> dict[str, Any] | None:
    """First source action code in 13-20, as a single {code, name} object.

    Returns None when the event has no such code — the event is still kept and
    emits `action: null`. Most source events carry an empty `actions` list, so
    dropping those events would discard the large majority of event supervision.
    """
    valid: list[str] = []
    for a in raw_actions or []:
        code = str((a.get("code") if isinstance(a, dict) else a) or "").strip()
        if not ACTION_CODE_RE.match(code):
            continue
        try:
            if not (ACTION_CODE_MIN <= int(code) <= ACTION_CODE_MAX):
                continue
        except ValueError:
            continue
        valid.append(code)
    if not valid:
        st.events_no_valid_action += 1
        return None
    if len(valid) > 1:
        st.events_multi_action_truncated += 1
    code = valid[0]
    name = canonical_action_name(code)
    if name is None:
        # Emit the numeric code anyway; only the name is unknown.
        st.action_name_null += 1
        st.unknown_action_codes[code] += 1
    st.events_with_action += 1
    return {"code": code, "name": name}


def clean_event_v2(ev: dict[str, Any], st: V2Stats) -> dict[str, Any]:
    st.events_in += 1
    actor1 = ev.get("actor1")
    if isinstance(actor1, str):
        actor1 = actor1.strip() or None
    actor2 = ev.get("actor2")
    if isinstance(actor2, str):
        actor2 = actor2.strip() or None
    if actor2 is None:
        # ~12.5% of source events genuinely have no second actor. Never
        # synthesize one; the reward must not penalize a true null.
        st.events_actor2_null += 1
    evidence = ev.get("evidence_text")
    if isinstance(evidence, str):
        evidence = evidence.strip() or None
    if not evidence:
        st.events_evidence_missing += 1
    return {
        "actor1": actor1,
        "actor2": actor2,
        "action": pick_action_v2(ev.get("actions"), st),
        "topics": clean_topics_v2(ev.get("topics"), st),
        "evidence_text": evidence,
    }


def build_target_v2(rec: dict[str, Any], st: V2Stats) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for e in rec.get("entities") or []:
        ce = clean_entity_v2(e, st)
        if not ce:
            continue
        key = (ce["text"].lower(), ce["type"], ce.get("subtype"))
        if key in seen:
            st.entities_dropped_dup += 1
            continue
        seen.add(key)
        entities.append(ce)
        st.entities_out += 1

    events: list[dict[str, Any]] = []
    for ev in rec.get("events") or []:
        cev = clean_event_v2(ev, st)
        # Keep any event that still carries something scorable. `action` alone
        # is not required (it is null for most source events).
        if cev["actor1"] or cev["topics"] or cev["action"] or cev["evidence_text"]:
            events.append(cev)
            st.events_out += 1
        else:
            st.events_dropped_no_content += 1
    return {"entities": entities, "events": events}


def build_user_v2(text: str, title: str | None) -> str:
    parts = []
    if title and str(title).strip():
        parts.append(f"Title: {str(title).strip()}")
    parts.append("Document:")
    parts.append(text.strip())
    parts.append(
        '\nExtract entities and events. Return ONLY a JSON object with keys '
        '"entities" and "events".'
    )
    return "\n".join(parts)


class PromptBudget:
    """Token-exact prompt-length guard for the rendered chat prompt.

    Why this exists
    ---------------
    v2's system prompt is longer than v1-minimal's 85 tokens (it has to specify
    four more fields), which pushed a slice of documents over
    `data.max_prompt_length`. Because `data.filter_overlong_prompts=True`, VERL
    **drops** those rows at load time without a per-row warning, so the
    regression was invisible. Measured at stride 20 over the whole corpus
    (4,029 docs) with the real tokenizer at a 1536-token budget:

        v1-minimal  (85-token system prompt)         2.36% over budget
        v2 original (221-token system prompt)        7.27% over budget
        v2 current  (165-token system prompt)        4.49% over budget
        v2 current + this guard                     *0.00%* over budget

    The middle row is the ~4,000-document silent loss this class exists to
    prevent; the last row is what it delivers, with 0 documents unfittable.

    Why the guard is in tokens and not characters
    ---------------------------------------------
    `--max_source_chars` cannot express this budget. The corpus is multilingual,
    and CJK text runs far more tokens per character, so a character cap that is
    generous for English still blows the token budget for other scripts.
    Measured over the same sample, tightening `--max_source_chars` from 6000 to
    4000 only moved the >1536-token rate from 7.27% to 1.41% — it never reaches
    zero, and it needlessly truncates English documents that were already fine.
    Raising `data.max_prompt_length` to 2048 instead still leaves 1.51%
    dropped. Fitting per-document in token space is the only option that
    reaches 0.00%, and it trims only the 4.49% of documents that need it.

    The `doc_text` recorded in `extra_info` is the text left **after** this
    trim, so `grpo/reward_ie_v2.py`'s evidence grounding keeps scoring against
    exactly what the model saw; gold spans past the cut are counted in
    `gold_evidence_outside_doc` rather than silently inflating or deflating the
    evidence term.
    """

    def __init__(self, tokenizer: Any, max_tokens: int, system: str) -> None:
        self.tok = tokenizer
        self.max_tokens = max_tokens
        self.system = system

    def n_tokens(self, user: str) -> int:
        """Exact token count of the rendered prompt VERL will measure."""
        rendered = self.tok.apply_chat_template(
            [
                {"role": "system", "content": self.system},
                {"role": "user", "content": user},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        return len(self.tok(rendered, add_special_tokens=False)["input_ids"])

    def fit(self, text: str, title: str | None) -> tuple[str, int, int]:
        """Shrink `text` until the rendered prompt fits. Returns (text, tokens, trimmed_chars).

        Binary-searches the longest character prefix that fits, so the trim is
        the smallest one that works rather than a blunt global cap. A document
        that cannot fit even with an empty body (a pathologically long title)
        returns tokens above the budget; the caller drops that row explicitly
        instead of letting VERL drop it silently.
        """
        n = self.n_tokens(build_user_v2(text, title))
        if n <= self.max_tokens:
            return text, n, 0

        lo, hi = 0, len(text)
        best: str | None = None
        best_n = n
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = text[:mid].rstrip() + TRUNCATION_MARK
            cand_n = self.n_tokens(build_user_v2(cand, title))
            if cand_n <= self.max_tokens:
                best, best_n = cand, cand_n
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            # Not fittable: report the empty-body cost so the caller can drop it.
            return text, best_n, 0
        return best, best_n, len(text) - len(best)


def split_ids(ids: list[Any], train_r: float, val_r: float, seed: int):
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return set(ids[:n_train]), set(ids[n_train : n_train + n_val]), set(ids[n_train + n_val :])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(ROOT / "data" / "ner_topic_llm_dataset.jsonl"))
    p.add_argument("--out_dir", default=str(ROOT / "data" / "grpo_v2"))
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--max_source_chars", type=int, default=6000,
                   help="Coarse pre-trim in characters. The real budget is "
                        "--max_prompt_tokens; this only bounds tokenizer work.")
    p.add_argument("--max_target_chars", type=int, default=8000)
    p.add_argument("--max_prompt_tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS,
                   help="Token budget for the rendered prompt. Must match "
                        "data.max_prompt_length in the launch script, because "
                        "filter_overlong_prompts silently DROPS longer rows.")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="Tokenizer used to enforce --max_prompt_tokens.")
    p.add_argument("--no_prompt_token_guard", action="store_true",
                   help="Disable the token-exact prompt guard (not recommended: "
                        "rows over budget are then dropped silently at load).")
    p.add_argument("--train_ratio", type=float, default=0.90)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_examples", type=int, default=0,
                   help="Print N rendered gold targets to stdout for eyeballing.")
    p.add_argument("--mlflow", action="store_true", default=True)
    p.add_argument("--no_mlflow", action="store_true")
    p.add_argument("--mlflow_uri", default=None)
    p.add_argument("--experiment", default="gspo-ie-qwen3-0.6b")
    args = p.parse_args(argv)

    use_mlflow = args.mlflow and not args.no_mlflow
    source = Path(args.source).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise SystemExit(f"Source dataset not found: {source}\nPass --source explicitly.")

    st = V2Stats()
    rows: list[dict[str, Any]] = []
    skipped = {"empty": 0, "no_labels": 0, "too_long": 0, "prompt_unfittable": 0}

    budget: PromptBudget | None = None
    if not args.no_prompt_token_guard:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            budget = PromptBudget(tok, args.max_prompt_tokens, SYSTEM_IE_V2)
            st.prompt_guard_active = 1
            sys_tokens = len(tok(SYSTEM_IE_V2, add_special_tokens=False)["input_ids"])
            print(
                f"Prompt guard ON: tokenizer={args.tokenizer} "
                f"budget={args.max_prompt_tokens} tokens "
                f"(system prompt alone = {sys_tokens})"
            )
        except Exception as e:  # noqa: BLE001 — must be loud, never silent
            print(
                f"\n*** WARNING: prompt-token guard DISABLED: could not load "
                f"tokenizer {args.tokenizer!r} ({type(e).__name__}: {e}).\n"
                f"*** Rows whose prompt exceeds data.max_prompt_length will be "
                f"SILENTLY DROPPED by VERL at load time "
                f"(filter_overlong_prompts=True). Pass --tokenizer explicitly.\n"
            )
    else:
        print(
            "*** WARNING: --no_prompt_token_guard set; overlong prompts will be "
            "silently dropped at load."
        )

    print(f"Reading {source}  (schema={SCHEMA_VERSION})")
    with open(source, encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="prepare-parquet-v2")):
            if args.max_docs is not None and i >= args.max_docs:
                break
            rec = json.loads(line)
            text = (rec.get("text") or "").strip()
            if not text:
                skipped["empty"] += 1
                continue
            if len(text) > args.max_source_chars:
                text = text[: args.max_source_chars] + TRUNCATION_MARK

            target = build_target_v2(rec, st)
            if not target["entities"] and not target["events"]:
                skipped["no_labels"] += 1
                continue
            gt = json.dumps(target, ensure_ascii=False)
            if len(gt) > args.max_target_chars:
                skipped["too_long"] += 1
                continue

            title = rec.get("title")
            if budget is not None:
                fitted, n_tok, trimmed = budget.fit(text, title)
                if n_tok > budget.max_tokens:
                    # Unfittable even with an empty body — drop it here, loudly
                    # in the stats, rather than letting VERL drop it in silence.
                    st.prompt_unfittable += 1
                    skipped["prompt_unfittable"] += 1
                    continue
                if trimmed:
                    st.prompt_over_budget += 1
                    st.prompt_truncated += 1
                    st.prompt_chars_trimmed += trimmed
                    text = fitted
                st.prompt_max_tokens_seen = max(st.prompt_max_tokens_seen, n_tok)
                # Gold evidence past the cut can no longer be grounded against
                # the document the model sees. Counted, never silently ignored.
                if trimmed:
                    for key in ("entities", "events"):
                        for item in target[key]:
                            ev = item.get("evidence_text")
                            if isinstance(ev, str) and ev and ev not in text:
                                st.gold_evidence_outside_doc += 1

            user = build_user_v2(text, title)
            rows.append(
                {
                    "data_source": "ner_topic_ie",
                    "prompt": [
                        {"role": "system", "content": SYSTEM_IE_V2},
                        {"role": "user", "content": user},
                    ],
                    "ability": "ie",
                    "reward_model": {"style": "rule", "ground_truth": gt},
                    "extra_info": {
                        "doc_id": rec.get("doc_id"),
                        "source_platform": rec.get("source_platform"),
                        "split": None,
                        "index": i,
                        "schema_version": SCHEMA_VERSION,
                        "n_entities": len(target["entities"]),
                        "n_events": len(target["events"]),
                        # Grounding target for the v2 evidence_text reward: the
                        # (possibly truncated) document the model actually saw.
                        "doc_text": text,
                        # gold completion for optional SFT
                        "answer": gt,
                    },
                }
            )

    train_ids, val_ids, test_ids = split_ids(
        [r["extra_info"]["doc_id"] for r in rows],
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for r in rows:
        did = r["extra_info"]["doc_id"]
        if did in train_ids:
            split = "train"
        elif did in val_ids:
            split = "val"
        else:
            split = "test"
        r["extra_info"]["split"] = split
        splits[split].append(r)

    try:
        import datasets as hf_datasets
    except ImportError:
        hf_datasets = None

    for name, part in splits.items():
        path = out_dir / f"{name}.parquet"
        if hf_datasets is not None:
            ds = hf_datasets.Dataset.from_list(part)
            ds.to_parquet(str(path))
        else:
            import pandas as pd

            pd.DataFrame(part).to_parquet(path, index=False)
        print(f"  {name}: {len(part)} → {path}")

    if rows:
        mid = rows[len(rows) // 2]
        (out_dir / "example_prompt.json").write_text(
            json.dumps(mid["prompt"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "example_gt.json").write_text(
            mid["reward_model"]["ground_truth"], encoding="utf-8"
        )
        try:
            (out_dir / "example_gt_pretty.json").write_text(
                json.dumps(
                    json.loads(mid["reward_model"]["ground_truth"]),
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except json.JSONDecodeError:
            pass

    if args.print_examples and rows:
        step = max(len(rows) // (args.print_examples + 1), 1)
        for k in range(args.print_examples):
            r = rows[min(step * (k + 1), len(rows) - 1)]
            print(f"\n===== EXAMPLE {k + 1} (doc_id={r['extra_info']['doc_id']}) =====")
            print(
                json.dumps(
                    json.loads(r["reward_model"]["ground_truth"]),
                    indent=2,
                    ensure_ascii=False,
                )
            )

    stats = {
        "schema_version": SCHEMA_VERSION,
        "num_rows": len(rows),
        "splits": {k: len(v) for k, v in splits.items()},
        "skipped": skipped,
        "source": str(source),
        "avg_entities": sum(r["extra_info"]["n_entities"] for r in rows) / max(len(rows), 1),
        "avg_events": sum(r["extra_info"]["n_events"] for r in rows) / max(len(rows), 1),
        "v2": st.as_dict(),
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))

    if use_mlflow:
        with mlflow_run(
            run_name="prepare_parquet_v2",
            tags={
                "stage": "data_prep",
                "task": "gspo_ie",
                "schema_version": SCHEMA_VERSION,
                "pipeline": "v2_subtype_taxonomy_cameo_names",
            },
            params={
                "schema_version": SCHEMA_VERSION,
                "source": str(source),
                "out_dir": str(out_dir),
                "max_docs": args.max_docs or -1,
                "max_source_chars": args.max_source_chars,
                "max_prompt_tokens": args.max_prompt_tokens,
                "prompt_token_guard": bool(budget is not None),
                "tokenizer": args.tokenizer,
                "seed": args.seed,
            },
            tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment,
        ):
            metrics = {
                "num_rows": stats["num_rows"],
                "num_train": stats["splits"]["train"],
                "num_val": stats["splits"]["val"],
                "num_test": stats["splits"]["test"],
                "skipped_empty": skipped["empty"],
                "skipped_no_labels": skipped["no_labels"],
                "skipped_too_long": skipped["too_long"],
                "skipped_prompt_unfittable": skipped["prompt_unfittable"],
                "avg_entities": stats["avg_entities"],
                "avg_events": stats["avg_events"],
            }
            metrics.update(
                {
                    f"v2_{k}": v
                    for k, v in st.as_dict().items()
                    if isinstance(v, (int, float))
                }
            )
            log_metrics(metrics)
            log_artifact(stats_path)
            for fn in ("example_gt_pretty.json", "example_prompt.json"):
                if (out_dir / fn).exists():
                    log_artifact(out_dir / fn)
            log_dict_as_json(stats, "prepare_stats_v2.json")
            print(f"MLflow: logged prepare_parquet_v2 → experiment {args.experiment}")

    print(f"Wrote {out_dir}")


def _self_test() -> None:
    """Structural checks on the prompt/target/taxonomy contract. No GPU, no data.

    Run with `python grpo/prepare_parquet_v2.py --self_test`. This is the third
    self-test suite alongside grpo/topic_taxonomy.py and grpo/subtype_taxonomy.py,
    and it exists because the failure this pipeline is most exposed to is silent
    drift between what the prompt asks for, what the target emits, and what
    grpo/reward_ie_v2.py scores.
    """
    from grpo import reward_ie_v2 as rw

    # 1. Every scored field is actually requested by the prompt.
    for f in SCORED_ENTITY_FIELDS + SCORED_EVENT_FIELDS:
        assert f'"{f}"' in SYSTEM_IE_V2, f"prompt never asks for scored field {f!r}"

    # 2. The prompt must not request a field nothing scores — that is pure
    #    response-token cost, the exact regression that sank the v1 full schema.
    for banned in ("other_actors", "normalized", "score", "resolution_method",
                   "char_start", "char_end", "confidence", "code_verified"):
        assert banned not in SYSTEM_IE_V2, f"prompt requests unscored field {banned!r}"

    # 3. build_target_v2 emits exactly the advertised field sets.
    st = V2Stats()
    rec = {
        "entities": [
            {"text": "Taiwan", "type": "LOCATION", "subtype": "country",
             "evidence_text": "near Taiwan", "score": 0.9, "normalized": "TW",
             "char_start": None},
            {"text": "x", "type": "PERSON", "subtype": "activist",
             "evidence_text": "x"},                      # SUBTYPE_DROP
            {"text": "y", "type": "HDL", "subtype": "UNKNOWN"},  # NOISE_TYPES
            {"text": "Ministry", "type": "ORGANIZATION",
             "subtype": "defense_ministry", "evidence_text": "the Ministry"},
            {"text": "Vote", "type": "EVENT", "subtype": "election",
             "evidence_text": "the Vote"},
        ],
        "events": [{
            "actor1": "China", "actor2": None,
            "actions": [{"code": "15"}, {"code": "17"}],
            "topics": [{"code": "01.1"}, {"code": "99.9"}, {"code": "EVT"}],
            "evidence_text": "China said", "other_actors": ["a"],
        }],
    }
    tgt = build_target_v2(rec, st)
    assert [e["text"] for e in tgt["entities"]] == ["Taiwan", "Ministry", "Vote"], tgt
    for e in tgt["entities"]:
        assert set(e) == set(SCORED_ENTITY_FIELDS), set(e) ^ set(SCORED_ENTITY_FIELDS)
    assert len(tgt["events"]) == 1
    assert set(tgt["events"][0]) == set(SCORED_EVENT_FIELDS)

    # 4. Subtype remap/drop behaviour, including the two bugs fixed here.
    assert tgt["entities"][1]["subtype"] == "government_body", "defense_ministry fold"
    assert tgt["entities"][2]["subtype"] == "election", "election must stay EVENT-shaped"
    assert st.entities_dropped_subtype == 1 and st.entities_dropped_noise_type == 1
    for e in tgt["entities"]:
        assert e["subtype"] in ALLOWED_SUBTYPES, e

    # 5. `actions` collapses to a singular in-range `action`; topics keep only NN.N.
    assert tgt["events"][0]["action"] == {"code": "15", "name": "Exhibit Force Posture"}
    assert st.events_multi_action_truncated == 1
    assert [t["code"] for t in tgt["events"][0]["topics"]] == ["01.1", "99.9"]
    assert st.topics_dropped_bad_code == 1  # "EVT"

    # 6. Names come from the source taxonomy, and an unnamed code emits null
    #    rather than dropping the code.
    names = {t["code"]: t["name"] for t in tgt["events"][0]["topics"]}
    assert names["01.1"] == "Bilateral Political Relations", names
    assert names["99.9"] is None and st.topic_name_null == 1
    # The CAMEO name this pipeline used to emit for 01.1 must never come back.
    assert names["01.1"] != "Decline Comment"

    # 7. A v2 gold target scores a perfect 1.0 against itself under the v2 reward
    #    (prompt/target/reward all agreeing is the whole point).
    doc = "China said it would act near Taiwan. the Ministry and the Vote followed."
    gt = json.dumps({
        "entities": [{"text": "Taiwan", "type": "LOCATION", "subtype": "country",
                      "evidence_text": "near Taiwan"}],
        "events": [{"actor1": "China", "actor2": None,
                    "action": {"code": "15", "name": "Exhibit Force Posture"},
                    "topics": [{"code": "01.1", "name": "Bilateral Political Relations"}],
                    "evidence_text": "China said it would act near Taiwan"}],
    }, ensure_ascii=False)
    m = rw.compute_score_detailed(
        solution_str=gt, ground_truth=gt, extra_info={"doc_text": doc})
    for k in ("format", "entity_f1", "subtype", "topic_f1", "topic_name",
              "action_f1", "action_name", "actor_f1", "evidence_text"):
        assert m[k] == 1.0, (k, m[k])

    print(f"SYSTEM_IE_V2: {len(SYSTEM_IE_V2)} chars")
    print(f"scored entity fields: {SCORED_ENTITY_FIELDS}")
    print(f"scored event fields:  {SCORED_EVENT_FIELDS}")
    print("prepare_parquet_v2 self-checks OK")


if __name__ == "__main__":
    if "--self_test" in sys.argv:
        _self_test()
    else:
        main()
