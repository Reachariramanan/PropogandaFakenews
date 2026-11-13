"""Propaganda tagger v2: paragraph-window chunking (no 8k truncation) + optional
recall-oriented prompt with few-shot fragments mined from the PTC *train* split only.

tag_spans_v2(doc_id, text, prompt="v1"|"v2") -> list[Span] with offsets into `text`.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from ..schema import PROPAGANDA_TECHNIQUES
from . import prompts as P
from . import validators as V
from .cache import cached_call

CHUNK_CHARS = 1500
MAX_TOKENS = 3500
ROOT = Path(__file__).resolve().parents[2]


def chunk_windows(text: str, size: int = CHUNK_CHARS) -> list[tuple[int, str]]:
    """Split at paragraph/sentence boundaries into windows <= ~size chars. -> [(offset, chunk)]"""
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + size)
        if end < n:
            cut = max(text.rfind("\n", start + size // 2, end), text.rfind(". ", start + size // 2, end))
            if cut > start:
                end = cut + 1
        if text[start:end].strip():
            out.append((start, text[start:end]))
        start = end
    return out


@lru_cache(maxsize=1)
def fewshot_block(k: int = 3, seed: int = 7) -> str:
    """k short gold fragments per technique, from train-split articles only."""
    from ..datasets import load_ptc
    split = json.loads((ROOT / "data/splits/ptc_v1.json").read_text())["split"]
    arts, spans = load_ptc()
    text = dict(zip(arts.article_id, arts.text))
    by: dict[str, list[str]] = {}
    for r in spans.itertuples():
        if split.get(r.article_id) != "train":
            continue
        frag = " ".join(text[r.article_id][r.start:r.end].split())
        if 12 <= len(frag) <= 110:
            by.setdefault(r.technique, []).append(frag)
    rng = random.Random(seed)
    lines = []
    for t in by:        # gold labels use the 14 merged class names
        pool = sorted(set(by[t]))
        rng.shuffle(pool)
        if pool:
            lines.append(f'- {t.replace(',', ' / ')}: ' + " | ".join(f'"{p}"' for p in pool[:k]))
    return "\n".join(lines)


def system_v2() -> str:
    return P.COMMON + """

TASK: annotate persuasion-technique fragments in this text passage (Da San Martino et al., 2019 taxonomy, 18 techniques). Use EXACTLY these technique names:
""" + P._TECH_LINES + """

Annotation rules (follow the PTC annotators' conventions):
- Be thorough: news and opinion articles typically contain many fragments (often 10-30 per 1500 characters of opinion text). Tag every fragment that uses a technique, not just the clearest ones.
- Loaded_Language and Name_Calling,Labeling are the most common but NOT the only techniques. Actively look for Exaggeration,Minimisation, Doubt, Appeal_to_fear-prejudice, Flag-Waving, Causal_Oversimplification, Slogans, Appeal_to_Authority, Black-and-White_Fallacy, Thought-terminating_Cliches, Whataboutism, Bandwagon, Red_Herring, Straw_Men, Repetition.
- The same words may carry two techniques (e.g. a label that is also loaded); report both entries.
- Repetition: tag each repeated occurrence of a word/phrase the passage keeps hammering on.
- Appeal_to_Authority: quoted experts/officials used as the sole justification. Doubt: insinuating questions or discrediting without evidence.
- Fragments are SHORT: usually 1-8 words (Loaded_Language / Name_Calling) up to one clause or sentence (Causal_Oversimplification, Flag-Waving, Whataboutism, Doubt).
- Copy each fragment VERBATIM from the passage (original language and script, no ellipsis).
- Plain neutral reporting of facts is not propaganda; but when in doubt about a persuasive phrase, include it with a lower confidence.

Examples of real annotated fragments per technique (from other articles):
""" + fewshot_block() + """

For each fragment give "fragment", "technique", "confidence" (0..1). Also "one_sided" (0..1) and "framing" (one English sentence, or "neutral").
Output schema:
{"spans": [{"fragment": "...", "technique": "Doubt", "confidence": 0.7}], "one_sided": 0.3, "framing": "..."}"""


def tag_spans_v2(doc_id: str, text: str, *, prompt: str = "v2", cache=None) -> list[dict]:
    system = system_v2() if prompt == "v2" else P.PROPAGANDA_SYSTEM
    version = f"propaganda_v2_{prompt}_c{CHUNK_CHARS}"
    spans: list[dict] = []
    for off, chunk in chunk_windows(text):
        user = P.render(P.PROPAGANDA_USER, NOTE="", TEXT=chunk)
        res = cached_call("propaganda_v2", version, system, user, V.validate_propaganda,
                          MAX_TOKENS, retry_failed=True, cache=cache)
        if not res["ok"]:
            continue
        got, _, _, _ = V.parse_spans(res["obj"], doc_id, chunk)
        for s in got:
            s = dict(s)
            s["start"] += off
            s["end"] += off
            s["text"] = text[s["start"]:s["end"]]
            spans.append(s)
    return sorted(spans, key=lambda s: (s["start"], s["end"]))
