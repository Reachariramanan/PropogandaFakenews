"""Per-source column maps for the news and social databases, and text normalisation.

Every source row becomes a dict with the same keys (see `fetch_candidates`). All queries here go
through `connect_source`, i.e. read-only sessions; there is no write path to the source DBs.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from psycopg2 import sql

from ..config import C


@dataclass(frozen=True)
class Src:
    kind: str                      # news | social
    platform: str
    table: str                     # schema-qualified, as in the source DB
    pk: str
    ts: str                        # ingestion time column (watermark key)
    pub: str                       # publication time column
    text: tuple[str, ...]          # body columns; first non-empty wins
    title: Optional[str] = None
    url: Optional[str] = None
    author: Optional[str] = None
    outlet: Optional[str] = None
    lang: tuple[str, ...] = ()
    truncate: bool = False              # over-long text is cut at a sentence boundary instead of being dropped
    url_expr: Optional[str] = None      # constant SQL expression building a link when there is no url column
    group_by_pk: bool = False           # items of one video share a split group (dup_key), so they never straddle train/val

    @property
    def db(self) -> str:
        return "shaktidb_news" if self.kind == "news" else "shaktidb_social"


_S = "public."
SOURCES: tuple[Src, ...] = (
    Src("news", "news", "raw.scraped_articles_trusted", "id", "scraped_at", "published_date",
        ("article_content",), title="article_title", url="sourceurl", author="author", outlet="source",
        lang=("language",)),
    Src("social", "twitter", _S + "twitter_posts", "id", "scraped_at", "posted_at", ("text",),
        url="post_url", author="author_username", lang=("language", "detected_language")),
    Src("social", "x", _S + "x_posts", "tweet_id", "scraped_at", "posted_at", ("text",),
        url="post_url", author="author_username", lang=("language", "detected_language")),
    Src("social", "telegram", _S + "telegram_posts", "id", "scraped_at", "posted_at", ("text",),
        url="post_url", author="channel_name", outlet="channel_name", lang=("language", "detected_language")),
    Src("social", "truthsocial", _S + "truthsocial_posts", "post_id", "collected_at", "created_at", ("text_plain",),
        url="url", author="author_handle", lang=("language", "detected_language")),
    Src("social", "snapchat", _S + "snapchat_posts", "id", "scraped_at", "posted_at", ("caption",),
        title="title", url="spotlight_url", author="creator", lang=("language", "detected_language")),
    Src("social", "tiktok", _S + "tiktok_posts", "post_id", "collected_at", "create_time_local", ("desc",),
        url="share_url", author="author_username", lang=("language", "detected_language")),
    Src("social", "mastodon", _S + "mastodon_posts", "id", "collected_at", "created_at", ("content_text",),
        url="url", author="account_acct", lang=("language", "detected_language")),
    Src("social", "linkedin", _S + "linkedin_posts", "id", "scraped_at", "posted_at", ("main_text",),
        title="title", url="post_url", author="author_page", lang=("language", "detected_language")),
    Src("social", "whatsapp", _S + "whatsapp_posts", "id", "scraped_at", "posted_at", ("text", "caption"),
        author="channel_id", lang=("language", "detected_language")),
    Src("social", "reddit", _S + "reddit_posts", "id", "scraped_at", "posted_at", ("body",),
        title="title", url="url", author="author", outlet="source", lang=("language", "detected_language")),
    Src("social", "instagram", _S + "instagram_posts", "id", "scraped_at", "posted_at", ("caption",),
        url="post_url", author="user_name", lang=("language", "detected_language")),
    Src("social", "facebook", _S + "facebook_posts", "id", "scraped_at", "posted_time", ("post_text",),
        url="url", author="author_name", lang=("language", "detected_language")),
    # YouTube: comments are short posts; a video yields two kinds of item, its description and the start of its
    # transcript (spoken content, often richer in claims than the description, which is mostly promo links).
    Src("social", "youtube_comment", _S + "youtube_comments", "comment_id", "scraped_at", "published_at",
        ("comment_text",), url_expr="'https://www.youtube.com/watch?v=' || video_id || '&lc=' || comment_id",
        author="author_display_name", lang=("language", "detected_language")),
    Src("social", "youtube_video", _S + "youtube_videos", "video_id", "scraped_at", "video_published_at",
        ("video_description",), title="video_title", url="video_url", author="channel_name", outlet="channel_name",
        lang=("language", "detected_language"), truncate=True, group_by_pk=True),
    Src("social", "youtube_transcript", _S + "youtube_videos", "video_id", "scraped_at", "video_published_at",
        ("transcript_text",), title="video_title", url="video_url", author="channel_name", outlet="channel_name",
        lang=("transcript_language", "language", "detected_language"), truncate=True, group_by_pk=True),
)
BY_KEY = {(s.kind, s.platform): s for s in SOURCES}


def length_bounds(src: Src) -> tuple[int, int]:
    k = "news" if src.kind == "news" else "social"
    return int(C(f"ann_{k}_min_chars")), int(C(f"ann_{k}_max_chars"))


# ------------------------------------------------------------------ SQL
def _tbl(name: str) -> sql.Composable:
    return sql.SQL(".").join(sql.Identifier(p) for p in name.split("."))


FETCH_FACTOR = 4   # truncating sources fetch only the first FETCH_FACTOR x cap characters (transcripts run to 50k+)


def _body(src: Src) -> sql.Composable:
    parts = [sql.SQL("NULLIF(btrim({}), '')").format(sql.Identifier(c)) for c in src.text]
    body = sql.SQL("COALESCE({})").format(sql.SQL(", ").join(parts))
    if src.truncate:
        body = sql.SQL("left({}, {})").format(body, sql.Literal(FETCH_FACTOR * length_bounds(src)[1]))
    return body


def select_sql(src: Src, where: sql.Composable, tail: sql.Composable, sample_pct: Optional[float] = None) -> sql.Composable:
    nul = sql.SQL("NULL")
    ident = lambda c: sql.Identifier(c) if c else nul
    url = sql.SQL(src.url_expr) if src.url_expr else ident(src.url)
    lang = sql.SQL("COALESCE({})").format(sql.SQL(", ").join(
        sql.SQL("NULLIF(btrim({}), '')").format(sql.Identifier(c)) for c in src.lang)) if src.lang else nul
    sample = sql.SQL(" TABLESAMPLE BERNOULLI ({})").format(sql.Literal(sample_pct)) if sample_pct else sql.SQL("")
    return sql.SQL("SELECT {pk}::text AS pk, {ts} AS ts, {pub} AS pub, {title} AS title, {body} AS body, "
                   "{url} AS url, {author}::text AS author, {outlet}::text AS outlet, {lang}::text AS lang "
                   "FROM {tbl}{sample} WHERE {ts} IS NOT NULL AND {body} IS NOT NULL AND {where} {tail}").format(
        pk=sql.Identifier(src.pk), ts=sql.Identifier(src.ts), pub=ident(src.pub), title=ident(src.title),
        body=_body(src), url=url, author=ident(src.author), outlet=ident(src.outlet), lang=lang,
        tbl=_tbl(src.table), sample=sample, where=where, tail=tail)


def length_where(src: Src) -> sql.Composable:
    lo, hi = length_bounds(src)
    if src.truncate:                    # no upper bound: long text is cut at a sentence boundary in _prepare
        return sql.SQL("char_length({}) >= {}").format(_body(src), sql.Literal(lo))
    # prefilter on the body column(s) only; the exact bounds are re-applied on the composed text
    return sql.SQL("char_length({}) BETWEEN {} AND {}").format(_body(src), sql.Literal(lo),
                                                              sql.Literal(hi))


# ------------------------------------------------------------------ text normalisation
_NUL = re.compile("\x00")
_URL = re.compile(r"https?://\S+")
_MENT = re.compile(r"@\w+")
_RT = re.compile(r"^\s*rt\s+@\w+\s*:\s*", re.I)
_NONWORD = re.compile(r"\W+", re.UNICODE)


def clean_text(s: str) -> str:
    """Only what the DB/UI cannot store verbatim: NULs removed, CRLF -> LF, outer whitespace stripped."""
    return _NUL.sub("", s).replace("\r\n", "\n").replace("\r", "\n").strip()


_SENT_END = re.compile(r"[.!?।॥؟…]+[\"'”’)\]]*(?=\s|$)")
_WS = re.compile(r"\s")


_ABBREV = frozenset("mr mrs ms dr prof sr jr st vs etc no gen col sen rep gov lt capt sgt hon rev mt fig approx".split())


def _abbrev_before(window: str, m: "re.Match") -> bool:
    """True when a lone '.' ends an abbreviation or initial (Mr. / Dr. / J. / U.S.), not a sentence."""
    if m.group() != ".":
        return False
    tok = window[: m.start()].split()[-1:] or [""]
    w = tok[0].strip("\"'“‘([").lower()
    return w in _ABBREV or len(w) == 1 or "." in w


def truncate_at_boundary(text: str, cap: int, min_frac: float = 0.6) -> tuple[str, Optional[str]]:
    """Cut `text` to at most `cap` characters at the nearest natural stopping point, keeping as much as possible.
    Preference: the last sentence end (. ! ? । ؟ ...) or paragraph break at/after `min_frac` x cap, then the last line
    break, then the last word boundary, and only then a hard cut. The result is always a verbatim prefix, so character
    offsets into the stored text stay valid. -> (text, how) with how in sentence|line|word|hard, or None if uncut."""
    if len(text) <= cap:
        return text, None
    floor = int(cap * min_frac)
    window = text[: cap + 1]                       # one extra char so a boundary at `cap` is recognised
    cut = max((m.end() for m in _SENT_END.finditer(window) if floor <= m.end() <= cap and not _abbrev_before(window, m)), default=0)
    para = window.rfind("\n\n", floor, cap + 1)
    cut = max(cut, para)
    if cut:
        return text[:cut].rstrip(), "sentence"
    nl = window.rfind("\n", floor, cap + 1)
    if nl > 0:
        return text[:nl].rstrip(), "line"
    ws = max((m.start() for m in _WS.finditer(window) if m.start() <= cap), default=0)
    if ws >= floor:
        return text[:ws].rstrip(), "word"
    return text[:cap].rstrip(), "hard"


def compose(title: Optional[str], body: str) -> tuple[str, str, int]:
    """-> (text, title, title_len). text = title + blank line + body so one offset space covers both."""
    body = clean_text(body or "")
    t = clean_text(title or "")
    if t and t != body and not body.startswith(t):
        return t + "\n\n" + body, t, len(t) + 2
    return body, "", 0


def norm(text: str) -> str:
    return _NONWORD.sub(" ", unicodedata.normalize("NFKC", text).lower()).strip()


def norm_hash(text: str) -> str:
    return hashlib.sha1(norm(text).encode()).hexdigest()


def dup_key(text: str) -> str:
    """Near-duplicate key (retweets / forwards / re-posts with a different tail): the first 12 words
    of the text after dropping RT prefixes, URLs and @mentions. Items sharing it are split together."""
    t = _MENT.sub(" ", _URL.sub(" ", _RT.sub("", text)))
    toks = norm(t).split()
    key = " ".join(toks[:12]) if len(toks) >= 4 else " ".join(toks)
    return hashlib.sha1(key.encode()).hexdigest()


_URL_MENTION = re.compile(r"https?://\S+|www\.\S+|@\w+")


def substance(body: str) -> int:
    """Letters left once URLs and @mentions are removed: a link-only post has none (hashtags count: they can be slogans)."""
    return sum(ch.isalpha() for ch in _URL_MENTION.sub(" ", body))


def body_key(text: str, title_len: int = 0) -> str:
    """Near-duplicate key of the body only (the title is dropped, so a video's title does not make its description and
    transcript look alike). Same construction as dup_key: first 12 words after RT / URL / @mention removal."""
    return dup_key(text[title_len:])


def video_group_key(video_id: str) -> str:
    """dup_key (char(40), a sha1) for every item derived from one video: its description and transcript split together."""
    return hashlib.sha1(("video:" + video_id).encode()).hexdigest()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def lang_group(lang: Optional[str], text: str) -> tuple[str, str]:
    """-> (lang code or 'unknown', 'en'|'hi'|'other'). Falls back to script detection when the source has none."""
    l = (lang or "").strip().lower()
    if l in ("english", "eng") or l.startswith("en"):
        return "en", "en"
    if l in ("hindi", "hin") or l.startswith("hi"):
        return "hi", "hi"
    if l and l not in ("unknown", "und", "xx", "null", "none"):
        return l[:12], "other"
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "unknown", "other"
    dev = sum("ऀ" <= ch <= "ॿ" for ch in letters) / len(letters)
    if dev > 0.3:
        return "hi", "hi"
    asc = sum(ch < "ɐ" for ch in letters) / len(letters)
    return ("en", "en") if asc > 0.9 else ("unknown", "other")


def flag_fraction(h: str, salt: str) -> float:
    """Deterministic uniform [0,1) from the item's norm_hash, so double/blind flags never depend on run order."""
    return int(hashlib.sha1((salt + h).encode()).hexdigest()[:8], 16) / 0x100000000
