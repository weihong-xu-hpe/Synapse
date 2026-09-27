"""Lexical query/index text transforms for CJK-aware full-text search.

The default unicode61 FTS5 tokenizer keeps unsegmented Chinese runs as single
tokens, so a query like ``会话摘要 重复`` never matches documents containing
``会话摘要重复写入`` unless the exact character run appears. The bigram
transform splits CJK runs into overlapping 2-grams (``会话摘要`` → ``会话
话摘 摘要``) which are segmented cleanly by unicode61; non-CJK text passes
through unchanged so English words and code identifiers are unaffected.
"""

from __future__ import annotations

import re

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")
_WORD_CHAR = re.compile(r"\w", re.UNICODE)

# English/zh stopwords and ultra-short tokens dropped from OR queries. The
# corpus's highest-df CJK function words are included per the lexical-fixes
# experiment (/tmp/sparse-feasibility, §11).
STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was",
        "for", "on", "it", "this", "that", "with", "as", "be", "at", "by",
        "from", "but", "not", "what", "how", "why", "do", "does", "can",
        "should", "would", "please", "tell", "me", "about",
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都",
        "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
        "着", "没有", "看", "好", "自己", "这", "先", "吧", "呢", "吗", "啊",
        "把", "给", "让", "用", "能", "没", "还", "再", "才", "只", "等",
        "被", "跟", "对", "下", "里", "来",
    }
)

# Hard cap on OR terms: a 4000-char bridge recall query must stay fast.
MAX_QUERY_TERMS = 24


def cjk_bigrams(text: str) -> str:
    """Return text with every CJK run replaced by space-joined bigrams.

    Non-CJK characters are preserved verbatim. Single CJK characters are kept
    as-is (no 2-gram possible).
    """

    def _expand(match: re.Match[str]) -> str:
        run = match.group(0)
        if len(run) == 1:
            return run
        return " ".join(run[i : i + 2] for i in range(len(run) - 1))

    return _CJK_RUN.sub(_expand, text)


def has_cjk(text: str) -> bool:
    return bool(_CJK_CHAR.search(text))


def _clean_token(token: str) -> str:
    return token.replace('"', "").strip()


def build_or_query(query: str, *, max_terms: int = MAX_QUERY_TERMS) -> str:
    """Build an FTS5 OR expression from a raw user query.

    Tokens are derived both from whitespace splitting and from CJK run
    extraction (Chinese sentences contain no spaces, so whitespace splitting
    leaves whole sentences as single unusable tokens). Each term is quoted
    (FTS5 literal). Terms without any word character, stopwords, and single
    characters are dropped; the term count is capped so long bridge recall
    queries stay fast. Returns "" when nothing usable remains.
    """

    terms: list[str] = []
    raw_tokens: list[str] = []
    for chunk in query.split():
        raw_tokens.append(chunk)
        # CJK runs inside a chunk become separate candidate tokens: a long
        # zh sentence must not enter the index as one quoted blob.
        raw_tokens.extend(_CJK_RUN.findall(chunk))
    for raw in raw_tokens:
        token = _clean_token(raw)
        if not token or not _WORD_CHAR.search(token):
            continue
        lowered = token.casefold()
        if lowered in STOPWORDS:
            continue
        if len(token) < 2:
            continue
        term = f'"{token}"'
        if has_cjk(token):
            # Query-side bigram expansion mirrors the index transform so CJK
            # tokens match bigram-indexed documents.
            term = f'"{cjk_bigrams(token)}"'
        if term not in terms:
            terms.append(term)
        if len(terms) >= max_terms:
            break
    return " OR ".join(terms)


def build_bigram_text(title: str, content: str, tags: str) -> tuple[str, str, str]:
    """Column tuple for the bigram shadow index row."""

    return cjk_bigrams(title), cjk_bigrams(content), cjk_bigrams(tags)
