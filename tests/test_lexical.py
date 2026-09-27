"""Tests for CJK-bigram lexical transforms and OR query building."""

from __future__ import annotations

from synapse.storage.lexical import build_bigram_text, build_or_query, cjk_bigrams, has_cjk


def test_cjk_bigrams_splits_runs_into_overlapping_pairs() -> None:
    assert cjk_bigrams("会话摘要") == "会话 话摘 摘要"
    # Single characters stay as-is.
    assert cjk_bigrams("库") == "库"
    # Non-CJK text is untouched.
    assert cjk_bigrams("vector cache") == "vector cache"
    # Mixed: only CJK runs expand.
    assert cjk_bigrams("记忆库 memory") == "记忆 忆库 memory"


def test_has_cjk_detects_cjk_only() -> None:
    assert has_cjk("记忆")
    assert not has_cjk("plain english")
    assert not has_cjk("nodes_vec")


def test_build_or_query_drops_stopwords_and_short_tokens() -> None:
    expr = build_or_query("the cache 的 a 记忆")
    terms = expr.split(" OR ")
    assert '"cache"' in terms
    assert '"记忆"' in terms  # single CJK char run: no bigram possible, kept as-is
    assert '"the"' not in expr and '"的"' not in expr and '"a"' not in expr


def test_build_or_query_expands_cjk_runs_to_bigrams() -> None:
    expr = build_or_query("会话摘要 重复")
    assert '"会话 话摘 摘要"' in expr
    assert '"重复"' in expr


def test_build_or_query_extracts_cjk_runs_from_sentences() -> None:
    # Chinese sentences have no spaces: runs must be extracted as separate terms.
    expr = build_or_query("记忆库已经基本被重复的会话摘要占满")
    # The run expands to its full bigram sequence (a single token).
    assert '"记忆 忆库 库已 已经 经基 基本 本被 被重 重复 复的 的会 会话 话摘 摘要 要占 占满"' in expr
    # The whole sentence is never quoted as one blob.
    assert '"记忆库已经基本被重复的会话摘要占满"' not in expr


def test_build_or_query_dedupes_and_caps_terms() -> None:
    expr = build_or_query("cache cache cache " + " ".join(f"t{i}term" for i in range(40)))
    assert expr.count('"cache"') == 1
    assert expr.count(" OR ") < 24


def test_build_or_query_returns_empty_for_noise() -> None:
    assert build_or_query("!!! ??? ...") == ""
    assert build_or_query("") == ""


def test_build_bigram_text_transforms_all_columns() -> None:
    title, content, tags = build_bigram_text("记忆库", "cache layer", "[]")
    assert title == "记忆 忆库"
    assert content == "cache layer"
    assert tags == "[]"


def test_code_identifiers_survive_or_building() -> None:
    expr = build_or_query("nodes_vec sanitizer sqlite.py")
    terms = expr.split(" OR ")
    assert '"nodes_vec"' in terms
    assert '"sqlite.py"' in terms
