"""Tests for the inverted-index abstract reconstruction logic."""

from openalex_med.abstract import build_text, reconstruct_abstract


def test_simple_reconstruction():
    idx = {"hello": [0], "world": [1]}
    assert reconstruct_abstract(idx) == "hello world"


def test_repeated_tokens():
    # "the cat sat on the mat"
    idx = {"the": [0, 4], "cat": [1], "sat": [2], "on": [3], "mat": [5]}
    assert reconstruct_abstract(idx) == "the cat sat on the mat"


def test_handles_gaps():
    # Position 1 is missing — should still produce a single-spaced string.
    idx = {"a": [0], "c": [2]}
    assert reconstruct_abstract(idx) == "a c"


def test_empty_inputs():
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_unicode_any_language():
    # Non-English tokens (Japanese + Spanish) round-trip fine.
    idx = {"医療": [0], "の": [1], "進歩": [2], "y": [3], "salud": [4]}
    assert reconstruct_abstract(idx) == "医療 の 進歩 y salud"


def test_build_text_modes():
    title = "Title"
    abstract = "Abstract body."

    assert build_text(title, abstract, mode="title") == "Title"
    assert build_text(title, abstract, mode="abstract") == "Abstract body."
    assert build_text(title, abstract, mode="title_abstract") == \
        "Title\n\nAbstract body."
    assert build_text(None, abstract, mode="title_abstract") == "Abstract body."
    assert build_text(title, None, mode="title_abstract") == "Title"
    assert build_text(None, None, mode="title_abstract") is None
