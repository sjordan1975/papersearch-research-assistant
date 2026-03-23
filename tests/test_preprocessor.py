"""Tests for text preprocessing.

Each preprocessing step is deterministic — ideal for unit testing.
Tests document the specific artifacts we're cleaning and verify
we don't destroy meaningful structure.
"""

import pytest

from src.ingestion.preprocessor import (
    dehyphenate,
    normalize_whitespace,
    preprocess,
    remove_headers_footers,
    replace_ligatures,
)


# ---------------------------------------------------------------------------
# Ligatures
# ---------------------------------------------------------------------------

class TestReplaceLigatures:
    def test_fi_ligature(self):
        assert replace_ligatures("signi\ufb01cant") == "significant"

    def test_fl_ligature(self):
        assert replace_ligatures("over\ufb02ow") == "overflow"

    def test_ff_ligature(self):
        assert replace_ligatures("e\ufb00ect") == "effect"

    def test_ffi_ligature(self):
        assert replace_ligatures("e\ufb03cient") == "efficient"

    def test_ffl_ligature(self):
        assert replace_ligatures("ba\ufb04e") == "baffle"

    def test_no_ligatures_unchanged(self):
        text = "normal text without ligatures"
        assert replace_ligatures(text) == text

    def test_multiple_ligatures(self):
        text = "e\ufb03cient classi\ufb01cation"
        assert replace_ligatures(text) == "efficient classification"


# ---------------------------------------------------------------------------
# Dehyphenation
# ---------------------------------------------------------------------------

class TestDehyphenate:
    def test_rejoins_split_word(self):
        assert dehyphenate("sig-\nnificant") == "significant"

    def test_preserves_intentional_hyphen(self):
        """Hyphens NOT followed by a newline are kept."""
        assert dehyphenate("state-of-the-art") == "state-of-the-art"

    def test_preserves_hyphen_before_uppercase(self):
        """Hyphens before uppercase (e.g., proper nouns) are kept — only
        lowercase continuation signals a line-break artifact."""
        result = dehyphenate("Monte-\nCarlo")
        assert result == "Monte-\nCarlo"

    def test_preserves_hyphen_at_end_of_line_no_letter(self):
        """A hyphen followed by a newline and non-word char is kept."""
        text = "value: 5-\n10 range"
        assert dehyphenate(text) == "value: 5-\n10 range"

    def test_multiple_dehyphenations(self):
        text = "classi-\nfication and opti-\nmization"
        assert dehyphenate(text) == "classification and optimization"


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_collapses_tabs(self):
        assert normalize_whitespace("hello\tworld") == "hello world"

    def test_strips_trailing_spaces(self):
        assert normalize_whitespace("hello   \nworld") == "hello\nworld"

    def test_collapses_excessive_blank_lines(self):
        assert normalize_whitespace("a\n\n\n\n\nb") == "a\n\nb"

    def test_preserves_paragraph_break(self):
        """A single blank line (paragraph break) is preserved."""
        assert normalize_whitespace("paragraph one\n\nparagraph two") == "paragraph one\n\nparagraph two"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("\n\n  hello  \n\n") == "hello"


# ---------------------------------------------------------------------------
# Header/footer removal
# ---------------------------------------------------------------------------

class TestRemoveHeadersFooters:
    def test_removes_repeated_header(self):
        page_texts = [
            "Zhang et al.\nPage 1 content",
            "Zhang et al.\nPage 2 content",
            "Zhang et al.\nPage 3 content",
            "Zhang et al.\nPage 4 content",
        ]
        full = "\n".join(page_texts)
        result = remove_headers_footers(full, page_texts)
        assert "Zhang et al." not in result
        assert "Page 1 content" in result

    def test_removes_repeated_footer(self):
        page_texts = [
            "Content 1\n5",
            "Content 2\n5",
            "Content 3\n5",
            "Content 4\n5",
        ]
        full = "\n".join(page_texts)
        result = remove_headers_footers(full, page_texts)
        # Page number "5" appears as last line on all pages
        assert "Content 1" in result

    def test_preserves_non_repeated_lines(self):
        page_texts = [
            "Unique header 1\nContent",
            "Unique header 2\nContent",
            "Unique header 3\nContent",
        ]
        full = "\n".join(page_texts)
        result = remove_headers_footers(full, page_texts)
        assert "Unique header 1" in result
        assert "Unique header 2" in result

    def test_skips_with_few_pages(self):
        """With <3 pages, can't reliably detect headers."""
        page_texts = ["Header\nContent 1", "Header\nContent 2"]
        full = "\n".join(page_texts)
        result = remove_headers_footers(full, page_texts)
        assert result == full

    def test_ignores_long_repeated_lines(self):
        """Lines >100 chars are probably real content, not headers."""
        long_line = "A" * 101
        page_texts = [
            f"{long_line}\nContent 1",
            f"{long_line}\nContent 2",
            f"{long_line}\nContent 3",
        ]
        full = "\n".join(page_texts)
        result = remove_headers_footers(full, page_texts)
        assert long_line in result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_combines_all_steps(self):
        text = "e\ufb03cient   classi-\nfication\n\n\n\nresults"
        result = preprocess(text)
        assert result == "efficient classification\n\nresults"

    def test_with_page_texts_removes_headers(self):
        page_texts = [
            "Repeated\nContent A with signi\ufb01cant results",
            "Repeated\nContent B",
            "Repeated\nContent C",
        ]
        full = "\n".join(page_texts)
        result = preprocess(full, page_texts=page_texts)
        assert "Repeated" not in result
        assert "significant" in result

    def test_without_page_texts_skips_header_removal(self):
        text = "Repeated\nsome content"
        result = preprocess(text)
        assert "Repeated" in result

    def test_empty_input(self):
        assert preprocess("") == ""

    def test_real_world_sample(self):
        """Simulate a realistic PDF extract."""
        text = (
            "arXiv:2412.15239v2  [cs.CL]  26 Mar 2025\n"
            "Modeling Story Expectations\n"
            "creating signi\ufb01cant chal-\n"
            "lenges for traditional   analysis methods.\n\n\n\n"
            "Section 2: Methods"
        )
        result = preprocess(text)
        assert "significant challenges" in result
        assert "   " not in result
        assert "\n\n\n" not in result
