"""Text preprocessing — clean PDF extraction artifacts before chunking.

PDF text extraction is messy. PyMuPDF does a good job, but the raw output
still has artifacts that would degrade chunking and retrieval quality:

- Ligatures: "ﬁ" instead of "fi", "ﬂ" instead of "fl"
- Hyphenated line breaks: "sig-\nnificant" should become "significant"
- Excessive whitespace: double spaces, trailing spaces, runs of blank lines
- Header/footer noise: repeated lines across pages (e.g., "Author et al.")

The preprocessor works on a full document's text (all pages concatenated).
It does NOT remove meaningful structure — paragraph breaks are preserved.
"""

import re

# Ligature replacements (most common in academic PDFs)
_LIGATURES = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
}


def replace_ligatures(text: str) -> str:
    """Replace Unicode ligature characters with their ASCII equivalents."""
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    return text


def dehyphenate(text: str) -> str:
    """Rejoin words split across lines by a hyphen.

    Matches: "word-\\n" followed by a lowercase letter → join them.
    Preserves intentional hyphens (e.g., "state-of-the-art") because those
    aren't followed by a newline.
    """
    return re.sub(r"([a-zA-Z])-\n([a-z])", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Clean up whitespace without destroying paragraph structure.

    - Collapse runs of spaces/tabs to a single space (per line)
    - Strip trailing whitespace per line
    - Collapse 3+ consecutive blank lines to 2 (preserving paragraph breaks)
    """
    # Collapse horizontal whitespace within lines
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip trailing spaces per line
    text = re.sub(r" +\n", "\n", text)
    # Collapse excessive blank lines (keep at most one blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_headers_footers(text: str, page_texts: list[str]) -> str:
    """Remove repeated header/footer lines that appear on most pages.

    Strategy: if a line appears on >50% of pages as the first or last line,
    it's likely a header or footer. Remove all occurrences.
    """
    if len(page_texts) < 3:
        return text

    threshold = len(page_texts) * 0.5

    # Count first/last lines across pages
    first_lines: dict[str, int] = {}
    last_lines: dict[str, int] = {}
    for page in page_texts:
        lines = [l.strip() for l in page.strip().split("\n") if l.strip()]
        if lines:
            first_lines[lines[0]] = first_lines.get(lines[0], 0) + 1
            last_lines[lines[-1]] = last_lines.get(lines[-1], 0) + 1

    noise = set()
    for line, count in first_lines.items():
        if count >= threshold and len(line) < 100:
            noise.add(line)
    for line, count in last_lines.items():
        if count >= threshold and len(line) < 100:
            noise.add(line)

    if not noise:
        return text

    cleaned_lines = []
    for line in text.split("\n"):
        if line.strip() not in noise:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def preprocess(text: str, page_texts: list[str] | None = None) -> str:
    """Run the full preprocessing pipeline on extracted text.

    Args:
        text: Full document text (all pages concatenated).
        page_texts: Optional list of per-page text, used for header/footer
            detection. If not provided, header/footer removal is skipped.

    Returns:
        Cleaned text ready for chunking.
    """
    text = replace_ligatures(text)
    text = dehyphenate(text)
    if page_texts:
        text = remove_headers_footers(text, page_texts)
    text = normalize_whitespace(text)
    return text
