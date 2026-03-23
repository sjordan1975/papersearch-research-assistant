"""Tests for PDF loader.

Tests both happy path and error handling (D4 pitfall #7: corrupt PDFs).
Uses real PDFs from the dev subset where possible.
"""

import pytest
from pathlib import Path
from uuid import UUID

from src.ingestion.loader import load_pdf, load_pdfs
from src.models import Document

PDF_DIR = Path(__file__).parent.parent / "data" / "pdf" / "arxiv" / "pdfs"


# ---------------------------------------------------------------------------
# load_pdf — happy path
# ---------------------------------------------------------------------------

class TestLoadPdf:
    @pytest.fixture
    def sample_pdf(self):
        """Use the smallest PDF in our dev set."""
        path = PDF_DIR / "2412.15239v2.pdf"
        if not path.exists():
            pytest.skip("Dev PDFs not downloaded — run scripts/download_data.py")
        return path

    def test_returns_document(self, sample_pdf):
        doc = load_pdf(sample_pdf)
        assert isinstance(doc, Document)

    def test_has_content(self, sample_pdf):
        doc = load_pdf(sample_pdf)
        assert len(doc.content) > 1000

    def test_has_uuid(self, sample_pdf):
        doc = load_pdf(sample_pdf)
        assert isinstance(doc.id, UUID)

    def test_metadata_source(self, sample_pdf):
        doc = load_pdf(sample_pdf)
        assert doc.metadata.source == "2412.15239v2.pdf"

    def test_metadata_page_count(self, sample_pdf):
        doc = load_pdf(sample_pdf)
        assert doc.metadata.page_count > 0

    def test_ligatures_cleaned(self, sample_pdf):
        """The preprocessing pipeline should have replaced ligatures."""
        doc = load_pdf(sample_pdf)
        assert "\ufb01" not in doc.content  # ﬁ
        assert "\ufb02" not in doc.content  # ﬂ

    def test_no_excessive_whitespace(self, sample_pdf):
        """No runs of 3+ blank lines should survive preprocessing."""
        doc = load_pdf(sample_pdf)
        assert "\n\n\n" not in doc.content


# ---------------------------------------------------------------------------
# load_pdf — error handling
# ---------------------------------------------------------------------------

class TestLoadPdfErrors:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path / "nonexistent.pdf")

    def test_corrupt_pdf(self, tmp_path):
        """A file that isn't a valid PDF should raise ValueError."""
        fake = tmp_path / "corrupt.pdf"
        fake.write_text("this is not a PDF")
        with pytest.raises(ValueError, match="Failed to open"):
            load_pdf(fake)

    def test_empty_pdf(self, tmp_path):
        """A valid PDF with no text should raise ValueError."""
        import pymupdf
        doc = pymupdf.open()
        doc.new_page()  # blank page, no text
        path = tmp_path / "empty.pdf"
        doc.save(str(path))
        doc.close()
        with pytest.raises(ValueError, match="No extractable text"):
            load_pdf(path)


# ---------------------------------------------------------------------------
# load_pdfs — batch loading
# ---------------------------------------------------------------------------

class TestLoadPdfs:
    def test_loads_from_directory(self):
        if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
            pytest.skip("Dev PDFs not downloaded")
        docs = load_pdfs(PDF_DIR)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_empty_directory(self, tmp_path):
        docs = load_pdfs(tmp_path)
        assert docs == []

    def test_skips_corrupt_files(self, tmp_path):
        """Corrupt files are skipped, valid ones are loaded."""
        import pymupdf

        # Create one valid PDF
        valid = pymupdf.open()
        page = valid.new_page()
        page.insert_text((72, 72), "Hello world")
        valid.save(str(tmp_path / "valid.pdf"))
        valid.close()

        # Create one corrupt file
        (tmp_path / "corrupt.pdf").write_text("not a pdf")

        docs = load_pdfs(tmp_path)
        assert len(docs) == 1
        assert docs[0].metadata.source == "valid.pdf"
