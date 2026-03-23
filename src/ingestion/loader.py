"""PDF loader — extract text and metadata from PDFs using PyMuPDF.

Converts a PDF file into a Document model, handling:
- Text extraction (all pages)
- Metadata extraction (title, author, page count)
- Malformed/corrupt PDF graceful handling (D4 pitfall #7)
- Text preprocessing via the preprocessor module
"""

import logging
from pathlib import Path

import pymupdf

from src.ingestion.preprocessor import preprocess
from src.models import Document, DocumentMetadata

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> Document:
    """Load a single PDF and return a Document.

    Args:
        path: Path to the PDF file.

    Returns:
        A Document with extracted and preprocessed text.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the PDF produces no extractable text.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        doc = pymupdf.open(str(path))
    except Exception as e:
        raise ValueError(f"Failed to open PDF {path.name}: {e}") from e

    # Extract per-page text (needed for header/footer detection)
    page_texts = []
    for page in doc:
        page_texts.append(page.get_text())

    full_text = "\n".join(page_texts)

    if not full_text.strip():
        doc.close()
        raise ValueError(f"No extractable text in {path.name} (scanned/image-only PDF?)")

    # Preprocess
    cleaned = preprocess(full_text, page_texts=page_texts)

    metadata = DocumentMetadata(
        source=path.name,
        title=doc.metadata.get("title") or None,
        author=doc.metadata.get("author") or None,
        page_count=doc.page_count,
    )

    doc.close()

    return Document(content=cleaned, metadata=metadata)


def load_pdfs(directory: str | Path) -> list[Document]:
    """Load all PDFs from a directory, skipping failures.

    Args:
        directory: Path to directory containing PDF files.

    Returns:
        List of successfully loaded Documents.
    """
    directory = Path(directory)
    pdf_paths = sorted(directory.glob("*.pdf"))

    if not pdf_paths:
        logger.warning("No PDFs found in %s", directory)
        return []

    documents = []
    failed = []

    for path in pdf_paths:
        try:
            doc = load_pdf(path)
            documents.append(doc)
        except (ValueError, Exception) as e:
            logger.warning("Skipping %s: %s", path.name, e)
            failed.append(path.name)

    logger.info(
        "Loaded %d/%d PDFs (%d failed)",
        len(documents), len(pdf_paths), len(failed),
    )

    return documents
