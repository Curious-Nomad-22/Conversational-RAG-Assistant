from pypdf import PdfReader


def extract_pages_from_pdf(pdf_path):
    """Extract text page by page so retrieval can cite page numbers."""
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        pages.append({
            "page_number": page_number,
            "text": extracted,
        })

    return pages


def extract_text_from_pdf(pdf_path):
    """Backward-compatible helper that returns the whole document as text."""
    return "\n\n".join(
        page["text"] for page in extract_pages_from_pdf(pdf_path)
        if page["text"].strip()
    )
