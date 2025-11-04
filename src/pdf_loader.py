from typing import List
import fitz  # PyMuPDF
from .utils import log


def load_pdf(pdf_bytes: bytes) -> str:
    """Extract full text from uploaded PDF bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[str] = [page.get_text("text") for page in doc]
    doc.close()
    text = "\n".join(pages)
    log.info(f"Extracted {len(pages)} pages, {len(text):,} chars")
    return text