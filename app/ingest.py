"""PDF loading and cleaning: text pages plus tables."""
from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)      # join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_reference_chunk(text: str) -> bool:
    """Bibliography chunks hurt retrieval, so they are dropped."""
    low = text.lower()
    numbered = len(re.findall(r"^\s*\[\d+\]\s", text, flags=re.M))
    years = len(re.findall(r"\b(19|20)\d{2}\b", text))
    return numbered >= 3 or low.count("arxiv preprint") >= 2 or (low.count("et al.") >= 6 and years >= 6)


def extract_tables(path: Path) -> list[Document]:
    """pypdf puts every table cell on its own line; pdfplumber keeps the rows together."""
    out: list[Document] = []
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            for page_no, page in enumerate(pdf.pages, 1):
                try:
                    for table in page.extract_tables():
                        rows = [" | ".join((c or "").replace("\n", " ").strip() for c in row)
                                for row in table if any(row)]
                        if len(rows) >= 2:
                            out.append(Document(page_content="Table:\n" + "\n".join(rows),
                                                metadata={"source": path.name, "page": page_no}))
                except Exception:
                    continue
    except Exception as exc:  # pdfplumber missing or broken PDF: text-only is still fine
        print(f"table extraction skipped for {path.name}: {exc}")
    return out


def load_pdf(path: Path, tables: bool = True) -> list[Document]:
    pages: list[Document] = []
    for doc in PyPDFLoader(str(path)).load():
        text = clean(doc.page_content)
        if len(text) < 50:                       # blank or figure-only page
            continue
        pages.append(Document(page_content=text,
                              metadata={"source": path.name, "page": int(doc.metadata.get("page", 0)) + 1}))
    if tables and pages:
        pages += extract_tables(path)
    return pages
