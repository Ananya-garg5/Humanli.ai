import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from app.config import Settings  # noqa: E402
from app.index import DocumentIndex  # noqa: E402
from make_sample_report import make  # noqa: E402


class FakeReranker:
    """Word-overlap score, enough to test the plumbing without downloading a model."""

    def predict(self, pairs):
        return [float(len(set(re.findall(r"\w+", q.lower())) & set(re.findall(r"\w+", d.lower())))) - 4
                for q, d in pairs]


class FakeLLM:
    def __init__(self, yes=0.9):
        self.yes, self.calls = yes, []

    def chat(self, messages, max_new_tokens=None):
        p = messages[-1]["content"]
        self.calls.append(p[:14])
        if p.startswith("Rewrite"):
            return "How many parameters does LoRA train?"
        if p.startswith("Current summary"):
            return "Short summary."
        return "The answer is in the passage [1, 2]."

    def yes_prob(self, messages):
        return 0.0 if "2019" in messages[-1]["content"].split("Question:")[-1] else self.yes


def make_pdf(path: Path, pages):
    import textwrap
    c = canvas.Canvas(str(path), pagesize=A4)
    for text in pages:
        y = 800
        for line in textwrap.wrap(text, 90):
            c.drawString(40, y, line)
            y -= 14
        c.showPage()
    c.save()


@pytest.fixture()
def cfg(tmp_path):
    return Settings(data_dir=tmp_path, aliases={
        "rag_lewis_2020.pdf": ["original rag"],
        "rag_survey_gao_2023.pdf": ["rag survey"],
        "acme_annual_report.pdf": ["acme"]})


@pytest.fixture()
def pdfs(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    make_pdf(d / "rag_lewis_2020.pdf", ["The original method retrieves passages with a dense retriever and conditions a generator on them. " * 4])
    make_pdf(d / "rag_survey_gao_2023.pdf", ["This survey describes the retrieval step as embedding the query and fetching top K chunks. " * 4])
    make_pdf(d / "attention_is_all_you_need.pdf", ["The Transformer uses 8 attention heads and self attention instead of recurrence. " * 4])
    make(d / "acme_annual_report.pdf")
    return sorted(d.glob("*.pdf"))


@pytest.fixture()
def index(cfg, pdfs):
    idx = DocumentIndex(DeterministicFakeEmbedding(size=64), FakeReranker(), cfg)
    idx.add_pdfs(pdfs)
    return idx
