from pathlib import Path

from app.index import MULTI_RE, DocumentIndex, tokenize
from app.ingest import clean, is_reference_chunk
from app.rag import NOT_FOUND_MSG, ConversationalRAG, strip_cites
from conftest import FakeLLM, FakeReranker

from langchain_core.embeddings import DeterministicFakeEmbedding


# ---------- ingestion helpers ----------
def test_clean_joins_hyphenated_words():
    assert clean("retrie-\nval  works") == "retrieval works"


def test_reference_chunks_are_detected():
    refs = "[1] A. B. arXiv preprint 2020\n[2] C. D. arXiv preprint 2019\n[3] E. F. 2018"
    assert is_reference_chunk(refs)
    assert not is_reference_chunk("The model uses eight attention heads in the base setting.")


def test_multi_regex_ignores_the_word_different():
    assert not MULTI_RE.search("Which one generates each token from a different document?")
    assert MULTI_RE.search("Compare A vs. B") and MULTI_RE.search("What are the differences between them")


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The RAG-Token model!") == ["rag", "token", "model"]


def test_strip_cites():
    assert strip_cites("It uses 8 heads [1, 2]. See also [3].") == "It uses 8 heads. See also."


# ---------- index ----------
def test_add_pdfs_reports_status_and_skips_duplicates(index, pdfs):
    assert set(index.sources) == {p.name for p in pdfs}
    again = index.add_pdfs(pdfs[:1])
    assert again[0]["status"] == "already indexed"


def test_tables_become_rows(index):
    tables = [c.page_content for c in index.chunks if "Table:" in c.page_content]
    assert any("Net profit | 212 | 268 | 331" in t for t in tables)


def test_chunk_ids_are_unique_after_incremental_add(index, tmp_path):
    from conftest import make_pdf
    extra = tmp_path / "extra.pdf"
    make_pdf(extra, ["Quarterly revenue grew by twelve percent driven by cloud services. " * 5])
    index.add_pdfs([extra])
    ids = [c.metadata["chunk_id"] for c in index.chunks]
    assert len(ids) == len(set(ids)) and "extra.pdf" in index.sources


def test_named_document_restricts_retrieval(index):
    srcs = {d.metadata["source"] for d, _ in index.retrieve("What are the formulations in the original RAG paper?")}
    assert srcs == {"rag_lewis_2020.pdf"}


def test_comparison_covers_both_named_documents(index):
    q = "Compare how the RAG survey and the original RAG paper describe the retrieval step."
    assert {d.metadata["source"] for d, _ in index.retrieve(q)} == {"rag_lewis_2020.pdf", "rag_survey_gao_2023.pdf"}


def test_save_and_load_roundtrip(index, cfg, tmp_path):
    index.save(tmp_path / "store")
    loaded = DocumentIndex.load(tmp_path / "store", DeterministicFakeEmbedding(size=64), FakeReranker(), cfg)
    assert len(loaded.chunks) == len(index.chunks)
    assert loaded.named_sources("tell me about acme") == ["acme_annual_report.pdf"]
    assert loaded.retrieve("net profit 2023")


# ---------- conversational layer ----------
def test_answer_has_sources_and_no_citations_in_memory(index, cfg):
    rag = ConversationalRAG(index, FakeLLM(), cfg)
    res = rag.ask("What was Acme's net profit in 2023?")
    assert res["grounded"] and res["sources"][0]["file"] == "acme_annual_report.pdf"
    assert "[" not in rag.history[-1][1]


def test_verifier_blocks_unanswerable_question(index, cfg):
    rag = ConversationalRAG(index, FakeLLM(), cfg)
    res = rag.ask("What was Acme's net profit in 2019?")
    assert res["answer"] == NOT_FOUND_MSG and res["sources"] == [] and not res["grounded"]


def test_no_hits_means_no_llm_call(index, cfg):
    llm = FakeLLM()
    cfg.rerank_min_score = 99
    res = ConversationalRAG(index, llm, cfg).ask("completely unrelated zzzz")
    assert res["answer"] == NOT_FOUND_MSG and llm.calls == []


def test_standalone_followup_skips_rewrite_but_pronoun_followup_is_rewritten(index, cfg):
    llm = FakeLLM()
    rag = ConversationalRAG(index, llm, cfg)
    rag.ask("What is attention in the Transformer?")
    r = rag.ask("And how does LoRA reduce the number of trainable parameters?")
    assert r["standalone"] == r["question"]
    r = rag.ask("How does it work?")
    assert r["standalone"] == "How many parameters does LoRA train?"


def test_old_turns_are_summarised(index, cfg):
    cfg.keep_turns = 2
    rag = ConversationalRAG(index, FakeLLM(), cfg)
    for i in range(4):
        rag.ask(f"What is the Transformer attention question number {i}?")
    assert len(rag.history) == 2 and rag.summary == "Short summary."


def test_confidence_check_is_skipped_for_comparisons(index, cfg):
    q = "Compare how the RAG survey and the original RAG paper describe the retrieval step."
    res = ConversationalRAG(index, FakeLLM(yes=0.0), cfg).ask(q)          # verifier says NO
    assert res["answer"] != NOT_FOUND_MSG and res["grounded"]
    res = ConversationalRAG(index, FakeLLM(yes=0.0), cfg).ask("What was Acme's net profit in 2023?")
    assert res["answer"] == NOT_FOUND_MSG                                   # but still blocks normal questions
