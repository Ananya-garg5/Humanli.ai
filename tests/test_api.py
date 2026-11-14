from fastapi.testclient import TestClient

from app.engine import Engine
from app.index import DocumentIndex
from app.main import app
from conftest import FakeLLM, FakeReranker, make_pdf

from langchain_core.embeddings import DeterministicFakeEmbedding


def client_for(cfg):
    index = DocumentIndex(DeterministicFakeEmbedding(size=64), FakeReranker(), cfg)
    app.state.engine = Engine(index, FakeLLM(), cfg)
    return TestClient(app)


def test_full_flow(cfg, tmp_path):
    pdf = tmp_path / "attention_is_all_you_need.pdf"
    make_pdf(pdf, ["The Transformer uses 8 attention heads and self attention instead of recurrence. " * 4])
    with client_for(cfg) as c:
        assert c.get("/health").json() == {"status": "ok", "documents": 0, "chunks": 0}
        assert c.post("/chat", json={"question": "hello?"}).status_code == 409   # nothing uploaded yet

        r = c.post("/upload", files=[("files", (pdf.name, pdf.read_bytes(), "application/pdf")),
                                     ("files", ("notes.txt", b"hello", "text/plain")),
                                     ("files", ("fake.pdf", b"not really a pdf", "application/pdf"))])
        statuses = {x["file"]: x["status"] for x in r.json()["results"]}
        assert statuses[pdf.name] == "indexed"
        assert statuses["notes.txt"].startswith("rejected") and statuses["fake.pdf"].startswith("rejected")
        assert [d["file"] for d in c.get("/documents").json()["documents"]] == [pdf.name]
        assert (cfg.store_dir / "faiss_index").exists()                        # persisted

        again = c.post("/upload", files=[("files", (pdf.name, pdf.read_bytes(), "application/pdf"))]).json()
        assert again["results"][0]["status"] == "already indexed"

        a = c.post("/chat", json={"question": "How many attention heads does the Transformer use?"}).json()
        assert a["sources"] and a["sources"][0]["file"] == pdf.name and a["session_id"]
        b = c.post("/chat", json={"question": "How does it work?", "session_id": a["session_id"]}).json()
        assert b["session_id"] == a["session_id"] and b["rewritten_question"] != "How does it work?"
        assert c.post("/reset", json={"session_id": a["session_id"]}).json() == {"reset": True}
        assert c.post("/reset", json={"session_id": a["session_id"]}).json() == {"reset": False}
        assert c.post("/chat", json={"question": ""}).status_code == 422
        assert c.get("/").status_code == 200
