"""Engine: shared index + LLM + per-session conversation memory."""
from __future__ import annotations

import threading
from collections import OrderedDict

from .config import Settings
from .index import DocumentIndex
from .rag import ConversationalRAG


class Engine:
    def __init__(self, index: DocumentIndex, llm, cfg: Settings, max_sessions: int = 200):
        self.index, self.llm, self.cfg = index, llm, cfg
        self.max_sessions = max_sessions
        self.sessions: OrderedDict[str, ConversationalRAG] = OrderedDict()
        self.lock = threading.Lock()             # one generation at a time (single model, single index)

    def session(self, session_id: str) -> ConversationalRAG:
        if session_id in self.sessions:
            self.sessions.move_to_end(session_id)
        else:
            self.sessions[session_id] = ConversationalRAG(self.index, self.llm, self.cfg)
            while len(self.sessions) > self.max_sessions:
                self.sessions.popitem(last=False)
        return self.sessions[session_id]

    def reset(self, session_id: str) -> bool:
        return self.sessions.pop(session_id, None) is not None

    @classmethod
    def build(cls, cfg: Settings | None = None) -> "Engine":
        from .models import LocalLLM, load_embeddings, load_reranker

        cfg = cfg or Settings()
        cfg.pdf_dir.mkdir(parents=True, exist_ok=True)
        emb, reranker = load_embeddings(cfg), load_reranker(cfg)
        if (cfg.store_dir / "faiss_index").exists():
            index = DocumentIndex.load(cfg.store_dir, emb, reranker, cfg)
        else:
            index = DocumentIndex(emb, reranker, cfg)
        new = [p for p in sorted(cfg.pdf_dir.glob("*.pdf")) if p.name not in index.sources]
        if new:                                  # PDFs dropped into data/pdfs are indexed at startup
            for r in index.add_pdfs(new):
                print(r)
            index.save(cfg.store_dir)
        return cls(index, LocalLLM(cfg), cfg)
