"""Document index: FAISS (dense) + BM25 (sparse) -> RRF fusion -> cross-encoder rerank."""
from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .ingest import is_reference_chunk, load_pdf

STOPWORDS = set("a an the of and or to in on for with is are was were be by as at from that this it its "
                "what which who how does do did".split())

# comparison questions need chunks from more than one document
MULTI_RE = re.compile(r"\b(compare|comparison|contrast|differences?|differs?|versus|vs\.?|both|across|"
                      r"each of|all (the )?(papers|documents))\b", re.I)


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class DocumentIndex:
    def __init__(self, emb, reranker, cfg: Settings):
        self.emb, self.reranker, self.cfg = emb, reranker, cfg
        self.aliases = cfg.aliases
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""])
        self.chunks: list[Document] = []
        self.sources: list[str] = []
        self.vs = None
        self.bm25 = None

    # ---------------- ingestion ----------------
    def _rebuild_bm25(self):
        self.bm25 = BM25Retriever.from_documents(self.chunks, preprocess_func=tokenize)
        self.bm25.k = self.cfg.k_bm25

    def add_pdfs(self, paths) -> list[dict]:
        """Index PDFs. Returns one status dict per file (used by the /upload endpoint)."""
        new, results = [], []
        for path in map(Path, paths):
            if path.name in self.sources:
                results.append({"file": path.name, "status": "already indexed"})
                continue
            try:
                pages = load_pdf(path, self.cfg.tables)
            except Exception as exc:
                results.append({"file": path.name, "status": f"failed: {exc}"})
                continue
            if not pages:
                results.append({"file": path.name, "status": "no extractable text (scanned PDF? OCR is not supported)"})
                continue
            chunks = [c for c in self.splitter.split_documents(pages) if not is_reference_chunk(c.page_content)]
            title = path.stem.replace("_", " ")
            for c in chunks:
                c.page_content = f"[Document: {title}]\n{c.page_content}"
                c.metadata["chunk_id"] = len(self.chunks) + len(new)
                new.append(c)
            self.sources.append(path.name)
            results.append({"file": path.name, "status": "indexed",
                            "pages": len({p.metadata["page"] for p in pages}), "chunks": len(chunks)})
        if new:
            if self.vs is None:
                self.vs = FAISS.from_documents(new, self.emb)
            else:
                self.vs.add_documents(new)
            self.chunks += new
            self._rebuild_bm25()
        return results

    def documents(self) -> list[dict]:
        info: dict[str, dict] = {}
        for c in self.chunks:
            d = info.setdefault(c.metadata["source"], {"file": c.metadata["source"], "chunks": 0, "pages": 0})
            d["chunks"] += 1
            d["pages"] = max(d["pages"], c.metadata["page"])
        return list(info.values())

    # ---------------- retrieval ----------------
    def named_sources(self, query: str) -> list[str]:
        q = query.lower()
        return [s for s in self.sources if any(a in q for a in self.aliases.get(s, []))]

    def _hybrid(self, query: str) -> list[Document]:
        dense = self.vs.similarity_search(query, k=self.cfg.k_dense)
        sparse = self.bm25.invoke(query)
        scores: dict[int, float] = {}
        docs: dict[int, Document] = {}
        for lst in (dense, sparse):
            for rank, d in enumerate(lst):
                cid = d.metadata["chunk_id"]
                docs[cid] = d
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)   # reciprocal rank fusion
        top = sorted(scores, key=scores.get, reverse=True)[:self.cfg.pool]
        return [docs[c] for c in top]

    def _in_source(self, query: str, source: str, k: int) -> list[Document]:
        return self.vs.similarity_search(query, k=k, filter={"source": source}, fetch_k=len(self.chunks))

    def _per_source(self, query: str, named: list[str], per: int = 2):
        """Best chunks from each document so one paper cannot crowd out the others."""
        best = {}
        for s in self.sources:
            cands = self._in_source(query, s, per * 4)
            if not cands:
                continue
            ce = self.reranker.predict([(query, d.page_content) for d in cands])
            ranked = sorted(zip(cands, ce), key=lambda x: x[1], reverse=True)[:per]
            best[s] = [(d, float(x)) for d, x in ranked]
        others = sorted([s for s in best if s not in named], key=lambda s: best[s][0][1], reverse=True)
        chosen = named if len(named) >= 2 else named + others[:3 - len(named)]
        out = [h for s in chosen if s in best for h in best[s]]
        return sorted(out, key=lambda x: x[1], reverse=True)[:max(self.cfg.k_final + 3, per * len(chosen))]

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """Returns [(chunk, rerank_score)], best first."""
        if self.vs is None:
            return []
        named = self.named_sources(query)
        if MULTI_RE.search(query) or len(named) > 1:
            return self._per_source(query, named)
        if len(named) == 1:                      # question is about one specific document
            cands = self._in_source(query, named[0], self.cfg.pool)
        else:
            cands = self._hybrid(query)
        ce = self.reranker.predict([(query, d.page_content) for d in cands])
        ranked = sorted(zip(cands, ce), key=lambda x: x[1], reverse=True)[:self.cfg.k_final]
        return [(d, float(s)) for d, s in ranked]

    # ---------------- persistence ----------------
    def save(self, folder: Path) -> None:
        if self.vs is None:
            return
        folder.mkdir(parents=True, exist_ok=True)
        self.vs.save_local(str(folder / "faiss_index"))
        (folder / "chunks.json").write_text(json.dumps(
            [dict(text=c.page_content, **c.metadata) for c in self.chunks]))

    @classmethod
    def load(cls, folder: Path, emb, reranker, cfg: Settings) -> "DocumentIndex":
        idx = cls(emb, reranker, cfg)
        # the pickle was written by save() above, so it is trusted
        idx.vs = FAISS.load_local(str(folder / "faiss_index"), emb, allow_dangerous_deserialization=True)
        for r in json.loads((folder / "chunks.json").read_text()):
            text = r.pop("text")
            idx.chunks.append(Document(page_content=text, metadata=r))
        idx.sources = sorted({c.metadata["source"] for c in idx.chunks})
        idx._rebuild_bm25()
        return idx
