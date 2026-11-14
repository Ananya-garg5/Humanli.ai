"""FastAPI app: upload PDFs, chat with memory, inspect and reset sessions."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .engine import Engine

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "engine"):         # tests inject their own engine
        app.state.engine = Engine.build()
    yield


app = FastAPI(title="Multi-Document RAG", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class Source(BaseModel):
    file: str
    page: int


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]
    grounded: bool
    rewritten_question: str
    seconds: float


class ResetRequest(BaseModel):
    session_id: str


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    eng: Engine = app.state.engine
    return {"status": "ok", "documents": len(eng.index.sources), "chunks": len(eng.index.chunks)}


@app.get("/documents")
def documents():
    return {"documents": app.state.engine.index.documents()}


@app.post("/upload")
def upload(files: list[UploadFile] = File(...)):
    eng: Engine = app.state.engine
    limit = eng.cfg.max_upload_mb * 1024 * 1024
    results, saved = [], []
    for f in files:
        name = Path(f.filename or "").name
        if not name.lower().endswith(".pdf"):
            results.append({"file": name or "?", "status": "rejected: only PDF files are accepted"})
            continue
        if name in eng.index.sources:
            results.append({"file": name, "status": "already indexed"})
            continue
        data = f.file.read(limit + 1)
        if len(data) > limit:
            results.append({"file": name, "status": f"rejected: larger than {eng.cfg.max_upload_mb} MB"})
            continue
        if not data.startswith(b"%PDF"):
            results.append({"file": name, "status": "rejected: not a valid PDF"})
            continue
        eng.cfg.pdf_dir.mkdir(parents=True, exist_ok=True)
        dest = eng.cfg.pdf_dir / name
        dest.write_bytes(data)
        saved.append(dest)
    if saved:
        with eng.lock:
            results += eng.index.add_pdfs(saved)
            eng.index.save(eng.cfg.store_dir)
    return {"results": results, "documents": eng.index.documents()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    eng: Engine = app.state.engine
    if not eng.index.sources:
        raise HTTPException(409, "No documents indexed yet. Upload PDFs first.")
    sid = req.session_id or uuid.uuid4().hex
    with eng.lock:
        res = eng.session(sid).ask(req.question)
    return ChatResponse(session_id=sid, answer=res["answer"], sources=res["sources"],
                        grounded=res["grounded"], rewritten_question=res["standalone"], seconds=res["seconds"])


@app.post("/reset")
def reset(req: ResetRequest):
    return {"reset": app.state.engine.reset(req.session_id)}
