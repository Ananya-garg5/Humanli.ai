"""Settings. Every value can be overridden with an environment variable (see .env.example)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _aliases() -> dict:
    path = Path(os.getenv("ALIASES_FILE", ROOT / "config" / "aliases.json"))
    return json.loads(path.read_text()) if path.exists() else {}


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


@dataclass
class Settings:
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

    chunk_size: int = 1000
    chunk_overlap: int = 150
    k_dense: int = 15
    k_bm25: int = 15
    pool: int = 20                 # fused candidates sent to the reranker
    k_final: int = 5               # chunks given to the LLM
    rerank_min_score: float = float(os.getenv("RERANK_MIN_SCORE", "-4.0"))
    max_new_tokens: int = 450
    keep_turns: int = 4            # recent turns kept verbatim, older ones are summarised

    tables: bool = _flag("TABLES", "1")          # extract tables with pdfplumber
    verify: bool = _flag("VERIFY", "1")          # second guard: LLM confidence that passages answer the question
    verify_min: float = float(os.getenv("VERIFY_MIN", "0.001"))

    data_dir: Path = Path(os.getenv("DATA_DIR", ROOT / "data"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    aliases: dict = field(default_factory=_aliases)   # optional: phrases that identify a document in a question

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def store_dir(self) -> Path:
        return self.data_dir / "store"

    def public(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "data_dir"}
        return d
