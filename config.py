"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Upload limits
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf"}

# Chunking defaults (see README for rationale)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# Model IDs (override via .env)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-multilingual-v3.0")

ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() in ("1", "true", "yes")
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() in ("1", "true", "yes")

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "academic_chunks")


def _require_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"Missing {name}. Copy .env.example to .env and set your API key."
        )
    return value


def get_gemini_api_key() -> str:
    return _require_key("GEMINI_API_KEY")


def get_cohere_api_key() -> str:
    return _require_key("COHERE_API_KEY")


def get_tavily_api_key() -> str | None:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    return key or None


@dataclass
class AppPaths:
    project_root: Path
    upload_dir: Path
    chroma_dir: Path


def ensure_directories() -> AppPaths:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return AppPaths(PROJECT_ROOT, UPLOAD_DIR, CHROMA_DIR)
