from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


WORKSPACE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(WORKSPACE_DIR / ".env", override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "").strip()
QDRANT_URL = os.getenv("QDRANT_URL", "")
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "uysm")

EMBED_BATCH_SIZE = 50
QDRANT_BATCH_SIZE = 100

SMART_PARENT_BLOCK_SIZE: int = int(os.getenv("SMART_PARENT_SIZE", "3000"))
SMART_CHILD_SIZE: int = int(os.getenv("SMART_CHILD_SIZE", "500"))
SMART_CHILD_OVERLAP: int = int(os.getenv("SMART_CHILD_OVERLAP", "100"))
SMART_BOUNDARY_LLM_MODEL: Optional[str] = os.getenv("BOUNDARY_LLM_MODEL", "qwen3:1.7b")


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    return None
