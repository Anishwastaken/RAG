"""
Shared configuration — single source of truth for models, chunking and storage.
Imported by ingest.py, core/pipeline.py, utils/upload_processor.py and the entry points.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

# Load here (not only in the entry points) so .env overrides below take effect
# no matter which module imports config first.
load_dotenv()

CHROMA_DIR = "db/chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
# Writes the final answer. Override with LLM_MODEL in .env, e.g. set it to
# gemini-3.5-flash-lite during development for a much larger free-tier quota.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
# Query rewriting and multi-query generation are simple tasks; running them on
# a lighter model spreads usage across two per-model free-tier quotas.
LIGHT_LLM_MODEL = "gemini-3.5-flash-lite"

# Runs in-process via sentence-transformers: no server, no API key, no quota.
# Changing this requires re-running ingest.py (old vectors are incompatible).
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once per process (first call downloads it)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
