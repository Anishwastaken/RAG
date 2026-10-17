"""
Shared configuration — single source of truth for models, chunking and storage.
Imported by ingest.py, core/pipeline.py, utils/upload_processor.py and the entry points.
"""

from functools import lru_cache

CHROMA_DIR = "db/chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
LLM_MODEL = "gemini-2.5-flash"

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
