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

# Public-demo guardrails to protect the API quota (0 = unlimited, the default
# for local use). Set them as secrets / environment variables on the host.
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "0"))
MAX_QUESTIONS_PER_DAY = int(os.getenv("MAX_QUESTIONS_PER_DAY", "0"))

# Runs in-process via FastEmbed (ONNX Runtime): no server, no API key, no
# quota, and no PyTorch — which keeps cloud installs small and startup fast.
# Changing this requires re-running ingest.py (old vectors are incompatible).
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once per process (first call downloads it)."""
    from langchain_community.embeddings import FastEmbedEmbeddings

    # FastEmbed's default batch_size (256) peaked at 2.7 GB while indexing the
    # 644 default chunks, which got the app killed on Streamlit Cloud's free
    # tier. batch_size=8 peaks at ~750 MB and is also ~2x faster.
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, batch_size=8)
