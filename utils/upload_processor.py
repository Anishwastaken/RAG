"""
Upload Processor — handles the full pipeline for user-uploaded documents.

Responsibilities:
  1. Parse uploaded files (delegates to data_loader.load_uploaded_files)
  2. Split into chunks (same splitter config as ingest.py)
  3. Build an in-memory ChromaDB collection scoped to the upload session
  4. Return chunks and vectorstore so the pipeline can use them directly

This module is intentionally stateless — callers (app.py) manage state via
st.session_state. All configuration constants mirror ingest.py for consistency.
"""

import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import CHUNK_SIZE, CHUNK_OVERLAP, get_embeddings
from utils.data_loader import load_uploaded_files

# Unique collection name prefix; a UUID suffix prevents collection collisions
# when a new file replaces the previous one within the same Chroma process.
UPLOAD_COLLECTION_PREFIX = "uploaded_docs_"


def process_uploaded_files(uploaded_files):
    """
    Full pipeline: raw UploadedFile(s) → chunked Documents + in-memory Chroma.

    This function is called from app.py whenever the user uploads files.
    It returns everything the pipeline needs to answer queries from the upload.

    Args:
        uploaded_files: A single or list of Streamlit UploadedFile objects.

    Returns:
        tuple: (chunks, vectorstore)
            chunks      — list of LangChain Document objects (split text chunks)
            vectorstore — in-memory Chroma collection populated with embeddings

    Raises:
        ValueError: Propagated from load_uploaded_files on bad file type.
        RuntimeError: Wraps any unexpected loading/embedding error.
    """
    # ── Step 1: Parse raw bytes into LangChain Documents ─────────────────────
    print("[UploadProcessor] Parsing uploaded file(s)...")
    raw_documents = load_uploaded_files(uploaded_files)
    print(f"[UploadProcessor] Parsed {len(raw_documents)} raw document segment(s)")

    # ── Step 2: Split into chunks (same config as ingest.py) ─────────────────
    print(f"[UploadProcessor] Splitting (chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(raw_documents)
    print(f"[UploadProcessor] Created {len(chunks)} chunks")

    if not chunks:
        raise RuntimeError(
            "The uploaded file(s) appear to be empty or could not be parsed. "
            "Please check the file content and try again."
        )

    # ── Step 3: Build an in-memory Chroma vectorstore ────────────────────────
    # We use a unique collection name each time so that replacing an uploaded
    # document always starts with a fresh, clean collection — no stale embeddings
    # from the previous upload bleed through.
    collection_name = UPLOAD_COLLECTION_PREFIX + uuid.uuid4().hex[:8]
    print(f"[UploadProcessor] Building in-memory vectorstore (collection='{collection_name}')...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        # No persist_directory → purely in-memory; disappears when process exits.
        # This means uploaded docs NEVER pollute the permanent ChromaDB store.
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"[UploadProcessor] In-memory vectorstore ready with {len(chunks)} chunks")
    return chunks, vectorstore
