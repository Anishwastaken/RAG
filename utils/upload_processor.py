"""
Document Processor — turns a document into something the pipeline can search.

Used for both kinds of documents the app works with:
  - files the user uploads (PDF, DOCX, TXT, MD)
  - the bundled sample documents in docs/

Steps (same for both):
  1. Check the document isn't too large to index on a small server
  2. Split into chunks (same splitter config as ingest.py)
  3. Embed into an in-memory ChromaDB collection
  4. Return chunks (for BM25) and vectorstore (for vector search)

This module is intentionally stateless — callers (app.py) manage state via
st.session_state and st.cache_resource.
"""

import os
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_DOCUMENT_CHARS, CHARS_PER_PAGE, get_embeddings
from utils.data_loader import load_uploaded_files, strip_reference_sections

SAMPLE_DOCS_PATH = "docs"

# Unique collection name prefix; a UUID suffix gives every document its own
# collection, so a new document never mixes with a previous one's embeddings.
COLLECTION_PREFIX = "doc_"


def build_index(raw_documents):
    """
    Chunk and embed already-loaded Documents into an in-memory vectorstore.

    Returns:
        tuple: (chunks, vectorstore)

    Raises:
        ValueError: If the document is too large or contains no text.
    """
    total_chars = sum(len(doc.page_content) for doc in raw_documents)
    if total_chars > MAX_DOCUMENT_CHARS:
        raise ValueError(
            f"This document is too large for the demo (about "
            f"{total_chars // CHARS_PER_PAGE:,} pages). The limit is about "
            f"{MAX_DOCUMENT_CHARS // CHARS_PER_PAGE} pages."
        )

    print(f"[Processor] Splitting (chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(raw_documents)
    print(f"[Processor] Created {len(chunks)} chunks")

    if not chunks:
        raise ValueError(
            "No text could be extracted. Scanned PDFs (images of pages) aren't "
            "supported. Please try a file with selectable text."
        )

    collection_name = COLLECTION_PREFIX + uuid.uuid4().hex[:8]
    print(f"[Processor] Embedding into in-memory collection '{collection_name}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        # No persist_directory → purely in-memory; nothing is written to disk.
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[Processor] Ready: {len(chunks)} chunks")
    return chunks, vectorstore


def process_uploaded_files(uploaded_files):
    """Index one or more Streamlit UploadedFile objects. Returns (chunks, vectorstore)."""
    print("[Processor] Parsing uploaded file(s)...")
    return build_index(load_uploaded_files(uploaded_files))


def list_sample_documents(docs_path=SAMPLE_DOCS_PATH):
    """Names of the bundled sample documents, e.g. ["Google", "Microsoft", ...]."""
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(docs_path)
        if name.endswith(".txt")
    )


def process_sample_document(name, docs_path=SAMPLE_DOCS_PATH):
    """Index one bundled sample document by name. Returns (chunks, vectorstore)."""
    # utf-8-sig drops the byte-order mark some of the exported files start with
    with open(os.path.join(docs_path, f"{name}.txt"), encoding="utf-8-sig") as f:
        text = strip_reference_sections(f.read())
    return build_index([Document(page_content=text, metadata={"source": f"{name}.txt"})])
