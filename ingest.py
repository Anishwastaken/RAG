"""
Document Ingestion Script.
Loads documents from the docs/ directory, splits them into chunks,
and stores the embeddings in ChromaDB.

Usage:
    python ingest.py
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from utils.data_loader import load_documents

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────
CHROMA_DIR = "db/chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "nomic-embed-text"


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into smaller chunks with overlap."""
    print(f"[Ingest] Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    print(f"[Ingest] Created {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, persist_directory=CHROMA_DIR):
    """Create and persist ChromaDB vector store from document chunks."""
    print(f"[Ingest] Creating embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"[Ingest] Vector store saved to '{persist_directory}'")
    return vectorstore


def main():
    """Run the full ingestion pipeline."""
    print("=" * 50)
    print("  Document Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Load documents
    documents = load_documents()

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Create vector store
    create_vector_store(chunks)

    print("\n✓ Ingestion complete!")


if __name__ == "__main__":
    main()
