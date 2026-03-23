"""
Hybrid Search — combines BM25 (keyword) + Vector (semantic) retrieval.
Uses LangChain's EnsembleRetriever to merge results with configurable weights.
"""

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
def create_hybrid_retriever(vectorstore, documents, k=5, bm25_weight=0.3, vector_weight=0.7):
    """
    Create a hybrid retriever combining vector search and BM25 keyword search.

    Args:
        vectorstore: ChromaDB vector store instance
        documents: List of Document objects (for BM25 indexing)
        k: Number of results to retrieve from each method
        bm25_weight: Weight for BM25 results (default: 0.3)
        vector_weight: Weight for vector results (default: 0.7)

    Returns:
        EnsembleRetriever combining both methods
    """
    # Semantic vector retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # Combine with weighted ensemble
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )

    return hybrid_retriever