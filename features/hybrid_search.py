from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever   

def create_hybrid_retriever(vectorstore, documents, k=3):
    """
    Creates a hybrid retriever combining:
    - Vector search (semantic)
    - BM25 (keyword)
    """

    # Vector retriever (you already use this)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # BM25 retriever (keyword search)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # Combine both
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    return hybrid_retriever