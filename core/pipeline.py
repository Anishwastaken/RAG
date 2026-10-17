"""
Core RAG Pipeline — centralized logic for the entire retrieval-augmented generation system.

Pipeline flow:
  User Query → Multi-Query Generation → Hybrid Retrieval per query → RRF Ranking → LLM → Answer + Sources

All RAG logic lives here. Entry points (query.py, chat.py, app.py) call run_rag().

v2 additions:
  run_rag() now accepts optional `documents` and `vectorstore` kwargs.
  When supplied (e.g. from a Streamlit file upload), those are used instead
  of loading from disk, enabling zero-restart document switching.
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import CHROMA_DIR, LLM_MODEL, LIGHT_LLM_MODEL, get_embeddings
from ingest import split_documents
from utils.data_loader import load_documents
from utils.prompt import get_qa_prompt
from features.multi_query import generate_multi_queries, retrieve_for_multiple_queries
from features.hybrid_search import create_hybrid_retriever
from features.rrf import reciprocal_rank_fusion

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────
TOP_K_RETRIEVAL = 3   # documents per query per retriever
TOP_N_FINAL = 3      # documents after RRF


def _get_llm(model=LLM_MODEL):
    """Initialize a Gemini LLM."""
    return ChatGoogleGenerativeAI(model=model, temperature=0)


def _get_vectorstore():
    """Load the existing ChromaDB vector store."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def run_rag(query, search_query=None, chat_history=None, documents=None, vectorstore=None,
            use_multi_query=False):
    """
    Run the full RAG pipeline.

    Pipeline:
      1. Multi-query generation (original + 3 alternatives)
      2. Hybrid retrieval (BM25 + vector) for each query
      3. RRF ranking across all result lists
      4. LLM answer generation with source attribution

    Args:
        query: The user's original question (used for answer generation)
        search_query: Optional rewritten query for retrieval (defaults to query)
        chat_history: Optional list of LangChain message objects for context
        documents: Optional list of pre-loaded LangChain Document objects.
                   When provided, these are used for BM25 indexing instead of
                   loading from disk. Pass chunked docs from an uploaded file.
        vectorstore: Optional pre-built Chroma vectorstore.
                     When provided, this is used for semantic retrieval instead
                     of loading the default persisted store.
        use_multi_query: Expand the query into 3 LLM-written variations.
                         Off by default: on eval/questions.json it lowered
                         Hit@1 versus plain hybrid, and costs one LLM call.

    Returns:
        dict with keys:
          - "answer": The LLM's response string
          - "sources": List of source document paths
          - "context_docs": List of top-k Document objects used
    """
    retrieval_query = search_query or query

    print(f"\n{'='*60}")
    print(f"[Pipeline] Original query: {query}")
    if search_query:
        print(f"[Pipeline] Retrieval query: {retrieval_query}")
    print(f"{'='*60}")

    llm = _get_llm()

    # Use the caller-supplied vectorstore (uploaded doc) or fall back to the
    # persisted default ChromaDB store.
    active_vectorstore = vectorstore if vectorstore is not None else _get_vectorstore()

    # Use the caller-supplied documents (uploaded doc chunks, for BM25) or fall
    # back to loading the default knowledge base from disk.
    if documents is not None:
        active_documents = documents
        print(f"[Pipeline] Using {len(active_documents)} caller-supplied documents (uploaded file mode)")
    else:
        # Chunk with the same settings as ingest.py so BM25 and the vector
        # store return identical chunks (whole files would bloat the context
        # and defeat RRF deduplication).
        active_documents = split_documents(load_documents())

    # ── Step 1: Multi-query generation ──────────────────────────────
    if use_multi_query:
        all_queries = generate_multi_queries(_get_llm(LIGHT_LLM_MODEL), retrieval_query)
    else:
        all_queries = [retrieval_query]
        print("[Multi-Query] Disabled — using the single query only")

    # ── Step 2: Hybrid retrieval for each query ─────────────────────
    hybrid_retriever = create_hybrid_retriever(
        active_vectorstore, active_documents, k=TOP_K_RETRIEVAL
    )
    ranked_lists = retrieve_for_multiple_queries(all_queries, hybrid_retriever)

    # ── Step 3: RRF ranking across all lists ────────────────────────
    top_docs = reciprocal_rank_fusion(ranked_lists, top_n=TOP_N_FINAL)

    if not top_docs:
        return {
            "answer": "I couldn't find any relevant documents to answer your question.",
            "sources": [],
            "context_docs": [],
        }

    # ── Step 4: LLM answer generation ──────────────────────────────
    qa_prompt = get_qa_prompt(query, top_docs)

    messages = [
        SystemMessage(content=(
            "You are a conversational assistant. "
            "Use chat history to resolve references (e.g., 'he', 'it'). "
            "Answer strictly using the provided documents."
        )),
    ]

    # Include chat history for conversational context (if available)
    if chat_history:
        messages.extend(chat_history)

    messages.append(HumanMessage(content=qa_prompt))

    print(f"[Pipeline] Sending {len(top_docs)} documents to LLM...")
    result = llm.invoke(messages)
    answer = result.text

    # Extract unique source paths
    sources = sorted(set(
        doc.metadata.get("source", "Unknown") for doc in top_docs
    ))

    print(f"[Pipeline] Answer generated. Sources: {sources}")
    return {
        "answer": answer,
        "sources": sources,
        "context_docs": top_docs,
    }
