"""
Core RAG Pipeline — centralized logic for the entire retrieval-augmented generation system.

Pipeline flow:
  User Query → Multi-Query Generation → Hybrid Retrieval per query → RRF Ranking → LLM → Answer + Sources

All RAG logic lives here. Entry points (query.py, chat.py, app.py) call run_rag().
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from utils.data_loader import load_documents
from utils.prompt import get_qa_prompt
from features.multi_query import generate_multi_queries, retrieve_for_multiple_queries
from features.hybrid_search import create_hybrid_retriever
from features.rrf import reciprocal_rank_fusion

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────
CHROMA_DIR = "db/chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemini-2.5-flash"
TOP_K_RETRIEVAL = 3   # documents per query per retriever
TOP_N_FINAL = 3      # documents after RRF


def _get_llm():
    """Initialize the Gemini LLM."""
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


def _get_vectorstore():
    """Load the existing ChromaDB vector store."""
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


def run_rag(query, search_query=None, chat_history=None):
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
    vectorstore = _get_vectorstore()

    # Load raw documents (needed for BM25 indexing)
    documents = load_documents()

    # ── Step 1: Multi-query generation ──────────────────────────────
    all_queries = generate_multi_queries(llm, retrieval_query)

    # ── Step 2: Hybrid retrieval for each query ─────────────────────
    hybrid_retriever = create_hybrid_retriever(
        vectorstore, documents, k=TOP_K_RETRIEVAL
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
    answer = result.content

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
