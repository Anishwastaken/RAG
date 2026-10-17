"""
Streamlit Web App — ChatGPT-style interface for the RAG system.
Provides a clean chat UI with answer display, source attribution, and persistent history.

v2 additions:
  - File upload panel in the sidebar (PDF, DOCX, TXT, MD)
  - In-memory vectorstore built from uploaded files on the fly
  - Pipeline automatically routes to uploaded docs when present,
    falls back to default KB otherwise
  - Uploading new files replaces the previous upload (no accumulation)

Usage:
    streamlit run app.py
"""

import sys

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from core.pipeline import run_rag
from utils.prompt import get_query_rewrite_prompt
from utils.upload_processor import process_uploaded_files   # NEW

load_dotenv()

# Windows consoles default to cp1252, which can't encode the "→" in pipeline
# log lines — force UTF-8 so print() never crashes a request.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

MAX_HISTORY = 6  # keep last 6 messages (3 turns) to prevent token overflow

# ─── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Document QA System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom Styling ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Typography and base */
    .block-container {
        max-width: 860px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header */
    .app-header {
        margin-bottom: 0.25rem;
    }
    .app-header h1 {
        font-size: 1.55rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0;
        letter-spacing: -0.01em;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 0.88rem;
        margin-top: 2px;
        margin-bottom: 0;
    }

    /* Chat message spacing */
    .stChatMessage {
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        width: 280px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    .sidebar-heading {
        font-size: 0.82rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }
    .sidebar-item {
        font-size: 0.88rem;
        color: #374151;
        padding: 0.2rem 0;
        line-height: 1.6;
    }
    .sidebar-label {
        color: #6b7280;
    }
    .sidebar-desc {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-top: 0.75rem;
        line-height: 1.5;
    }

    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        .app-header h1 { color: #f3f4f6; }
        .app-subtitle { color: #9ca3af; }
        .sidebar-item { color: #d1d5db; }
        .sidebar-label { color: #9ca3af; }
        .sidebar-desc { color: #6b7280; }
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────
st.markdown('<div class="app-header"><h1>Document Question Answering System</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Query your document set using retrieval-augmented generation</p>', unsafe_allow_html=True)
st.divider()

# ─── Session State ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # display messages: {"role", "content", "sources"}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # LangChain message objects for pipeline
if "llm" not in st.session_state:
    st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# NEW — Upload-related session state keys
# uploaded_chunks:      list[Document] – chunked text from uploaded file(s)
# uploaded_vectorstore: Chroma         – in-memory store for uploaded docs
# uploaded_file_names:  list[str]      – display names shown in sidebar
# last_uploaded_key:    str            – fingerprint to detect new uploads
if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = None
if "uploaded_vectorstore" not in st.session_state:
    st.session_state.uploaded_vectorstore = None
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
if "last_uploaded_key" not in st.session_state:
    st.session_state.last_uploaded_key = None
# Bumped by "Clear Upload" to give the file_uploader a fresh key, which is the
# only way to empty it programmatically.
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def rewrite_query_if_needed(question):
    """Rewrite a follow-up question into standalone form using chat history.
    Returns the original question unchanged if rewriting fails."""
    if not st.session_state.chat_history:
        return None

    try:
        history = st.session_state.chat_history[-MAX_HISTORY:]
        history_text = ""
        for msg in history:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_text += f"{role}: {msg.content}\n"

        prompt = get_query_rewrite_prompt(history_text, question)
        result = st.session_state.llm.invoke(prompt)
        rewritten = result.content.strip()
        return rewritten if rewritten else None
    except Exception:
        return None


# ─── Display Chat History ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    filename = src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    st.markdown(f"- {filename}")

# ─── Chat Input ─────────────────────────────────────────────────────
if user_input := st.chat_input("Enter your question..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # Query rewriting for follow-ups
            search_query = rewrite_query_if_needed(user_input)

            # NEW — Route to uploaded-doc mode or default KB:
            # If the user has uploaded files, pass the pre-built vectorstore
            # and chunked documents to run_rag so it uses them exclusively.
            # If no upload is active, run_rag() receives no extra kwargs and
            # falls back to the persisted ChromaDB + docs/ directory — exactly
            # as it worked before this feature was added.
            if (
                st.session_state.uploaded_chunks is not None
                and st.session_state.uploaded_vectorstore is not None
            ):
                result = run_rag(
                    query=user_input,
                    search_query=search_query,
                    chat_history=st.session_state.chat_history[-MAX_HISTORY:],
                    documents=st.session_state.uploaded_chunks,          # NEW
                    vectorstore=st.session_state.uploaded_vectorstore,   # NEW
                )
            else:
                # Default path — unchanged from original
                result = run_rag(
                    query=user_input,
                    search_query=search_query,
                    chat_history=st.session_state.chat_history[-MAX_HISTORY:],
                )

            answer = result["answer"]
            sources = result["sources"]

        # Display answer
        st.markdown(answer)

        # Display sources
        if sources:
            with st.expander("Sources", expanded=False):
                for src in sources:
                    filename = src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    st.markdown(f"- {filename}")

    # Update session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))

# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-heading">System Info</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-item"><span class="sidebar-label">Retrieval:</span> Hybrid (BM25 + Vector)</div>'
        '<div class="sidebar-item"><span class="sidebar-label">Ranking:</span> RRF</div>'
        '<div class="sidebar-item"><span class="sidebar-label">Query Expansion:</span> Enabled</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="sidebar-desc">'
        'This system retrieves and ranks relevant document chunks before generating answers.'
        '</p>',
        unsafe_allow_html=True,
    )

    st.markdown("")  # spacer

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    # ── NEW: Document Upload Section ─────────────────────────────────────────
    st.divider()
    st.markdown('<p class="sidebar-heading">Upload Document</p>', unsafe_allow_html=True)

    st.markdown(
        '<p class="sidebar-desc">'
        'Upload a file to query it directly. Overrides the default knowledge base '
        'while active. Supports PDF, DOCX, TXT, and MD.'
        '</p>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        label="Choose file(s)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
        help="Upload PDF, DOCX, TXT, or MD files. Multiple files are merged into one searchable knowledge base.",
    )

    # ── Detect whether the uploader contents have changed ────────────────────
    # We build a simple fingerprint from filenames + sizes.
    # Comparing this to last_uploaded_key lets us re-process only when something
    # actually changed — not on every Streamlit rerun.
    if uploaded_files:
        current_key = "|".join(
            f"{f.name}:{f.size}" for f in uploaded_files
        )
    else:
        current_key = None

    if uploaded_files and current_key != st.session_state.last_uploaded_key:
        # ── New upload detected — process the files ───────────────────────────
        file_names = [f.name for f in uploaded_files]
        status_placeholder = st.empty()

        try:
            status_placeholder.info("⏳ Uploading...")

            status_placeholder.info("📄 Extracting text...")
            # (extraction happens inside process_uploaded_files)

            status_placeholder.info("🔢 Creating embeddings...")
            chunks, vectorstore = process_uploaded_files(uploaded_files)

            # Replace previous upload — clear old vectorstore from session state
            # before storing the new one so Python can GC the old in-memory Chroma.
            st.session_state.uploaded_chunks = None
            st.session_state.uploaded_vectorstore = None

            st.session_state.uploaded_chunks = chunks
            st.session_state.uploaded_vectorstore = vectorstore
            st.session_state.uploaded_file_names = file_names
            st.session_state.last_uploaded_key = current_key

            status_placeholder.success(
                f"✅ Document ready. ({len(chunks)} chunks from {len(file_names)} file(s))"
            )

        except ValueError as exc:
            # Unsupported file type or parsing error
            status_placeholder.error(f"❌ File error: {exc}")
        except Exception as exc:
            # Any other unexpected error (embedding failure, etc.)
            status_placeholder.error(f"❌ Processing failed: {exc}")

    elif not uploaded_files and st.session_state.uploaded_chunks is not None:
        # ── User cleared the uploader — fall back to default KB ─────────────
        st.session_state.uploaded_chunks = None
        st.session_state.uploaded_vectorstore = None
        st.session_state.uploaded_file_names = []
        st.session_state.last_uploaded_key = None
        st.info("ℹ️ Upload cleared. Using default knowledge base.")

    # ── Show active knowledge base indicator ─────────────────────────────────
    st.markdown("")  # spacer
    if st.session_state.uploaded_file_names:
        st.markdown('<p class="sidebar-heading">Active Knowledge Base</p>', unsafe_allow_html=True)
        for name in st.session_state.uploaded_file_names:
            st.markdown(
                f'<div class="sidebar-item">📎 {name}</div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear Upload", use_container_width=True, key="clear_upload_btn"):
            st.session_state.uploaded_chunks = None
            st.session_state.uploaded_vectorstore = None
            st.session_state.uploaded_file_names = []
            st.session_state.last_uploaded_key = None
            st.session_state.uploader_key += 1
            st.rerun()
    else:
        st.markdown(
            '<div class="sidebar-item"><span class="sidebar-label">KB:</span> Default (docs/)</div>',
            unsafe_allow_html=True,
        )
