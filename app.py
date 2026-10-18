"""
Streamlit Web App — ChatGPT-style document Q&A over one document at a time.

The visitor either uploads their own file(s) (PDF, DOCX, TXT, MD) or picks one
of the bundled sample documents. The chosen document is chunked and embedded
into an in-memory index on the spot, so the app does no work at startup.

Usage:
    streamlit run app.py
"""

import datetime
import re
import sys

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from config import LIGHT_LLM_MODEL, MAX_QUESTIONS_PER_SESSION, MAX_QUESTIONS_PER_DAY

from core.pipeline import run_rag
from utils.prompt import get_query_rewrite_prompt
from utils.upload_processor import (
    list_sample_documents,
    process_sample_document,
    process_uploaded_files,
)

load_dotenv()

# Windows consoles default to cp1252, which can't encode the "→" in pipeline
# log lines — force UTF-8 so print() never crashes a request.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

MAX_HISTORY = 6  # keep last 6 messages (3 turns) to prevent token overflow
REPO_URL = "https://github.com/Anishwastaken/RAG"

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
st.markdown('<p class="app-subtitle">Ask questions about any document — answers come only from its text, with sources</p>', unsafe_allow_html=True)
st.divider()

# ─── Session State ───────────────────────────────────────────────────
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0    # counts toward MAX_QUESTIONS_PER_SESSION
if "messages" not in st.session_state:
    st.session_state.messages = []          # display messages: {"role", "content", "sources"}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # LangChain message objects for pipeline
if "llm" not in st.session_state:
    # Only used for follow-up rewriting, so the light model is enough
    st.session_state.llm = ChatGoogleGenerativeAI(model=LIGHT_LLM_MODEL, temperature=0)

# The active document — every question is answered from this one.
# doc_chunks:       list[Document] – chunks, for BM25
# doc_vectorstore:  Chroma         – in-memory index, for vector search
# doc_names:        list[str]      – display names shown in sidebar
# doc_is_sample:    bool           – samples are shared via cache, never deleted
# last_uploaded_key: str           – fingerprint of the processed upload
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = None
    st.session_state.doc_vectorstore = None
    st.session_state.doc_names = []
    st.session_state.doc_is_sample = False
if "last_uploaded_key" not in st.session_state:
    st.session_state.last_uploaded_key = None
# Bumped to give the file_uploader a fresh key, which is the only way to empty
# it programmatically.
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


@st.cache_resource(show_spinner=False)
def load_sample(name):
    """Index a sample document once per server; every visitor shares it."""
    return process_sample_document(name)


def clear_chat():
    st.session_state.messages = []
    st.session_state.chat_history = []


def set_active_document(chunks, vectorstore, names, is_sample):
    """Switch to a new document. The old chat referred to the old one, so clear it."""
    clear_active_document()
    st.session_state.doc_chunks = chunks
    st.session_state.doc_vectorstore = vectorstore
    st.session_state.doc_names = names
    st.session_state.doc_is_sample = is_sample


def clear_active_document():
    # An upload's in-memory collection belongs to this visitor only; free it.
    # Sample collections are shared through the cache and must stay.
    if st.session_state.doc_vectorstore is not None and not st.session_state.doc_is_sample:
        try:
            st.session_state.doc_vectorstore.delete_collection()
        except Exception:
            pass
    st.session_state.doc_chunks = None
    st.session_state.doc_vectorstore = None
    st.session_state.doc_names = []
    st.session_state.doc_is_sample = False
    clear_chat()


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
        rewritten = result.text.strip()
        return rewritten if rewritten else None
    except Exception:
        return None


def quota_error_message(exc):
    """Return a friendly message if exc is a Gemini rate-limit error, else None."""
    text = str(exc)
    if "RESOURCE_EXHAUSTED" not in text and "429" not in text:
        return None
    model = re.search(r"model: ([\w.\-]+)", text)
    retry = re.search(r"retry in ((?:\d+h)?(?:\d+m)?)", text)
    msg = "Gemini API limit reached"
    if model:
        msg += f" for `{model.group(1)}`"
    msg += "."
    if retry and retry.group(1):
        msg += f" Try again in about {retry.group(1)}."
    msg += " Turning off Query Expansion in the sidebar uses fewer requests per question."
    return msg


@st.cache_resource
def daily_usage():
    """Question counter shared by every visitor of this server process."""
    return {"date": None, "count": 0}


def demo_limit_message():
    """Return why the demo is closed for this visitor, or None if they can ask."""
    usage = daily_usage()
    today = datetime.date.today()
    if usage["date"] != today:
        usage["date"], usage["count"] = today, 0

    if MAX_QUESTIONS_PER_DAY and usage["count"] >= MAX_QUESTIONS_PER_DAY:
        return "The demo has reached its daily question limit. Please come back tomorrow."
    if MAX_QUESTIONS_PER_SESSION and st.session_state.questions_asked >= MAX_QUESTIONS_PER_SESSION:
        return f"You've used all {MAX_QUESTIONS_PER_SESSION} questions for this demo session."
    return None


# ─── Sidebar: document picker ───────────────────────────────────────
# Runs before the main area so a newly uploaded file is active this run.
with st.sidebar:
    st.markdown('<p class="sidebar-heading">Your Document</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Choose file(s)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
        help="PDF, DOCX, TXT or MD. Multiple files are searched together as one document.",
    )

    # Fingerprint from filenames + sizes, so files are processed once — not on
    # every Streamlit rerun.
    current_key = "|".join(f"{f.name}:{f.size}" for f in uploaded_files) if uploaded_files else None

    if uploaded_files and current_key != st.session_state.last_uploaded_key:
        st.session_state.last_uploaded_key = current_key
        try:
            with st.spinner("Reading and indexing your document..."):
                chunks, vectorstore = process_uploaded_files(uploaded_files)
            set_active_document(chunks, vectorstore, [f.name for f in uploaded_files], is_sample=False)
        except Exception as exc:
            st.error(f"Couldn't use this file: {exc}")

    elif not uploaded_files and st.session_state.last_uploaded_key is not None:
        # The visitor removed their upload with the uploader's ✕
        st.session_state.last_uploaded_key = None
        if not st.session_state.doc_is_sample:
            clear_active_document()

    st.markdown(
        '<p class="sidebar-desc">Up to ~100 pages. Your text is sent to Google Gemini '
        'to generate answers, so please don\'t upload anything confidential.</p>',
        unsafe_allow_html=True,
    )

    if st.session_state.doc_names:
        st.markdown('<p class="sidebar-heading">Active Document</p>', unsafe_allow_html=True)
        for name in st.session_state.doc_names:
            st.markdown(f'<div class="sidebar-item">📎 {name}</div>', unsafe_allow_html=True)
        if st.button("Close Document", use_container_width=True):
            clear_active_document()
            st.session_state.last_uploaded_key = None
            st.session_state.uploader_key += 1
            st.rerun()


# ─── Main: empty state with samples, or the chat ────────────────────
has_document = st.session_state.doc_chunks is not None

if not has_document:
    st.markdown(
        "**Upload a PDF, Word or text file in the sidebar**, or try a sample "
        "document (a company's Wikipedia page):"
    )
    samples = list_sample_documents()
    for column, name in zip(st.columns(len(samples)), samples):
        if column.button(name, use_container_width=True, key=f"sample_{name}"):
            with st.spinner(f"Indexing the {name} sample..."):
                chunks, vectorstore = load_sample(name)
            set_active_document(chunks, vectorstore, [f"{name} (sample)"], is_sample=True)
            # Drop any half-picked upload so the sidebar matches the sample
            st.session_state.last_uploaded_key = None
            st.session_state.uploader_key += 1
            st.rerun()
    st.caption(
        "Behind the scenes: hybrid search (BM25 keywords + embeddings) finds the "
        "relevant passages, then Gemini answers using only those passages. "
        f"[How well the retrieval works, measured]({REPO_URL}#results)"
    )

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
limit_message = demo_limit_message()
if limit_message:
    st.info(limit_message)

if not has_document:
    placeholder = "Load a document first, then ask about it..."
else:
    placeholder = f"Ask about {', '.join(st.session_state.doc_names)}..."

if user_input := st.chat_input(placeholder, disabled=not has_document or limit_message is not None):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            # Query rewriting for follow-ups
            search_query = rewrite_query_if_needed(user_input)

            try:
                result = run_rag(
                    query=user_input,
                    search_query=search_query,
                    chat_history=st.session_state.chat_history[-MAX_HISTORY:],
                    documents=st.session_state.doc_chunks,
                    vectorstore=st.session_state.doc_vectorstore,
                    use_multi_query=st.session_state.get("use_multi_query", False),
                )
            except Exception as exc:
                friendly = quota_error_message(exc)
                if friendly is None:
                    raise
                # Drop the unanswered question so the history stays consistent
                st.session_state.messages.pop()
                st.warning(friendly)
                st.stop()

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

    # Count only answered questions toward the demo limits
    st.session_state.questions_asked += 1
    daily_usage()["count"] += 1

# ─── Sidebar: settings ──────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown('<p class="sidebar-heading">System Info</p>', unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-item"><span class="sidebar-label">Retrieval:</span> Hybrid (BM25 + Vector)</div>'
        '<div class="sidebar-item"><span class="sidebar-label">Ranking:</span> RRF</div>',
        unsafe_allow_html=True,
    )

    st.toggle(
        "Query Expansion",
        value=False,
        key="use_multi_query",
        help="Rewrites each question 3 ways before searching. Off by default: "
             "in our evaluation it did not beat plain hybrid search, and it costs "
             "one extra Gemini request per question.",
    )

    if st.button("Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()
