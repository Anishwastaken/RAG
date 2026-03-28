"""
Streamlit Web App — ChatGPT-style interface for the RAG system.
Provides a clean chat UI with answer display, source attribution, and persistent history.

Usage:
    streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from core.pipeline import run_rag
from utils.prompt import get_query_rewrite_prompt

load_dotenv()

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

            # Run centralized pipeline (original query for answer, rewritten for retrieval)
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
