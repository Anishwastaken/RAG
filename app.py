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
    page_title="RAG Chat",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom Styling ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean, minimal styling */
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.85em;
        border-left: 3px solid #4A90D9;
    }
    .header-subtitle {
        color: #6b7280;
        font-size: 0.9em;
        margin-top: -10px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────
st.title("🔍 RAG Chat")
st.markdown('<p class="header-subtitle">Ask questions about your documents — powered by multi-query retrieval, hybrid search, and RRF ranking.</p>', unsafe_allow_html=True)
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
            with st.expander("📄 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# ─── Chat Input ─────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a question about your documents..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
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
            with st.expander("📄 Sources", expanded=False):
                for src in sources:
                    st.markdown(f"- `{src}`")

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
    st.header("⚙️ About")
    st.markdown("""
    **Pipeline:**
    1. Multi-query generation
    2. Hybrid retrieval (BM25 + Vector)
    3. Reciprocal Rank Fusion
    4. LLM answer (Gemini)
    """)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
