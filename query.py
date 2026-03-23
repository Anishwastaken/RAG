"""
CLI Query Interface — interactive conversational RAG from the terminal.
Maintains chat history, rewrites follow-up questions, and loops until exit.

Usage:
    python query.py
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from core.pipeline import run_rag
from utils.prompt import get_query_rewrite_prompt

load_dotenv()

MAX_HISTORY = 6  # keep last 6 messages (3 turns) to prevent token overflow


def rewrite_query(llm, chat_history, new_question):
    """
    Rewrite a follow-up question into a standalone query using chat history.
    Returns the original question unchanged if rewriting fails.
    """
    try:
        history_text = ""
        for msg in chat_history:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_text += f"{role}: {msg.content}\n"

        prompt = get_query_rewrite_prompt(history_text, new_question)
        result = llm.invoke(prompt)
        rewritten = result.content.strip()

        if rewritten:
            print(f"[Query] Rewritten query: {rewritten}")
            return rewritten
    except Exception as e:
        print(f"[Query] Rewriting failed ({e}), using original query")

    return new_question


def main():
    """Run an interactive query loop with conversation memory."""
    print("=" * 50)
    print("  RAG Query System")
    print("=" * 50)
    print("Ask questions about your documents. Type 'quit' to exit.\n")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    chat_history = []

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Rewrite the query if there's conversation history
        if chat_history:
            search_query = rewrite_query(llm, chat_history[-MAX_HISTORY:], question)
        else:
            search_query = None

        # Run the centralized pipeline (original query for answer, rewritten for retrieval)
        result = run_rag(
            query=question,
            search_query=search_query,
            chat_history=chat_history[-MAX_HISTORY:],
        )

        # Display the answer
        print(f"\nAssistant: {result['answer']}")

        if result["sources"]:
            print("\n📄 Sources:")
            for src in result["sources"]:
                print(f"  - {src}")

        # Update chat history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=result["answer"]))
        print()


if __name__ == "__main__":
    main()
