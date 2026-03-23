"""
Reusable prompt templates for the RAG pipeline.
All prompts enforce grounded answers, prevent hallucination, and include source attribution.
"""


def get_qa_prompt(query, context_docs):
    """
    Build the final QA prompt with retrieved context.
    Forces the LLM to answer ONLY from the provided documents
    and list the source document names.

    Args:
        query: The user's question
        context_docs: List of LangChain Document objects

    Returns:
        Formatted prompt string
    """
    # Build context with source labels
    context_parts = []
    sources_seen = set()

    for i, doc in enumerate(context_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        sources_seen.add(source)
        context_parts.append(f"[Document {i} — {source}]\n{doc.page_content}")

    context_text = "\n\n".join(context_parts)
    sources_list = "\n".join(f"- {s}" for s in sorted(sources_seen))

    return f"""You are a precise question-answering system.

STRICT RULES:
1. Answer using ONLY the provided context below.
2. If the answer is not in the context, respond exactly: "I don't have enough information in the provided documents to answer this question."
3. Do NOT make up or infer information beyond what is explicitly stated.
4. Keep your answer concise and direct.
5. At the end of your answer, list the source documents you used.

QUESTION:
{query}

CONTEXT:
{context_text}

AVAILABLE SOURCES:
{sources_list}

Provide your answer below. End with a "Sources:" section listing only the documents you actually used."""


def get_query_rewrite_prompt(chat_history_text, new_question):
    """
    Prompt to rewrite a follow-up question into a standalone query
    using conversation history for context.

    Args:
        chat_history_text: Formatted string of previous conversation
        new_question: The user's latest question

    Returns:
        Formatted prompt string
    """
    return f"""Given the following conversation history, rewrite the new question 
so it is a standalone, self-contained search query. 
Return ONLY the rewritten question — no explanation, no extra text.

CONVERSATION HISTORY:
{chat_history_text}

NEW QUESTION: {new_question}

REWRITTEN QUESTION:"""


def get_multi_query_prompt(query):
    """
    Prompt to generate alternative versions of a query for multi-query retrieval.

    Args:
        query: The original user query

    Returns:
        Formatted prompt string
    """
    return f"""You are a helpful assistant that generates alternative search queries.

Given the following question, generate 3 alternative versions of it.
Each alternative should approach the question from a different angle or use different keywords,
while preserving the original intent.

Return ONLY the 3 alternative queries, one per line, numbered 1-3.
Do NOT include the original question. Do NOT add explanations.

ORIGINAL QUESTION: {query}

ALTERNATIVE QUERIES:"""
