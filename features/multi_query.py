"""
Multi-Query Retrieval — generates alternative queries to improve recall.
Uses the LLM to produce 3 variations of the user's question,
then retrieves documents for each using the hybrid retriever.
"""

import re
from utils.prompt import get_multi_query_prompt


def generate_multi_queries(llm, query):
    """
    Use the LLM to generate 3 alternative versions of the query.

    Args:
        llm: LangChain LLM instance (Gemini)
        query: Original user query

    Returns:
        List of queries (original + 3 alternatives)
    """
    prompt = get_multi_query_prompt(query)
    response = llm.invoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)

    # Parse numbered lines (e.g. "1. ...", "2. ...", "3. ...")
    lines = raw_text.strip().split("\n")
    alternatives = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if cleaned:
            alternatives.append(cleaned)

    # Always include the original query + up to 3 alternatives
    all_queries = [query] + alternatives[:0]
    print(f"[Multi-Query] Generated {len(all_queries)} queries (1 original + {len(alternatives[:3])} alternatives)")
    return all_queries


def retrieve_for_multiple_queries(queries, hybrid_retriever):
    """
    Run hybrid retrieval for each query and return all ranked lists.

    Args:
        queries: List of query strings
        hybrid_retriever: The hybrid (BM25 + vector) retriever instance

    Returns:
        List of ranked document lists (one list per query)
    """
    all_ranked_lists = []

    for i, q in enumerate(queries):
        docs = hybrid_retriever.invoke(q)
        all_ranked_lists.append(docs)
        print(f"[Multi-Query] Query {i + 1}: retrieved {len(docs)} documents")

    return all_ranked_lists
