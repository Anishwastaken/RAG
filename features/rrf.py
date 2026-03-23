"""
Reciprocal Rank Fusion (RRF) — merges multiple ranked document lists into one.
Assigns scores based on rank position across all lists, then returns top-k results.
"""


def reciprocal_rank_fusion(ranked_lists, k=60, top_n=5):
    """
    Apply RRF scoring across multiple ranked document lists.

    For each document, the RRF score = sum of 1/(k + rank) across all lists
    where it appears. Higher scores = more consistently highly ranked.

    Args:
        ranked_lists: List of lists, where each inner list is a ranked
                      sequence of LangChain Document objects
        k: RRF constant (default: 60, standard value from the original paper)
        top_n: Number of top documents to return

    Returns:
        List of top-n Document objects sorted by RRF score (descending)
    """
    rrf_scores = {}  # page_content -> cumulative score
    doc_map = {}     # page_content -> Document object (for deduplication)

    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            key = doc.page_content
            score = 1.0 / (k + rank + 1)  # rank is 0-indexed, so +1

            if key in rrf_scores:
                rrf_scores[key] += score
            else:
                rrf_scores[key] = score
                doc_map[key] = doc

    # Sort by RRF score (highest first) and return top-n
    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    top_docs = [doc_map[key] for key in sorted_keys[:top_n]]

    print(f"[RRF] Merged {len(ranked_lists)} ranked lists → {len(rrf_scores)} unique docs → top {len(top_docs)} selected")
    return top_docs
