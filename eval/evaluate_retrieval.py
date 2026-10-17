"""
Retrieval Evaluation — measures how often each retrieval strategy finds the
chunk that answers a question.

A question counts as answered at rank r if the r-th retrieved chunk contains the
question's evidence phrase (verbatim text from the source doc, whitespace- and
case-insensitive). Matching on text instead of chunk IDs keeps the eval valid
when chunking settings change.

Strategies compared:
  vector       — dense embeddings only
  bm25         — keyword search only
  hybrid       — BM25 + vector (EnsembleRetriever, same weights as the app)
  hybrid+mq    — hybrid + multi-query expansion + RRF (the full app pipeline)

Metrics: Hit@1, Hit@3 (what the app sends to the LLM), Hit@5, MRR@10.

LLM usage: only hybrid+mq needs Gemini, to generate query variations. They are
cached in eval/multi_queries.json, so only the first run makes API calls
(one per question). Use --skip-llm to leave that strategy out.

Usage:
    python eval/evaluate_retrieval.py
    python eval/evaluate_retrieval.py --skip-llm
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
import uuid

# Allow `python eval/evaluate_retrieval.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from config import LIGHT_LLM_MODEL, get_embeddings
from ingest import split_documents
from utils.data_loader import load_documents
from features.hybrid_search import create_hybrid_retriever
from features.multi_query import generate_multi_queries
from features.rrf import reciprocal_rank_fusion

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_PATH = os.path.join(EVAL_DIR, "questions.json")
MQ_CACHE_PATH = os.path.join(EVAL_DIR, "multi_queries.json")
RESULTS_PATH = os.path.join(EVAL_DIR, "results.md")

K = 10                   # depth retrieved per strategy (for MRR@10)
LLM_CALL_INTERVAL = 4.5  # seconds between Gemini calls (free tier: 15 req/min)


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def first_hit_rank(docs, evidence):
    """1-based rank of the first chunk containing the evidence, or None."""
    target = normalize(evidence)
    for rank, doc in enumerate(docs[:K], 1):
        if target in normalize(doc.page_content):
            return rank
    return None


def quietly(fn, *args, **kwargs):
    """Run fn while hiding the pipeline's progress prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def load_multi_query_cache(questions):
    """Return {question_id: [queries]}, generating and caching any missing ones."""
    cache = {}
    if os.path.exists(MQ_CACHE_PATH):
        with open(MQ_CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)

    missing = [q for q in questions if q["id"] not in cache]
    if not missing:
        return cache

    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI

    load_dotenv()
    llm = ChatGoogleGenerativeAI(model=LIGHT_LLM_MODEL, temperature=0)
    print(f"[Eval] Generating multi-queries for {len(missing)} question(s) "
          f"with {LIGHT_LLM_MODEL} (~{len(missing) * LLM_CALL_INTERVAL / 60:.0f} min)...")

    for i, q in enumerate(missing, 1):
        cache[q["id"]] = quietly(generate_multi_queries, llm, q["question"])
        # Save after every call so an interrupted run loses nothing
        with open(MQ_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"  {i}/{len(missing)} {q['id']}")
        if i < len(missing):
            time.sleep(LLM_CALL_INTERVAL)

    return cache


def evaluate(questions, strategies):
    """Return {strategy: {question_id: rank or None}}."""
    ranks = {name: {} for name in strategies}
    for q in questions:
        for name, retrieve in strategies.items():
            docs = quietly(retrieve, q)
            ranks[name][q["id"]] = first_hit_rank(docs, q["evidence"])
    return ranks


def summarize(rank_map):
    values = list(rank_map.values())
    n = len(values)
    hit = lambda k: sum(r is not None and r <= k for r in values) / n
    mrr = sum(1 / r for r in values if r is not None) / n
    return {"Hit@1": hit(1), "Hit@3": hit(3), "Hit@5": hit(5), "MRR@10": mrr}


def format_table(ranks):
    lines = [
        "| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@10 |",
        "|---|---|---|---|---|",
    ]
    for name, rank_map in ranks.items():
        s = summarize(rank_map)
        lines.append(
            f"| {name} | {s['Hit@1']:.1%} | {s['Hit@3']:.1%} | {s['Hit@5']:.1%} | {s['MRR@10']:.3f} |"
        )
    return "\n".join(lines)


def format_per_question(questions, ranks):
    names = list(ranks)
    lines = [
        "| ID | Question | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for q in questions:
        cells = [str(ranks[n][q["id"]] or "–") for n in names]
        lines.append(f"| {q['id']} | {q['question']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skip-llm", action="store_true",
                        help="skip the hybrid+mq strategy (no Gemini calls)")
    args = parser.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    # Build a fresh in-memory index from the current loader/chunker instead of
    # reading db/chroma_db, so results always reflect the code as it is now
    # (and the running app's locked store is never touched).
    chunks = quietly(split_documents, quietly(load_documents))
    print(f"[Eval] Embedding {len(chunks)} chunks into a temporary index...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"eval_{uuid.uuid4().hex[:8]}",
        collection_metadata={"hnsw:space": "cosine"},
    )

    vector = vectorstore.as_retriever(search_kwargs={"k": K})
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = K
    hybrid = create_hybrid_retriever(vectorstore, chunks, k=K)

    strategies = {
        "vector": lambda q: vector.invoke(q["question"]),
        "bm25": lambda q: bm25.invoke(q["question"]),
        "hybrid": lambda q: hybrid.invoke(q["question"])[:K],
    }

    if not args.skip_llm:
        mq_cache = load_multi_query_cache(questions)
        strategies["hybrid+mq"] = lambda q: reciprocal_rank_fusion(
            [hybrid.invoke(query) for query in mq_cache[q["id"]]], top_n=K
        )

    print(f"[Eval] Evaluating {len(questions)} questions x {len(strategies)} strategies...")
    ranks = evaluate(questions, strategies)

    table = format_table(ranks)
    print("\n" + table + "\n")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation Results\n\n")
        f.write(f"{len(questions)} questions over {len(chunks)} chunks. "
                f"A hit means a retrieved chunk contains the answer's evidence phrase.\n\n")
        f.write(table + "\n\n")
        f.write("## Rank of first correct chunk per question (– = not in top 10)\n\n")
        f.write(format_per_question(questions, ranks) + "\n")
    print(f"[Eval] Results written to {os.path.relpath(RESULTS_PATH)}")


if __name__ == "__main__":
    main()
