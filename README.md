# RAG System 

**Live demo:** https://anish-rag-demo.streamlit.app/ (upload your own PDF, Word or text file, or click a sample document to try it instantly. If it's been idle, it takes a minute or two to wake up.)

This is a Retrieval-Augmented Generation (RAG) system I built to actually understand how modern AI systems retrieve and reason over documents instead of just blindly calling APIs.

The goal wasn’t just to “make a chatbot”, but to experiment with better retrieval techniques like multi-query generation, hybrid search, and ranking strategies.

---

## What this project does

Given a set of documents, the system:

* Finds relevant chunks using both **semantic + keyword search**
* Improves recall using **multiple rewritten queries**
* Combines results using **Reciprocal Rank Fusion (RRF)**
* Generates answers using an LLM (Gemini) with **source grounding**

It also supports follow-up questions in a conversational setting.

---

## Why I built this

Most beginner RAG projects are just:

> query → vector search → answer

I wanted to go a bit deeper and explore:

* how to improve retrieval quality
* how ranking affects final answers
* how real-world systems avoid missing context

---

## Features (stuff I actually focused on)

* Hybrid retrieval (BM25 + embeddings)
* Reciprocal Rank Fusion for better ranking
* Multi-query generation (LLM rewrites the question into multiple variations) — optional, off by default (see results)
* Basic conversational memory (rewrites follow-up queries)
* Upload your own PDF / DOCX / TXT / MD files and query them
* A retrieval evaluation set to measure all of the above
* Streamlit UI for testing

---

## Results

I built a small evaluation set to check whether these techniques actually help instead of assuming they do: 42 questions over the 5 company docs, each paired with an evidence phrase copied from the source. A retrieval "hits" if a returned chunk contains that phrase.

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| Vector only | 59.5% | 73.8% | 81.0% | 0.698 |
| BM25 only | 57.1% | 83.3% | 85.7% | 0.702 |
| **Hybrid (BM25 + vector)** | **66.7%** | **85.7%** | **90.5%** | **0.769** |
| Hybrid + multi-query | 52.4% | 76.2% | 90.5% | 0.666 |

What I found:

* **Hybrid beats either method alone.** It finds the right chunk in the top 3 for 86% of questions, vs 74% for vector-only.
* **Multi-query made things worse, which I didn't expect.** The LLM rewrites drop the exact wording from the docs and sometimes add details that aren't there (e.g. "etymology", "trademark issues"), and with equal-weight fusion they outvote the original question. Weighting the original 5× only brought it back to plain hybrid, never above it. For fact-lookup questions the original wording is already the best query, so I turned multi-query off by default.
* **65% of the knowledge base was citation lists.** The docs are Wikipedia exports, and everything after the References heading is URLs and "Retrieved on…" lines. Stripping it cut the index from 1,870 to 644 chunks and raised vector-only Hit@1 from 52.4% to 59.5%.

With 42 questions each one is worth ~2.4 points, so small differences are noise; the hybrid vs vector gap is the solid one.

Reproduce with `python eval/evaluate_retrieval.py`. Per-question ranks are in [eval/results.md](eval/results.md).

---

## Project Structure (kept it modular on purpose)

```
rag-project/

config.py        # models, chunk sizes, paths (one place)
ingest.py        # loads + embeds documents
query.py         # single query
chat.py          # conversational mode
app.py           # streamlit UI

core/
  pipeline.py    # main RAG pipeline

features/
  multi_query.py
  hybrid_search.py
  rrf.py

utils/
  prompt.py
  data_loader.py       # loads docs, strips reference sections
  upload_processor.py  # chunks + embeds an uploaded or sample document

eval/
  questions.json         # 42 questions with evidence phrases
  evaluate_retrieval.py  # compares retrieval strategies
  results.md             # latest results

docs/            # input files
db/              # vector store
```

---

## How to run

1. Install dependencies

```
pip install -r requirements.txt
```

2. Add your Gemini API key in `.env`

3. Start the web app, then upload a file or pick a sample document

```
streamlit run app.py
```

The app indexes whichever document you choose on the spot, so there's no setup step.

**Command-line mode** (asks questions across all the docs in `docs/` at once):

```
python ingest.py   # build the index; re-run after changing docs/
python query.py    # single questions
python chat.py     # conversation
```

---

## Example

```
Q: what is tesla?

→ System generates multiple query variations
→ Retrieves documents using BM25 + vector search
→ Merges rankings using RRF
→ Passes top chunks to LLM

Answer:Tesla, Inc. is an American multinational automotive and clean energy company. It is headquartered in Austin, Texas, and designs, manufactures, and sells battery electric vehicles (BEVs), stationary battery energy storage devices from home to grid-scale, solar panels and solar shingles, and related products and services. Tesla was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning as Tesla Motors, and its name is a tribute to inventor and electrical engineer Nikola Tesla.
Source: Tesla.txt

Q: Who owns it ?
Answer: Retrieval query: Who owns Tesla, Inc.?
 Elon Musk owns 12.9% of Tesla. Other major shareholders include The Vanguard Group (7.2%), BlackRock (4.5%), State Street Corporation (3.4%), Geode Capital Management (1.7%), Capital Research & Management (World Investors) (1.3%), BlackRock Life (1.2%), Eaton Vance (1.0%), Norges Bank (1.0%), and Fidelity Investments (0.9%). The remaining 64.9% is owned by others.
```

---

## Things I learned

* Retrieval quality matters more than the LLM
* Simple vector search misses obvious results sometimes
* Combining keyword and vector search helps more than either alone
* Measure before believing: multi-query sounded like an upgrade and wasn't
* Data cleaning counts: removing citation noise was a free win
* Prompting for grounded answers reduces hallucination

---

## What I’d add next
* Deployment is currently being worked on and will be available soon.
* A cross-encoder reranker after fusion, measured against the current results
* Tests + CI
* Answer-quality evaluation (faithfulness), not just retrieval

---

## Tech stack

LangChain, ChromaDB, FastEmbed (BGE embeddings on ONNX Runtime), Gemini, Streamlit, BM25

---

## Final note

This project was mainly to understand how RAG systems work beyond tutorials.
If you have suggestions or ideas to improve it, I’d love to try them out.
