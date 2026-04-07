# RAG System (Built from Scratch)

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

* Multi-query generation (LLM rewrites the question into multiple variations)
* Hybrid retrieval (BM25 + embeddings)
* Reciprocal Rank Fusion for better ranking
* Basic conversational memory (rewrites follow-up queries)
* Streamlit UI for testing

---

## Project Structure (kept it modular on purpose)

```
rag-project/

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
  data_loader.py

docs/            # input files
db/              # vector store
```

---

## How to run

1. Install dependencies

```
pip install -r requirements.txt
```

2. Run Ollama (for embeddings)

```
ollama serve
```


3. Add your Gemini API key in `.env`

4. Add documents in `docs/`

5. Run ingestion

```
python ingest.py
```

6. Ask query

```
python query.py
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
* Combining rankings (RRF) actually helps a lot
* Prompting for grounded answers reduces hallucination

---

## What I’d improve next

* Add evaluation (measure accuracy vs baseline RAG)
* Try reranking models instead of RRF
* Support PDFs / larger datasets

---

## Tech stack

LangChain, ChromaDB, Ollama, Gemini, Streamlit, BM25

---

## Final note

This project was mainly to understand how RAG systems work beyond tutorials.
If you have suggestions or ideas to improve it, I’d love to try them out.
