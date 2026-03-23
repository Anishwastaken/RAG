# 🔍 RAG System — Production-Style Retrieval-Augmented Generation

A modular, production-grade RAG (Retrieval-Augmented Generation) system built with LangChain, ChromaDB, Ollama embeddings, and Google Gemini. Features advanced retrieval techniques including multi-query generation, hybrid search, and Reciprocal Rank Fusion (RRF).

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-Query Retrieval** | Generates 3+ alternative queries via LLM to improve recall |
| **Hybrid Search** | Combines BM25 keyword search + vector semantic search |
| **Reciprocal Rank Fusion** | Merges ranked lists across all queries for optimal ranking |
| **Conversational RAG** | History-aware query rewriting for follow-up questions |
| **Grounded Answers** | Strict prompting to prevent hallucination with source attribution |
| **Streamlit Web UI** | ChatGPT-style chat interface with source display |
| **Modular Architecture** | Clean separation of concerns — easy to extend and maintain |

---

## 🏗️ Architecture

```
User Query
  → Query Rewriting (if chat history exists)
  → Multi-Query Generation (original + 3 alternatives)
  → Hybrid Retrieval per query (BM25 + Vector via EnsembleRetriever)
  → Reciprocal Rank Fusion across all ranked lists
  → Top-k Document Selection
  → LLM Generation (Google Gemini)
  → Answer + Sources
```

### Project Structure

```
rag-project/
│
├── ingest.py               # Document ingestion pipeline
├── query.py                # Single-query entry point
├── chat.py                 # Conversational chat entry point
├── app.py                  # Streamlit web app
│
├── core/
│   └── pipeline.py         # Centralized RAG pipeline (run_rag)
│
├── features/
│   ├── multi_query.py      # Multi-query generation
│   ├── hybrid_search.py    # BM25 + vector hybrid retrieval
│   └── rrf.py              # Reciprocal Rank Fusion
│
├── utils/
│   ├── prompt.py           # Reusable prompt templates
│   └── data_loader.py      # Shared document loading
│
├── docs/                   # Your source documents (.txt files)
├── db/                     # ChromaDB storage (auto-generated)
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running locally ([install guide](https://ollama.com))
- **Google Gemini API key** ([get one here](https://aistudio.google.com/apikey))

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd rag-project

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Add your documents

Place `.txt` files in the `docs/` directory. The system will index all text files it finds.

### 5. Run ingestion

```bash
python ingest.py
```

This loads your documents, splits them into chunks, and stores embeddings in ChromaDB.

---

## 💻 Usage

### Option 1: Single Query (Terminal)

```bash
python query.py
```

Enter a question and get an answer with sources.

### Option 2: Chat Mode (Terminal)

```bash
python chat.py
```

Interactive multi-turn conversation with history-aware follow-ups.

### Option 3: Web App (Streamlit)

```bash
streamlit run app.py
```

Opens a ChatGPT-style web interface at `http://localhost:8501`.

---

## 📖 Example

```
You: How much did Microsoft pay to acquire GitHub?

[Pipeline] Processing query: How much did Microsoft pay to acquire GitHub?
[Multi-Query] Generated 4 queries (1 original + 3 alternatives)
[Multi-Query] Query 1: retrieved 5 documents
[Multi-Query] Query 2: retrieved 5 documents
[Multi-Query] Query 3: retrieved 5 documents
[Multi-Query] Query 4: retrieved 5 documents
[RRF] Merged 4 ranked lists → 12 unique docs → top 5 selected
[Pipeline] Answer generated. Sources: ['docs\Microsoft.txt']

Answer: Microsoft paid $7.5 billion to acquire GitHub.
Sources:
  - docs\Microsoft.txt
```

---

## 🛠️ Tech Stack

- **LangChain** — orchestration framework
- **ChromaDB** — vector database
- **Ollama** — local embeddings (nomic-embed-text)
- **Google Gemini** — LLM for generation
- **Streamlit** — web interface
- **rank-bm25** — keyword search

---

## 📝 License

MIT
