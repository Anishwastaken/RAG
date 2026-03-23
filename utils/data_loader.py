"""
Shared document loading utility.
Provides a single function to load documents, used by ingestion and BM25 retrieval.
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_documents(docs_path="docs"):
    """
    Load all .txt files from the specified directory.

    Args:
        docs_path: Path to the documents directory (default: "docs")

    Returns:
        List of loaded Document objects
    """
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Directory '{docs_path}' not found. Create it and add your .txt files."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in '{docs_path}'. Add your documents first."
        )

    print(f"[Loader] Loaded {len(documents)} document(s) from '{docs_path}'")
    return documents
