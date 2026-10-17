"""
Shared document loading utility.
Provides functions to load documents from disk (default KB) and from
Streamlit-uploaded in-memory file bytes (PDF, DOCX, TXT, MD).
"""

import os
import io
import re
import tempfile
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


# A line that is exactly one of these headings starts the back matter of a
# Wikipedia-style article (footnotes, then citations, external links, ...).
_REFERENCE_HEADING = re.compile(r"^(Notes|References|Citations)[ \t]*$", re.MULTILINE)


def strip_reference_sections(text):
    """
    Drop everything from the first Notes/References/Citations heading onward.

    In the default docs this back matter is ~65% of the text (URLs, "Retrieved
    on ..." lines) and only adds noise chunks that compete with real content.
    """
    match = _REFERENCE_HEADING.search(text)
    return text[:match.start()].rstrip() if match else text


def load_documents(docs_path="docs", strip_references=True):
    """
    Load all .txt files from the specified directory (default knowledge base).

    Args:
        docs_path: Path to the documents directory (default: "docs")
        strip_references: Remove reference/citation sections (default: True)

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

    if strip_references:
        before = sum(len(d.page_content) for d in documents)
        for doc in documents:
            doc.page_content = strip_reference_sections(doc.page_content)
        after = sum(len(d.page_content) for d in documents)
        print(f"[Loader] Stripped reference sections: {before:,} → {after:,} characters")

    print(f"[Loader] Loaded {len(documents)} document(s) from '{docs_path}'")
    return documents


def load_uploaded_files(uploaded_files):
    """
    Parse one or more Streamlit UploadedFile objects into LangChain Documents.

    Supported types: .pdf, .docx, .txt, .md

    Strategy: write each file to a NamedTemporaryFile so that library-specific
    loaders (PyPDFLoader, Docx2txtLoader) can open it by path, then delete the
    temp file immediately after loading to avoid disk clutter.

    Args:
        uploaded_files: A single UploadedFile or list of UploadedFile objects
                        from Streamlit's st.file_uploader().

    Returns:
        List of LangChain Document objects with metadata["source"] set to the
        original filename.

    Raises:
        ValueError: If an unsupported file extension is provided.
        RuntimeError: If loading fails for any file.
    """
    # Accept a single file or a list — normalise to list
    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]

    all_documents = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        ext = os.path.splitext(filename)[1].lower()
        print(f"[Loader] Processing uploaded file: {filename} (type={ext})")

        # ── Read raw bytes from the UploadedFile ───────────────────────
        raw_bytes = uploaded_file.read()

        if ext == ".pdf":
            docs = _load_pdf(raw_bytes, filename)
        elif ext == ".docx":
            docs = _load_docx(raw_bytes, filename)
        elif ext in (".txt", ".md"):
            docs = _load_text(raw_bytes, filename)
        else:
            raise ValueError(
                f"Unsupported file type '{ext}' for '{filename}'. "
                "Supported: .pdf, .docx, .txt, .md"
            )

        print(f"[Loader] '{filename}' → {len(docs)} page/chunk(s) loaded")
        all_documents.extend(docs)

    print(f"[Loader] Total uploaded documents loaded: {len(all_documents)}")
    return all_documents


# ─── Private helpers ──────────────────────────────────────────────────────────

def _load_pdf(raw_bytes: bytes, filename: str) -> list:
    """
    Load a PDF from raw bytes using PyPDFLoader.
    PyPDFLoader requires a file path, so we write to a NamedTemporaryFile.
    Each page becomes one Document; metadata["source"] is set to filename.
    """
    from langchain_community.document_loaders import PyPDFLoader

    # suffix must be .pdf so PyPDFLoader recognises the format
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        # Override source metadata to show original filename, not temp path
        for doc in docs:
            doc.metadata["source"] = filename
    finally:
        os.unlink(tmp_path)  # always clean up, even on error

    return docs


def _load_docx(raw_bytes: bytes, filename: str) -> list:
    """
    Load a DOCX from raw bytes using Docx2txtLoader.
    Returns a single Document containing all extracted text.
    """
    from langchain_community.document_loaders import Docx2txtLoader

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        loader = Docx2txtLoader(tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
    finally:
        os.unlink(tmp_path)

    return docs


def _load_text(raw_bytes: bytes, filename: str) -> list:
    """
    Load a plain-text or Markdown file from raw bytes.
    Decodes as UTF-8 (with latin-1 fallback) and returns one Document.
    """
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")

    doc = Document(
        page_content=text,
        metadata={"source": filename},
    )
    return [doc]
