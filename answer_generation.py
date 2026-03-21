from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "How much did Google pay to acquire DeepMind?"

retriever = db.as_retriever(search_kwargs={"k": 3})

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


combined_input = f"""
You are a question answering system.

Rules:
- Use ONLY the provided context.
- Answer the question directly.
- Do NOT add extra commentary.
- Do NOT summarize the document.
- If the answer exists, state only the answer.
- If the answer does not exist, say:
"I don't have enough information in the provided documents."

Question:
{query}

Context:
{chr(10).join([doc.page_content for doc in relevant_docs])}

Answer:
"""

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

messages = [
    SystemMessage(content="You answer questions using only the provided documents."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)
print("\n--- Generated Response ---")
print(result.content)