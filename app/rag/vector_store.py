import re
import uuid

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vector_store(chunks, collection_name=None):
    """Create a Chroma collection for the supplied document chunks."""
    if not chunks:
        raise ValueError("Cannot create a vector store without text chunks.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    texts = [chunk["content"] for chunk in chunks]
    metadatas = []

    for chunk in chunks:
        metadata = {
            "chunk_id": int(chunk["chunk_id"]),
            "source": str(chunk.get("source") or "Unknown document"),
        }
        if chunk.get("page_number") is not None:
            metadata["page_number"] = int(chunk["page_number"])
        metadatas.append(metadata)

    if collection_name is None:
        collection_name = f"rag_{uuid.uuid4().hex[:20]}"
    collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name)[:60]
    if len(collection_name) < 3:
        collection_name = f"rag_{uuid.uuid4().hex[:20]}"

    return Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory="app/chroma_db",
    )


def delete_vector_store_collection(vector_store):
    """Delete a previous app-owned Chroma collection after a replacement is ready."""
    collection = getattr(vector_store, "_collection", None)
    collection_name = getattr(collection, "name", None)
    if not collection_name or not collection_name.startswith("rag_"):
        return
    client = getattr(vector_store, "_client", None)
    if client is None:
        raise RuntimeError("Cannot access the Chroma client to clean up the old index.")
    client.delete_collection(name=collection_name)
