from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_vector_store(collection_name=None):
    """Load a known Chroma collection. A collection name is required."""
    if not collection_name:
        raise ValueError("A Chroma collection name is required to load an index.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        collection_name=collection_name,
        persist_directory="app/chroma_db",
        embedding_function=embedding_model,
    )


def retrieve_documents(query, vector_store, k=6):
    """Retrieve ranked passages from the active vector store."""
    if vector_store is None:
        raise ValueError("No vector store is available. Upload a PDF first.")
    if not query or not query.strip():
        return []

    results = vector_store.similarity_search_with_score(query.strip(), k=k)
    return [(doc, float(score)) for doc, score in results]
