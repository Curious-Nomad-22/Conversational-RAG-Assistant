from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vector_store(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [chunk["content"] for chunk in chunks]

    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": "resume.pdf"
        }
        for chunk in chunks
    ]

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory="app/chroma_db"
    )

    return vector_store