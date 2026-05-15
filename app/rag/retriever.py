from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_vector_store():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory="app/chroma_db",
        embedding_function=embedding_model
    )

    return vector_store


def retrieve_documents(query):

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query,
        k=8
    )

    filtered_results = []

    for doc, score in results:

        print("SCORE:", score)

        filtered_results.append(doc)

    return filtered_results