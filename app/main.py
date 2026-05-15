import streamlit as st
from services.file_service import save_uploaded_files
from rag.pdf_loader import extract_text_from_pdf
from rag.text_splitter import split_text_into_chunks
from rag.vector_store import create_vector_store
from rag.retriever import retrieve_documents
from rag.llm import generate_response
from dotenv import load_dotenv
import os


load_dotenv()

if "message" not in st.session_state:
    st.session_state.message = []

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    layout="wide"
)

st.title("Enterprise RAG Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files and "vector_store" not in st.session_state:
    saved_files = save_uploaded_files(uploaded_files)

    st.success(f"{len(saved_files)} file(s) uploaded successfully.")

    st.write("Saved Files:")
    
    for file in saved_files:
        st.write(file)

    st.subheader("Extracted Text Preview")

    extracted_text = extract_text_from_pdf(saved_files[0])
    chunks = split_text_into_chunks(extracted_text)

    vector_store = create_vector_store(chunks)

    st.text_area(
        "PDF Content",
        extracted_text[:3000],
        height=300
    )
    st.subheader("Chunk Information")

    st.write(f"Total Chunks Created: {len(chunks)}")


    st.write(chunks[0])
    st.success("Vector store created successfully!")


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.chat_input(
    "Ask a question about your documents..."
)
if user_question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })

    st.chat_message("user").write(user_question)

    retrieved_docs = retrieve_documents(user_question)

    unique_chunks = []

    for doc in retrieved_docs:

        if doc.page_content not in unique_chunks:
            unique_chunks.append(doc.page_content)

    context = "\n\n".join(unique_chunks)

    chat_history = ""

    for message in st.session_state.chat_history:
        chat_history += f"{message['role']}: {message['content']}\n"

    answer = generate_response(
        user_question,
        context,
        chat_history
)


    st.chat_message("assistant").write(answer)

    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    with st.expander("View Retrieved Sources"):

        for i, chunk in enumerate(unique_chunks):

            st.write(f"Source Chunk {i+1}")

            st.write(chunk)

            st.divider()