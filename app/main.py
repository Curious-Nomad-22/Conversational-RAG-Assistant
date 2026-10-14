import hashlib
import logging
import os
import streamlit as st
from dotenv import load_dotenv

from services.file_service import save_uploaded_files
from rag.pdf_loader import extract_pages_from_pdf
from rag.text_splitter import split_text_into_chunks
from rag.vector_store import create_vector_store, delete_vector_store_collection
from rag.retriever import retrieve_documents
from rag.llm import generate_response

load_dotenv()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Conversational RAG Assistant", layout="wide")
st.title("Conversational RAG Assistant")
st.caption("Upload PDFs and ask questions grounded in their content.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_fingerprint" not in st.session_state:
    st.session_state.indexed_fingerprint = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)


def get_upload_fingerprint(files):
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8", errors="replace"))
        digest.update(file.getvalue())
    return digest.hexdigest()


if uploaded_files:
    fingerprint = get_upload_fingerprint(uploaded_files)
    if fingerprint != st.session_state.indexed_fingerprint:
        with st.spinner("Reading PDFs and building the search index..."):
            try:
                saved_paths = save_uploaded_files(uploaded_files)
                all_chunks = []
                failures = []
                indexed_names = []

                for uploaded_file, file_path in zip(uploaded_files, saved_paths):
                    try:
                        pages = extract_pages_from_pdf(file_path)
                        file_chunks = []
                        for page in pages:
                            page_text = page["text"].strip()
                            if not page_text:
                                continue
                            chunks = split_text_into_chunks(
                                page_text,
                                source=uploaded_file.name,
                                page_number=page["page_number"],
                                start_chunk_id=len(all_chunks) + len(file_chunks) + 1,
                            )
                            file_chunks.extend(chunks)
                        if not file_chunks:
                            failures.append(
                                f"{uploaded_file.name}: no extractable text was found. "
                                "Scanned PDFs may require OCR."
                            )
                            continue
                        all_chunks.extend(file_chunks)
                        indexed_names.append(uploaded_file.name)
                    except Exception:
                        logger.exception("Failed to extract uploaded PDF: %s", uploaded_file.name)
                        failures.append(
                            f"{uploaded_file.name}: extraction failed. "
                            "Check that the PDF is valid and contains readable text."
                        )

                if not all_chunks:
                    st.session_state.vector_store = None
                    st.session_state.indexed_fingerprint = None
                    st.session_state.indexed_files = []
                    st.error("No searchable text could be extracted from the uploaded PDFs.")
                    for failure in failures:
                        st.warning(failure)
                else:
                    previous_vector_store = st.session_state.vector_store
                    new_vector_store = create_vector_store(
                        all_chunks,
                        collection_name=f"rag_{fingerprint[:20]}",
                    )
                    st.session_state.vector_store = new_vector_store
                    st.session_state.indexed_fingerprint = fingerprint
                    st.session_state.indexed_files = indexed_names
                    st.session_state.chat_history = []

                    if previous_vector_store is not None:
                        try:
                            delete_vector_store_collection(previous_vector_store)
                        except Exception:
                            logger.exception("Could not remove the previous Chroma collection")
                            st.warning(
                                "The new index is ready, but an older local index "
                                "could not be cleaned up."
                            )

                    st.success(
                        f"Indexed {len(indexed_names)} PDF(s) into "
                        f"{len(all_chunks)} searchable chunks."
                    )
                    for failure in failures:
                        st.warning(failure)
            except Exception:
                logger.exception("Failed to build the PDF index")
                st.session_state.vector_store = None
                st.session_state.indexed_fingerprint = None
                st.session_state.indexed_files = []
                st.error(
                    "Could not index the uploaded documents. Check the PDF files "
                    "and application logs, then try again."
                )

if st.session_state.indexed_files:
    st.sidebar.caption("Indexed documents")
    for filename in st.session_state.indexed_files:
        st.sidebar.write(f"• {filename}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if not os.getenv("GROQ_API_KEY"):
    st.info("Add GROQ_API_KEY to your environment or Streamlit secrets to enable answers.")

question = st.chat_input(
    "Ask a question about your documents...",
    disabled=st.session_state.vector_store is None,
)

if question:
    previous_history = st.session_state.chat_history.copy()
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    source_items = []
    try:
        with st.spinner("Searching your documents and generating an answer..."):
            retrieved = retrieve_documents(
                question,
                st.session_state.vector_store,
                k=6,
            )
            if not retrieved:
                answer = (
                    "I couldn't find relevant passages in the uploaded documents "
                    "for that question. Try rephrasing it or uploading a document "
                    "that covers the topic."
                )
            else:
                context_blocks = []
                for index, (doc, distance) in enumerate(retrieved, start=1):
                    metadata = doc.metadata or {}
                    source = metadata.get("source", "Unknown document")
                    page = metadata.get("page_number")
                    chunk_id = metadata.get("chunk_id", index)
                    source_id = f"S{index}"
                    page_label = f", page {page}" if page is not None else ""
                    context_blocks.append(
                        f"[{source_id}] Source: {source}{page_label}; "
                        f"chunk {chunk_id}\n{doc.page_content}"
                    )
                    source_items.append({
                        "id": source_id,
                        "source": source,
                        "page": page,
                        "chunk_id": chunk_id,
                        "distance": distance,
                        "content": doc.page_content,
                    })

                history_text = "\n".join(
                    f"{item['role']}: {item['content']}"
                    for item in previous_history[-8:]
                )
                answer = generate_response(
                    question,
                    "\n\n".join(context_blocks),
                    history_text,
                )
    except Exception:
        logger.exception("Failed to answer a document question")
        answer = (
            "I couldn't complete that request. Check the Groq API configuration "
            "and application logs, then try again."
        )
        source_items = []

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        if source_items:
            with st.expander("Inspect retrieved sources"):
                for item in source_items:
                    page_label = f", page {item['page']}" if item["page"] is not None else ""
                    st.markdown(
                        f"**[{item['id']}] {item['source']}{page_label}** "
                        f"— chunk {item['chunk_id']} "
                        f"(distance: {item['distance']:.4f})"
                    )
                    st.write(item["content"])
                    st.divider()
