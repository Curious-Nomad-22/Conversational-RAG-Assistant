# Conversational RAG Assistant

A Streamlit application that lets users upload PDF documents, search their contents with vector retrieval, and ask questions answered by a Groq-hosted language model.

## Live Demo

https://conversational-rag-assistant-rohg5mhpsmwzvpnstuicpe.streamlit.app/

## Current features

- Upload and index multiple PDFs in one session.
- Extract text page by page and preserve document/page metadata.
- Split extracted text into overlapping chunks.
- Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2` and store them in ChromaDB.
- Retrieve relevant chunks and show source details in the interface.
- Generate context-grounded answers using Groq's `llama-3.3-70b-versatile` model.
- Maintain chat history for the active Streamlit session.

## Architecture

PDF upload → page-level text extraction → chunking with source metadata → embedding generation → ChromaDB → semantic retrieval → Groq LLM answer with source IDs.

## Tech stack

- Python and Streamlit
- LangChain text splitters and Chroma integration
- ChromaDB
- Hugging Face Sentence Transformers
- pypdf
- Groq API

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the repository root:

   ```text
   GROQ_API_KEY=your_groq_api_key
   ```

4. Start the application from the repository root:

   ```bash
   streamlit run app/main.py
   ```

You can also add `GROQ_API_KEY` through your Streamlit deployment's secrets configuration.

## Notes and limitations

- Scanned PDFs that contain images rather than selectable text need OCR, which is not included.
- Chat history is held in Streamlit session state and is not durable across sessions.
- The local Chroma directory is not a managed cloud database. Deployment persistence depends on the hosting environment.
- Similarity scores are surfaced for inspection; they are distance values, not calibrated confidence probabilities.
- The application has not been security-audited or load-tested for production use.

## Planned improvements

Hybrid retrieval, reranking, automated RAG evaluation, persistent multi-user history, authentication, and deeper request/response observability.
