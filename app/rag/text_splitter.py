from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(
    text,
    source="Unknown document",
    page_number=None,
    start_chunk_id=1,
):
    """Split text into chunks while preserving source and page metadata."""
    if not text or not text.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_chunks = text_splitter.split_text(text)

    return [
        {
            "content": chunk,
            "chunk_id": start_chunk_id + index,
            "source": source,
            "page_number": page_number,
        }
        for index, chunk in enumerate(split_chunks)
    ]
