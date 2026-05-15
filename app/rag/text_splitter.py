from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    split_chunks = text_splitter.split_text(text)

    final_chunks = []

    chunk_id = 1

    for chunk in split_chunks:

        final_chunks.append({
            "content": chunk,
            "chunk_id": chunk_id
        })

        chunk_id += 1

    return final_chunks