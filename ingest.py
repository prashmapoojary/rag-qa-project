import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
DOCS_DIR = "docs"

def load_and_chunk():
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    for filename in os.listdir(DOCS_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DOCS_DIR, filename)
            print(f"Loading {filename}...")
            loader = PyPDFLoader(path)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            for chunk in chunks:
                chunk.metadata["source"] = filename
            all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"\nTotal chunks created: {len(chunks)}")
    print("\nSample chunk:\n")
    print(chunks[0].page_content[:500])