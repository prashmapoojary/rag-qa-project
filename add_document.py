import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"

def add_document(filename):
    path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {filename}...")
    loader = PyPDFLoader(path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = filename

    print(f"Created {len(chunks)} chunks from {filename}")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Loading existing index...")
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    print("Adding new chunks to index...")
    vectorstore.add_documents(chunks)

    vectorstore.save_local(INDEX_DIR)
    print(f"Done. '{filename}' added to the index.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_document.py <filename.pdf>")
        print("(the file must already be inside the 'docs' folder)")
        sys.exit(1)

    add_document(sys.argv[1])