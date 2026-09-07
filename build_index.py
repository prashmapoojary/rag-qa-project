from ingest import load_and_chunk
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_index():
    print("Loading and chunking documents...")
    chunks = load_and_chunk()
    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model (first run downloads ~90MB, be patient)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS index (embeds every chunk — may take a few minutes)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("faiss_index")
    print("Done. Index saved to the 'faiss_index' folder.")

if __name__ == "__main__":
    build_index()