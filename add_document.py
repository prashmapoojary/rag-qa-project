import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import ask

load_dotenv()

st.set_page_config(page_title="RAG Research Assistant", page_icon="📚")
st.title("📚 RAG Research Assistant")
st.caption("Ask questions about a curated set of Retrieval-Augmented Generation papers.")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY"),
    )

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = FAISS.load_local(
        "faiss_index", get_embeddings(), allow_dangerous_deserialization=True
    )

llm = get_llm()

# --- Sidebar: add a new document live ---
st.sidebar.header("Add a document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    if st.sidebar.button("Add to knowledge base"):
        os.makedirs("docs", exist_ok=True)
        save_path = os.path.join("docs", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.sidebar.status("Processing document..."):
            loader = PyPDFLoader(save_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(pages)
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name

            st.session_state.vectorstore.add_documents(chunks)
            st.session_state.vectorstore.save_local("faiss_index")

        st.sidebar.success(f"Added '{uploaded_file.name}' ({len(chunks)} chunks) to the knowledge base!")

# --- Main chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the papers...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask(question, st.session_state.vectorstore, llm)
            used = ", ".join(sorted(set(s.metadata.get("source", "unknown") for s in sources)))
            full_response = f"{answer}\n\n**Sources:** {used}"
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})