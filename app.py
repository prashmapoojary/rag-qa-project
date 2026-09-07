import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import load_vectorstore, ask

load_dotenv()

st.set_page_config(page_title="RAG Research Assistant", page_icon="📚")
st.title("📚 RAG Research Assistant")
st.caption("Ask questions about a curated set of Retrieval-Augmented Generation papers.")

@st.cache_resource
def get_resources():
    vectorstore = load_vectorstore()
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY"),
    )
    return vectorstore, llm

vectorstore, llm = get_resources()

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
            answer, sources = ask(question, vectorstore, llm)
            used = ", ".join(sorted(set(s.metadata.get("source", "unknown") for s in sources)))
            full_response = f"{answer}\n\n**Sources:** {used}"
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})