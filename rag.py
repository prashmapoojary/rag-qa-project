import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def build_prompt(question, context_chunks):
    context_text = "\n\n---\n\n".join(
        f"[Source: {c.metadata.get('source', 'unknown')}]\n{c.page_content}"
        for c in context_chunks
    )
    return f"""You are a helpful research assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information in the provided documents to answer that."
Mention which source(s) you used.

Context:
{context_text}

Question: {question}

Answer:"""

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)

def ask(question, vectorstore, llm, k=4):
    docs = vectorstore.similarity_search(question, k=k)
    prompt = build_prompt(question, docs)
    response = llm.invoke(prompt)
    return extract_text(response.content), docs

if __name__ == "__main__":
    print("Loading vector store...")
    vectorstore = load_vectorstore()

    print("Connecting to Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    print("\nReady! Type a question (or 'quit' to exit).\n")
    while True:
        question = input("You: ")
        if question.lower() in ("quit", "exit"):
            break
        answer, sources = ask(question, vectorstore, llm)
        print(f"\nAssistant: {answer}\n")
        used = ", ".join(sorted(set(s.metadata.get("source", "unknown") for s in sources)))
        print(f"Sources used: {used}\n")