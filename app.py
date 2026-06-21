"""
Technical Documentation Q&A Assistant (RAG)
-------------------------------------------
Chunk -> embed (local MiniLM) -> FAISS retrieval -> grounded answer via Groq.

Run:
    export GROQ_API_KEY=...        # or paste it in the sidebar
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 70B = better answers, 8B = even faster. Swap if you want speed over quality.
GROQ_MODEL = "llama-3.3-70b-versatile"
SAMPLE_DOC = "sample_docs/quanta_queue_docs.md"

EXAMPLE_QUESTIONS = [
    "How does the retry logic work?",
    "What is the default job timeout and what's the maximum?",
    "What happens when I exceed the rate limit?",
    "What's the difference between the admin and producer scopes?",
]

st.set_page_config(page_title="Tech Docs Q&A (RAG)", page_icon="📄", layout="wide")


# ---- Cache the embedding model so it loads only once -------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ---- Turn raw uploaded/sample text into a FAISS retriever --------------------
def build_vectorstore(documents):
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    return vectorstore, len(chunks)


def load_uploaded_file(uploaded_file):
    """Load a Streamlit UploadedFile into LangChain Documents."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix == ".pdf":
        # PyPDFLoader needs a real path, so write to a temp file first.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        return PyPDFLoader(tmp_path).load()
    # .txt / .md
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(uploaded_file.getvalue().decode("utf-8"))
        tmp_path = tmp.name
    return TextLoader(tmp_path, encoding="utf-8").load()


def build_rag_chain(retriever, api_key):
    prompt = ChatPromptTemplate.from_template(
        """You are a technical documentation assistant. Answer the question
using ONLY the context below. If the answer is not in the context, say
"I couldn't find that in the documentation." Be concise and precise.

Context:
{context}

Question: {input}"""
    )
    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)


# ============================== UI ============================================
st.title("📄 Technical Documentation Q&A Assistant")
st.caption("RAG over your docs — chunking + MiniLM embeddings + FAISS + Groq (Llama 3.3)")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Groq API key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Free key at console.groq.com. Or set GROQ_API_KEY in your env.",
    )
    st.divider()
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF / .txt / .md", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    use_sample = st.button("📦 Load sample doc instead")

# Decide what to index, and remember it across reruns via session_state.
if use_sample:
    docs = TextLoader(SAMPLE_DOC, encoding="utf-8").load()
    vs, n_chunks = build_vectorstore(docs)
    st.session_state.vectorstore = vs
    st.session_state.source_name = "Sample: Quanta Queue docs"
    st.session_state.n_chunks = n_chunks

elif uploaded:
    docs = []
    for f in uploaded:
        docs.extend(load_uploaded_file(f))
    vs, n_chunks = build_vectorstore(docs)
    st.session_state.vectorstore = vs
    st.session_state.source_name = ", ".join(f.name for f in uploaded)
    st.session_state.n_chunks = n_chunks

if "vectorstore" not in st.session_state:
    st.info(
        "👈 Upload a document or click **Load sample doc** to get started.\n\n"
        "Then ask a question below."
    )
    st.stop()

st.success(
    f"Indexed **{st.session_state.source_name}** "
    f"into {st.session_state.n_chunks} chunks. Ask away 👇"
)

# Example-question buttons so a demo never starts from a blank screen.
st.write("**Try an example:**")
cols = st.columns(len(EXAMPLE_QUESTIONS))
clicked = None
for col, q in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(q, use_container_width=True):
        clicked = q

question = st.text_input("Your question", value=clicked or "")

if question:
    if not api_key:
        st.error("Add your Groq API key in the sidebar first.")
        st.stop()

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = build_rag_chain(retriever, api_key)

    with st.spinner("Retrieving and generating..."):
        result = chain.invoke({"input": question})

    st.markdown("### Answer")
    st.write(result["answer"])

    # Showing the retrieved chunks is what proves this is real RAG.
    with st.expander(f"🔍 Sources ({len(result['context'])} chunks retrieved)"):
        for i, doc in enumerate(result["context"], 1):
            st.markdown(f"**Chunk {i}**")
            st.text(doc.page_content)
            st.divider()
