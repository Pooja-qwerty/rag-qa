"""
Technical Documentation Q&A Assistant (RAG)
-------------------------------------------
Chunk -> embed (local MiniLM) -> FAISS retrieval -> grounded answer via Groq.

Run:
    export GROQ_API_KEY=...        # or paste it in the sidebar
    streamlit run app.py
"""

import math
import os
import re
import tempfile
import uuid
from collections import Counter
from typing import Literal

import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# 70B = better answers, 8B = even faster. Swap if you want speed over quality.
GROQ_MODEL = "llama-3.3-70b-versatile"
SAMPLE_DOC = "sample_docs/quanta_queue_docs.md"

EXAMPLE_QUESTIONS = [
    "How does the retry logic work?",
    "What is the default job timeout and what's the maximum?",
    "What happens when I exceed the rate limit?",
    "What's the difference between the admin and producer scopes?",
    "Run a thermal analysis on aluminum with fixed base at 40°C and fine mesh",
    "What are the von Mises stress results for simulation SIM-001?",
]

st.set_page_config(page_title="Tech Docs Q&A (RAG)", page_icon="📄", layout="wide")


# ---- Cache the embedding model so it loads only once -------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# How much the final score weights cosine similarity vs. the BM25 keyword
# score. 1.0 = pure embedding search, 0.0 = pure keyword search.
HYBRID_WEIGHT = 0.5


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _min_max_normalize(scores):
    """Squash scores to [0, 1] so cosine similarity and BM25 -- which live on
    completely different scales -- can be combined with a weighted sum."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# ---- Minimal Okapi BM25 keyword scorer (no extra dependency) ----------------
class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        doc_tokens = [_tokenize(c.page_content) for c in chunks]
        self.term_freqs = [Counter(toks) for toks in doc_tokens]
        self.doc_lengths = np.array([len(toks) for toks in doc_tokens])
        self.avg_doc_length = self.doc_lengths.mean()
        self.n_docs = len(chunks)

        doc_freq = Counter()
        for toks in doc_tokens:
            doc_freq.update(set(toks))
        self.idf = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in doc_freq.items()
        }

    def score(self, query):
        scores = np.zeros(self.n_docs)
        for term in _tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.n_docs):
                freq = self.term_freqs[i].get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[i] / self.avg_doc_length
                )
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores


# ---- From-scratch numpy retrieval: cosine similarity + BM25 hybrid ---------
class NumpyVectorStore:
    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings
        vectors = np.array(embeddings.embed_documents([c.page_content for c in chunks]))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.matrix = vectors / norms
        self.bm25 = BM25(chunks)

    def retrieve(self, query, k=4, hybrid_weight=HYBRID_WEIGHT):
        qvec = np.array(self.embeddings.embed_query(query))
        qvec = qvec / np.linalg.norm(qvec)
        cosine_sims = self.matrix @ qvec
        bm25_scores = self.bm25.score(query)

        final_scores = hybrid_weight * _min_max_normalize(
            cosine_sims
        ) + (1 - hybrid_weight) * _min_max_normalize(bm25_scores)

        top_k = np.argsort(final_scores)[::-1][:k]
        return [self.chunks[i] for i in top_k]


# ---- Simple fixed-size chunker: character windows with overlap -------------
def chunk_text(text, chunk_size=1000, chunk_overlap=150):
    """Slide a window over `text`, stepping by (chunk_size - chunk_overlap)
    so consecutive chunks share `chunk_overlap` characters of context."""
    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def chunk_documents(documents, chunk_size=1000, chunk_overlap=150):
    chunks = []
    for doc in documents:
        for piece in chunk_text(doc.page_content, chunk_size, chunk_overlap):
            chunks.append(Document(page_content=piece, metadata=doc.metadata))
    return chunks


# ---- Turn raw uploaded/sample text into a numpy-backed vector store ---------
def build_vectorstore(documents):
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=150)
    vectorstore = NumpyVectorStore(chunks, get_embeddings())
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


def build_doc_chain(api_key):
    prompt = ChatPromptTemplate.from_template(
        """You are a technical documentation assistant. Answer the question
using ONLY the context below. If the answer is not in the context, say
"I couldn't find that in the documentation." Be concise and precise.

Context:
{context}

Question: {input}"""
    )
    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0)
    return create_stuff_documents_chain(llm, prompt)


# ---- Function-calling layer: two fake tools, LLM decides when to call each -
@tool
def setup_simulation(
    analysis_type: str,
    material: str,
    boundary_conditions: str,
    mesh_density: Literal["coarse", "medium", "fine"],
    solver_settings: str,
) -> dict:
    """Configure and start a new simulation run.

    Args:
        analysis_type: The kind of analysis to run, e.g. "thermal", "structural", "fluid".
        material: The material being simulated, e.g. "aluminum 6061", "structural steel".
        boundary_conditions: The boundary conditions to apply, e.g.
            "fixed base temperature 40C".
        mesh_density: Mesh resolution -- "coarse", "medium", or "fine".
        solver_settings: Free-text solver options, e.g.
            "max_iterations=500, convergence_tolerance=1e-6".
    """
    return {
        "status": "configured",
        "simulation_id": f"sim-{uuid.uuid4().hex[:8]}",
        "analysis_type": analysis_type,
        "material": material,
        "boundary_conditions": boundary_conditions,
        "mesh_density": mesh_density,
        "solver_settings": solver_settings,
        "note": "stub — nothing actually ran",
    }


# Stub metric table so query_simulation_results has something to look up.
SIMULATION_METRICS = {
    "max_temperature": ("87.3", "degC"),
    "min_temperature": ("21.0", "degC"),
    "von_mises_stress": ("142.6", "MPa"),
    "max_displacement": ("0.84", "mm"),
    "safety_factor": ("2.3", "unitless"),
}


@tool
def query_simulation_results(simulation_id: str, metric: str) -> dict:
    """Look up a result metric for a previously configured simulation.

    Args:
        simulation_id: The ID returned by setup_simulation, e.g. "sim-a1b2c3d4".
        metric: The metric to look up, e.g. "max_temperature", "von_mises_stress".
    """
    value, units = SIMULATION_METRICS.get(metric, ("0.0", "unknown"))
    return {
        "status": "ok",
        "simulation_id": simulation_id,
        "metric": metric,
        "value": value,
        "units": units,
        "note": "stub — no real simulation data",
    }


TOOLS_BY_NAME = {t.name: t for t in (setup_simulation, query_simulation_results)}

ROUTER_SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are a router with access to two tools:\n"
        "1. setup_simulation -- configures and starts a new simulation run. Call "
        "this when the user asks to configure, set up, or start a simulation. "
        "Extract analysis_type, material, boundary_conditions, mesh_density, and "
        "solver_settings from what the user actually stated; if they didn't "
        "mention one, use a reasonable default value instead of leaving it out.\n"
        "2. query_simulation_results -- looks up a result metric (e.g. "
        "max_temperature, von_mises_stress) for a simulation the user refers to "
        "by simulation_id. Call this when the user asks for a simulation result "
        "or metric.\n"
        "For anything else (documentation questions, general questions), do NOT "
        "call any tool -- just reply normally."
    )
)


def build_router_llm(api_key):
    """A model with both tools bound. It only emits a tool call when the
    user's message matches one of them; otherwise it returns plain text (in
    which case we fall back to RAG)."""
    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0)
    return llm.bind_tools([setup_simulation, query_simulation_results])


# ============================== UI ============================================
st.title("📄 Technical Documentation Q&A Assistant")
st.caption("RAG over your docs — chunking + MiniLM embeddings + numpy retrieval + Groq (Llama 3.3)")

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
for col, q in zip(cols, EXAMPLE_QUESTIONS):
    if col.button(q, use_container_width=True):
        st.session_state.question_input = q

# key="question_input" makes this widget own its value in session_state, so it
# holds what the user typed across reruns instead of resetting to "".
question = st.text_input("Your question", key="question_input")

if question:
    if not api_key:
        st.error("Add your Groq API key in the sidebar first.")
        st.stop()

    # Route: let the LLM decide if this is a "set up a simulation" request
    # (tool call) or a documentation question (RAG).
    router_llm = build_router_llm(api_key)
    with st.spinner("Routing..."):
        router_response = router_llm.invoke([ROUTER_SYSTEM_MESSAGE, ("human", question)])

    if router_response.tool_calls:
        st.markdown("### 🔧 Tool Call")
        for call in router_response.tool_calls:
            st.code(f"{call['name']}({call['args']})", language="python")
            result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            st.json(result)

    else:
        doc_chain = build_doc_chain(api_key)

        with st.spinner("Retrieving and generating..."):
            retrieved = st.session_state.vectorstore.retrieve(question, k=4)
            result = doc_chain.invoke({"input": question, "context": retrieved})

        st.markdown("### Answer")
        st.write(result)

        # Showing the retrieved chunks is what proves this is real RAG.
        with st.expander(f"🔍 Sources ({len(retrieved)} chunks retrieved)"):
            for i, doc in enumerate(retrieved, 1):
                st.markdown(f"**Chunk {i}**")
                st.text(doc.page_content)
                st.divider()
