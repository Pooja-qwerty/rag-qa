"""
Eval script for the RAG pipeline (separate from the Streamlit UI).

Measures two things per question, over a fixed set of Q/A pairs on the sample doc:
  - retrieval hit rate: was the chunk that contains the answer in the top-k retrieved?
  - answer faithfulness: does the generated answer match the expected answer,
    judged by the LLM with a yes/no?

Also measures a third thing over a fixed set of router test cases:
  - routing accuracy: did the router LLM call (or not call) a tool when it
    should (or shouldn't) have?

Run:
    export GROQ_API_KEY=...
    python eval.py
"""

import uuid
from typing import Literal

import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

GROQ_MODEL = "llama-3.3-70b-versatile"
SAMPLE_DOC = "sample_docs/quanta_queue_docs.md"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

# Each pair is tagged with the index of the chunk (0-based, in build order)
# that contains the answer, so we can check whether retrieval actually found it.
QA_PAIRS = [
    {
        "question": "When a job exhausts its retries, what state does it move to and where is it sent?",
        "expected_answer": "It moves to the `dead` state and is sent to the dead-letter queue (DLQ).",
        "expected_chunk": 0,
    },
    {
        "question": "What is the default job_timeout_seconds and what's the maximum allowed value?",
        "expected_answer": "Default is 300 seconds (5 minutes); the maximum allowed value is 3600 seconds.",
        "expected_chunk": 1,
    },
    {
        "question": "Is the dead-letter queue enabled by default?",
        "expected_answer": "Yes, `dlq_enabled` defaults to true.",
        "expected_chunk": 1,
    },
    {
        "question": "How is the retry delay computed, and what are the default base_delay and max_delay?",
        "expected_answer": "delay = min(base_delay * (2 ** n) + random_jitter, max_delay); base_delay defaults to 2 seconds and max_delay defaults to 300 seconds.",
        "expected_chunk": 2,
    },
    {
        "question": "What's the difference between the admin and producer API key scopes?",
        "expected_answer": "An admin-scoped key can enqueue to any namespace; a producer-scoped key can only enqueue to its own namespace and cannot read job results.",
        "expected_chunk": 3,
    },
    {
        "question": "What does a rising quanta_dlq_depth metric usually indicate?",
        "expected_answer": "It usually indicates a bug in a job handler rather than an infrastructure problem.",
        "expected_chunk": 4,
    },
]

# Each case is tagged with whether the router should call a tool for it.
# The two "should trigger" cases target setup_simulation specifically; the two
# "should not trigger" cases are ordinary documentation questions.
ROUTING_TEST_CASES = [
    {
        "question": "Set up a thermal analysis on titanium with fixed base temperature 60C and a coarse mesh",
        "expect_tool_call": True,
        "expected_tool": "setup_simulation",
    },
    {
        "question": "Configure a structural simulation on steel with a clamped edge boundary condition and fine mesh",
        "expect_tool_call": True,
        "expected_tool": "setup_simulation",
    },
    {
        "question": "How does the retry logic work?",
        "expect_tool_call": False,
        "expected_tool": None,
    },
    {
        "question": "What's the difference between the admin and producer API key scopes?",
        "expect_tool_call": False,
        "expected_tool": None,
    },
]


# ---- Chunking (same fixed-size window approach as app.py) -------------------
def chunk_text(text, chunk_size=1000, chunk_overlap=150):
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


# ---- Same numpy retrieval as app.py -----------------------------------------
class NumpyVectorStore:
    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings
        vectors = np.array(embeddings.embed_documents([c.page_content for c in chunks]))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.matrix = vectors / norms

    def retrieve(self, query, k=4):
        qvec = np.array(self.embeddings.embed_query(query))
        qvec = qvec / np.linalg.norm(qvec)
        sims = self.matrix @ qvec
        top_k = np.argsort(sims)[::-1][:k]
        return list(top_k), [self.chunks[i] for i in top_k]


# ---- Same function-calling layer as app.py -----------------------------------
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


@tool
def query_simulation_results(simulation_id: str, metric: str) -> dict:
    """Look up a result metric for a previously configured simulation.

    Args:
        simulation_id: The ID returned by setup_simulation, e.g. "sim-a1b2c3d4".
        metric: The metric to look up, e.g. "max_temperature", "von_mises_stress".
    """
    return {
        "status": "ok",
        "simulation_id": simulation_id,
        "metric": metric,
        "value": "0.0",
        "units": "unknown",
        "note": "stub — no real simulation data",
    }


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


def build_router_llm():
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return llm.bind_tools([setup_simulation, query_simulation_results])


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a technical documentation assistant. Answer the question
using ONLY the context below. If the answer is not in the context, say
"I couldn't find that in the documentation." Be concise and precise.

Context:
{context}

Question: {input}"""
)

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are grading a generated answer against an expected answer.
Expected answer: {expected}
Generated answer: {generated}

Does the generated answer convey the same key facts as the expected answer?
Reply with exactly one word: "yes" or "no"."""
)


def run_eval():
    docs = TextLoader(SAMPLE_DOC, encoding="utf-8").load()
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = NumpyVectorStore(chunks, embeddings)

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    doc_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)
    judge = JUDGE_PROMPT | llm

    results = []
    for qa in QA_PAIRS:
        top_indices, retrieved_chunks = store.retrieve(qa["question"], k=TOP_K)
        hit = qa["expected_chunk"] in top_indices

        generated = doc_chain.invoke({"input": qa["question"], "context": retrieved_chunks})

        verdict = judge.invoke({"expected": qa["expected_answer"], "generated": generated})
        faithful = verdict.content.strip().lower().startswith("yes")

        results.append(
            {
                "question": qa["question"],
                "expected_chunk": qa["expected_chunk"],
                "top_indices": top_indices,
                "hit": hit,
                "generated": generated,
                "faithful": faithful,
            }
        )
    return results


def run_routing_eval():
    router = build_router_llm()

    results = []
    for case in ROUTING_TEST_CASES:
        response = router.invoke([ROUTER_SYSTEM_MESSAGE, ("human", case["question"])])
        called_tool = response.tool_calls[0]["name"] if response.tool_calls else None
        made_call = called_tool is not None

        correct = made_call == case["expect_tool_call"]
        if correct and case["expected_tool"] is not None:
            correct = called_tool == case["expected_tool"]

        results.append(
            {
                "question": case["question"],
                "expect_tool_call": case["expect_tool_call"],
                "expected_tool": case["expected_tool"],
                "called_tool": called_tool,
                "correct": correct,
            }
        )
    return results


def print_report(results, routing_results):
    n = len(results)
    hits = sum(r["hit"] for r in results)
    faithful = sum(r["faithful"] for r in results)

    header = f"{'#':<3}{'Question':<55}{'Retrieval':<12}{'Faithful':<10}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, 1):
        q = r["question"] if len(r["question"]) <= 52 else r["question"][:49] + "..."
        retrieval_mark = "PASS" if r["hit"] else "FAIL"
        faithful_mark = "PASS" if r["faithful"] else "FAIL"
        print(f"{i:<3}{q:<55}{retrieval_mark:<12}{faithful_mark:<10}")

    print("-" * len(header))
    print(f"Retrieval hit rate:  {hits}/{n} ({100 * hits / n:.0f}%)")
    print(f"Faithfulness rate:   {faithful}/{n} ({100 * faithful / n:.0f}%)")

    print()
    n_routing = len(routing_results)
    correct = sum(r["correct"] for r in routing_results)

    header2 = f"{'#':<3}{'Question':<55}{'Expected':<20}{'Called':<20}{'Result':<8}"
    print(header2)
    print("-" * len(header2))
    for i, r in enumerate(routing_results, 1):
        q = r["question"] if len(r["question"]) <= 52 else r["question"][:49] + "..."
        expected = r["expected_tool"] or "(no tool)"
        called = r["called_tool"] or "(no tool)"
        mark = "PASS" if r["correct"] else "FAIL"
        print(f"{i:<3}{q:<55}{expected:<20}{called:<20}{mark:<8}")

    print("-" * len(header2))
    print(f"Routing accuracy:    {correct}/{n_routing} ({100 * correct / n_routing:.0f}%)")


if __name__ == "__main__":
    print_report(run_eval(), run_routing_eval())
