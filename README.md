# Technical Docs Assistant — RAG + Agentic Tool-Routing

A Streamlit app that combines grounded document Q&A with an **agentic
tool-routing layer**: a router LLM decides, per message, whether to answer
from retrieved documentation or to call a structured tool that configures an
engineering simulation. Both paths are backed by a from-scratch retrieval
stack (no FAISS, no LangChain retriever) and validated by a standalone eval
harness.

> Built with Python, LangChain (for prompts/tool schemas only), numpy, and
> Groq (Llama 3.3). Embeddings run locally with `all-MiniLM-L6-v2`.

<!-- TODO: replace with your own screenshot/GIF of the running app -->
![demo](demo.png)

## Architecture

```
                       +----------+
                       |   User   |
                       +----------+
                             |
                             v
                    +-----------------+
                    |    Router LLM   |
                    |  (tools bound)  |
                    +-----------------+
                             |
             +---------------+----------------+
   tool call                                    no tool call
             |                                |
             v                                v
+-------------------------+      +-------------------------+
|     Simulation Tool     |      |       RAG Pipeline      |
|    setup_simulation /   |      |  hybrid retrieve (k=4)  |
| query_simulation_results|      |    -> Groq generation   |
+-------------------------+      +-------------------------+
             |                                |
             +---------------+----------------+
                             |
                             v
                      +------------+
                      |  Response  |
                      +------------+
```

## Headline feature: agentic tool-routing

Every user message first goes to a Groq model with two tools bound via
`bind_tools`:

- **`setup_simulation(analysis_type, material, boundary_conditions,
  mesh_density, solver_settings)`** — extracts a full simulation config from
  a natural-language request and returns a structured JSON dict (including a
  generated `simulation_id`).
- **`query_simulation_results(simulation_id, metric)`** — looks up a result
  metric (`max_temperature`, `von_mises_stress`, etc.) for a given
  simulation.

A system message tells the router when each tool applies and when to stay
silent (i.e. fall through to RAG). There's no keyword-matching or regex
classifier — routing is entirely the LLM's decision based on tool
descriptions and the message content, so it generalizes to phrasings the
router wasn't explicitly programmed for.

```
"Run a thermal analysis on aluminum with fixed base at 40°C and fine mesh"
  → setup_simulation(analysis_type="thermal", material="aluminum",
                      boundary_conditions="fixed base temperature 40C",
                      mesh_density="fine", solver_settings="...")

"What are the von Mises stress results for simulation SIM-001?"
  → query_simulation_results(simulation_id="SIM-001", metric="von_mises_stress")

"How does the retry logic work?"
  → (no tool call — falls through to RAG)
```

Both tools currently return stub data — no real solver is wired up — but the
extraction and routing logic is the reusable part: swap the stub bodies for
calls into a real simulation backend and the routing layer doesn't change.

## Production-readiness detail: hybrid retrieval

Retrieval is a from-scratch numpy implementation, not FAISS or a LangChain
retriever:

1. Chunks are embedded with local MiniLM and L2-normalized into a matrix.
2. A minimal from-scratch **Okapi BM25** scorer runs over the same chunks
   (tokenize, term frequency, IDF — no extra dependency).
3. Both signals are min-max normalized to `[0, 1]` and combined with a
   weighted sum: `HYBRID_WEIGHT * cosine + (1 - HYBRID_WEIGHT) * bm25`
   (`HYBRID_WEIGHT = 0.5` by default, tunable).

Pure embedding similarity is good at semantic relatedness but can under-rank
a chunk that contains an exact identifier (an error code, a config key, a
part number) if the surrounding language isn't very "on topic." BM25 fixes
that: querying `"RETRYABLE"` moves the chunk containing that literal token
from 3rd to 1st place versus pure-embedding ranking. This matters for
engineering docs specifically, since they're full of exact tokens (material
codes, solver flags, file names) that matter more than their frequency in a
general-purpose embedding space suggests.

## Proof of engineering rigor: the eval harness

`eval.py` is a standalone script (no Streamlit dependency) that scores the
pipeline against fixed test sets — not a demo, a regression check you can
run after any retrieval, prompt, or routing change:

| Metric | What it checks | How |
|---|---|---|
| Retrieval hit rate | Was the chunk containing the answer in the top-k? | Each QA pair is tagged with its source chunk index |
| Answer faithfulness | Does the generated answer match the expected answer? | LLM-as-judge, yes/no verdict |
| Routing accuracy | Did the router call a tool exactly when it should have? | 2 simulation-setup cases + 2 documentation cases |

```bash
export GROQ_API_KEY=your_key_here
python eval.py
```

```
#  Question                                               Retrieval   Faithful
--------------------------------------------------------------------------------
1  When a job exhausts its retries, what state does ...   PASS        PASS
...
Retrieval hit rate:  6/6 (100%)
Faithfulness rate:   6/6 (100%)

#  Question                                               Expected            Called              Result
----------------------------------------------------------------------------------------------------------
1  Set up a thermal analysis on titanium with fixed ...   setup_simulation    setup_simulation    PASS
...
Routing accuracy:    4/4 (100%)
```

## How this maps to SimLab workflows

The pattern in this repo — **router LLM decides tool-vs-RAG, tool calls
extract structured args from natural language, an eval harness scores both
retrieval and routing** — is directly reusable for LLM-assisted simulation
setup in a platform like SimLab:

- **Tool schemas as the config surface.** `setup_simulation`'s parameters
  (`analysis_type`, `material`, `boundary_conditions`, `mesh_density`,
  `solver_settings`) are a stand-in for a real simulation job schema. Point
  the tool body at an actual job-submission API and the LLM becomes a
  natural-language front end for simulation configuration, with the router
  handling disambiguation between "configure a new run" and "ask about an
  existing one."
- **Hybrid retrieval for engineering-dense docs.** Simulation documentation
  is exactly the case hybrid search is built for — solver keywords, material
  IDs, and boundary-condition syntax are exact tokens that pure embedding
  search under-ranks. The same BM25 + cosine combination would ground
  answers about solver options or material properties without missing
  literal matches.
- **Eval-driven confidence for tool-arg extraction.** Routing accuracy here
  is a 4-case sanity check, but the same shape (labeled test cases + pass/fail
  scoring) is what you'd scale up to validate that an LLM reliably extracts
  correct, complete simulation parameters before anything is submitted to a
  solver — a prerequisite for trusting an LLM in front of an expensive or
  irreversible compute job.
- **Stub-first design.** Both tools return fake-but-structured data before
  any real backend integration. That's deliberate: it lets the routing and
  extraction logic be validated (via the eval harness) independently of
  solver availability, so the natural-language layer can be built and tested
  before the systems it will eventually call are ready.

## Quickstart

```bash
git clone <your-repo-url>
cd rag-qa
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here        # free at https://console.groq.com
streamlit run app.py
```

Then either upload your own PDF/`.md`/`.txt`, or click **Load sample doc** to try
it against the included `sample_docs/quanta_queue_docs.md`.

## Example questions (against the sample doc)

Documentation (routes to RAG):
- How does the retry logic work?
- What is the default job timeout and what's the maximum?
- What happens when I exceed the rate limit?
- What's the difference between the admin and producer scopes?

Simulation (routes to a tool call):
- Run a thermal analysis on aluminum with fixed base at 40°C and fine mesh
- What are the von Mises stress results for simulation SIM-001?

## Tech stack

| Component | Choice |
|---|---|
| Chunking | Fixed-size character windows (1000 / 150 overlap), from scratch |
| Embeddings | `all-MiniLM-L6-v2` (local, free) |
| Retrieval | From-scratch numpy: cosine similarity + BM25 hybrid, top-k=4 |
| Tool routing | Groq `bind_tools`, system-message-steered, no keyword classifier |
| Generation | Groq `llama-3.3-70b-versatile` |
| Eval | Standalone script — retrieval hit rate, faithfulness, routing accuracy |
| UI | Streamlit |

## Possible extensions

- Wire `setup_simulation` / `query_simulation_results` into a real solver backend
- Persist the chunk embeddings to disk so re-indexing survives restarts
- Add reranking on top of hybrid retrieval
- Stream the answer token-by-token
- Expand the routing eval set with adversarial/ambiguous phrasings
