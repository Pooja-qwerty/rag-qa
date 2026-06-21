# Technical Documentation Q&A Assistant (RAG)

A Retrieval-Augmented Generation pipeline that answers engineering-style
questions about technical documents. Upload a PDF/Markdown doc, ask a question,
and get an answer **grounded in the source text** — with the retrieved chunks
shown so you can verify it.

> Built with Python, LangChain, FAISS, and Groq (Llama 3.3). Embeddings run
> locally with `all-MiniLM-L6-v2`; generation runs on Groq's free tier.

<!-- TODO: replace with your own screenshot/GIF of the running app -->
![demo](demo.png)

## How it works

```
Document → chunk (RecursiveCharacterTextSplitter)
         → embed (MiniLM, local)
         → index (FAISS)
         → retrieve top-k chunks for the question
         → answer with Groq LLM, constrained to the retrieved context
```

The prompt forces the model to answer **only** from retrieved context and to say
when it doesn't know — this is what keeps answers grounded and prevents
hallucination.

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

- How does the retry logic work?
- What is the default job timeout and what's the maximum?
- What happens when I exceed the rate limit?
- What's the difference between the admin and producer scopes?

## Tech stack

| Component | Choice |
|---|---|
| Chunking | `RecursiveCharacterTextSplitter` (1000 / 150 overlap) |
| Embeddings | `all-MiniLM-L6-v2` (local, free) |
| Vector store | FAISS (in-memory) |
| Retrieval | top-k=4 similarity search |
| Generation | Groq `llama-3.3-70b-versatile` |
| UI | Streamlit |

## Possible extensions

- Persist the FAISS index to disk (`vectorstore.save_local`) so it survives restarts
- Add reranking or hybrid (keyword + vector) retrieval
- Stream the answer token-by-token
- Evaluate retrieval quality with a small labeled question set
