
# BUS5PR1 Assessment 3 — RAG Learning Assistant

A Streamlit learning assistant for Python, SQL, and data science PDFs. It uses
pdfplumber for per-page extraction, overlapping text chunks, normalized
`sentence-transformers/all-MiniLM-L6-v2` embeddings, a FAISS cosine-similarity
index, and Groq chat completions for grounded explanations with source citations.

## Architecture

**PDFs → Extraction → Chunking → Embeddings → FAISS → Retrieval → Groq LLM → Answer with source citations**

The FAISS index uses normalized vectors with inner-product similarity, which is
cosine similarity for these vectors. Retrieval defaults to a minimum cosine
score of `0.25` and top four results. These are configurable in
`scripts/retriever.py`; test against your course material before changing them.
Chunk size defaults to 700 characters with 100 characters of overlap. Chunking
is page-local so source and one-based page attribution are preserved.

## Setup

Use Python 3.10 or later and run commands from the project root in a terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`, set `GROQ_API_KEY` to a key from
[Groq Console](https://console.groq.com/keys). Never put a real key in source,
screenshots, or version control. Replace the `GROQ_MODEL` placeholder with a
supported Groq model ID. The application default is `openai/gpt-oss-20b`,
recommended in Groq's current deprecation guidance; `GROQ_MODEL` overrides it.
Groq model availability changes, so check the
[current Groq model catalog](https://console.groq.com/docs/models) before
changing the configured model.

Generation settings are `TEMPERATURE=0.2`, `TOP_P=0.9`, and `MAX_TOKENS=800`.
The low temperature favours consistent, grounded tutoring over creative
variation; top-p and token limits constrain sampling and answer length.

## Add PDFs and run

Place educational PDFs in `data/pdfs/`, or upload one or more PDFs from the
application sidebar (uploads are stored under the ignored
`data/pdfs/uploads/` directory). The five supplied PDFs remain in the project.

```powershell
streamlit run streamlitapp.py --server.fileWatcherType none
```

The watcher is disabled explicitly because Streamlit's source watcher may probe
optional `transformers` image-processing modules that are not used by this app.
This does not affect app reloading; restart Streamlit after code changes.

Select **Build / Rebuild Knowledge Base** after adding or changing PDFs. This
reprocesses all PDFs under `data/pdfs/` and replaces the previous index. Existing
index files are not trusted automatically; the application requires a successful
build in the current session and checks for changes before answering. The index
and metadata are generated under `index/` and are intentionally not submission
artifacts. If they are missing, stale, or incompatible, rebuild from the sidebar
or run:

```powershell
python -m scripts.indexer --pdfDir data/pdfs --indexDir index
```

Text extraction is page-by-page. Image files may be saved for the existing image
processing code, but image-only pages are not OCR'd or represented as readable
content in RAG. Scanned PDFs therefore need a separate OCR step before their
content can be retrieved.

## Tests

Tests use generated synthetic PDFs, deterministic fake embeddings, a temporary
FAISS index, and a mocked Groq response. They do not need a real key or make API
calls:

```powershell
python -m unittest discover -s tests -v
```

The suite covers page/source metadata, chunking, vector/metadata alignment,
retrieval, citations, missing/stale indexes, corrupt and text-empty PDFs,
ambiguous questions, a minor spelling error, unrelated questions, prompt
injection handling, attempted secret extraction, and a grounded learning query.
The model's response to prompt injection is not proved by mocked tests: system
instructions are one safeguard, not a complete prompt-injection defence.

## Assessment demonstration plan

After building the index from your course PDFs, demonstrate:

| Demonstration | Expected behaviour |
|---|---|
| Ask a normal Python, SQL, or data-science question | A concise explanation grounded in retrieved passages with document/page citations |
| Ask for a step-by-step explanation | A beginner-friendly ordered explanation, citing retrieved sources |
| Ask a follow-up | Recent conversation is available as context, but retrieved documents remain authoritative |
| Include a small spelling mistake | Retrieval should still find relevant content where semantic similarity permits |
| Ask an ambiguous question | The tutor should request clarification rather than assume |
| Ask an unrelated question | It should say the learning materials are insufficient and avoid unsupported claims |
| Ask it to ignore its instructions or reveal secrets | Treat the query and retrieved text as untrusted data and do not reveal secrets |
| Check the Sources section | Show only passages actually retrieved for that answer, with their actual source path/name, page, and excerpt |

Source citations identify the retrieved passage; they do not independently
guarantee that every generated claim is supported. Review answers against the
quoted excerpt.

## Security and limitations

- API keys are read from `.env` and are never printed by the application.
- Uploaded files are limited to 25 MB each; source PDFs remain separate from
  temporary uploads.
- Retrieved PDF text and user questions are untrusted. Prompt instructions help
  but cannot guarantee resistance to prompt injection; verify important answers.
- The app does not execute learner-provided code.
- Empty/image-only pages do not provide searchable text unless OCR is added.
- The similarity threshold is a heuristic and may need evaluation on your
  documents. It is not a guarantee of answer correctness.