# Academic Q&A System using RAG

A beginner-friendly **Retrieval-Augmented Generation (RAG)** web app for students.
Upload academic PDFs, ask questions grounded in your documents, view supporting passages
with **filename and page-number citations**, and generate **multiple-choice quizzes** for exam prep.

---

## RAG Data Flow

```
PDF Upload
    │
    ▼
PyMuPDF — extract text per page (filename + page number preserved)
    │
    ▼
Text quality check
    │
    ├── Readable text → use extracted text
    │
    └── Garbled/scanned text → Tesseract OCR fallback
    │
    ▼
Chunking with overlap (800 chars / 120 char overlap by default)
    │
    ▼
Cohere embed-multilingual-v3.0 — "search_document" embeddings
    │
    ▼
ChromaDB (local persistent storage in chroma_db/)
    │
    ┌──────────────────────┐
    │   Student question   │
    └──────────┬───────────┘
               │
               ▼
    Cohere — "search_query" embedding
               │
               ▼
    ChromaDB — cosine similarity → top-K passages + metadata
               │
               ▼
    Google Gemini — evidence-only grounded answer
               │
               ▼
    Answer + source filename + page number displayed in Streamlit
```

**Key RAG principle:** Gemini is instructed to answer **only** from the retrieved passages.
If evidence is insufficient, it responds with an `INSUFFICIENT_EVIDENCE:` prefix — the app
displays a warning instead of an invented answer.

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF extraction | PyMuPDF |
| OCR fallback | Tesseract OCR, pytesseract, Pillow |
| Embeddings | Cohere `embed-multilingual-v3.0` (English + Hindi-friendly) |
| Vector database | ChromaDB (local `chroma_db/` folder) |
| LLM | Google Gemini via `google-genai` SDK |
| Config | python-dotenv |
| Tests | pytest |
| Optional reranking | Cohere `rerank-multilingual-v3.0` (`ENABLE_RERANK=true`) |

---

## Chunk Size and Overlap

| Setting | Default | Why |
|---|---|---|
| `CHUNK_SIZE` | 800 characters | Fits ~1–2 paragraphs from textbooks; small enough for precise retrieval |
| `CHUNK_OVERLAP` | 120 characters | Reduces mid-sentence cuts at chunk boundaries so citations stay coherent |

Both are configurable via `.env`. Duplicate chunks (same SHA-256 hash) are skipped automatically.

---

## Project Layout

```
academic-qa-rag/
├── app.py            Streamlit UI (upload, chat, quiz, health tabs)
├── ingest.py         PDF extraction, OCR fallback, validation, chunking + metadata
├── retrieval.py      Cohere embeddings, ChromaDB persistence, semantic search
├── generation.py     Gemini grounded Q&A, answer modes, quiz JSON parsing
├── evaluation.py     Retrieval hit@k, latency metrics, CLI runner
├── config.py         All settings via environment variables (.env)
├── requirements.txt  Pinned dependencies
├── .env.example      Variable names only — copy to .env and add keys
├── .gitignore        Excludes .env, .venv, chroma_db, uploads
├── README.md         This file
├── data/
│   ├── uploads/      Temporary upload staging (gitignored)
│   └── eval_sample.json  Sample evaluation queries
├── chroma_db/        Persistent vector store (gitignored)
└── tests/
    ├── conftest.py
    ├── test_chunking.py
    ├── test_ingestion.py
    ├── test_retrieval.py
    └── test_generation.py
```

---

## Setup (One Time)

### 1. Python environment

Requires **Python 3.10+**.

> **Windows note:** `chromadb >= 1.0` (included) ships pre-built wheels for Windows and does not
> require MSVC build tools or compiling `chroma-hnswlib` from source.

```powershell
# Go to the project folder
cd path\to\academic-qa-rag

# Create virtual environment
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

If `pip install` fails, upgrade pip first:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1b. Tesseract OCR (for the scanned-PDF fallback)

`pytesseract` is only a Python wrapper — the **Tesseract OCR engine itself must be installed
separately** and be available on your system `PATH`.

- **Windows:** install Tesseract (e.g. the UB Mannheim installer) and add its install folder to `PATH`.
- **macOS:** `brew install tesseract`
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`

Verify with `tesseract --version`. OCR currently uses `lang="eng"` (English only).

---

### 2. API Keys

```powershell
copy .env.example .env
```

Open `.env` and fill in:

```
GEMINI_API_KEY=your_gemini_key_here
COHERE_API_KEY=your_cohere_key_here
```

| Variable | Required | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | Yes | https://aistudio.google.com/apikey |
| `COHERE_API_KEY` | Yes | https://dashboard.cohere.com/ |
| `GEMINI_MODEL` | No | Default: `gemini-2.0-flash` |
| `COHERE_EMBED_MODEL` | No | Default: `embed-multilingual-v3.0` |
| `ENABLE_RERANK` | No | Set `true` to enable Cohere Rerank |

> **Privacy notice:** Uploaded PDF text is sent to Cohere (for embeddings) and Google Gemini
> (for answers and quizzes) when you index documents and ask questions.
> Documents are **not** uploaded to any web search service.

---

### 3. Launch the App

```powershell
# Make sure .venv is active
.\.venv\Scripts\Activate.ps1

# Start Streamlit
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage Guide

### Documents tab
1. Click **Browse files** and select one or more PDF files.
2. Wait for the green success message confirming how many chunks were indexed.
3. If a PDF contains garbled or image-based text, the app attempts
   OCR using Tesseract. OCR accuracy depends on document quality
   and the language configured for recognition.

### Q&A Chat tab
1. Choose an **Answer mode** in the sidebar (each mode changes the actual Gemini instructions).
2. Type your question in the chat box and press Enter.
3. The answer is generated only from retrieved passages — if the documents don't contain
   enough information you'll see an **Insufficient evidence** warning.
4. Expand **Supporting source passages** to see which parts of your PDF were used.
5. Click **Clear chat** to start a fresh conversation.

### Quiz tab
1. Optionally enter a focus topic (e.g. "Newton's laws").
2. Set the number of questions (1–10).
3. Click **Generate quiz** — questions come strictly from your uploaded documents.
4. Answer all questions, then click **Reveal answers and score**.

### Sidebar controls
- **Top-K**: number of passages retrieved per question (3–15).
- **Answer mode**: Simple Explanation, Short Answer, Detailed Answer, Exam-Style, Example-Based.
- **Reset vector index**: safely deletes and recreates the local ChromaDB collection.

### Health tab
Shows API key status, vector count, and current model/config settings.

---

## Answer Modes

Each mode changes the **system prompt sent to Gemini** — not just a UI label:

| Mode | Gemini instruction style |
|---|---|
| Simple Explanation | Plain language, short paragraphs, beginner-friendly |
| Short Answer | 2–4 sentences, direct, no fluff |
| Detailed Answer | Thorough, structured with headings or bullets |
| Exam-Style Answer | Definition + key points + brief conclusion |
| Example-Based Explanation | At least one concrete example from the evidence |

---

## Running Tests

```powershell
pytest -q
```

Expected output: **27 passed** (all offline — no API calls made during tests).

```powershell
# Verbose output
pytest -v

# Single module
pytest tests/test_chunking.py -v
```

### What the tests cover

| File | Tests |
|---|---|
| `test_chunking.py` | Split with overlap, page metadata, unique IDs, overlap-size error, empty pages |
| `test_ingestion.py` | Non-PDF rejection, text extraction, scanned-PDF detection |
| `test_retrieval.py` | Hit@k metric, duplicate skip, empty-index error, empty-query error (all mocked) |
| `test_generation.py` | Quiz JSON parsing, deduplication, invalid index, empty-chunk insufficient-evidence |

---

## Evaluation

After indexing PDFs that contain matching text, run the evaluation CLI:

```powershell
python -m evaluation
```

This loads `data/eval_sample.json` and reports:

**Automated metrics:**
- `retrieval_hit_at_k` — whether relevant chunk substrings appear in the top-K results
- `latency` — retrieval and end-to-end answer latency in seconds

**Human-judgment metrics (not automated):**
- Answer faithfulness to retrieved evidence
- Answer correctness vs reference answers
- Answer relevance to the question

> The evaluation script does **not** fabricate quality scores for subjective metrics.
> The sample dataset contains 5 queries based on a Process Synchronization PDF — update
> `data/eval_sample.json` with substrings from your own indexed PDFs for meaningful hit@k scores.

### Latest Evaluation

- Evaluation dataset: 5 queries based on Process Synchronization.
- Retrieval hit rate: 1.00 (5/5).
- Average retrieval latency: 0.5668 seconds.
- Average answer latency: 0.8098 seconds.

All 27 automated tests passed.

**Note:** Retrieval hit rate measures whether relevant content was found
in the top-k retrieved chunks. Answer faithfulness, correctness, and
relevance require separate human evaluation.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Missing COHERE_API_KEY` error on startup | Add keys to `.env` and restart Streamlit |
| `Missing GEMINI_API_KEY` error on answer | Add key to `.env` and restart Streamlit |
| Scanned PDF / no text warning | OCR fallback was attempted but found little or no text; try a higher-quality scan or a text-based PDF |
| `TesseractNotFoundError` | Install the Tesseract OCR engine and make sure it is on your `PATH` (see Setup 1b) |
| Empty retrieval result | Confirm indexing succeeded; rephrase your question |
| Embedding dimension mismatch | Different embed model was used previously — click **Reset vector index** in sidebar |
| `chromadb` install fails on Windows | Run `python -m pip install --upgrade pip` then reinstall |
| Streamlit port already in use | Run `streamlit run app.py --server.port 8502` |
| `INSUFFICIENT_EVIDENCE` on every answer | The uploaded documents may not cover the topic; try uploading more relevant PDFs |

---

## Limitations

- **OCR limitations** — OCR fallback is available for pages with
  garbled or insufficient extractable text. Recognition may be
  inaccurate for low-quality scans, complex layouts, or unsupported
  languages.
- **Citations are retrieved passages** — not guaranteed sentence-level attribution. The system shows
  which chunk (filename + page) was retrieved; it does not highlight the exact sentence Gemini used.
- **Optional features** — Cohere Rerank is off by default (`ENABLE_RERANK=true` to enable).
  Tavily web search fallback is not implemented in this version.
- **Evaluation subjectivity** — faithfulness and correctness require human review; they are
  documented but not automated.
- **Session-scoped index tracking** — the `indexed_files` list in the UI resets on page refresh,
  but the ChromaDB vectors persist on disk (so you don't need to re-index after a refresh).
- **Hosted deployments** — local ChromaDB persistence is not guaranteed on cloud platforms
  (e.g. Streamlit Cloud); storage may or may not persist depending on the hosting environment.