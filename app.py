"""
Streamlit UI - Academic Q&A System using RAG
=============================================
Students upload academic PDFs, ask questions grounded in those documents,
view supporting passages with source & page-number citations, and generate
multiple-choice quizzes for exam preparation.

External services used (disclosed to user on load):
  - Cohere  : text embeddings (indexing + query)
  - Google Gemini : answer and quiz generation
"""

from __future__ import annotations

import io

import streamlit as st

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TOP_K,
    ENABLE_RERANK,
    GEMINI_MODEL,
    COHERE_EMBED_MODEL,
    ensure_directories,
)
from generation import (
    AnswerMode,
    GenerationError,
    generate_grounded_answer,
    generate_quiz,
)
from ingest import (
    IngestionError,
    chunk_pages,
    extract_text_from_pdf,
    validate_upload,
)
from retrieval import RetrievalError, VectorStore

# Page config must be the first Streamlit call
st.set_page_config(
    page_title="Academic Q&A - RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()

# Custom CSS for a cleaner look
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session-state initialisation ─────────────────────────────────────────────
def init_session_state() -> None:
    defaults: dict = {
        "chat_messages": [],
        "indexed_files": {},
        "last_retrieved": [],
        "quiz_bundle": None,
        "quiz_answers": {},
        "quiz_revealed": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")

        top_k = st.slider(
            "Top-K retrieved passages",
            min_value=3,
            max_value=15,
            value=DEFAULT_TOP_K,
            help="Number of document passages fetched from ChromaDB before answering.",
        )

        mode_label = st.selectbox(
            "Answer mode",
            [m.value for m in AnswerMode],
            index=0,
            help=(
                "Each mode sends different instructions to Gemini "
                "(not just a label change)."
            ),
        )

        st.markdown("---")
        st.caption("**Model info**")
        st.markdown(f"- Gemini: `{GEMINI_MODEL}`")
        st.markdown(f"- Embed: `{COHERE_EMBED_MODEL}`")
        st.markdown(f"- Rerank: {'on' if ENABLE_RERANK else 'off'}")

        st.markdown("---")
        st.caption(
            f"Chunk size: **{CHUNK_SIZE}** chars | Overlap: **{CHUNK_OVERLAP}** chars  \n"
            "See README for rationale."
        )

        st.markdown("---")
        if st.button("Reset vector index", type="secondary"):
            _reset_index()

    return {"top_k": top_k, "mode": AnswerMode(mode_label)}


def _reset_index() -> None:
    try:
        store = VectorStore()
        store.reset_index()
        st.session_state.indexed_files = {}
        st.session_state.last_retrieved = []
        st.sidebar.success("Local Chroma index cleared.")
    except Exception as exc:
        st.sidebar.error(f"Reset failed: {exc}")


# ── Documents tab ─────────────────────────────────────────────────────────────
def index_uploaded_file(uploaded_file, store: VectorStore) -> tuple[bool, str]:
    """Validate -> extract -> chunk -> embed -> store. Returns (success, message)."""
    name = uploaded_file.name

    if name in st.session_state.indexed_files:
        return False, f"'{name}' was already indexed in this session."

    # 1. Validate
    try:
        validate_upload(name, uploaded_file.size)
    except IngestionError as exc:
        return False, str(exc)

    # 2. Extract
    try:
        data = uploaded_file.getvalue()
        doc = extract_text_from_pdf(io.BytesIO(data), name)
    except IngestionError as exc:
        return False, f"Could not read '{name}': {exc}"

    # 3. Scanned-PDF guard
    if doc.is_likely_scanned:
        return False, (
            f"'{name}' appears to be a scanned PDF with no extractable text.  \n"
            "OCR is not supported in this version. Please use a text-based PDF "
            "(one where you can select/copy text in a PDF viewer)."
        )

    # 4. Chunk
    try:
        chunks = chunk_pages(doc)
    except ValueError as exc:
        return False, f"Chunking error: {exc}"

    if not chunks:
        return False, f"'{name}' produced no text chunks (file may be empty or whitespace only)."

    # 5. Index (embed + store)
    try:
        added = store.index_chunks(chunks)
    except RetrievalError as exc:
        return False, f"Embedding/indexing error: {exc}"
    except ValueError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Unexpected indexing error: {exc}"

    if added == 0 and store.count() == 0:
        return False, "Indexing did not add any vectors to the database."

    st.session_state.indexed_files[name] = {
        "pages": len(doc.pages),
        "chunks": len(chunks),
        "chars": doc.total_chars,
        "new_added": added,
    }
    skipped = len(chunks) - added
    skip_note = f" ({skipped} duplicate chunk(s) skipped)" if skipped else ""
    return True, (
        f"**'{name}'** indexed: {added} chunk(s) stored{skip_note}  \n"
        f"   {len(doc.pages)} page(s) | {doc.total_chars:,} characters"
    )


def render_upload_tab(store: VectorStore) -> None:
    st.subheader("Upload academic PDFs")
    st.caption(
        "Uploaded document text is sent to **Cohere** (embeddings) and "
        "**Google Gemini** (answers/quizzes) when you index and ask questions. "
        "Do not upload confidential material."
    )

    uploads = st.file_uploader(
        "Choose one or more PDF files (max 25 MB each)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if uploads:
        for uf in uploads:
            with st.spinner(f"Processing {uf.name}..."):
                ok, msg = index_uploaded_file(uf, store)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)

    # Indexed files list
    if st.session_state.indexed_files:
        st.markdown("### Documents indexed this session")
        for fname, meta in st.session_state.indexed_files.items():
            cols = st.columns([4, 1, 1, 1])
            cols[0].markdown(f"**{fname}**")
            cols[1].metric("Pages", meta["pages"])
            cols[2].metric("Chunks", meta["chunks"])
            cols[3].metric("Chars", f"{meta['chars']:,}")
    else:
        st.info("No documents indexed yet. Upload a PDF to begin.")

    st.markdown("---")
    total = store.count()
    sources = store.list_sources()
    st.caption(
        f"**Vector store:** {total} vector(s) | "
        f"**Sources:** {', '.join(sources) if sources else 'none'}"
    )


# ── Q&A Chat tab ──────────────────────────────────────────────────────────────
def build_recent_chat_context(max_turns: int = 3) -> str | None:
    """Return last N question/answer pairs as plain text (conversation continuity)."""
    msgs = st.session_state.chat_messages
    if not msgs:
        return None
    recent = msgs[-(max_turns * 2):]
    lines = []
    for m in recent:
        role = m.get("role", "user").upper()
        content = m.get("content", "")[:400]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _render_citations(citations: list[dict]) -> None:
    """Render retrieved source passages in a collapsible expander."""
    if not citations:
        return
    label = (
        "Supporting source passages "
        "(retrieved excerpts - not guaranteed sentence-level attribution)"
    )
    with st.expander(label):
        for c in citations:
            st.markdown(f"**{c['source_filename']}** - page {c['page_number']}")
            st.caption(c["text"][:800])
            st.markdown("---")


def render_chat_tab(store: VectorStore, settings: dict) -> None:
    st.subheader("Ask about your documents")

    # Replay chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg.get("insufficient"):
                st.warning(
                    "**Insufficient evidence** — the uploaded documents did not "
                    "provide enough information for this question."
                )
            st.markdown(msg["content"])
            if msg.get("citations"):
                _render_citations(msg["citations"])

    # Clear chat
    if st.session_state.chat_messages:
        if st.button("Clear chat", key="clear_chat_btn"):
            st.session_state.chat_messages = []
            st.session_state.last_retrieved = []
            st.rerun()

    # Question input
    question = st.chat_input("Ask a question about your uploaded documents...")
    if not question:
        return

    if store.count() == 0:
        st.warning(
            "No documents are indexed yet. "
            "Go to the **Documents** tab and upload a PDF first."
        )
        return

    # Append user message
    st.session_state.chat_messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving passages and generating grounded answer..."):
            try:
                chunks = store.retrieve(question, top_k=settings["top_k"])
                st.session_state.last_retrieved = chunks

                result = generate_grounded_answer(
                    question,
                    chunks,
                    mode=settings["mode"],
                    chat_context=build_recent_chat_context(),
                )

                citations = [
                    {
                        "source_filename": c.source_filename,
                        "page_number": c.page_number,
                        "text": c.text,
                    }
                    for c in chunks
                ]

                if result.insufficient_evidence:
                    st.warning(
                        "**Insufficient evidence** — the uploaded documents did not "
                        "provide enough information for this question."
                    )
                st.markdown(result.answer)
                _render_citations(citations)

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "citations": citations,
                        "insufficient": result.insufficient_evidence,
                    }
                )

            except RetrievalError as exc:
                err = str(exc)
                st.error(f"Retrieval error: {err}")
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": f"[Retrieval error: {err}]"}
                )
            except GenerationError as exc:
                err = str(exc)
                st.error(f"Generation error: {err}")
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": f"[Generation error: {err}]"}
                )
            except Exception as exc:
                err = str(exc)
                st.error(f"Unexpected error: {err}")
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": f"[Error: {err}]"}
                )


# ── Quiz tab ──────────────────────────────────────────────────────────────────
def render_quiz_tab(store: VectorStore, settings: dict) -> None:
    st.subheader("Generate a practice quiz")
    st.caption(
        "Questions are generated **strictly from retrieved document passages**. "
        "General knowledge outside the uploaded documents is not used."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "Focus topic (optional)",
            placeholder="e.g. Newton's laws, photosynthesis, data structures...",
            help="Used as a retrieval query to select relevant passages from your documents.",
        )
    with col2:
        num_q = st.number_input(
            "Questions",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

    if st.button("Generate quiz", type="primary"):
        if store.count() == 0:
            st.warning(
                "No documents indexed. Upload a PDF in the Documents tab first."
            )
            return
        query = topic.strip() or "key concepts from the uploaded study material"
        with st.spinner("Retrieving content and generating quiz..."):
            try:
                chunks = store.retrieve(query, top_k=max(settings["top_k"], 6))
                bundle = generate_quiz(chunks, num_questions=int(num_q))
                st.session_state.quiz_bundle = bundle
                st.session_state.quiz_answers = {}
                st.session_state.quiz_revealed = False
            except RetrievalError as exc:
                st.error(f"Retrieval error: {exc}")
                return
            except GenerationError as exc:
                st.error(f"Generation error: {exc}")
                return
            except Exception as exc:
                st.error(f"Quiz generation failed: {exc}")
                return

    bundle = st.session_state.quiz_bundle
    if bundle is None:
        st.info("Click 'Generate quiz' to create questions from your documents.")
        return

    if bundle.parse_errors:
        for err in bundle.parse_errors:
            st.warning(err)

    if not bundle.questions:
        st.info(
            "No valid quiz questions were produced. "
            "Try a different topic or upload more material."
        )
        return

    st.markdown(f"**{len(bundle.questions)} question(s)** — choose your answers, then reveal the results.")
    st.markdown("---")

    for i, q in enumerate(bundle.questions):
        st.markdown(f"**Q{i + 1}.** {q.question}")
        choice = st.radio(
            "Your answer:",
            q.options,
            key=f"quiz_q_{i}",
            index=None,
            horizontal=False,
        )
        if choice is not None:
            st.session_state.quiz_answers[i] = q.options.index(choice)
        st.markdown("")

    col_rev, col_new = st.columns([1, 1])
    with col_rev:
        if st.button("Reveal answers and score"):
            st.session_state.quiz_revealed = True
    with col_new:
        if st.button("New quiz"):
            st.session_state.quiz_bundle = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_revealed = False
            st.rerun()

    if st.session_state.quiz_revealed:
        st.markdown("---")
        st.markdown("### Results")
        score = 0
        for i, q in enumerate(bundle.questions):
            picked = st.session_state.quiz_answers.get(i)
            correct = q.correct_index
            answered = picked is not None
            is_correct = answered and (picked == correct)
            if is_correct:
                score += 1
                st.success(
                    f"**Q{i + 1}: Correct**  \n"
                    f"_{q.explanation}_"
                )
            else:
                correct_label = q.options[correct]
                picked_label = q.options[picked] if answered else "*(not answered)*"
                st.error(
                    f"**Q{i + 1}: Incorrect**  \n"
                    f"You chose: {picked_label}  \n"
                    f"Correct answer: **{correct_label}**  \n"
                    f"_{q.explanation}_"
                )
        pct = int(score / len(bundle.questions) * 100)
        if pct == 100:
            st.balloons()
        st.markdown(f"### Score: {score} / {len(bundle.questions)} ({pct}%)")


# ── Health tab ────────────────────────────────────────────────────────────────
def render_health_tab(store: VectorStore) -> None:
    import os

    st.subheader("System health")
    st.success("Streamlit app is running.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vectors in index", store.count())
    with col2:
        sources = store.list_sources()
        st.metric("Indexed documents", len(sources))

    st.markdown("---")
    st.markdown("**API key status**")

    gemini_ok = bool(os.getenv("GEMINI_API_KEY", "").strip())
    cohere_ok = bool(os.getenv("COHERE_API_KEY", "").strip())

    c1, c2 = st.columns(2)
    with c1:
        if gemini_ok:
            st.success("GEMINI_API_KEY - present (value hidden)")
        else:
            st.error("GEMINI_API_KEY - MISSING")
    with c2:
        if cohere_ok:
            st.success("COHERE_API_KEY - present (value hidden)")
        else:
            st.error("COHERE_API_KEY - MISSING")

    if not gemini_ok or not cohere_ok:
        st.info(
            "Copy `.env.example` to `.env`, add your keys, then restart Streamlit.  \n"
            "Gemini keys: https://aistudio.google.com/apikey  \n"
            "Cohere keys: https://dashboard.cohere.com/"
        )

    st.markdown("---")
    st.markdown("**Current configuration**")
    config_data = {
        "Gemini model": GEMINI_MODEL,
        "Cohere embed model": COHERE_EMBED_MODEL,
        "Chunk size (chars)": CHUNK_SIZE,
        "Chunk overlap (chars)": CHUNK_OVERLAP,
        "Default top-K": DEFAULT_TOP_K,
        "Reranking": "enabled" if ENABLE_RERANK else "disabled",
    }
    for k, v in config_data.items():
        st.text(f"  {k}: {v}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    init_session_state()

    st.title("Academic Q&A System using RAG")
    st.markdown(
        "Upload PDF study materials, ask questions grounded in your documents, "
        "view supporting passages with page numbers, and generate practice quizzes.  \n"
        "Document text is sent to **Cohere** (embeddings) and **Google Gemini** "
        "(answers) when you index and ask questions."
    )
    st.markdown("---")

    settings = render_sidebar()

    # Initialise vector store (fails gracefully on missing Cohere key)
    try:
        store = VectorStore()
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        st.info(
            "Add your API keys to a `.env` file (see `.env.example`) "
            "and restart the app."
        )
        return
    except Exception as exc:
        st.error(f"Could not initialise vector store: {exc}")
        return

    tab_docs, tab_chat, tab_quiz, tab_health = st.tabs(
        ["Documents", "Q&A Chat", "Quiz", "Health"]
    )

    with tab_docs:
        render_upload_tab(store)

    with tab_chat:
        render_chat_tab(store, settings)

    with tab_quiz:
        render_quiz_tab(store, settings)

    with tab_health:
        render_health_tab(store)


if __name__ == "__main__":
    main()
