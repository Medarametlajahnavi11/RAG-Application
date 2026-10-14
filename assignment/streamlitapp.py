import logging
import os
from pathlib import Path

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st

from scripts.indexer import build_knowledge_base
from scripts.llm import GroqLLM
from scripts.retriever import Retriever


ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "data" / "pdfs"
UPLOAD_DIR = PDF_DIR / "uploads"
INDEX_DIR = ROOT / "index"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CHAT_HISTORY_MESSAGES = 20
logger = logging.getLogger(__name__)


def _upload_signature(files):
    return tuple(sorted((file.name, file.size) for file in files or []))


def _index_signature():
    files = sorted(
        path
        for path in PDF_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ) if PDF_DIR.exists() else []
    return tuple(
        (path.relative_to(PDF_DIR).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in files
    )


def _safe_upload_path(filename):
    basename = Path(filename).name
    if not basename or Path(basename).suffix.lower() != ".pdf":
        raise ValueError("Only files with a .pdf extension can be uploaded.")
    return UPLOAD_DIR / basename


st.set_page_config(page_title="Learning Assistant", layout="wide")
st.title("Python, SQL & Data Science Learning Assistant")
st.write(
    "Ask questions grounded in your learning PDFs. Answers include citations "
    "to the source document and page."
)

for key, initial in (
    ("chat_history", []),
    ("index_ready", False),
    ("indexed_signature", None),
    ("indexed_uploads", None),
):
    if key not in st.session_state:
        st.session_state[key] = initial

with st.sidebar:
    st.header("Knowledge base")
    uploadedFiles = st.file_uploader(
        "Add one or more educational PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    uploadSignature = _upload_signature(uploadedFiles)
    for uploaded in uploadedFiles or []:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error(f"{uploaded.name} exceeds the 25 MB per-file limit.")

    if st.session_state.index_ready and (
        st.session_state.indexed_signature != _index_signature()
        or st.session_state.indexed_uploads != uploadSignature
    ):
        st.session_state.index_ready = False

    buildButton = st.button(
        "Build / Rebuild Knowledge Base",
        disabled=any(file.size > MAX_UPLOAD_BYTES for file in uploadedFiles or []),
    )
    if buildButton:
        st.session_state.index_ready = False
        try:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            for uploaded in uploadedFiles or []:
                uploadPath = _safe_upload_path(uploaded.name)
                uploadPath.write_bytes(uploaded.getvalue())
            with st.spinner("Extracting PDFs and building the FAISS index..."):
                summary = build_knowledge_base(
                    pdfDir=str(PDF_DIR), indexDir=str(INDEX_DIR)
                )
            st.session_state.index_ready = True
            st.session_state.indexed_signature = _index_signature()
            st.session_state.indexed_uploads = uploadSignature
            st.sidebar.success(
                f"Indexed {summary['documents']} PDFs, {summary['pages']} pages, "
                f"and {summary['chunks']} chunks."
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("Knowledge-base build failed: %s", exc)
            st.sidebar.error(f"Could not build the knowledge base: {exc}")

    if st.session_state.index_ready:
        st.success("Knowledge base is ready.")
    else:
        st.info("Build the knowledge base before asking a question.")

st.subheader("Ask your learning question")
with st.form("question_form", clear_on_submit=False):
    query = st.text_input(
        "Question",
        placeholder="For example: Explain Python lists step by step.",
    )
    colAsk, colClear = st.columns([1, 4])
    with colAsk:
        askButton = st.form_submit_button("Ask")
    with colClear:
        clearButton = st.form_submit_button("Clear chat")

if clearButton:
    st.session_state.chat_history = []

if askButton and query.strip():
    if not st.session_state.index_ready:
        st.warning("Build or rebuild the knowledge base before asking.")
    else:
        try:
            with st.spinner("Searching the learning materials..."):
                retriever = Retriever(indexDir=str(INDEX_DIR), minimumScore=0.25)
                sources = retriever.search(query.strip(), topK=4)
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.session_state.index_ready = False
            logger.warning("Knowledge-base index is unavailable: %s", exc)
            st.error(f"The knowledge-base index is missing or incompatible: {exc}")
            st.info("Build or rebuild the knowledge base, then try again.")
        else:
            try:
                if sources:
                    with st.spinner("Preparing a grounded explanation..."):
                        llm = GroqLLM()
                        answer = llm.generate_answer(
                            query.strip(),
                            sources,
                            chat_history=st.session_state.chat_history[-6:],
                        )
                else:
                    answer = (
                        "The available learning materials do not provide enough "
                        "relevant information to answer that question. Try asking "
                        "about a topic covered by the uploaded PDFs."
                    )
                st.session_state.chat_history.extend(
                    [
                        {"role": "user", "content": query.strip()},
                        {"role": "assistant", "content": answer, "sources": sources},
                    ]
                )
                st.session_state.chat_history = st.session_state.chat_history[
                    -CHAT_HISTORY_MESSAGES:
                ]
            except (RuntimeError, ValueError, OSError) as exc:
                logger.warning("Question processing failed: %s", exc)
                st.error(f"Could not answer this question: {exc}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=False)
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.text(f"{source['source']} — page {source['page']}")
                    excerpt = source["text"].strip().replace("\n", " ")
                    st.text(excerpt[:350] + ("…" if len(excerpt) > 350 else ""))

if st.sidebar.button("Clear chat history"):
    st.session_state.chat_history = []
