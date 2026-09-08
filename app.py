import os
import re
from typing import List, Tuple

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide",
)

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load the open-source embedding model once per Streamlit process."""
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    """Create the Groq client from Streamlit secrets or an environment variable."""
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it in Streamlit Cloud → "
            "App settings → Secrets."
        )

    return Groq(api_key=api_key)


# -----------------------------
# PDF processing
# -----------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from all pages of the uploaded PDF."""
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n\n".join(pages)


def clean_text(text: str) -> str:
    """Normalize unnecessary whitespace while preserving readable text."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Create overlapping word-based chunks.

    Word-based chunking is simple and predictable for a first RAG application.
    The embedding model performs its own model-specific tokenization internally.
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += step

    return chunks


def count_tokens(text: str, model) -> int:
    """
    Count tokens using the embedding model's tokenizer.

    This is mainly for transparency/debugging. SentenceTransformer still
    performs tokenization internally when generating embeddings.
    """
    try:
        return len(model.tokenizer.encode(text, add_special_tokens=True))
    except Exception:
        return 0


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def create_faiss_index(
    chunks: List[str],
    embedding_model: SentenceTransformer,
) -> Tuple[faiss.Index, np.ndarray]:
    """Embed chunks and store normalized vectors in a FAISS cosine-similarity index."""
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve_chunks(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List[Tuple[int, float, str]]:
    """Retrieve the most relevant chunks for a user query."""
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((int(idx), float(score), chunks[int(idx)]))

    return results


# -----------------------------
# Generation
# -----------------------------
def generate_answer(
    question: str,
    retrieved_chunks: List[Tuple[int, float, str]],
) -> str:
    """Generate a grounded answer using only the retrieved document context."""
    context_parts = []

    for chunk_id, score, chunk in retrieved_chunks:
        context_parts.append(
            f"[Chunk {chunk_id + 1} | similarity={score:.3f}]\n{chunk}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a document question-answering assistant.

Answer the user's question using the provided document context.

Rules:
1. Use the retrieved context as the primary source of truth.
2. Do not invent facts that are not supported by the context.
3. If the answer cannot be found in the context, clearly say:
   "I couldn't find that information in the uploaded document."
4. Keep the answer clear and concise.
5. When useful, mention the relevant chunk number(s).
"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    return response.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 PDF RAG Chat")
st.caption(
    "Upload a PDF → extract text → chunk → tokenize → embed → FAISS retrieval → "
    "Groq open-weight LLM answer"
)

with st.sidebar:
    st.header("⚙️ RAG Settings")
    top_k = st.slider("Retrieved chunks (Top-K)", 1, 10, TOP_K)
    chunk_size = st.slider("Chunk size (words)", 300, 1500, CHUNK_SIZE, 50)
    chunk_overlap = st.slider("Chunk overlap (words)", 0, 300, CHUNK_OVERLAP, 20)

    st.divider()
    st.markdown("**LLM:** `openai/gpt-oss-120b`")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown("**Vector DB:** FAISS")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="For best results, use a text-based PDF. Scanned/image-only PDFs need OCR.",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    # Rebuild the index only when a new PDF/settings combination is selected.
    file_signature = (
        uploaded_file.name,
        uploaded_file.size,
        chunk_size,
        chunk_overlap,
    )

    if st.session_state.get("file_signature") != file_signature:
        with st.spinner("Processing PDF and building the vector index..."):
            raw_text = extract_pdf_text(uploaded_file)
            text = clean_text(raw_text)

            if not text:
                st.error(
                    "No extractable text was found. This may be a scanned/image-only PDF."
                )
                st.stop()

            chunks = chunk_text(
                text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

            embedding_model = load_embedding_model()
            index, _ = create_faiss_index(chunks, embedding_model)

            token_counts = [
                count_tokens(chunk, embedding_model) for chunk in chunks[:100]
            ]
            avg_tokens = (
                sum(token_counts) / len(token_counts) if token_counts else 0
            )

            st.session_state.file_signature = file_signature
            st.session_state.document_name = uploaded_file.name
            st.session_state.document_text = text
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.embedding_model = embedding_model
            st.session_state.avg_tokens = avg_tokens
            st.session_state.messages = []

        st.success(f"Indexed **{len(chunks)} chunks** from `{uploaded_file.name}`.")

    # Document statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Characters", f"{len(st.session_state.document_text):,}")
    col2.metric("Chunks", f"{len(st.session_state.chunks):,}")
    col3.metric("Avg. tokens/chunk", f"{st.session_state.avg_tokens:.0f}")
    col4.metric("Embedding model", "MiniLM-L6")

    st.divider()

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Retrieved context"):
                    for source in message["sources"]:
                        chunk_id, score, chunk = source
                        st.markdown(
                            f"**Chunk {chunk_id + 1}** — similarity `{score:.3f}`"
                        )
                        st.write(chunk)

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                retrieved = retrieve_chunks(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.embedding_model,
                    top_k=top_k,
                )

                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, retrieved)

                st.markdown(answer)

                with st.expander("🔎 Retrieved context"):
                    for chunk_id, score, chunk in retrieved:
                        st.markdown(
                            f"**Chunk {chunk_id + 1}** — similarity `{score:.3f}`"
                        )
                        st.write(chunk)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved,
                    }
                )

            except Exception as exc:
                st.error(f"Error: {exc}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Sorry, an error occurred: {exc}",
                    }
                )

else:
    st.info("Upload a PDF to build your RAG knowledge base.")

    with st.expander("How this RAG pipeline works"):
        st.markdown(
            """
            **1. PDF extraction** → `pypdf` extracts text.

            **2. Chunking** → the text is divided into overlapping chunks.

            **3. Tokenization** → the embedding model's tokenizer converts text
            into model tokens internally; the app also reports approximate token
            counts for transparency.

            **4. Embeddings** → `all-MiniLM-L6-v2` converts each chunk into a
            numerical vector.

            **5. Vector database** → FAISS stores the vectors and performs
            similarity search.

            **6. Retrieval** → the user's question is embedded and the most
            similar chunks are retrieved.

            **7. Generation** → the retrieved context is sent to Groq's
            `openai/gpt-oss-120b` model to produce a grounded answer.
            """
        )
