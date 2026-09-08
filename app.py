```python
import os
import re
from typing import List, Tuple

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="centered",
)

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading AI model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# =========================================================
# GROQ CLIENT
# =========================================================

def get_groq_client():
    api_key = st.secrets.get(
        "GROQ_API_KEY",
        os.getenv("GROQ_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it in "
            "Streamlit Cloud → App settings → Secrets."
        )

    return Groq(api_key=api_key)


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_pages(uploaded_file):
    """
    Extract text page-by-page so that every chunk
    can remember its original PDF page number.
    """

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = clean_text(text)

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================
# CHUNKING
# =========================================================

def chunk_pdf_pages(
    pages,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """
    Create chunks while preserving the original PDF page number.
    """

    all_chunks = []

    step = max(1, chunk_size - overlap)

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        words = text.split()

        if not words:
            continue

        start = 0

        while start < len(words):

            end = min(start + chunk_size, len(words))

            chunk_text_value = " ".join(
                words[start:end]
            ).strip()

            if chunk_text_value:
                all_chunks.append(
                    {
                        "page": page_number,
                        "text": chunk_text_value,
                    }
                )

            if end >= len(words):
                break

            start += step

    return all_chunks


# =========================================================
# FAISS VECTOR DATABASE
# =========================================================

def create_faiss_index(
    chunks,
    embedding_model,
):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_chunks(
    question,
    index,
    chunks,
    embedding_model,
    top_k=TOP_K,
):
    """
    Find the most relevant chunks from the PDF.
    """

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0],
    ):

        if index_number != -1:

            chunk = chunks[int(index_number)]

            results.append(
                {
                    "chunk_id": int(index_number),
                    "page": chunk["page"],
                    "score": float(score),
                    "text": chunk["text"],
                }
            )

    return results


# =========================================================
# ANSWER GENERATION
# =========================================================

def generate_answer(
    question,
    retrieved_chunks,
):
    """
    Generate an answer ONLY when the retrieved context
    contains enough information.
    """

    context_parts = []

    for source in retrieved_chunks:

        context_parts.append(
            f"""
SOURCE:
Page: {source["page"]}
Similarity: {source["score"]:.3f}

Content:
{source["text"]}
"""
        )

    context = "\n\n-------------------\n\n".join(
        context_parts
    )

    system_prompt = """
You are a strict PDF question-answering assistant.

Your job is to answer questions ONLY from the uploaded PDF.

IMPORTANT RULES:

1. Use ONLY the provided PDF context.
2. Never use your general knowledge to answer.
3. Never invent or assume information.
4. If the answer is clearly present in the context, answer it accurately.
5. If the answer is NOT present in the context, respond EXACTLY with:

Sorry, I couldn't find this information in the uploaded PDF.

6. If the question is unrelated to the PDF, respond with:

Sorry, I couldn't find this information in the uploaded PDF.

7. Keep answers clear and reasonably concise.
8. Do not mention chunks or similarity scores to the user.
9. Do not make up page numbers.
"""

    user_prompt = f"""
UPLOADED PDF CONTEXT:

{context}

USER QUESTION:

{question}
"""

    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.0,
        max_tokens=1200,
    )

    return response.choices[0].message.content.strip()


# =========================================================
# SOURCE DETECTION
# =========================================================

def get_relevant_sources(retrieved_chunks):
    """
    Return unique PDF page numbers used by the answer.
    """

    pages = []

    for source in retrieved_chunks:

        page = source["page"]

        if page not in pages:
            pages.append(page)

    return sorted(pages)


# =========================================================
# UI
# =========================================================

st.title("📚 PDF RAG Chat")

st.caption(
    "Upload a PDF and ask questions about its content."
)


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    file_signature = (
        uploaded_file.name,
        uploaded_file.size,
    )

    if (
        st.session_state.file_signature
        != file_signature
    ):

        with st.spinner(
            "Reading your PDF..."
        ):

            pages = extract_pdf_pages(
                uploaded_file
            )

            if not pages:

                st.error(
                    "No readable text was found in this PDF. "
                    "It may be a scanned/image-only PDF."
                )

                st.stop()

            chunks = chunk_pdf_pages(
                pages,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            if not chunks:

                st.error(
                    "Could not create searchable content "
                    "from this PDF."
                )

                st.stop()

            embedding_model = (
                load_embedding_model()
            )

            index = create_faiss_index(
                chunks,
                embedding_model,
            )

            # Store everything in session state
            st.session_state.file_signature = (
                file_signature
            )

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.chunks = chunks

            st.session_state.index = index

            st.session_state.embedding_model = (
                embedding_model
            )

            # Clear previous conversation
            st.session_state.messages = []

        st.success(
            f"✅ `{uploaded_file.name}` is ready. "
            "Ask a question below."
        )


    # =====================================================
    # CHAT HISTORY
    # =====================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message.get("sources"):

                pages = message["sources"]

                page_text = ", ".join(
                    [f"Page {page}" for page in pages]
                )

                st.caption(
                    f"📍 Source: {page_text}"
                )


    # =====================================================
    # QUESTION INPUT
    # =====================================================

    question = st.chat_input(
        "Ask a question about your PDF..."
    )


    # =====================================================
    # ANSWER
    # =====================================================

    if question:

        # Show user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)


        with st.chat_message("assistant"):

            try:

                # Retrieve relevant PDF content
                retrieved = retrieve_chunks(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.embedding_model,
                    top_k=TOP_K,
                )


                # Generate grounded answer
                with st.spinner(
                    "Searching the PDF..."
                ):

                    answer = generate_answer(
                        question,
                        retrieved,
                    )


                st.markdown(answer)


                # Only show source if answer was found
                if (
                    "couldn't find this information"
                    not in answer.lower()
                ):

                    source_pages = (
                        get_relevant_sources(
                            retrieved
                        )
                    )

                    if source_pages:

                        page_text = ", ".join(
                            [
                                f"Page {page}"
                                for page in source_pages
                            ]
                        )

                        st.caption(
                            f"📍 Source: {page_text}"
                        )

                else:

                    source_pages = []


                # Save assistant message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_pages,
                    }
                )


            except Exception as exc:

                error_message = (
                    f"Sorry, an error occurred: {exc}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

else:

    st.info(
        "👆 Upload a PDF to start asking questions."
    )
```
