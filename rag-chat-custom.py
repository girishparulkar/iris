import os
import tempfile

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# CAI Inference Configuration
# -----------------------------

MODEL_ID = "meta/llama-3.2-1b-instruct"

BASE_URL = "https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/raggirish/v1"

API_KEY = os.getenv("CDP_TOKEN", "")


# -----------------------------
# Streamlit Page Config
# -----------------------------

st.set_page_config(
    page_title="Cloudera AI PDF RAG Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("Cloudera AI PDF RAG Assistant")
st.caption("PDF Q&A powered by FAISS + CAI Inference Llama 3.2 1B")


# -----------------------------
# Sidebar Settings
# -----------------------------

with st.sidebar:
    st.header("Settings")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("Max tokens", 128, 2048, 512, 128)
    top_k = st.slider("Number of chunks to retrieve", 1, 6, 3, 1)

    st.markdown("---")
    st.write("LLM Endpoint")
    st.code(MODEL_ID)

    st.markdown("---")
    show_context = st.checkbox("Show retrieved context", value=True)


# -----------------------------
# Cached Models / Clients
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm_client():
    if not API_KEY:
        st.error("CDP_TOKEN environment variable is not set.")
        st.stop()

    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )


# -----------------------------
# PDF Processing
# -----------------------------

def extract_text_from_pdf(uploaded_file):
    text = ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    reader = PdfReader(tmp_file_path)

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text += f"\n\n--- Page {page_num} ---\n"
            text += page_text

    return text


def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []

    text = text.replace("\n", " ")
    words = text.split()

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

        if start >= len(words):
            break

    return chunks


def build_faiss_index(chunks, embedding_model):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve_relevant_chunks(question, chunks, index, embedding_model, top_k=3):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(question_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "chunk": chunks[idx],
                "score": float(score)
            }
        )

    return results


# -----------------------------
# LLM Call
# -----------------------------

def generate_answer(question, retrieved_chunks, temperature=0.2, max_tokens=512):
    client = get_llm_client()

    context = "\n\n".join(
        [
            f"Context Chunk {i + 1}:\n{item['chunk']}"
            for i, item in enumerate(retrieved_chunks)
        ]
    )

    system_prompt = """
You are an enterprise document assistant.

Answer the user's question using only the provided context.
If the answer is not present in the context, say:
"I could not find this information in the uploaded document."

Do not make up facts.
Be concise and clear.
"""

    user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature,
        top_p=0.7,
        max_tokens=max_tokens,
        stream=False
    )

    return response.choices[0].message.content


# -----------------------------
# Session State
# -----------------------------

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Main UI
# -----------------------------

uploaded_pdf = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_pdf is not None:
    if st.button("Process PDF"):
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_pdf)

        if not text.strip():
            st.error("Could not extract text from this PDF. It may be scanned/image-based.")
            st.stop()

        with st.spinner("Splitting document into chunks..."):
            chunks = chunk_text(text)

        with st.spinner("Loading embedding model and building FAISS index..."):
            embedding_model = load_embedding_model()
            faiss_index, _ = build_faiss_index(chunks, embedding_model)

        st.session_state.chunks = chunks
        st.session_state.faiss_index = faiss_index
        st.session_state.pdf_processed = True
        st.session_state.messages = []

        st.success(f"PDF processed successfully. Created {len(chunks)} text chunks.")

if st.session_state.pdf_processed:
    st.markdown("### Ask questions from the uploaded PDF")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about the PDF...")

    if user_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating answer..."):
                try:
                    embedding_model = load_embedding_model()

                    retrieved_chunks = retrieve_relevant_chunks(
                        question=user_question,
                        chunks=st.session_state.chunks,
                        index=st.session_state.faiss_index,
                        embedding_model=embedding_model,
                        top_k=top_k
                    )

                    answer = generate_answer(
                        question=user_question,
                        retrieved_chunks=retrieved_chunks,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    st.markdown(answer)

                    if show_context:
                        with st.expander("Retrieved context used for this answer"):
                            for i, item in enumerate(retrieved_chunks, start=1):
                                st.markdown(f"#### Chunk {i} | Similarity score: {item['score']:.4f}")
                                st.write(item["chunk"])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:
                    st.error(f"Error while processing question: {str(e)}")

else:
    st.info("Upload a PDF and click **Process PDF** to start.")
