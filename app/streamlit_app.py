from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("data/index/faiss.index")
METADATA_PATH = Path("data/index/chunk_metadata.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 4


@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_faiss_index():
    return faiss.read_index(str(INDEX_PATH))


@st.cache_data
def load_metadata():
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(query: str, embed_model, index, metadata: list[dict], top_k: int = TOP_K) -> list[dict]:
    query_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i}] {chunk['source']} pages {chunk['page_start']}-{chunk['page_end']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    return f"""
You are a course assistant for a Generative AI course.

Use ONLY the retrieved course material below.
Do not use outside knowledge.
If the answer is not supported by the retrieved material, say so.

Instructions:
- Answer clearly and concisely.
- Be faithful to the retrieved text.
- End with a Sources section in this format:
  - lecture_xx.pdf, page y

Student question:
{query}

Retrieved course material:
{context}
""".strip()


def ask_gemini(prompt: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def ask_gemini_baseline(query: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
You are a course assistant for a Generative AI course.

Answer the student's question as clearly and concisely as possible.
If you are unsure, say so.

Student question:
{query}
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def main():
    st.set_page_config(page_title="RAG Course Assistant", layout="wide")

    st.title("RAG-Based Course Assistant")
    st.write("Ask a question about the course lectures and get an answer grounded in the lecture materials.")

    if "GEMINI_API_KEY" not in os.environ:
        st.error("GEMINI_API_KEY is not set in this terminal session.")
        st.stop()

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        st.error("Missing FAISS index or metadata. Build the retriever first.")
        st.stop()

    embed_model = load_embed_model()
    index = load_faiss_index()
    metadata = load_metadata()

    query = st.text_input(
        "Enter your course question:",
        placeholder="Example: What is the difference between semantic search and RAG?"
    )

    show_baseline = st.checkbox("Also show Gemini baseline (no retrieval)", value=False)

    if st.button("Ask") and query.strip():
        with st.spinner("Retrieving relevant lecture chunks..."):
            retrieved = retrieve(query, embed_model, index, metadata, TOP_K)

        with st.spinner("Generating grounded answer..."):
            rag_prompt = build_rag_prompt(query, retrieved)
            rag_answer = ask_gemini(rag_prompt)

        st.subheader("RAG Answer")
        st.write(rag_answer)

        if show_baseline:
            with st.spinner("Generating baseline answer..."):
                baseline_answer = ask_gemini_baseline(query)

            st.subheader("Baseline Answer (No Retrieval)")
            st.write(baseline_answer)

        st.subheader("Retrieved Sources")
        for i, item in enumerate(retrieved, start=1):
            with st.expander(
                f"[{i}] {item['source']} | pages {item['page_start']}-{item['page_end']} | score={item['score']:.4f}"
            ):
                st.write(item["text"])


if __name__ == "__main__":
    main()