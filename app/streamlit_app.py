from __future__ import annotations

import json
import os
import html
import time
from pathlib import Path

import faiss
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("data/index/faiss.index")
METADATA_PATH = Path("data/index/chunk_metadata.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
AGICTO_MODEL = "gemini-2.5-flash"
DEFAULT_TOP_K = 4

SAMPLE_QUESTIONS = [
    "",
    "What is RAG?",
    "What is RLHF?",
    "What is the bias-variance tradeoff?",
    "What is the difference between semantic search and RAG?",
    "What are some LLM benchmarking metrics?",
]


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


def retrieve(query: str, embed_model, index, metadata: list[dict], top_k: int):
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
- Prefer short paragraphs or bullet points.
- Be faithful to the retrieved text.
- End with a Sources section in this format:
  - lecture_xx.pdf, page y

Student question:
{query}

Retrieved course material:
{context}
""".strip()


def ask_agicto(prompt: str) -> str:
    client = OpenAI(
        api_key=os.environ["AGICTO_API_KEY"],
        base_url="https://api.agicto.cn/v1",
    )

    response = client.chat.completions.create(
        model=AGICTO_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def ask_agicto_baseline(query: str) -> str:
    client = OpenAI(
        api_key=os.environ["AGICTO_API_KEY"],
        base_url="https://api.agicto.cn/v1",
    )

    prompt = f"""
    You are a course assistant for a Generative AI course.

    Answer the student's question as clearly and concisely as possible.
    If you are unsure, say so.

    Student question:
    {query}
    """.strip()

    response = client.chat.completions.create(
        model=AGICTO_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def score_badge(score: float) -> str:
    if score >= 0.65:
        return "Strong match"
    if score >= 0.40:
        return "Medium match"
    return "Weak match"


def escape_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def render_source_card(item: dict, rank: int, show_chunk_text: bool) -> None:
    label = score_badge(item["score"])
    score_class = (
        "score-strong" if label == "Strong match"
        else "score-medium" if label == "Medium match"
        else "score-weak"
    )

    chunk_html = ""
    if show_chunk_text:
        chunk_html = f"""
        <div class="chunk-text">
            {escape_text(item["text"])}
        </div>
        """

    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-top">
                <div class="source-rank">[{rank}]</div>
                <div class="source-main">
                    <div class="source-title">{html.escape(item["source"])}</div>
                    <div class="source-meta">
                        <span class="meta-pill">Page {item["page_start"]}</span>
                        <span class="meta-pill">Method: {html.escape(item["method"])}</span>
                        <span class="meta-pill">Chunk: {html.escape(item["chunk_id"])}</span>
                    </div>
                </div>
                <div class="source-score">
                    <span class="score-pill {score_class}">{label} · {item["score"]:.3f}</span>
                </div>
            </div>
            {chunk_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="RAG Course Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #fafbff 0%, #f4f6fb 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #ffffff 0%, #eef3ff 100%);
            border: 1px solid #e3e8f5;
            box-shadow: 0 8px 24px rgba(20, 35, 90, 0.06);
            margin-bottom: 1.2rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #16213e;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            color: #52607a;
            font-size: 1rem;
            line-height: 1.5;
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.6rem;
            color: #1b2540;
        }
        .answer-card {
            padding: 1.1rem 1.2rem;
            border-radius: 16px;
            border: 1px solid #e5e9f2;
            box-shadow: 0 6px 18px rgba(20, 35, 90, 0.05);
            background: #ffffff;
            min-height: 280px;
        }
        .answer-card.rag {
            border-left: 6px solid #4f7cff;
        }
        .answer-card.base {
            border-left: 6px solid #9c7cff;
        }
        .card-label {
            font-size: 0.9rem;
            font-weight: 700;
            color: #5e6b87;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.7rem;
        }
        .card-body {
            color: #1f2937;
            line-height: 1.65;
            font-size: 1rem;
        }
        .source-card {
            background: #ffffff;
            border: 1px solid #e5e9f2;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 4px 14px rgba(20, 35, 90, 0.04);
        }
        .source-top {
            display: flex;
            align-items: flex-start;
            gap: 0.8rem;
        }
        .source-rank {
            font-weight: 800;
            color: #4f7cff;
            min-width: 28px;
        }
        .source-main {
            flex: 1;
        }
        .source-title {
            font-weight: 700;
            color: #18243d;
            margin-bottom: 0.35rem;
        }
        .source-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .meta-pill {
            display: inline-block;
            background: #f3f6fc;
            color: #4e5a73;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            font-size: 0.8rem;
        }
        .score-pill {
            display: inline-block;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .score-strong {
            background: #e8f7ee;
            color: #1f7a43;
        }
        .score-medium {
            background: #fff5df;
            color: #9a6a00;
        }
        .score-weak {
            background: #fdeaea;
            color: #b23838;
        }
        .chunk-text {
            margin-top: 0.85rem;
            padding: 0.85rem 0.95rem;
            background: #f8faff;
            border-radius: 10px;
            color: #334155;
            font-size: 0.95rem;
            line-height: 1.6;
            border: 1px solid #edf2fa;
        }
        .tip-box {
            background: #f7faff;
            border: 1px solid #e2ebff;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            color: #4a5a77;
            font-size: 0.95rem;
            margin-top: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "query" not in st.session_state:
        st.session_state["query"] = ""

    with st.sidebar:
        st.header("Demo Settings")
        top_k = st.slider("Top-k retrieved chunks", min_value=2, max_value=8, value=DEFAULT_TOP_K)
        show_baseline = st.checkbox("Show baseline answer", value=True)
        show_chunk_text = st.checkbox("Show retrieved chunk text", value=True)

        st.markdown("---")
        st.markdown("### About")
        st.write(
            "This demo retrieves lecture content from course PDFs and generates a grounded answer using Gemini."
        )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📚 RAG-Based Course Assistant for GR5293</div>
            <div class="hero-subtitle">
                Ask a question about the course and get an answer grounded in the lecture materials.
                The app retrieves relevant lecture chunks first, then generates a response using only those sources.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "GEMINI_API_KEY" not in os.environ:
        st.error("GEMINI_API_KEY is not set in this terminal session.")
        st.stop()

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        st.error("Missing FAISS index or metadata. Build the retriever first.")
        st.stop()

    embed_model = load_embed_model()
    index = load_faiss_index()
    metadata = load_metadata()

    st.markdown('<div class="section-title">Sample Question</div>', unsafe_allow_html=True)
    selected_sample = st.selectbox(
        "Choose a sample question or keep it blank and type your own:",
        options=SAMPLE_QUESTIONS,
        index=0,
    )

    if selected_sample:
        st.session_state["query"] = selected_sample

    query = st.text_input(
        "Enter your course question:",
        value=st.session_state["query"],
        placeholder="Example: What are some LLM benchmarking metrics?",
    )

    ask_clicked = st.button("Ask the Assistant", use_container_width=True)

    if ask_clicked and query.strip():
        st.session_state["query"] = query.strip()

        with st.spinner("Retrieving relevant lecture chunks..."):
            retrieved = retrieve(query, embed_model, index, metadata, top_k=top_k)
        best_score = retrieved[0]["score"] if retrieved else 0.0
        if best_score < 0.40:
            st.warning("Retrieved evidence looks weak, so the answer may be incomplete or less reliable.")

        with st.spinner("Generating grounded answer..."):
            rag_prompt = build_rag_prompt(query, retrieved)
            rag_answer = ask_agicto(rag_prompt)

        baseline_answer = None
        if show_baseline:
            with st.spinner("Generating baseline answer..."):
                baseline_answer = ask_agicto_baseline(query)

        st.markdown('<div class="section-title">Answers</div>', unsafe_allow_html=True)

        if show_baseline:
            col1, col2 = st.columns(2)
        else:
            col1, col2 = st.columns([1, 0.001])

        with col1:
            st.markdown(
                f"""
                <div class="answer-card rag">
                    <div class="card-label">RAG Answer</div>
                    <div class="card-body">{escape_text(rag_answer)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if show_baseline:
            with col2:
                st.markdown(
                    f"""
                    <div class="answer-card base">
                        <div class="card-label">Baseline Answer</div>
                        <div class="card-body">{escape_text(baseline_answer)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-title">Retrieved Sources</div>', unsafe_allow_html=True)
        for i, item in enumerate(retrieved, start=1):
            render_source_card(item, i, show_chunk_text)

        st.markdown(
            """
            <div class="tip-box">
                Demo tip: start with reliable questions like “What is RAG?” or “What is RLHF?” before showing harder cases.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()