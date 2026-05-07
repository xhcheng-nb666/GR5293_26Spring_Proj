from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

INDEX_PATH = Path("data/index/faiss.index")
METADATA_PATH = Path("data/index/chunk_metadata.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"

QUESTIONS = [
    "What is RAG?",
    "What is RLHF?",
    "What is the bias-variance tradeoff?",
    "What is LoRA?",
    "What is the difference between semantic search and RAG?",
]

TOP_K_VALUES = [2, 4, 6]


def load_metadata():
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(question, embed_model, index, metadata, top_k):
    query_emb = embed_model.encode(
        [question],
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


def build_prompt(question, retrieved):
    context_parts = []
    for i, chunk in enumerate(retrieved, start=1):
        context_parts.append(
            f"[Source {i}] {chunk['source']} pages {chunk['page_start']}-{chunk['page_end']}\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)
    return f"""
Use ONLY the retrieved course material below.
If the answer is not supported, say so.

Question:
{question}

Retrieved course material:
{context}
""".strip()


def ask_agicto(prompt: str) -> str:
    client = OpenAI(
        api_key=os.environ["AGICTO_API_KEY"],
        base_url="https://api.agicto.cn/v1",
    )

    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    metadata = load_metadata()
    index = faiss.read_index(str(INDEX_PATH))
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    for question in QUESTIONS:
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")

        for k in TOP_K_VALUES:
            print(f"\n--- top_k = {k} ---")
            retrieved = retrieve(question, embed_model, index, metadata, k)
            for i, item in enumerate(retrieved, start=1):
                print(f"[{i}] {item['source']} page {item['page_start']} score={item['score']:.4f}")
            answer = ask_agicto(build_prompt(question, retrieved))
            print("\nAnswer:")
            print(answer[:1000])


if __name__ == "__main__":
    main()