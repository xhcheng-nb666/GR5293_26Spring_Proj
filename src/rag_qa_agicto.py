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
AGICTO_MODEL = "gemini-2.5-flash"
TOP_K = 4


def load_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(query: str, model: SentenceTransformer, index, metadata: list[dict], top_k: int = TOP_K) -> list[dict]:
    query_emb = model.encode(
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


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
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


def main() -> None:
    if "AGICTO_API_KEY" not in os.environ:
        print("Missing AGICTO_API_KEY in environment.")
        return

    if not INDEX_PATH.exists():
        print(f"Missing index: {INDEX_PATH.resolve()}")
        return
    if not METADATA_PATH.exists():
        print(f"Missing metadata: {METADATA_PATH.resolve()}")
        return

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading metadata...")
    metadata = load_metadata(METADATA_PATH)

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    while True:
        query = input("\nAsk a course question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        retrieved = retrieve(query, embed_model, index, metadata, TOP_K)

        print("\nRetrieved chunks:")
        for i, item in enumerate(retrieved, start=1):
            preview = item["text"][:180].replace("\n", " ")
            print(
                f"[{i}] score={item['score']:.4f} "
                f"{item['source']} pages {item['page_start']}-{item['page_end']} | {preview}..."
            )

        prompt = build_prompt(query, retrieved)

        print("\nGenerating answer...\n")
        answer = ask_agicto(prompt)
        print(answer)


if __name__ == "__main__":
    main()