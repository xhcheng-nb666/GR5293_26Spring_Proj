from __future__ import annotations

import json
import os
import time
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai.errors import ClientError


QUESTIONS_PATH = Path("eval/questions.json")
RESULTS_PATH = Path("eval/results_first_pass.json")

INDEX_PATH = Path("data/index/faiss.index")
METADATA_PATH = Path("data/index/chunk_metadata.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 4

# Gemini free tier can be tight, so stay conservative.
REQUEST_DELAY_SECONDS = 15
RATE_LIMIT_SLEEP_SECONDS = 65
MAX_RETRIES = 6


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ask_gemini(prompt: str, client: genai.Client) -> str:
    """
    Safe Gemini caller with retry + delay for free-tier limits and temporary server overload.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Calling Gemini... (attempt {attempt}/{MAX_RETRIES})")
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            # Slow down a bit between successful calls.
            time.sleep(REQUEST_DELAY_SECONDS)
            return response.text

        except ClientError as e:
            error_text = str(e)

            # 429 = quota/rate limit
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                wait_time = min(90 * attempt, 300)
                print(
                    f"Rate limit hit. Sleeping for {wait_time} seconds before retrying..."
                )
                time.sleep(wait_time)
                continue

            # 503 = model temporarily overloaded / unavailable
            if "503" in error_text or "UNAVAILABLE" in error_text:
                wait_time = min(20 * attempt, 90)
                print(
                    f"Gemini temporarily unavailable. Sleeping for {wait_time} seconds before retrying..."
                )
                time.sleep(wait_time)
                continue

            raise

    raise RuntimeError("Gemini request failed after maximum retries.")


def baseline_answer(question: str, client: genai.Client) -> str:
    prompt = f"""
You are a course assistant for a Generative AI course.

Answer the student's question as clearly and concisely as possible.
If you are unsure, say so.

Student question:
{question}
""".strip()

    return ask_gemini(prompt, client)


def retrieve(question: str, embed_model, index, metadata, top_k: int = TOP_K):
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


def rag_answer(question: str, retrieved_chunks: list[dict], client: genai.Client) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i}] {chunk['source']} pages {chunk['page_start']}-{chunk['page_end']}\n"
            f"{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
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
{question}

Retrieved course material:
{context}
""".strip()

    return ask_gemini(prompt, client)


def load_existing_results() -> list[dict]:
    if RESULTS_PATH.exists():
        return load_json(RESULTS_PATH)
    return []


def main() -> None:
    if "GEMINI_API_KEY" not in os.environ:
        print("Missing GEMINI_API_KEY in environment.")
        return

    if not QUESTIONS_PATH.exists():
        print(f"Missing questions file: {QUESTIONS_PATH.resolve()}")
        return
    if not INDEX_PATH.exists():
        print(f"Missing index: {INDEX_PATH.resolve()}")
        return
    if not METADATA_PATH.exists():
        print(f"Missing metadata: {METADATA_PATH.resolve()}")
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print("Loading questions...")
    questions = load_json(QUESTIONS_PATH)

    print("Loading metadata...")
    metadata = load_json(METADATA_PATH)

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    results = load_existing_results()
    completed_ids = {item["id"] for item in results}

    print(f"Loaded {len(results)} existing results.")
    print(f"Will skip already completed questions: {sorted(completed_ids)}")

    for item in questions:
        qid = item["id"]
        question = item["question"]

        if qid in completed_ids:
            print(f"Skipping {qid} (already completed)")
            continue

        print(f"\nRunning {qid}: {question}")

        try:
            baseline = baseline_answer(question, client)
            retrieved = retrieve(question, embed_model, index, metadata, TOP_K)
            rag = rag_answer(question, retrieved, client)

            result_item = {
                "id": qid,
                "question": question,
                "expected_topics": item.get("expected_topics", []),
                "expected_sources": item.get("expected_sources", []),
                "baseline_answer": baseline,
                "retrieved_chunks": retrieved,
                "rag_answer": rag,
            }

            results.append(result_item)

            # Save after every successful question.
            save_json(RESULTS_PATH, results)
            print(f"Saved progress after {qid} -> {RESULTS_PATH}")

            print("Cooling down before next question...")
            time.sleep(30)

        except Exception as e:
            print(f"Error while processing {qid}: {e}")
            print("Saving partial results and stopping.")
            save_json(RESULTS_PATH, results)
            raise

    print(f"\nDone. Final results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()