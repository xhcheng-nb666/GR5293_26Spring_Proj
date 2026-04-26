from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("data/processed/all_chunks.json")
INDEX_DIR = Path("data/index")
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "chunk_metadata.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_embeddings(texts: list[str], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH.resolve()}")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading chunks...")
    chunks = load_chunks(INPUT_PATH)
    if not chunks:
        print("No chunks found.")
        return

    texts = [chunk["text"] for chunk in chunks]

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Building embeddings...")
    embeddings = build_embeddings(texts, model)

    dim = embeddings.shape[1]
    print(f"Embedding shape: {embeddings.shape}")

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)  # cosine similarity since embeddings are normalized
    index.add(embeddings)

    print(f"Saving index to: {INDEX_PATH}")
    faiss.write_index(index, str(INDEX_PATH))

    print(f"Saving chunk metadata to: {METADATA_PATH}")
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Indexed {len(chunks)} chunks.")
    print(f"FAISS index saved at: {INDEX_PATH}")
    print(f"Metadata saved at: {METADATA_PATH}")


if __name__ == "__main__":
    main()