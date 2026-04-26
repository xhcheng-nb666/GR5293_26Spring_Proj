from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("data/index/faiss.index")
METADATA_PATH = Path("data/index/chunk_metadata.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
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
    model = SentenceTransformer(EMBED_MODEL_NAME)

    while True:
        query = input("\nEnter a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        query_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = index.search(query_emb, TOP_K)

        print(f"\nTop {TOP_K} results:")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            item = metadata[idx]
            preview = item["text"][:300].replace("\n", " ")
            print(f"\n[{rank}] score={score:.4f}")
            print(f"source={item['source']}  pages={item['page_start']}-{item['page_end']}  chunk_id={item['chunk_id']}")
            print(f"text={preview}...")


if __name__ == "__main__":
    main()