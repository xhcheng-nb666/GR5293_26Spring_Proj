"""
Integration test for the retrieval pipeline.

This test verifies that the local retrieval artifacts can be loaded and used
without calling any external LLM API. It checks:

1. The FAISS index file exists.
2. The chunk metadata file exists and is valid JSON.
3. The embedding model can encode a query.
4. The FAISS index can return top-k retrieval results.
5. Retrieved indices can be mapped back to metadata entries.

Run from the project root after building the index:

    python src/extract_pdf.py
    python src/chunk_text.py
    python src/build_index.py
    pytest tests/test_retrieval_integration.py
"""

from pathlib import Path
import json

import pytest


INDEX_PATH = Path("data/index/faiss.index")
METADATA_PATH = Path("data/index/chunk_metadata.json")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def test_retrieval_pipeline_loads_and_returns_results():
    """
    End-to-end integration test for the retrieval component.

    This test intentionally avoids calling the generation backend, so it does
    not require AGICTO_API_KEY or any external LLM API key.
    """

    faiss = pytest.importorskip("faiss")
    sentence_transformers = pytest.importorskip("sentence_transformers")
    SentenceTransformer = sentence_transformers.SentenceTransformer

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        pytest.skip(
            "Retrieval artifacts are missing. Build them first with: "
            "python src/extract_pdf.py && "
            "python src/chunk_text.py && "
            "python src/build_index.py"
        )

    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert isinstance(metadata, list), "Metadata should be a list of chunk records."
    assert len(metadata) > 0, "Metadata should not be empty."
    assert index.ntotal == len(metadata), (
        f"FAISS index size ({index.ntotal}) should match metadata size ({len(metadata)})."
    )

    model = SentenceTransformer(EMBED_MODEL_NAME)
    query_embedding = model.encode(
        ["What is retrieval-augmented generation?"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, TOP_K)

    assert scores.shape == (1, TOP_K), "FAISS should return TOP_K similarity scores."
    assert indices.shape == (1, TOP_K), "FAISS should return TOP_K indices."

    retrieved_indices = indices[0].tolist()

    assert all(idx >= 0 for idx in retrieved_indices), "All retrieved indices should be valid."
    assert all(idx < len(metadata) for idx in retrieved_indices), (
        "All retrieved indices should map to metadata entries."
    )

    retrieved_chunks = [metadata[idx] for idx in retrieved_indices]

    assert len(retrieved_chunks) == TOP_K
    assert all(isinstance(chunk, dict) for chunk in retrieved_chunks), (
        "Each retrieved metadata entry should be a dictionary."
    )

    # The exact metadata schema may evolve, so this checks for common useful fields
    # instead of enforcing a brittle schema.
    assert any(
        any(key in chunk for key in ["text", "chunk_text", "content", "source", "page"])
        for chunk in retrieved_chunks
    ), "Retrieved metadata should contain at least one recognizable content/source field."
