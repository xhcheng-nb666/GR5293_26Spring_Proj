import numpy as np
import faiss

from src.rag_qa_agicto import build_prompt, retrieve


class DummyEmbedModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        # very simple deterministic fake embeddings
        # one vector per text
        arr = np.array([[1.0, 0.0] for _ in texts], dtype="float32")
        return arr


def test_build_prompt_contains_retrieved_context():
    query = "What is RAG?"
    retrieved_chunks = [
        {
            "source": "lecture_04.pdf",
            "page_start": 3,
            "page_end": 3,
            "text": "RAG combines retrieval with generation."
        },
        {
            "source": "lecture_04.pdf",
            "page_start": 4,
            "page_end": 4,
            "text": "Semantic search retrieves by meaning."
        },
    ]

    prompt = build_prompt(query, retrieved_chunks)

    assert "What is RAG?" in prompt
    assert "RAG combines retrieval with generation." in prompt
    assert "lecture_04.pdf pages 3-3" in prompt
    assert "Semantic search retrieves by meaning." in prompt


def test_retrieve_returns_exact_top_k_count():
    # build a tiny FAISS index
    dim = 2
    index = faiss.IndexFlatIP(dim)

    embeddings = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype="float32")
    index.add(embeddings)

    metadata = [
        {"chunk_id": "c1", "source": "lecture_01.pdf", "page_start": 1, "page_end": 1, "text": "chunk 1"},
        {"chunk_id": "c2", "source": "lecture_01.pdf", "page_start": 2, "page_end": 2, "text": "chunk 2"},
        {"chunk_id": "c3", "source": "lecture_01.pdf", "page_start": 3, "page_end": 3, "text": "chunk 3"},
        {"chunk_id": "c4", "source": "lecture_01.pdf", "page_start": 4, "page_end": 4, "text": "chunk 4"},
    ]

    model = DummyEmbedModel()
    results = retrieve("test query", model, index, metadata, top_k=3)

    assert len(results) == 3
    assert all("score" in r for r in results)