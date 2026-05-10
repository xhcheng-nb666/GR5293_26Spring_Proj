import numpy as np
import faiss
from unittest.mock import patch

from src.rag_qa_agicto import retrieve, build_prompt, ask_agicto


class DummyEmbedModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        # deterministic fake embedding
        return np.array([[1.0, 0.0]], dtype="float32")


def test_query_retrieve_build_prompt_smoke():
    dim = 2
    index = faiss.IndexFlatIP(dim)

    toy_embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
    ], dtype="float32")
    index.add(toy_embeddings)

    toy_metadata = [
        {
            "chunk_id": "chunk_1",
            "source": "lecture_04.pdf",
            "page_start": 3,
            "page_end": 3,
            "text": "RAG combines retrieval with generation."
        },
        {
            "chunk_id": "chunk_2",
            "source": "lecture_04.pdf",
            "page_start": 4,
            "page_end": 4,
            "text": "Semantic search retrieves documents by meaning."
        },
    ]

    model = DummyEmbedModel()
    results = retrieve("What is RAG?", model, index, toy_metadata, top_k=2)
    prompt = build_prompt("What is RAG?", results)

    assert len(results) == 2
    assert "What is RAG?" in prompt
    assert "RAG combines retrieval with generation." in prompt


@patch("src.rag_qa_agicto.OpenAI")
def test_agicto_call_is_mocked(mock_openai):
    mock_client = mock_openai.return_value
    mock_response = mock_client.chat.completions.create.return_value

    # mock the structure returned by the OpenAI SDK
    mock_response.choices = [
        type(
            "Choice",
            (),
            {
                "message": type("Msg", (), {"content": "Mocked answer from AGICTO"})()
            },
        )()
    ]

    answer = ask_agicto("Test prompt")
    assert answer == "Mocked answer from AGICTO"
    mock_client.chat.completions.create.assert_called_once()