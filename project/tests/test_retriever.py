"""
test_retriever.py

Purpose
-------
Unit tests for `src.retriever.Retriever`.

Responsibilities
-----------------
- Verify that `retrieve()` correctly orchestrates an `Embedder` and a
  `VectorStore` to produce a `RetrievalResult` with the expected shape and
  content.
- Use a lightweight fake `Embedder` (rather than the real
  BAAI/bge-small-en-v1.5 model) so these tests run instantly and without
  network access, while still exercising the real `VectorStore`
  implementation.

Inputs
------
Fixtures from `tests/conftest.py` (`base_config`).

Outputs
-------
Pass/fail test results reported by pytest.
"""

from __future__ import annotations

from src.models import Document, Question
from src.retriever import Retriever
from src.vector_store import VectorStore


class _FakeEmbedder:
    """A minimal stand-in for `Embedder` used to isolate Retriever tests.

    Returns a fixed, hand-chosen embedding for any query, so tests can
    assert on retrieval ordering without depending on a real embedding
    model or network access.
    """

    def __init__(self, query_embedding):
        self._query_embedding = query_embedding

    def embed_query(self, text: str):
        return self._query_embedding


def _build_populated_vector_store(config):
    """Builds a VectorStore with two documents at known embedding positions."""
    documents = [
        Document(
            document_id="doc1",
            title="Metformin mechanism",
            text="Metformin increases insulin sensitivity.",
            source="PubMed",
            year=2020,
        ),
        Document(
            document_id="doc2",
            title="ACE inhibitor mechanism",
            text="ACE inhibitors block angiotensin conversion.",
            source="PubMed",
            year=2019,
        ),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    store = VectorStore(config)
    store.build(documents, embeddings)
    return store


def test_retrieve_returns_populated_retrieval_result(base_config):
    """retrieve() should return a RetrievalResult with chunks ordered by relevance."""
    vector_store = _build_populated_vector_store(base_config)
    # This fake query embedding is closest to doc1's [1.0, 0.0, 0.0].
    fake_embedder = _FakeEmbedder(query_embedding=[0.95, 0.05, 0.0])

    retriever = Retriever(base_config, embedder=fake_embedder, vector_store=vector_store)
    question = Question(
        question_id="q1",
        question="What is the mechanism of action of metformin?",
        ground_truth_answer="It increases insulin sensitivity.",
        metadata={},
    )

    result = retriever.retrieve(question)

    assert result.question_id == "q1"
    assert result.question == question.question
    assert result.corpus_name == base_config["corpus"]["name"]
    assert len(result.retrieved_chunks) == base_config["retriever"]["top_k"]
    assert len(result.retrieval_scores) == len(result.retrieved_chunks)
    # The nearest document (doc1, about metformin) should be ranked first.
    assert "insulin sensitivity" in result.retrieved_chunks[0]


def test_retrieve_respects_top_k(base_config):
    """retrieve() should only return top_k chunks even if more documents exist."""
    base_config["retriever"]["top_k"] = 1
    vector_store = _build_populated_vector_store(base_config)
    fake_embedder = _FakeEmbedder(query_embedding=[0.0, 1.0, 0.0])

    retriever = Retriever(base_config, embedder=fake_embedder, vector_store=vector_store)
    question = Question(
        question_id="q2",
        question="What is the first-line treatment for hypertension?",
        ground_truth_answer="ACE inhibitors.",
        metadata={},
    )

    result = retriever.retrieve(question)

    assert len(result.retrieved_chunks) == 1
    assert "angiotensin" in result.retrieved_chunks[0]
