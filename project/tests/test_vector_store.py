"""
test_vector_store.py

Purpose
-------
Unit tests for `src.vector_store.VectorStore`.

Responsibilities
-----------------
- Verify that `build()` creates a persistent collection populated with the
  given documents/embeddings.
- Verify that `search()` returns the nearest document first, using
  hand-crafted low-dimensional embeddings whose expected nearest-neighbor
  ordering is known in advance (so the test does not depend on the real
  BAAI/bge-small-en-v1.5 model or any network access).
- Verify that `load()` can reopen a previously-built collection in a fresh
  `VectorStore` instance (simulating a separate pipeline run).
- Verify that mismatched `documents`/`embeddings` lengths raise
  `ValueError`.
- Verify that calling `search()` before `build()`/`load()` raises
  `RuntimeError`.

Inputs
------
Fixtures from `tests/conftest.py` (`base_config`), plus hand-built
`Document` objects and synthetic embeddings defined in this file.

Outputs
-------
Pass/fail test results reported by pytest.

Note
----
These tests require the `chromadb` package (see requirements.txt) to be
installed in the test environment.
"""

from __future__ import annotations

import pytest

from src.models import Document
from src.vector_store import VectorStore


def _make_documents_and_embeddings():
    """Builds two documents with clearly-separated 3D embeddings.

    Returns:
        A tuple `(documents, embeddings)` where `embeddings[0]` is close to
        `[1, 0, 0]` and `embeddings[1]` is close to `[0, 1, 0]`, so a query
        embedding of `[1, 0, 0]` should unambiguously retrieve documents[0]
        first.
    """
    documents = [
        Document(
            document_id="doc1",
            title="Doc One",
            text="This is document one about topic A.",
            source="TestCorpus",
            year=2020,
        ),
        Document(
            document_id="doc2",
            title="Doc Two",
            text="This is document two about topic B.",
            source="TestCorpus",
            year=2021,
        ),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    return documents, embeddings


def test_build_and_search_returns_nearest_document_first(base_config):
    """search() should rank the embedding-nearest document first."""
    documents, embeddings = _make_documents_and_embeddings()
    store = VectorStore(base_config)
    store.build(documents, embeddings)

    results = store.search(query_embedding=[0.9, 0.1, 0.0], top_k=2)

    assert len(results) == 2
    top_text, top_title, _top_score = results[0]
    assert top_title == "Doc One"
    assert "topic A" in top_text


def test_build_raises_on_mismatched_lengths(base_config):
    """build() should reject documents/embeddings lists of different lengths."""
    documents, embeddings = _make_documents_and_embeddings()
    store = VectorStore(base_config)

    with pytest.raises(ValueError):
        store.build(documents, embeddings[:1])  # One embedding short.


def test_search_before_build_raises_runtime_error(base_config):
    """search() should fail clearly if called before build()/load()."""
    store = VectorStore(base_config)

    with pytest.raises(RuntimeError):
        store.search(query_embedding=[1.0, 0.0, 0.0], top_k=1)


def test_load_reopens_existing_collection(base_config):
    """A second VectorStore instance should be able to load a built collection."""
    documents, embeddings = _make_documents_and_embeddings()
    builder_store = VectorStore(base_config)
    builder_store.build(documents, embeddings)

    # Simulate a fresh pipeline run reusing a previously-built collection.
    loader_store = VectorStore(base_config)
    loader_store.load()

    results = loader_store.search(query_embedding=[0.0, 1.0, 0.0], top_k=1)
    assert results[0][1] == "Doc Two"


def test_load_without_build_raises_value_error(base_config):
    """load() should raise a clear error if the collection was never built."""
    store = VectorStore(base_config)

    with pytest.raises(ValueError):
        store.load()
