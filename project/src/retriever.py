"""
===============================================================================
retriever.py
===============================================================================

Purpose
-------
Find the most relevant document chunks for a given question.

This class connects the Embedder and the VectorStore. It converts a
question into an embedding, searches the vector database, and returns the
retrieved context in a structured format.

Pipeline Position
-----------------
Question
    ↓
Embedder
    ↓
Query Embedding
    ↓
VectorStore
    ↓
Top-K Retrieved Chunks
    ↓
RetrievalResult   ← You are here
    ↓
Generator

Retrieve the top-k PubMed passages for a question.

The Retriever uses the shared Embedder to convert the question into a query
embedding and the shared VectorStore to perform similarity search. The
results are returned as a RetrievalResult for use by the downstream
generation and evaluation stages.

Conflict injection is handled separately by FailureInjector after retrieval.
"""

from __future__ import annotations

from typing import Any, Dict

from src.embedder import Embedder
from src.logger import get_logger
from src.models import Question, RetrievalResult
from src.vector_store import VectorStore

logger = get_logger(__name__)


class Retriever:
    """
    Handles the complete retrieval stage of the RAG pipeline.

    Given a Question, this class returns the most relevant document chunks
    needed by the Generator.
    """

    def __init__(self, config: Dict[str, Any], embedder: Embedder, vector_store: VectorStore) -> None:
        """
        Prepare the Retriever for searching.

        This constructor only stores configuration and references to the already
        created Embedder and VectorStore.

        No retrieval happens here.
        """
        # Number of documents to retrieve for each question.
        self.top_k: int = config["retriever"]["top_k"]
        
        # Keep track of which corpus the retrieved documents came from.
        self.corpus_name: str = config["corpus"]["name"]
        
        # Reuse the existing Embedder instead of creating a new one.
        self._embedder = embedder
        
        # Reuse the existing VectorStore connection.
        self._vector_store = vector_store

    def retrieve(self, question: Question) -> RetrievalResult:
        """
        Retrieve the most relevant document chunks for one question.

        This is the main function of the Retriever and is called once for every
        question processed by the pipeline.
        """
        # Convert the question into an embedding vector.
        query_embedding = self._embedder.embed_query(question.question)
        
        # Search the vector database for the most similar documents.
        search_results = self._vector_store.search(query_embedding, top_k=self.top_k)

        # Extract only the document text from the search results.
        retrieved_chunks = [text for text, _title, _score in search_results]
        
        # Extract the similarity score for each retrieved document.
        retrieval_scores = [float(score) for _text, _title, score in search_results]

        # Extract the similarity score for each retrieved document.
        logger.debug(
            "Retrieved %d chunks for question_id=%s.",
            len(retrieved_chunks),
            question.question_id,
        )

    # The Generator only works with RetrievalResult objects.
    # Keeping retrieval output in one standard format makes it easy to insert
    # new stages (e.g. Failure Injection) without changing either Retriever
    # or Generator.
        return RetrievalResult(
            question_id=question.question_id,
            question=question.question,
            retrieved_chunks=retrieved_chunks,
            retrieval_scores=retrieval_scores,
            corpus_name=self.corpus_name,
        )
