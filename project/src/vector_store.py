"""
===============================================================================
vector_store.py
===============================================================================

Purpose
Persistent vector store for the PubMed retrieval corpus.

This module wraps ChromaDB so that the rest of the pipeline does not need to
depend on ChromaDB's API directly. It provides three operations:

- build(): create a collection from documents and embeddings;
- load(): reuse an existing persisted collection; and
- search(): retrieve the nearest documents for a query embedding.

The dissertation experiment uses cosine distance for BGE-small embeddings.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from src.logger import get_logger
from src.models import Document

logger = get_logger(__name__)

# Supported similarity metrics.
#
# We currently use cosine for BGE embeddings, but keeping the other
# options makes it easy to experiment with different embedding models
# without changing the code.
_DISTANCE_METRIC_MAP = {
    "cosine": "cosine",
    "l2": "l2",
    "ip": "ip",
}


class VectorStore:
    """
    Handles all communication with the vector database.

    The rest of the project should only use:

    • build()
    • load()
    • search()

    Everything else is handled internally.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Read the vector database settings from config.yaml and create a connection
        to ChromaDB.

        This only prepares the database.

        No documents are stored yet.
        """
        # Read only the vector database settings.
        vs_cfg = config["vector_store"]
        
        # Folder where Chroma stores the database.
        self.persist_directory: str = vs_cfg["persist_directory"]
        
        # Name of the collection inside the database.
        self.collection_name: str = vs_cfg["collection_name"]
        
        # Read the similarity metric from the configuration.
        distance_metric: str = vs_cfg.get("distance_metric", "cosine")

        if distance_metric not in _DISTANCE_METRIC_MAP:
            raise ValueError(
                f"Unsupported distance_metric '{distance_metric}'. "
                f"Must be one of {sorted(_DISTANCE_METRIC_MAP.keys())}."
            )
        # Convert our config value into the format expected by Chroma.
        self._hnsw_space = _DISTANCE_METRIC_MAP[distance_metric]

        # Create the database folder if it doesn't already exist.
        os.makedirs(self.persist_directory, exist_ok=True)

  
        import chromadb
# Connect to the persistent Chroma database.
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = None  # Populated by build() or load().

    def build(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """
        Create a new searchable vector database from the document corpus.

        Called only when building the database for the first time or after the
        corpus has changed.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents ({len(documents)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )


        existing_collections = [c.name for c in self._client.list_collections()]
        if self.collection_name in existing_collections:
            logger.info("Deleting existing collection '%s' before rebuild.", self.collection_name)
            self._client.delete_collection(self.collection_name)

        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self._hnsw_space},
        )

        ids = [doc.document_id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [
            {"title": doc.title, "source": doc.source, "year": doc.year} for doc in documents
        ]

        batch_size = self._client.get_max_batch_size()
        logger.info("Using Chroma batch size: %d", batch_size)

        for i in range(0, len(ids), batch_size):
            self._collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )
            logger.info(
                    "Added batch %d-%d of %d documents",
                    i + 1,
                    min(i + batch_size, len(ids)),
                    len(ids),
                )
        logger.info(
            "Built collection '%s' with %d documents (metric=%s).",
            self.collection_name,
            len(documents),
            self._hnsw_space,
        )
    def load(self) -> None:
        """
        Load an existing vector database instead of rebuilding it.

        Useful when the corpus has already been indexed.
        """
        existing_collections = [c.name for c in self._client.list_collections()]
        if self.collection_name not in existing_collections:
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist in "
                f"'{self.persist_directory}'. Call build() first."
            )
        self._collection = self._client.get_collection(self.collection_name)
        logger.info("Loaded existing collection '%s'.", self.collection_name)

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[str, str, float]]:
        """
        Search the vector database for the most relevant documents.

        Called every time the Retriever processes a new question.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore has no active collection. Call build() or load() first.")

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

# Extract documents, metadata and distances for each retrieved document.
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

# Return only the information needed by the Retriever.
        return [
            (doc_text, metadata.get("title", ""), distance)
            for doc_text, metadata, distance in zip(documents, metadatas, distances)
        ]
