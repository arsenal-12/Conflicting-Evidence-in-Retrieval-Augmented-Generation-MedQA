"""
embedder.py
                 main.py
                     │
                     ▼
          embedder = Embedder(config)
                     │
                     ▼
          Load SentenceTransformer
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
embed_documents()           embed_query()
        │                         │
Corpus Documents           User Question
        │                         │
        ▼                         ▼
Document Vectors          Question Vector
        └────────────┬────────────┘
                     ▼
                 Retriever
===============================================================================
embedder.py
===============================================================================

Generate dense embeddings for corpus documents and search queries.

The Embedder wraps the BGE-small SentenceTransformer model so that the rest
of the pipeline does not need to interact with the embedding library directly.

Documents are embedded for indexing, while questions are embedded at
retrieval time using the BGE query instruction.

"""

from __future__ import annotations

from typing import Any, Dict, List

from src.logger import get_logger
from src.utils import resolve_device

logger = get_logger(__name__)

# BGE models are explicitly trained with this instruction prefix on the
# *query* side only. Omitting it does not break retrieval, but the model
# authors report a measurable recall improvement when it is included, so we
# apply it here rather than leaving it up to callers to remember.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """
    Loads the embedding model and provides simple methods to generate
    embeddings for documents and questions.

    The rest of the project should only interact with this class, never
    directly with SentenceTransformer.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Read the embedding settings from config.yaml and load the model.

        The model is loaded only once and reused throughout the pipeline.
        """
        # Read only the embedding-related settings.
        embedding_cfg = config["embedding"]
        
        self.model_name: str = embedding_cfg["model_name"]
        
        # Number of texts to embed together.
        # Larger batches are usually faster but use more memory.
        self.batch_size: int = embedding_cfg.get("batch_size", 32)
        
        # Normalize vectors so cosine similarity works correctly.
        self.normalize_embeddings: bool = embedding_cfg.get("normalize_embeddings", True)
        
        # Decide whether to run on CPU or GPU.
        self.device: str = resolve_device(embedding_cfg.get("device", "auto"))

        logger.info("Loading embedding model '%s' on device '%s'...", self.model_name, self.device)
        """Imported lazily (rather than at module top) so that modules which
        only need `Embedder`'s type for a signature (e.g. during type checking) 
        don't force an eager, slow import of sentence_transformers."""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Embedding model loaded successfully.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Loads the embedding model and provides simple methods to generate
        embeddings for documents and questions.

        The rest of the project should only interact with this class, never
        directly with SentenceTransformer.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Convert a single question into an embedding vector.

        Called every time the Retriever searches for relevant documents.
        """
        # The BGE query instruction is prepended here, and only here, so
        # that document embeddings computed via `embed_documents` remain
        # directly comparable to query embeddings computed via this method.
        prefixed_text = _BGE_QUERY_INSTRUCTION + text
        embedding = self._model.encode(
            prefixed_text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding.tolist()
