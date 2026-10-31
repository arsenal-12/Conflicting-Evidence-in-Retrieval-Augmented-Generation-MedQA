"""
===============================================================================
document_loader.py
===============================================================================

PURPOSE
-------
This module is responsible for loading the retrieval corpus from disk and
converting every raw JSON document into a standardized `Document` object.

Think of this module as the "translator" between:

        Raw JSON Files
              │
              ▼
      Python Document Objects

The rest of the RAG pipeline NEVER works directly with JSON files.
Instead, every downstream component (Embedder, Vector Store, Retriever,
Generator) only sees `Document` objects.

-------------------------------------------------------------------------------
WHERE THIS MODULE FITS IN THE RAG PIPELINE
-------------------------------------------------------------------------------

                config.yaml
                     │
                     ▼
              DocumentLoader
                     │
                     ▼
        corpus/pubmed_sample/*.json
                     │
                     ▼
          List[Document] Objects
                     │
                     ▼
               Embedder
                     │
                     ▼
               ChromaDB
                     │
                     ▼
                Retriever

-------------------------------------------------------------------------------
WHY THIS MODULE EXISTS
-------------------------------------------------------------------------------

Different document sources use different formats.

PubMed:
{
    "document_id": "...",
    "title": "...",
    "text": "..."
}

Wikipedia:
{
    "page_id": "...",
    "heading": "...",
    "content": "..."
}

ArXiv:
{
    "paper_id": "...",
    "abstract": "..."
}

Instead of forcing every downstream module to understand every possible
document format, this module converts ALL of them into ONE common
representation:

Document(
    document_id,
    title,
    text,
    source,
    year,
    metadata
)

This makes the retrieval pipeline completely corpus-agnostic.

===============================================================================
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

from src.io_utils import read_json, read_jsonl
from src.logger import get_logger
from src.models import Document

logger = get_logger(__name__)

# ============================================================================
# Custom Exception
# ============================================================================
#
# Instead of raising generic exceptions such as KeyError or ValueError,
# this module raises DocumentLoadError whenever a document cannot be loaded.
#
# This makes debugging much easier because the caller immediately knows
# the failure originated from the corpus loading stage.
#
# Example:
#
# GOOD
# ----
# DocumentLoadError:
#     paper23.json is missing required field "text"
#
# BAD
# ---
# KeyError: 'text'
#
# ============================================================================
class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or is malformed."""
    pass


# ============================================================================
# Helper Function
# ============================================================================
#
# PURPOSE
# -------
# Validates that ONE required field exists inside a document dictionary.
#
# This function does NOT know anything about PubMed.
# It simply validates dictionaries.
#
# INPUT
# -----
# {
#     "title":"Hypertension"
# }
#
# OUTPUT
# ------
# "Hypertension"
#
# WHY NOT ACCESS THE FIELD DIRECTLY?
#
# Instead of writing:
#
#     entry["title"]
#
# which raises an unhelpful KeyError,
#
# we validate the field first and raise a descriptive error that includes
# the filename.
#
# This helper follows the DRY (Don't Repeat Yourself) principle because
# the same validation logic is reused for document_id, title and text.
#
# ============================================================================

def _require_field(entry: Dict[str, Any], field_name: str, path: str) -> Any:

    value = entry.get(field_name)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise DocumentLoadError(f"File '{path}' is missing required field '{field_name}': {entry}")
    return value

# ============================================================================
# PubMed Parser
# ============================================================================
#
# PURPOSE
# -------
# Reads ONE PubMed JSON file and converts it into ONE Document object.
#
# INPUT
# -----
#
# paper001.json
#
# {
#     "document_id":"PM001",
#     "title":"Hypertension",
#     "text":"...",
#     "source":"PubMed",
#     "year":2023
# }
#
# OUTPUT
# ------
#
# Document(
#     document_id="PM001",
#     title="Hypertension",
#     text="...",
#     source="PubMed",
#     year=2023
# )
#
# INTERNAL STEPS
# --------------
#
# Step 1
# Read JSON
#
# Step 2
# Validate required fields
#
# Step 3
# Validate optional fields
#
# Step 4
# Construct Document object
#
# Step 5
# Return Document
#
# NOTE
# ----
# This parser ONLY understands PubMed.
#
# Later we may add:
#
# _parse_wikipedia_file()
#
# _parse_arxiv_file()
#
# without modifying the retrieval pipeline.
#
# ============================================================================
def _parse_pubmed_file(path: str) -> Document:

    entry = read_json(path)
    if not isinstance(entry, dict):
        raise DocumentLoadError(f"File '{path}' does not contain a JSON object.")

    document_id = _require_field(entry, "document_id", path)
    title = _require_field(entry, "title", path)
    text = _require_field(entry, "text", path)
    source = entry.get("source", "PubMed")

    # `year` is treated as best-effort metadata rather than a hard
    # requirement: some corpora (e.g. certain preprints) may not reliably
    # report a publication year, and failing the entire load for that
    # reason would be overly strict for a research corpus.
    year = entry.get("year", -1)
    try:
        year = int(year)
    except (TypeError, ValueError):
        logger.warning("Document '%s' has non-integer year (%r); defaulting to -1.", document_id, year)
        year = -1

    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        logger.warning(
            "Document '%s' has non-dict metadata (%r); replacing with empty dict.",
            document_id,
            metadata,
        )
        metadata = {}

    return Document(
        document_id=str(document_id),
        title=str(title),
        text=str(text),
        source=str(source),
        year=year,
        metadata=metadata,
    )
def _parse_pubmed_entry(entry: Dict[str, Any], index: int) -> Document:
    if not isinstance(entry, dict):
        raise DocumentLoadError(f"Record {index} is not a JSON object.")

    document_id = _require_field(entry, "pmid", f"record {index}")
    title = _require_field(entry, "title", f"record {index}")
    text = _require_field(entry, "abstract", f"record {index}")

    return Document(
        document_id=str(document_id),
        title=str(title),
        text=str(text),
        source="PubMed",
        year=-1,
        metadata={},
    )
# ============================================================================
# Parser Registry
# ============================================================================
#
# PURPOSE
# -------
# Maps the loader name from config.yaml to the parser function that knows
# how to read that corpus.
#
# Example:
#
# config.yaml
#
# corpus:
#     loader: "pubmed"
#
# DocumentLoader automatically selects:
#
#     _parse_pubmed_file()
#
# If later we add Wikipedia:
#
# _PARSERS = {
#     "pubmed": _parse_pubmed_file,
#     "wikipedia": _parse_wikipedia_file
# }
#
# No other source file changes.
#
# This follows the Open/Closed Principle:
#
# Open for extension
# Closed for modification
#
# ============================================================================
_PARSERS: Dict[str, Callable[[str], Document]] = {
    "pubmed": _parse_pubmed_file,
}

# ============================================================================
# Main Loader Class
# ============================================================================
#
# PURPOSE
# -------
# Coordinates the entire document loading process.
#
# It DOES NOT parse documents itself.
#
# Instead it:
#
# 1. Reads configuration
#
# 2. Chooses the correct parser
#
# 3. Walks through every JSON file
#
# 4. Calls the parser
#
# 5. Returns a List[Document]
#
# Think of this class as the manager.
#
# ============================================================================
class DocumentLoader:
# ============================================================================
# Constructor
# ============================================================================
#
# PURPOSE
# -------
# Reads the corpus configuration from config.yaml.
#
# Nothing is loaded from disk yet.
#
# It simply remembers:
#
# • Which folder contains the corpus
#
# • Which parser should be used
#
# After this constructor finishes,
#
# the loader is READY,
#
# but no documents have been loaded.
#
# ============================================================================
    def __init__(self, config: Dict[str, Any]) -> None:

        corpus_cfg = config["corpus"]
        self.directory: str = corpus_cfg["path"]
        self.loader_name: str = corpus_cfg["loader"]
        self.max_documents = corpus_cfg.get("max_documents")
        if self.loader_name not in _PARSERS:
            raise ValueError(
                f"Unknown corpus loader '{self.loader_name}'. "
                f"Available loaders: {sorted(_PARSERS.keys())}"
            )
        self._parse_file = _PARSERS[self.loader_name]
# ============================================================================
# Main Loading Function
# ============================================================================
#
# PURPOSE
# -------
# This is where the actual loading happens.
#
# EXECUTION FLOW
# --------------
#
# Check folder exists
#        │
#        ▼
# Find every JSON file
#        │
#        ▼
# Sort filenames
#        │
#        ▼
# For every file
#        │
#        ▼
# Parse JSON
#        │
#        ▼
# Create Document object
#        │
#        ▼
# Append to list
#        │
#        ▼
# Return List[Document]
#
# WHY SORT FILES?
#
# Different operating systems may return filenames in different orders.
#
# Sorting guarantees deterministic loading,
# which makes experiments reproducible.
#
# ============================================================================
    def load(self) -> List[Document]:

        documents: List[Document] = []

        # -----------------------------
        # JSONL corpus
        # -----------------------------
        if os.path.isfile(self.directory):

            if not self.directory.endswith(".jsonl"):
                raise DocumentLoadError(
                    f"Unsupported corpus file: {self.directory}"
                )

            entries = read_jsonl(self.directory)
            # TEMPORARY: load only the first 100 documents for testing
            if self.max_documents is not None:
                entries = entries[: self.max_documents]
            for idx, entry in enumerate(entries):
                documents.append(_parse_pubmed_entry(entry, idx))

        # -----------------------------
        # Folder of JSON files
        # -----------------------------
        elif os.path.isdir(self.directory):

            filenames = sorted(
                f for f in os.listdir(self.directory)
                if f.endswith(".json")
            )

            if not filenames:
                raise DocumentLoadError(
                    f"No .json document files found in corpus directory: {self.directory}"
                )

            for filename in filenames:
                full_path = os.path.join(self.directory, filename)
                documents.append(self._parse_file(full_path))

        else:
            raise FileNotFoundError(
                f"Corpus path not found: {self.directory}"
            )

        logger.info(
            "Loaded %d documents from %s using loader '%s'.",
            len(documents),
            self.directory,
            self.loader_name,
        )

        return documents