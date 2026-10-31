"""
===============================================================================
models.py
===============================================================================

Purpose
-------
Define the shared data objects used throughout the RAG pipeline.

These dataclasses act as the "language" spoken between different modules.
Every stage receives one object and returns another, making the pipeline
easy to understand and extend.

Pipeline Flow
-------------
QuestionLoader
        ↓
Question
        ↓
Retriever
        ↓
RetrievalResult
        ↓
Generator
        ↓
GenerationResult
        ↓
io_utils
        ↓
results.jsonl

Responsibilities
----------------
• Define the Question object.
• Define the Document object.
• Define the RetrievalResult object.
• Define the GenerationResult object.

Why this file exists
--------------------
Instead of passing dictionaries or many separate variables between modules,
the project uses strongly structured objects.

This keeps every stage independent while ensuring everyone works with the
same data format.

When should I modify this file?
-------------------------------
Modify this file if:

• A new pipeline stage needs additional information.
• A new field should be stored throughout the experiment.
• The experiment output schema changes.

Examples:
• Failure Injection
• Faithfulness Score
• Safety Label
• Retrieved Metadata

Be careful:
Changing these objects affects every module that uses them.

Software Engineering
--------------------
✓ Data Transfer Objects (DTOs)
    These classes only carry data.
    They do not contain business logic.

✓ Loose Coupling
    Modules communicate only through these objects.

✓ Extensibility
    New fields can be added without redesigning the whole pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Question:
    """
    Represents one question from the benchmark dataset.

    Created by:
        QuestionLoader

    Used by:
        Retriever

    This is the standard question format used throughout the project,
    regardless of which dataset the question originally came from.
    """
# Unique identifier used to track this question throughout the pipeline.
    question_id: str

# The actual question asked to the RAG system.
    question: str
    ground_truth_answer: str
    
# Optional dataset-specific information.
# Safe place to store extra fields without changing the class structure.    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """
    Represents one document in the retrieval corpus.

    Created by:
        DocumentLoader

    Used by:
        Embedder
        VectorStore
    """

    document_id: str
    title: str
    text: str
    source: str
    year: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    Represents everything returned by the Retriever.

    Created by:
        Retriever

    Used by:
        Generator

    The Generator should only work with this object instead of calling the
    Retriever directly.
    """

    question_id: str
    question: str
    retrieved_chunks: List[str]
    retrieval_scores: List[float]
    corpus_name: str
    


# This is the final object produced by the pipeline.
# Every experiment eventually becomes one GenerationResult before being
# written to results.jsonl.
@dataclass
class GenerationResult:
    """
    Represents the final output of one RAG experiment.

    Created by:
        Generator

    Used by:
        io_utils

    Everything needed to reproduce or evaluate one experiment should be stored
    inside this object.
    """

    question_id: str
    question: str
    retrieved_chunks: List[str]
    prompt: str
    generator_model: str
    answer: str
    generation_time: float
    failure_mode: str = "baseline"
    faithfulness_score: Optional[float] = None
    safety_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the GenerationResult object into a normal dictionary.

        Called before saving the experiment result as JSON.
"""
# Convert every dataclass field into a dictionary so it can be written
# directly to a JSON file.
        return asdict(self)
