"""
===============================================================================
question_loader.py
===============================================================================

Purpose
-------
Loads the raw question dataset and converts every entry into our standard
`Question` object.

Every dataset has its own structure (MedQA, PubMedQA, TriviaQA, etc.).
This file hides those differences so the rest of the pipeline always works
with the same `Question` model.

Pipeline Position
-----------------
config.yaml
      ↓
QuestionLoader   ← You are here
      ↓
Retriever
      ↓
Generator

Responsibilities
----------------
• Read the dataset file.
• Validate each question.
• Convert raw JSON into Question objects.
• Return a clean list of questions to the pipeline.

Why this file exists
--------------------
Without this loader, every other module would need to understand the format
of every dataset.

Instead, we convert everything into one common format here, so the rest of
the pipeline stays simple.

When should I modify this file?
-------------------------------
Modify this file if:

• You want to support a new dataset.
• The dataset JSON structure changes.
• You need additional fields inside the Question object.

Do NOT modify this file if you want to change:

• Retrieval
• Embeddings
• LLM generation
• Evaluation

Those belong in their own modules.

How to add a new dataset
------------------------
1. Write a new parser function.
2. Register it inside `_PARSERS`.
3. Change `dataset.loader` in config.yaml.

No other file needs to change.

Software Engineering
--------------------
✓ Single Responsibility Principle
    Only responsible for loading questions.

✓ Open/Closed Principle
    New datasets are added without changing existing code.

✓ Modularity
    Other modules never need to know which dataset is being used.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.io_utils import read_jsonl
from src.logger import get_logger
from src.models import Question

logger = get_logger(__name__)


class QuestionLoadError(Exception):
    """Raised when a question dataset entry is malformed or unparseable."""


def _require_field(entry: Dict[str, Any], field_name: str, entry_index: int) -> Any:
    """"
Safely read a required field from a dataset entry.

This helper keeps validation in one place so every parser doesn't have to
repeat the same checks.

If a required field is missing, we fail immediately with a clear error
instead of letting the pipeline crash later.
"""
    
    value = entry.get(field_name)
    # An empty string is treated the same as a missing field: a question
    # with an empty "question" field is just as unusable as one with no
    # "question" key at all.
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise QuestionLoadError(
            f"Entry at index {entry_index} is missing required field '{field_name}': {entry}"
        )
    return value


def _parse_medqa(entry: Dict[str, Any], entry_index: int) -> Question:
    """

Convert one MedQA JSON entry into our standard Question object.

This is the only place that knows the MedQA JSON structure.

The rest of the project never works directly with raw JSON.
It only works with Question objects.
    """
    question_id = str(entry_index)

    question_text = _require_field(entry, "question", entry_index)
    ground_truth = _require_field(entry, "answer", entry_index)

    metadata = {
        "options": entry.get("options", {}),
        "answer_idx": entry.get("answer_idx"),
        "meta_info": entry.get("meta_info"),
    }

    return Question(
        question_id=str(question_id),
        question=str(question_text),
        ground_truth_answer=str(ground_truth),
        metadata=metadata,
    )


# Registry mapping `dataset.loader` config values to parser functions.
# Adding a new dataset format is a one-line addition here plus a new
# `_parse_<name>` function -- see module docstring.
_PARSERS: Dict[str, Callable[[Dict[str, Any], int], Question]] = {
    "medqa": _parse_medqa,
}


class QuestionLoader:
    """
    Loads questions using the parser selected in config.yaml.

    The rest of the pipeline only calls `load()`.
    It doesn't need to know which dataset is being used.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Read the dataset configuration and prepare the correct parser.

        Nothing is loaded yet.

        This constructor only stores configuration and decides which parser
        should be used later.
        """
        
        dataset_cfg = config["dataset"]
        self.path: str = dataset_cfg["path"]
        self.loader_name: str = dataset_cfg["loader"]
        self.sample_size: Optional[int] = dataset_cfg.get("sample_size")

        if self.loader_name not in _PARSERS:
            raise ValueError(
                f"Unknown dataset loader '{self.loader_name}'. "
                f"Available loaders: {sorted(_PARSERS.keys())}"
            )
        self._parse_entry = _PARSERS[self.loader_name]

    def load(self) -> List[Question]:
        """
        Load the dataset from disk and return validated Question objects.

        Every question is checked before entering the pipeline.

        If any entry is invalid, stop immediately so problems are caught early
        instead of causing confusing errors later.
        """
        raw_data = read_jsonl(self.path)

        if not isinstance(raw_data, list):
            raise ValueError(
                f"Expected a JSON array of question entries in {self.path}, "
                f"got {type(raw_data).__name__}."
            )

        questions: List[Question] = []
        for index, raw_entry in enumerate(raw_data):
            if not isinstance(raw_entry, dict):
                raise QuestionLoadError(
                    f"Entry at index {index} is not a JSON object: {raw_entry!r}"
                )
            questions.append(self._parse_entry(raw_entry, index))

        if self.sample_size is not None:
            questions = questions[: self.sample_size]

        logger.info(
            "Loaded %d questions from %s using loader '%s'.",
            len(questions),
            self.path,
            self.loader_name,
        )
        return questions
