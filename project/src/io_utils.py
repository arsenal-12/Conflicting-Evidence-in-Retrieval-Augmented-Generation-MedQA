"""
io_utils.py

Purpose
-------
Centralizes all file I/O that is not specific to a single pipeline stage:
reading JSON files, writing JSONL result records, and ensuring output
directories exist. Keeping this logic in one place means every module
serializes data the same way, and there is exactly one place to fix a bug
in, e.g., JSONL writing.

Responsibilities
-----------------
- Read a JSON file into a Python object, with clear error messages on
  malformed input.
- Append `GenerationResult` records to a JSONL results file, one JSON
  object per line, building the full experiment record shape described in
  the project specification (experiment_id, timestamp, dataset, corpus,
  model names, plus the GenerationResult fields, plus the evaluator's
  EvaluationRecord).
- Ensure a directory exists before writing to it.

Inputs
------
- File paths supplied by calling modules (never hardcoded here).
- `GenerationResult` objects (from `models.py`) to be persisted.
- The evaluator's `EvaluationRecord`, already serialized to a plain dict
  by the caller (this module intentionally does not import anything from
  `src.evaluator`, to keep I/O decoupled from evaluation logic).

Outputs
-------
- Parsed Python objects (from JSON files).
- A `.jsonl` file on disk, one line per experiment record.

Interaction with the rest of the project
-----------------------------------------
`question_loader.py` and `document_loader.py` use `read_json` to load their
respective raw source files. `main.py` uses `ensure_dir` and
`append_jsonl_result` to persist pipeline output after each generation +
evaluation step, which is more robust than buffering all results in memory
and writing once at the end (a crash midway through a long run still leaves
partial, valid results on disk).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from src.logger import get_logger
from src.models import GenerationResult
from src.utils import generate_experiment_id, utc_timestamp

logger = get_logger(__name__)


def ensure_dir(path: str) -> None:
    """Creates a directory (and any missing parents) if it does not exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Any:
    """Reads and parses a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content (typically a list or dict).

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file content is not valid JSON.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

def read_jsonl(path: str) -> list[dict]:
    """
    Reads a JSONL (JSON Lines) file.

    Each line in the file must contain one valid JSON object.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of dictionaries, one per line.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any line contains malformed JSON.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_number} in {path}: {exc}"
                ) from exc

    return records
def build_experiment_record(
    result: GenerationResult,
    original_retrieval,
    modified_retrieval,
    dataset_name: str,
    corpus_name: str,
    embedding_model: str,
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds the full experiment record dictionary for a single result.

    This assembles the exact schema specified in the project requirements:
    an experiment-level envelope (id, timestamp, dataset/corpus/model
    names) wrapping the per-question `GenerationResult` fields, plus the
    evaluator's output for that answer.

    Args:
        result: The `GenerationResult` produced by the Generator stage.
        dataset_name: Name of the question dataset used (e.g. "MedQA").
        corpus_name: Name of the retrieval corpus used (e.g. "PubMed").
        embedding_model: Name of the embedding model used.
        evaluation: The evaluator's `EvaluationRecord` for this answer,
            already serialized to a plain dict by the caller (i.e.
            `evaluation_record.to_dict()`). Stored verbatim under the
            "evaluation" key.

    Returns:
        A JSON-serializable dictionary matching the project's result
        schema.
    """
    return {
        "experiment_id": generate_experiment_id(),
        "timestamp": utc_timestamp(),
        "dataset": dataset_name,
        "corpus": corpus_name,
        "embedding_model": embedding_model,
        "generator_model": result.generator_model,
        "question": result.question,

        "original_retrieved_chunks":
            original_retrieval.retrieved_chunks,

        "modified_retrieved_chunks":
            modified_retrieval.retrieved_chunks,

        "answer": result.answer,
        "generation_time": result.generation_time,
        "failure_mode": result.failure_mode,
        "faithfulness_score": result.faithfulness_score,
        "safety_label": result.safety_label,

        "evaluation": evaluation,
    }


def append_jsonl_result(record: Dict[str, Any], output_path: str) -> None:
    """Appends a single JSON record as one line to a JSONL file.

    Writing one line per result (rather than buffering the whole run in
    memory) means a crash partway through a long experiment still leaves
    all previously-completed results safely persisted on disk.

    Args:
        record: A JSON-serializable dictionary to append.
        output_path: Path to the `.jsonl` file to append to. Created if it
            does not already exist.
    """
    ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug("Appended result for question_id=%s to %s", record.get("question_id"), output_path)