"""
===============================================================================
test_question_loader.py
===============================================================================

Purpose
-------
Verify that QuestionLoader behaves correctly under both normal and error
conditions.

This file ensures that questions are loaded correctly from a dataset and
that invalid inputs are rejected with clear exceptions instead of causing
unexpected failures later in the pipeline.

Pipeline Position
-----------------
Question Dataset
        │
        ▼
QuestionLoader
        │
        ▼
Question Objects

What this file tests
--------------------
✓ Loading a valid dataset.

✓ Converting raw JSON into Question objects.

✓ Respecting the configured sample_size.

✓ Detecting malformed dataset entries.

✓ Rejecting unknown dataset loader names.

✓ Raising an error when the dataset file does not exist.

Why this file exists
--------------------
The QuestionLoader is the first stage of the RAG pipeline.

If questions are loaded incorrectly, every downstream component
(Retriever, Generator, Evaluation) will also receive incorrect input.

These tests ensure that QuestionLoader is reliable before the rest of the
pipeline depends on it.

Test Strategy
-------------
Each test focuses on one specific behaviour.

Temporary files and configuration are provided by fixtures from
`conftest.py`, ensuring that no real datasets or project files are modified
during testing.

Software Engineering Principles
-------------------------------
✓ Unit Testing
    Tests one module (QuestionLoader) in isolation.

✓ Test Isolation
    Uses temporary files and configurations instead of real project data.

✓ Fail Fast
    Confirms that invalid configurations and malformed datasets produce
    clear exceptions immediately.

✓ Repeatability
    Every test produces the same result regardless of the machine or
    environment.
"""

from __future__ import annotations

import pytest

from src.models import Question
from src.question_loader import QuestionLoader, QuestionLoadError


def test_load_returns_expected_questions(base_config):
    """A well-formed MedQA dataset should parse into Question objects."""

    loader = QuestionLoader(base_config)
    questions = loader.load()

    assert len(questions) == 2
    assert all(isinstance(q, Question) for q in questions)

    first = questions[0]

    assert first.question_id == "0"

    assert first.question.startswith(
        "A junior orthopaedic surgery resident"
    )

    assert (
        first.ground_truth_answer
        == "Tell the attending that he cannot fail to disclose this mistake"
    )

    assert first.metadata["answer_idx"] == "C"
    assert first.metadata["meta_info"] == "step1"
    assert first.metadata["options"]["C"] == (
        "Tell the attending that he cannot fail to disclose this mistake"
    )


def test_sample_size_truncates_questions(base_config):
    base_config["dataset"]["sample_size"] = 1

    loader = QuestionLoader(base_config)
    questions = loader.load()

    assert len(questions) == 1
    assert questions[0].question_id == "0"


def test_malformed_entry_raises_question_load_error(base_config, malformed_questions_file):
    """An entry missing a required field should raise QuestionLoadError."""
    base_config["dataset"]["path"] = malformed_questions_file
    loader = QuestionLoader(base_config)

    with pytest.raises(QuestionLoadError):
        loader.load()


def test_unknown_loader_raises_value_error(base_config):
    """Configuring an unregistered loader name should fail fast at construction."""
    base_config["dataset"]["loader"] = "not_a_real_loader"

    with pytest.raises(ValueError):
        QuestionLoader(base_config)


def test_missing_file_raises_file_not_found_error(base_config):
    """Pointing at a nonexistent dataset file should raise FileNotFoundError."""
    base_config["dataset"]["path"] = "/nonexistent/path/questions.json"
    loader = QuestionLoader(base_config)

    with pytest.raises(FileNotFoundError):
        loader.load()
