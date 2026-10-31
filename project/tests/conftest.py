from __future__ import annotations

import json
from typing import Any, Dict

import pytest


@pytest.fixture
def sample_questions_file(tmp_path) -> str:
    """
    Creates a temporary valid MedQA JSONL file.
    """

    questions = [
        {
            "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair...",
            "answer": "Tell the attending that he cannot fail to disclose this mistake",
            "options": {
                "A": "Disclose the error to the patient but leave it out of the operative report",
                "B": "Disclose the error to the patient and put it in the operative report",
                "C": "Tell the attending that he cannot fail to disclose this mistake",
                "D": "Report the physician to the ethics committee",
                "E": "Refuse to dictate the operative report",
            },
            "meta_info": "step1",
            "answer_idx": "C",
        },
        {
            "question": "What is the first-line treatment for hypertension?",
            "answer": "ACE inhibitors",
            "options": {
                "A": "ACE inhibitors",
                "B": "Beta blockers",
                "C": "Diuretics",
                "D": "Calcium channel blockers",
            },
            "meta_info": "step2",
            "answer_idx": "A",
        },
    ]

    path = tmp_path / "questions.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    return str(path)


@pytest.fixture
def malformed_questions_file(tmp_path) -> str:
    """
    Creates an invalid MedQA JSONL file.
    """

    questions = [
        {
            "question": "Valid question",
            "answer": "Correct answer",
            "options": {"A": "Correct answer"},
            "answer_idx": "A",
        },
        {
            # Missing "question"
            "answer": "ACE inhibitors",
            "options": {"A": "ACE inhibitors"},
            "answer_idx": "A",
        },
    ]

    path = tmp_path / "malformed_questions.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    return str(path)


@pytest.fixture
def sample_corpus_dir(tmp_path) -> str:
    """
    Creates a temporary corpus directory containing a few valid PubMed
    document files.

    Used by DocumentLoader and other pipeline components during testing.
    """
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    docs = [
        {
            "document_id": "doc1",
            "title": "Metformin mechanism review",
            "text": "Metformin increases insulin sensitivity in peripheral tissues.",
            "source": "PubMed",
            "year": 2020,
        },
        {
            "document_id": "doc2",
            "title": "ACE inhibitors in hypertension",
            "text": "ACE inhibitors block conversion of angiotensin I to angiotensin II.",
            "source": "PubMed",
            "year": 2019,
        },
    ]

    for doc in docs:
        (corpus_dir / f"{doc['document_id']}.json").write_text(
            json.dumps(doc),
            encoding="utf-8",
        )

    return str(corpus_dir)


@pytest.fixture
def base_config(
    tmp_path,
    sample_questions_file,
    sample_corpus_dir,
) -> Dict[str, Any]:
    """
    Builds a reusable temporary configuration for testing.

    Every path points to temporary files and folders so that tests never
    modify the real dataset, vector database, logs, or results.
    """
    vector_db_dir = tmp_path / "vector_db"
    output_dir = tmp_path / "results"
    log_dir = tmp_path / "logs"

    return {
        "dataset": {
            "name": "MedQA",
            "loader": "medqa",
            "path": sample_questions_file,
            "sample_size": None,
        },
        "corpus": {
            "name": "PubMed",
            "loader": "pubmed",
            "path": sample_corpus_dir,
        },
        "embedding": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "device": "cpu",
            "batch_size": 8,
            "normalize_embeddings": True,
        },
        "vector_store": {
            "backend": "chroma",
            "persist_directory": str(vector_db_dir),
            "collection_name": "test_collection",
            "distance_metric": "cosine",
        },
        "retriever": {
            "top_k": 2,
        },
        "generator": {
            "model_name": "mock-model",
            "provider": "mock",
            "temperature": 0.2,
            "max_new_tokens": 64,
            "top_p": 0.9,
            "device": "cpu",
            "prompt_template": (
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            ),
        },
        "experiment": {
            "output_dir": str(output_dir),
            "results_filename": "results.jsonl",
        },
        "logging": {
            "log_dir": str(log_dir),
            "log_filename": "test.log",
            "level": "DEBUG",
        },
    }