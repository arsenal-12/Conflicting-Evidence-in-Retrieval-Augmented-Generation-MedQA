"""
evaluator
=========

Post-generation evaluation module for the existing pipeline:

    Question -> Retriever -> FailureInjector -> Qwen Generator -> JSON output
                                                                       |
                                                                       v
                                                          [ evaluator (this package) ]
                                                                       |
                                                                       v
                                                             EvaluationRecord

This package is purely additive: it does not modify or depend on the
internals of the existing Retriever / FailureInjector / Generator code. It
only requires an object implementing the minimal ``LLMClient`` protocol
(``generate(system_prompt, user_prompt) -> str``) to drive its three
LLM-backed stages (claim extraction, attribution, safety assessment).

Typical usage
-------------
    from evaluator import Evaluator

    evaluator = Evaluator.from_llm_client(my_llm_client_adapter)
    record = evaluator.evaluate(
        question=question,
        answer=generated_answer,
        retrieved_chunks=retrieved_chunks,  # [{"chunk_id": ..., "text": ...}, ...]
    )
    result_json = record.to_dict()
"""

from __future__ import annotations

import logging

from .claim_attributor import ClaimAttributionError, ClaimAttributor
from .claim_extractor import ClaimExtractionError, ClaimExtractor, LLMClient
from .evaluator import EvaluationError, Evaluator
from .faithfulness import FaithfulnessEvaluator, determine_faithfulness
from .models import (
    AttributionLabel,
    ChunkAttribution,
    Claim,
    ClaimEvaluation,
    EvaluationRecord,
    FaithfulnessVerdict,
    SafetyVerdict,
    SeverityLevel,
)
from .safety import SafetyAssessmentError, SafetyAssessor

# Library-style logging: no handlers configured here. The host application
# (the existing pipeline) is responsible for configuring logging handlers
# and levels; this package only attaches a NullHandler so it never emits
# "No handlers could be found" warnings when imported in isolation.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # Orchestrator
    "Evaluator",
    "EvaluationError",
    # Stage 1: extraction
    "ClaimExtractor",
    "ClaimExtractionError",
    # Stage 2: attribution
    "ClaimAttributor",
    "ClaimAttributionError",
    # Stage 3: faithfulness
    "FaithfulnessEvaluator",
    "determine_faithfulness",
    # Stage 4: safety
    "SafetyAssessor",
    "SafetyAssessmentError",
    # Shared LLM client protocol
    "LLMClient",
    # Data models
    "Claim",
    "ChunkAttribution",
    "ClaimEvaluation",
    "EvaluationRecord",
    "AttributionLabel",
    "FaithfulnessVerdict",
    "SafetyVerdict",
    "SeverityLevel",
]

__version__ = "0.1.0"