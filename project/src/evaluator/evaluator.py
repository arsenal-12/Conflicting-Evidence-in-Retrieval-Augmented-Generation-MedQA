"""
evaluator.py
============

Top-level orchestrator for the evaluation pipeline:

    Answer
      -> Claim Extraction   (claim_extractor.ClaimExtractor)
      -> Attribution        (claim_attributor.ClaimAttributor)
      -> Faithfulness       (faithfulness.determine_faithfulness)
      -> Safety             (safety.SafetyAssessor)
      -> Aggregate EvaluationRecord

This module contains no LLM-call logic of its own; it wires together the
already-implemented stage classes and handles per-claim error isolation so
that a failure evaluating one claim does not abort evaluation of the rest.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional, Sequence

from .claim_attributor import ClaimAttributionError, ClaimAttributor
from .claim_extractor import ClaimExtractionError, ClaimExtractor, LLMClient
from .faithfulness import determine_faithfulness
from .models import Claim, ClaimEvaluation, EvaluationRecord, FaithfulnessVerdict
from .safety import SafetyAssessmentError, SafetyAssessor

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot proceed at all (e.g. claim extraction failed).

    Per-claim failures in attribution/safety are handled gracefully and do
    NOT raise this; they are instead recorded on the affected
    ``ClaimEvaluation`` (see ``Evaluator.evaluate``'s ``fail_fast`` option
    for stricter behavior).
    """


class Evaluator:
    """Orchestrates the full claim-level evaluation pipeline for one answer."""

    def __init__(
        self,
        claim_extractor: ClaimExtractor,
        claim_attributor: ClaimAttributor,
        safety_assessor: SafetyAssessor,
        fail_fast: bool = False,
    ) -> None:
        """
        Args:
            claim_extractor: Stage 1 — extracts atomic claims from the answer.
            claim_attributor: Stage 2 — classifies each claim against chunks.
            safety_assessor: Stage 4 — assesses risk for non-supported claims.
            fail_fast: If True, any per-claim stage failure (attribution or
                safety) raises immediately, aborting the whole evaluation.
                If False (default), the failure is logged, the offending
                claim is recorded with ``faithfulness=UNSUPPORTED`` and a
                diagnostic reason, and evaluation continues with the
                remaining claims.
        """
        self._claim_extractor = claim_extractor
        self._claim_attributor = claim_attributor
        self._safety_assessor = safety_assessor
        self._fail_fast = fail_fast

    @classmethod
    def from_llm_client(
        cls, llm_client: LLMClient, max_retries: int = 2, fail_fast: bool = False
    ) -> "Evaluator":
        """Convenience constructor: build all three LLM-backed stages from a
        single shared ``LLMClient``.

        Args:
            llm_client: Shared client used by extraction, attribution, and
                safety assessment.
            max_retries: Retry count applied to each stage.
            fail_fast: See ``Evaluator.__init__``.
        """
        return cls(
            claim_extractor=ClaimExtractor(llm_client, max_retries=max_retries),
            claim_attributor=ClaimAttributor(llm_client, max_retries=max_retries),
            safety_assessor=SafetyAssessor(llm_client, max_retries=max_retries),
            fail_fast=fail_fast,
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_chunks: Sequence[Mapping[str, Any]],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> EvaluationRecord:
        """Run the full pipeline and return an aggregated ``EvaluationRecord``.

        Args:
            question: The original user question.
            answer: The generated answer to evaluate (post FailureInjector /
                Qwen generation, per the existing pipeline).
            retrieved_chunks: The chunks retrieved for this question, each a
                mapping with at least a ``"text"`` key and ideally a
                ``"chunk_id"`` key.
            metadata: Optional extra metadata to attach to the resulting
                ``EvaluationRecord`` (e.g. run id, injected failure type).

        Returns:
            An ``EvaluationRecord`` containing a ``ClaimEvaluation`` for
            every claim extracted from ``answer``.

        Raises:
            EvaluationError: If claim extraction itself fails (there is
                nothing to evaluate downstream). Per-claim failures in
                attribution/safety are isolated and do not raise unless
                ``fail_fast=True`` was set at construction time.
        """
        logger.info("Starting evaluation for question: %r", question)

        try:
            claims: List[Claim] = self._claim_extractor.extract(answer)
        except ClaimExtractionError as exc:
            logger.error("Claim extraction failed; aborting evaluation: %s", exc)
            raise EvaluationError(f"Claim extraction failed: {exc}") from exc

        claim_evaluations: List[ClaimEvaluation] = []
        for claim in claims:
            try:
                claim_evaluations.append(
                    self._evaluate_single_claim(question, claim, retrieved_chunks)
                )
            except (ClaimAttributionError, SafetyAssessmentError) as exc:
                if self._fail_fast:
                    raise
                logger.error(
                    "Evaluation failed for claim %r; recording as UNSUPPORTED "
                    "with diagnostic reason. Error: %s",
                    claim.text,
                    exc,
                )
                claim_evaluations.append(
                    ClaimEvaluation(
                        claim=claim,
                        attributions=[],
                        faithfulness=FaithfulnessVerdict.UNSUPPORTED,
                        safety_verdict=None,
                        severity=None,
                        reason=f"Evaluation pipeline error: {exc}",
                    )
                )

        record = EvaluationRecord(
            question=question,
            answer=answer,
            claim_evaluations=claim_evaluations,
            metadata=dict(metadata or {}),
        )

        logger.info(
            "Evaluation complete: %d claim(s), %d unfaithful, %d unsafe.",
            record.num_claims,
            record.num_unfaithful_claims,
            record.num_unsafe_claims,
        )
        return record

    def _evaluate_single_claim(
        self,
        question: str,
        claim: Claim,
        retrieved_chunks: Sequence[Mapping[str, Any]],
    ) -> ClaimEvaluation:
        """Run attribution -> faithfulness -> safety for a single claim."""
        attributions = self._claim_attributor.attribute(question, claim, retrieved_chunks)
        faithfulness = determine_faithfulness(attributions)

        claim_evaluation = ClaimEvaluation(
            claim=claim,
            attributions=attributions,
            faithfulness=faithfulness,
        )

        # SafetyAssessor internally no-ops (no LLM call) for SUPPORTED claims.
        self._safety_assessor.assess(question, claim_evaluation)
        return claim_evaluation

    def evaluate_changed_claims(
        self,
        question: str,
        old_record: EvaluationRecord,
        new_answer: str,
        retrieved_chunks: Sequence[Mapping[str, Any]],
    ) -> EvaluationRecord:
        """
        Reuse evaluations for unchanged claims.

        Only newly-added or modified claims are evaluated again.
        """

        new_claims = self._claim_extractor.extract(new_answer)

        old_by_text = {
            ce.claim.text.strip(): ce
            for ce in old_record.claim_evaluations
        }

        new_evaluations = []

        reused = 0
        reevaluated = 0

        for claim in new_claims:

            cached = old_by_text.get(claim.text.strip())

            if cached is not None:
                new_evaluations.append(cached)
                reused += 1
                continue

            new_evaluations.append(
                self._evaluate_single_claim(
                    question,
                    claim,
                    retrieved_chunks,
                )
            )

            reevaluated += 1

        logger.info(
            "Evaluator reused %d claims, reevaluated %d.",
            reused,
            reevaluated,
        )

        logger.info(
            "Claim reuse rate: %.1f%%",
            100.0 * reused / max(len(new_claims), 1),
        )

        return EvaluationRecord(
            question=question,
            answer=new_answer,
            claim_evaluations=new_evaluations,
            metadata=dict(old_record.metadata),
        )