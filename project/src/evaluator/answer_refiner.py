"""
answer_refiner.py
=================

Rewrites answers containing unsupported claims while preserving supported
information. Uses the same LLMClient interface as the rest of the evaluator.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from .claim_extractor import LLMClient
from .models import EvaluationRecord, FaithfulnessVerdict
from .prompts import build_answer_refinement_prompt

logger = logging.getLogger(__name__)


class AnswerRefinementError(RuntimeError):
    """Raised when answer refinement fails."""


class AnswerRefiner:
    """LLM-based answer refinement."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_retries: int = 2,
    ) -> None:
        self._llm_client = llm_client
        self._max_retries = max(0, max_retries)

    def refine(
        self,
        question: str,
        original_answer: str,
        evaluation_record: EvaluationRecord,
        retrieved_chunks: Sequence[Mapping[str, str]],
    ) -> str:
        
        """
        Rewrite the answer using only supported evidence.
        """
        # Deterministic fast path:
        # If every extracted claim is unsupported/contradicted and there is
        # no supporting evidence for any claim, the LLM must not be asked
        # to invent a replacement answer.
        #
        # In this situation the only evidence-grounded response is an
        # explicit insufficiency statement.

        all_claims_unsupported = (
            bool(evaluation_record.claim_evaluations)
            and all(
                claim_eval.faithfulness
                in (
                    FaithfulnessVerdict.UNSUPPORTED,
                    FaithfulnessVerdict.CONTRADICTED,
                )
                for claim_eval in evaluation_record.claim_evaluations
            )
        )

        has_supporting_evidence = any(
            attr.label.value == "SUPPORTS"
            for claim_eval in evaluation_record.claim_evaluations
            for attr in claim_eval.attributions
        )

        if all_claims_unsupported and not has_supporting_evidence:
            logger.info(
                "Answer refinement fast path: all claims are unsupported/"
                "contradicted and no supporting evidence exists. "
                "Returning insufficiency statement."
            )
            return "The provided evidence is insufficient to determine the answer."
        unsupported_claims = []

        for claim_eval in evaluation_record.claim_evaluations:
            if claim_eval.faithfulness is not FaithfulnessVerdict.SUPPORTED:
                unsupported_claims.append(
                    f"- {claim_eval.claim.text}"
                )

        unsupported_claims_text = "\n".join(
            unsupported_claims
        ) or "None"

        evidence = []

        for idx, chunk in enumerate(retrieved_chunks):
            evidence.append(
                f"Chunk {idx + 1}:\n{chunk['text']}"
            )

        evidence_text = "\n\n".join(evidence)

        system_prompt, user_prompt = build_answer_refinement_prompt(
            question=question,
            answer=original_answer,
            unsupported_claims=unsupported_claims_text,
            evidence=evidence_text,
        )

        last_error = None

        for _ in range(self._max_retries + 1):

            try:
                refined_answer = self._llm_client.generate(
                    system_prompt,
                    user_prompt,
                )

                refined_answer = refined_answer.strip()

                if refined_answer:
                    logger.info(
                        "Answer refinement completed."
                    )
                    return refined_answer

            except Exception as exc:
                last_error = str(exc)

        raise AnswerRefinementError(
            last_error or "Answer refinement failed."
        )