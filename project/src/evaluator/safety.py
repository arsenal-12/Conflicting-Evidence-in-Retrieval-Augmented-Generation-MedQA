"""
safety.py
=========

Stage 4 of the evaluator pipeline: assess the safety risk of claims that
were NOT fully supported by the retrieved sources.

Only claims with a faithfulness verdict of CONTRADICTED or UNSUPPORTED are
sent to the LLM for safety assessment — SUPPORTED claims are left
untouched (no LLM call, no cost) since they carry no faithfulness-related
risk by definition.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from .claim_extractor import LLMClient
from .models import ClaimEvaluation, FaithfulnessVerdict, SafetyVerdict, SeverityLevel
from .prompts import build_safety_assessment_prompt

logger = logging.getLogger(__name__)

#: Faithfulness verdicts that require a safety assessment.
_ASSESSABLE_VERDICTS = frozenset(
    {FaithfulnessVerdict.CONTRADICTED, FaithfulnessVerdict.UNSUPPORTED}
)


class SafetyAssessmentError(RuntimeError):
    """Raised when safety assessment fails after all retries are exhausted."""


# --------------------------------------------------------------------------- #
# Robust JSON parsing helpers (kept local/self-contained; mirrors the pattern
# used in claim_extractor.py and claim_attributor.py).
# --------------------------------------------------------------------------- #

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text).strip()


def _extract_json_object(text: str) -> Optional[str]:
    match = _JSON_OBJECT_RE.search(text)
    return match.group(0) if match else None


def _safe_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    candidate = _strip_code_fences(raw_text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    extracted = _extract_json_object(candidate)
    if extracted is not None:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError as exc:
            logger.debug("Failed to parse extracted JSON block: %s", exc)

    logger.error("Unable to parse JSON from LLM response: %r", raw_text[:500])
    return None


# --------------------------------------------------------------------------- #
# SafetyAssessor
# --------------------------------------------------------------------------- #


class SafetyAssessor:
    """Assesses safety risk for claims that failed to be fully SUPPORTED."""

    def __init__(self, llm_client: LLMClient, max_retries: int = 2) -> None:
        """
        Args:
            llm_client: Any object implementing ``LLMClient``.
            max_retries: Number of additional attempts if the LLM response
                cannot be parsed as valid JSON matching the expected schema.
        """
        self._llm_client = llm_client
        self._max_retries = max(0, max_retries)

    def assess(self, question: str, claim_evaluation: ClaimEvaluation) -> ClaimEvaluation:
        """Populate safety fields on ``claim_evaluation`` in place, if applicable.

        Args:
            question: The original user question (provides context to the LLM).
            claim_evaluation: A ``ClaimEvaluation`` that already has
                ``faithfulness`` populated (see ``faithfulness.py``).

        Returns:
            The same ``ClaimEvaluation`` instance, mutated in place: for
            SUPPORTED claims, safety fields are left as ``None``
            (unchanged); for CONTRADICTED/UNSUPPORTED claims,
            ``safety_verdict``, ``severity``, and ``reason`` are populated.

        Raises:
            SafetyAssessmentError: If the LLM response could not be parsed
                into the expected schema after ``max_retries`` attempts.
            ValueError: If ``claim_evaluation.faithfulness`` is unset (the
                faithfulness stage must run before the safety stage).
        """
        if claim_evaluation.faithfulness is None:
            raise ValueError(
                "ClaimEvaluation.faithfulness must be set before safety assessment "
                "(run faithfulness.determine_faithfulness first)."
            )

        if claim_evaluation.faithfulness not in _ASSESSABLE_VERDICTS:
            logger.debug(
                "Skipping safety assessment for claim %r (faithfulness=%s).",
                claim_evaluation.claim.text,
                claim_evaluation.faithfulness.value,
            )
            return claim_evaluation

        claim_text = claim_evaluation.claim.text
        attribution_dicts = [
            {
                "chunk_id": a.chunk_id,
                "label": a.label.value,
                "rationale": a.rationale,
            }
            for a in claim_evaluation.attributions
        ]

        system_prompt, user_prompt = build_safety_assessment_prompt(
            question=question,
            claim=claim_text,
            faithfulness_verdict=claim_evaluation.faithfulness.value,
            attributions=attribution_dicts,
        )

        last_error: Optional[str] = None
        for attempt in range(self._max_retries + 1):
            logger.debug(
                "SafetyAssessor: attempt %d/%d for claim %r",
                attempt + 1,
                self._max_retries + 1,
                claim_text,
            )
            try:
                raw_response = self._llm_client.generate(system_prompt, user_prompt)
            except Exception as exc:  # pragma: no cover - defensive, client-specific
                last_error = f"LLM call raised an exception: {exc}"
                logger.warning(last_error)
                continue

            parsed = _safe_parse_json(raw_response)
            if parsed is None:
                last_error = "LLM response was not valid JSON."
                continue

            result = self._parse_safety_payload(parsed)
            if result is None:
                last_error = f"LLM response JSON did not match expected schema: {parsed!r}"
                logger.warning(last_error)
                continue

            safety_verdict, severity, reason = result
            claim_evaluation.safety_verdict = safety_verdict
            claim_evaluation.severity = severity
            claim_evaluation.reason = reason

            logger.info(
                "SafetyAssessor verdict=%s severity=%s for claim %r.",
                safety_verdict.value,
                severity.value,
                claim_text,
            )
            return claim_evaluation

        raise SafetyAssessmentError(
            f"Safety assessment failed after {self._max_retries + 1} attempt(s) "
            f"for claim {claim_text!r}: {last_error}"
        )

    @staticmethod
    def _parse_safety_payload(
        payload: Dict[str, Any]
    ) -> Optional[tuple[SafetyVerdict, SeverityLevel, str]]:
        """Validate and convert a parsed JSON payload into safety fields.

        Returns ``None`` if required keys are missing from the payload.
        """
        if "safety_verdict" not in payload or "severity" not in payload:
            return None

        safety_verdict = SafetyVerdict(payload.get("safety_verdict"))
        severity = SeverityLevel(payload.get("severity"))
        reason = str(payload.get("reason", "")).strip()

        return safety_verdict, severity, reason