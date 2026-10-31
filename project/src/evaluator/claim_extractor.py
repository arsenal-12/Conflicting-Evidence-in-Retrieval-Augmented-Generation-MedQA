"""
claim_extractor.py
===================

Stage 1 of the evaluator pipeline: decompose a generated answer into a list
of atomic factual claims via an LLM call.

This module intentionally does not depend on any concrete LLM client
implementation (e.g. the existing Qwen generator). Instead it defines a
minimal ``LLMClient`` protocol so it can be wired up to whatever generator
the surrounding project already uses, without modifying existing code.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .models import Claim
from .prompts import build_claim_extraction_prompt

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface required of any LLM client used by the evaluator.

    Any existing generator wrapper can be adapted to this protocol by
    providing a ``generate(system_prompt, user_prompt) -> str`` method
    (or wrapped in a thin adapter) without touching its original code.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw text completion for the given prompt pair."""
        ...


class ClaimExtractionError(RuntimeError):
    """Raised when claim extraction fails after all retries are exhausted."""


# --------------------------------------------------------------------------- #
# Robust JSON parsing helpers
# --------------------------------------------------------------------------- #

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences, if present."""
    return _CODE_FENCE_RE.sub("", text).strip()


def _extract_json_object(text: str) -> Optional[str]:
    """Best-effort extraction of the first top-level JSON object substring."""
    match = _JSON_OBJECT_RE.search(text)
    return match.group(0) if match else None


def _safe_parse_json(raw_text: str) -> Optional[Any]:
    """Attempt to parse ``raw_text`` as a JSON object, tolerating common
    LLM formatting quirks (markdown fences, leading/trailing prose).

    Returns ``None`` if parsing fails after all fallback strategies.
    """
    candidate = _strip_code_fences(raw_text)

    # Attempt 1: direct parse.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the first {...} block and parse that.
    extracted = _extract_json_object(candidate)
    if extracted is not None:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError as exc:
            logger.debug("Failed to parse extracted JSON block: %s", exc)

    logger.error("Unable to parse JSON from LLM response: %r", raw_text[:500])
    return None


# --------------------------------------------------------------------------- #
# ClaimExtractor
# --------------------------------------------------------------------------- #


class ClaimExtractor:
    """Extracts atomic factual claims from a generated answer using an LLM."""

    def __init__(self, llm_client: LLMClient, max_retries: int = 2) -> None:
        """
        Args:
            llm_client: Any object implementing ``LLMClient`` (a
                ``generate(system_prompt, user_prompt) -> str`` method).
            max_retries: Number of additional attempts if the LLM response
                cannot be parsed as valid JSON matching the expected schema.
        """
        self._llm_client = llm_client
        self._max_retries = max(0, max_retries)

    def extract(self, answer: str) -> List[Claim]:
        """Extract atomic claims from ``answer``.

        Args:
            answer: The generated answer text to decompose.

        Returns:
            A list of ``Claim`` objects. Returns an empty list if the
            answer is empty/whitespace-only, or if extraction ultimately
            yields no claims.

        Raises:
            ClaimExtractionError: If the LLM response could not be parsed
                into the expected schema after ``max_retries`` attempts.
        """
        if not answer or not answer.strip():
            logger.info("Empty answer provided to ClaimExtractor; returning no claims.")
            return []

        system_prompt, user_prompt = build_claim_extraction_prompt(answer)

        last_error: Optional[str] = None
        for attempt in range(self._max_retries + 1):
            logger.debug(
                "ClaimExtractor: attempt %d/%d", attempt + 1, self._max_retries + 1
            )
            try:
                raw_response = self._llm_client.generate(system_prompt, user_prompt)
                logger.debug("Raw LLM response: %s", raw_response)
            except Exception as exc:  # pragma: no cover - defensive, client-specific
                last_error = f"LLM call raised an exception: {exc}"
                logger.warning(last_error)
                continue

            parsed = _safe_parse_json(raw_response)
            if parsed is None:
                last_error = "LLM response was not valid JSON."
                continue

            claims = self._parse_claims_payload(parsed)
            if claims is None:
                last_error = f"LLM response JSON did not match expected schema: {parsed!r}"
                logger.warning(last_error)
                continue

            logger.info("ClaimExtractor extracted %d claim(s).", len(claims))
            return claims

        raise ClaimExtractionError(
            f"Claim extraction failed after {self._max_retries + 1} attempt(s): {last_error}"
        )

    @staticmethod
    def _parse_claims_payload(payload: Any) -> Optional[List[Claim]]:
        """Validate and convert a parsed JSON payload into ``Claim`` objects.

        Accepts the intended schema, ``{"claims": [{"text": ...,
        "source_span": ...}, ...]}``, and also tolerates a bare JSON array
        (``[{"text": ..., "source_span": ...}, ...]``) with no wrapping
        key — small instruct models occasionally drop the wrapper despite
        the prompt's explicit schema. Both shapes carry the same
        information, so both are accepted here without changing the
        intended schema or any other behavior.

        Returns ``None`` if the payload conforms to neither shape.
        """
        if isinstance(payload, list):
            raw_claims = payload
        elif isinstance(payload, dict):
            raw_claims = payload.get("claims")
        else:
            return None

        if raw_claims is None or not isinstance(raw_claims, list):
            return None

        claims: List[Claim] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                logger.warning("Skipping malformed claim entry (not an object): %r", item)
                continue
            text = item.get("text")
            if not text or not isinstance(text, str) or not text.strip():
                logger.warning("Skipping claim entry with empty/missing text: %r", item)
                continue
            source_span = item.get("source_span")
            if source_span is not None and not isinstance(source_span, str):
                source_span = None
            claims.append(Claim(text=text.strip(), source_span=source_span))

        return claims