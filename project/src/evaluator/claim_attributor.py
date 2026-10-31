"""
src/evaluator/claim_attributor.py
==================================
Stage 2 of the evaluator pipeline.

ClaimExtractor asks : What are the facts?
ClaimAttributor asks : Which evidence supports or contradicts each fact?

For a single claim, classifies its relationship
(SUPPORTS / CONTRADICTS / NEUTRAL) against every retrieved chunk,
using one LLM call per chunk.

One call per chunk (not one call for all chunks) is intentional:
Qwen2.5-1.5B is unreliable when asked to classify multiple items
in a single call. One decision per call eliminates that failure mode.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .claim_extractor import LLMClient
from .models import AttributionLabel, ChunkAttribution, Claim
from .prompts import build_claim_attribution_prompt_v2

logger = logging.getLogger(__name__)


class ClaimAttributionError(RuntimeError):
    """Raised when attribution fails after all retries are exhausted."""


# ---------------------------------------------------------------------------
# Robust JSON parsing helpers
# ---------------------------------------------------------------------------

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

    logger.error(
        "Unable to parse JSON from LLM response "
        "(length=%d, first500=%r, last500=%r)",
        len(raw_text),
        raw_text[:500],
        raw_text[-500:],
    )
    return None


def _is_injected_conflict_chunk(chunk: Mapping[str, Any]) -> bool:
    """Return True if this chunk was injected as a controlled conflict.

    Checks for the marker set by EvidenceCache / FailureInjector.
    """
    return bool(
        chunk.get("is_conflict", False)
        or chunk.get("failure_mode") == "conflicting"
    )


# ---------------------------------------------------------------------------
# ClaimAttributor
# ---------------------------------------------------------------------------


class ClaimAttributor:
    """Classifies a claim against every retrieved chunk using an LLM.

    One LLM call is made per chunk (not per claim). This is intentional:
    small models (1.5B) are unreliable when asked to classify multiple
    items in a single call. One decision per call is more reliable.

    For chunks that are known controlled conflict injections, the
    CONTRADICTS label is assigned deterministically — the ground truth
    is known by construction so no LLM call is needed.
    """

    def __init__(self, llm_client: LLMClient, max_retries: int = 0) -> None:
        """
        Args:
            llm_client: Any object implementing the LLMClient protocol.
            max_retries: Number of additional attempts per chunk if the
                LLM response cannot be parsed. 0 means one attempt only.
        """
        self._llm_client = llm_client
        self._max_retries = max(0, max_retries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(
        self,
        question: str,
        claim: Union[Claim, str],
        retrieved_chunks: Sequence[Mapping[str, Any]],
    ) -> List[ChunkAttribution]:
        """Classify ``claim`` against every chunk in ``retrieved_chunks``.

        Args:
            question: The original user question (provides context).
            claim:    A Claim object or plain claim text string.
            retrieved_chunks: Sequence of mappings, each with at least a
                ``"text"`` key. Optionally ``"chunk_id"`` and
                ``"is_conflict"`` / ``"failure_mode"`` keys.

        Returns:
            A list of ChunkAttribution, one per input chunk, in the same
            order as ``retrieved_chunks``. Always has exactly
            ``len(retrieved_chunks)`` entries — failures fall back to
            NEUTRAL rather than raising.

        Raises:
            ClaimAttributionError: Only if a chunk fails AND
                ``max_retries`` is exhausted AND the caller has set
                strict mode. In default mode, failures log a warning
                and default to NEUTRAL.
        """
        claim_text = claim.text if isinstance(claim, Claim) else str(claim)

        # Normalise chunks: ensure stable chunk_id and string text.
        normalised: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(retrieved_chunks):
            normalised.append({
                "chunk_id":    str(chunk.get("chunk_id", f"chunk_{idx}")),
                "text":        str(chunk.get("text", "")),
                "is_conflict": chunk.get("is_conflict", False),
                "failure_mode": chunk.get("failure_mode", ""),
            })

        if not normalised:
            logger.info(
                "No retrieved chunks for claim %r; returning empty list.",
                claim_text,
            )
            return []

        results: List[ChunkAttribution] = []
        for chunk in normalised:
            attribution = self._attribute_single_chunk(
                claim_text=claim_text,
                chunk=chunk,
            )
            results.append(attribution)

        logger.info(
            "ClaimAttributor: classified %d chunk(s) for claim %r.",
            len(results),
            claim_text,
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _attribute_single_chunk(
        self,
        claim_text: str,
        chunk: Dict[str, Any],
    ) -> ChunkAttribution:
        """Classify one chunk against one claim.

        Uses a deterministic CONTRADICTS override for controlled conflict
        injections. For all other chunks, makes one LLM call.

        Args:
            claim_text: Plain text of the claim being evaluated.
            chunk:      Normalised chunk dict with keys:
                        chunk_id, text, is_conflict, failure_mode.

        Returns:
            A ChunkAttribution with the resolved label and rationale.
            Falls back to NEUTRAL with a logged warning on failure.
        """
        chunk_id   = chunk["chunk_id"]
        chunk_text = chunk["text"]

        # ── Deterministic override for injected conflict chunks ────────
        # Ground truth is known by construction — no LLM call needed.
        # Documented in dissertation methodology section.
        if _is_injected_conflict_chunk(chunk):
            logger.info(
                "chunk %s is a controlled conflict injection — "
                "assigning CONTRADICTS deterministically.",
                chunk_id,
            )
            return ChunkAttribution(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                label=AttributionLabel.CONTRADICTS,
                rationale=(
                    "Deterministic: chunk was a controlled conflict injection. "
                    "Label assigned by construction, not by LLM."
                ),
            )

        # ── LLM classification for all other chunks ────────────────────
        system_prompt, user_prompt = build_claim_attribution_prompt_v2(
            claim=claim_text,
            chunk_text=chunk_text,
        )

        last_error: Optional[str] = None

        for attempt in range(self._max_retries + 1):
            logger.debug(
                "ClaimAttributor: chunk %s attempt %d/%d",
                chunk_id, attempt + 1, self._max_retries + 1,
            )

            try:
                raw_response = self._llm_client.generate(
                    system_prompt, user_prompt
                )
            except Exception as exc:
                last_error = f"LLM call raised an exception: {exc}"
                logger.warning(
                    "chunk %s attempt %d: %s", chunk_id, attempt + 1, last_error
                )
                continue

            parsed = _safe_parse_json(raw_response)
            if parsed is None:
                last_error = "Response was not valid JSON."
                logger.warning(
                    "chunk %s attempt %d: %s", chunk_id, attempt + 1, last_error
                )
                continue

            raw_label = str(parsed.get("label", "")).strip().upper()
            rationale = str(parsed.get("rationale", ""))

            try:
                label = AttributionLabel(raw_label)
            except ValueError:
                last_error = f"Unknown label {raw_label!r}."
                logger.warning(
                    "chunk %s attempt %d: %s", chunk_id, attempt + 1, last_error
                )
                continue

            logger.debug("chunk %s → %s | %s", chunk_id, label.value, rationale)
            return ChunkAttribution(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                label=label,
                rationale=rationale,
            )

        # All retries exhausted — fall back to NEUTRAL.
        logger.warning(
            "Attribution failed for chunk %s after %d attempt(s): %s. "
            "Defaulting to NEUTRAL.",
            chunk_id, self._max_retries + 1, last_error,
        )
        return ChunkAttribution(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            label=AttributionLabel.NEUTRAL,
            rationale=(
                f"Classification failed after {self._max_retries + 1} "
                f"attempt(s): {last_error}"
            ),
        )

    @staticmethod
    def _parse_attributions_payload(
        payload: Dict[str, Any],
        normalised_chunks: List[Dict[str, str]],
    ) -> Optional[List[ChunkAttribution]]:
        """Legacy helper — kept for backwards compatibility.

        Not called by the current one-chunk-per-call implementation.
        Safe to leave in place; will be removed in a future cleanup.
        """
        raw_attributions = payload.get("attributions")
        if raw_attributions is None or not isinstance(raw_attributions, list):
            return None

        by_chunk_id: Dict[str, Dict[str, Any]] = {}
        for item in raw_attributions:
            if not isinstance(item, dict):
                logger.warning("Skipping malformed attribution entry: %r", item)
                continue
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                logger.warning(
                    "Skipping attribution entry with missing chunk_id: %r", item
                )
                continue
            by_chunk_id[str(chunk_id)] = item

        results: List[ChunkAttribution] = []
        for chunk in normalised_chunks:
            chunk_id = chunk["chunk_id"]
            item = by_chunk_id.get(chunk_id)

            if item is None:
                logger.warning(
                    "LLM did not return a classification for chunk_id=%r; "
                    "defaulting to NEUTRAL.",
                    chunk_id,
                )
                results.append(ChunkAttribution(
                    chunk_id=chunk_id,
                    chunk_text=chunk["text"],
                    label=AttributionLabel.NEUTRAL,
                    rationale="No classification returned by LLM; defaulted to NEUTRAL.",
                ))
                continue

            raw_label = (
                str(item.get("label", AttributionLabel.NEUTRAL.value))
                .strip()
                .upper()
            )
            try:
                label = AttributionLabel(raw_label)
            except ValueError:
                logger.warning(
                    "Unknown attribution label %r for chunk %s; defaulting to NEUTRAL.",
                    raw_label, chunk_id,
                )
                label = AttributionLabel.NEUTRAL

            results.append(ChunkAttribution(
                chunk_id=chunk_id,
                chunk_text=chunk["text"],
                label=label,
                rationale=str(item.get("rationale", "")),
            ))

        unknown_ids = set(by_chunk_id) - {c["chunk_id"] for c in normalised_chunks}
        if unknown_ids:
            logger.warning(
                "LLM returned attributions for unknown chunk_id(s), ignoring: %s",
                sorted(unknown_ids),
            )

        return results