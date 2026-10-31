"""
faithfulness.py
================

Stage 3 of the evaluator pipeline: derive a claim-level faithfulness
verdict from its per-chunk attributions.

This module is pure Python — no LLM calls, no I/O — by design, since the
verdict is a deterministic function of the attribution labels already
produced by ``claim_attributor.py``.

Rules (evaluated in priority order):
    1. If ANY chunk attribution is CONTRADICTS -> CONTRADICTED.
    2. Else, if ANY chunk attribution is SUPPORTS -> SUPPORTED.
    3. Else (all NEUTRAL, or no attributions at all) -> UNSUPPORTED.
"""

from __future__ import annotations

import logging
from typing import Sequence

from .models import AttributionLabel, ChunkAttribution, Claim, ClaimEvaluation, FaithfulnessVerdict

logger = logging.getLogger(__name__)


def determine_faithfulness(attributions: Sequence[ChunkAttribution]) -> FaithfulnessVerdict:
    """Apply the deterministic faithfulness rule set to a claim's attributions.

    Args:
        attributions: The per-chunk ``ChunkAttribution`` results for a
            single claim (as produced by ``ClaimAttributor.attribute``).

    Returns:
        ``FaithfulnessVerdict.CONTRADICTED`` if any attribution is
        CONTRADICTS; otherwise ``FaithfulnessVerdict.SUPPORTED`` if any
        attribution is SUPPORTS; otherwise ``FaithfulnessVerdict.UNSUPPORTED``
        (including the case of an empty attribution list, e.g. no chunks
        were retrieved for this claim).
    """
    if not attributions:
        logger.info("No attributions provided; defaulting faithfulness to UNSUPPORTED.")
        return FaithfulnessVerdict.UNSUPPORTED

    labels = [a.label for a in attributions]

    if AttributionLabel.CONTRADICTS in labels:
        verdict = FaithfulnessVerdict.CONTRADICTED
    elif AttributionLabel.SUPPORTS in labels:
        verdict = FaithfulnessVerdict.SUPPORTED
    else:
        verdict = FaithfulnessVerdict.UNSUPPORTED

    logger.debug(
        "Faithfulness verdict %s derived from labels: %s",
        verdict.value,
        [label.value for label in labels],
    )
    return verdict


class FaithfulnessEvaluator:
    """Thin, stateless wrapper around ``determine_faithfulness``.

    Provided for API symmetry with the other pipeline stages
    (``ClaimExtractor``, ``ClaimAttributor``, ``SafetyAssessor``), which are
    all classes with a primary method. Holds no state and requires no
    dependencies since faithfulness derivation is a pure function.
    """

    def evaluate(self, claim: Claim, attributions: Sequence[ChunkAttribution]) -> ClaimEvaluation:
        """Build a ``ClaimEvaluation`` for ``claim`` with faithfulness populated.

        Safety fields (``safety_verdict``, ``severity``, ``reason``) are left
        as ``None`` here; they are populated later by ``SafetyAssessor`` for
        claims that are not SUPPORTED.

        Args:
            claim: The ``Claim`` being evaluated.
            attributions: This claim's per-chunk ``ChunkAttribution`` results.

        Returns:
            A ``ClaimEvaluation`` with ``faithfulness`` set.
        """
        verdict = determine_faithfulness(attributions)
        return ClaimEvaluation(
            claim=claim,
            attributions=list(attributions),
            faithfulness=verdict,
        )