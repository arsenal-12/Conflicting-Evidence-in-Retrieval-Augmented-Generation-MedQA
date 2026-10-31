"""
models.py
=========

Core data structures used throughout the ``evaluator`` package.

This module defines:
    - Enumerations for chunk-level attribution labels, claim-level
      faithfulness verdicts, and safety verdicts/severity levels.
    - Dataclasses representing a single extracted claim (``Claim``),
      the fully-evaluated result for a single claim (``ClaimEvaluation``),
      and the aggregated result for an entire answer (``EvaluationRecord``).

Design notes
------------
- All dataclasses are intentionally "dumb" containers: they hold data and
  provide (de)serialization helpers (``to_dict`` / ``from_dict``), but they
  do not perform any LLM calls or business logic. That logic lives in
  ``claim_extractor.py``, ``claim_attributor.py``, ``faithfulness.py``, and
  ``safety.py``.
- Enums are defined with plain string values so that serialized JSON stays
  human-readable and stable across versions.
- Every dataclass supports round-tripping through ``to_dict``/``from_dict``
  so records can be persisted as JSON alongside the existing pipeline
  output without any additional adapter code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import logging
import uuid

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class AttributionLabel(str, Enum):
    """Per-chunk classification of a claim against a single retrieved chunk."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"

    @classmethod
    def _missing_(cls, value: object) -> "AttributionLabel":
        # Be forgiving of case/whitespace variance coming back from an LLM.
        if isinstance(value, str):
            normalized = value.strip().upper()
            for member in cls:
                if member.value == normalized:
                    return member
        logger.warning(
            "Unrecognized AttributionLabel value %r; defaulting to NEUTRAL", value
        )
        return cls.NEUTRAL


class FaithfulnessVerdict(str, Enum):
    """Claim-level faithfulness verdict derived from chunk attributions."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


class SafetyVerdict(str, Enum):
    """Outcome of safety assessment for a non-supported claim."""

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value: object) -> "SafetyVerdict":
        if isinstance(value, str):
            normalized = value.strip().upper()
            for member in cls:
                if member.value == normalized:
                    return member
        logger.warning(
            "Unrecognized SafetyVerdict value %r; defaulting to UNKNOWN", value
        )
        return cls.UNKNOWN


class SeverityLevel(str, Enum):
    """Severity of a safety concern, when applicable."""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @classmethod
    def _missing_(cls, value: object) -> "SeverityLevel":
        if isinstance(value, str):
            normalized = value.strip().upper()
            for member in cls:
                if member.value == normalized:
                    return member
        logger.warning(
            "Unrecognized SeverityLevel value %r; defaulting to NONE", value
        )
        return cls.NONE


# --------------------------------------------------------------------------- #
# Chunk attribution (per-chunk result nested inside a ClaimEvaluation)
# --------------------------------------------------------------------------- #


@dataclass
class ChunkAttribution:
    """Result of classifying one claim against one retrieved chunk."""

    chunk_id: str
    chunk_text: str
    label: AttributionLabel
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "label": self.label.value,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkAttribution":
        return cls(
            chunk_id=str(data.get("chunk_id", "")),
            chunk_text=str(data.get("chunk_text", "")),
            label=AttributionLabel(data.get("label", AttributionLabel.NEUTRAL.value)),
            rationale=str(data.get("rationale", "")),
        )


# --------------------------------------------------------------------------- #
# Claim
# --------------------------------------------------------------------------- #


@dataclass
class Claim:
    """A single atomic factual claim extracted from a generated answer."""

    text: str
    claim_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_span: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "source_span": self.source_span,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        return cls(
            text=str(data.get("text", "")),
            claim_id=str(data.get("claim_id", uuid.uuid4().hex[:12])),
            source_span=data.get("source_span"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


# --------------------------------------------------------------------------- #
# ClaimEvaluation
# --------------------------------------------------------------------------- #


@dataclass
class ClaimEvaluation:
    """Full evaluation result for a single claim.

    Combines the extracted claim, its per-chunk attributions, the derived
    faithfulness verdict, and (when applicable) a safety assessment.
    """

    claim: Claim
    attributions: List[ChunkAttribution] = field(default_factory=list)
    faithfulness: Optional[FaithfulnessVerdict] = None

    # Safety fields are only populated for CONTRADICTED / UNSUPPORTED claims.
    safety_verdict: Optional[SafetyVerdict] = None
    severity: Optional[SeverityLevel] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "attributions": [a.to_dict() for a in self.attributions],
            "faithfulness": self.faithfulness.value if self.faithfulness else None,
            "safety_verdict": self.safety_verdict.value if self.safety_verdict else None,
            "severity": self.severity.value if self.severity else None,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimEvaluation":
        faithfulness = data.get("faithfulness")
        safety_verdict = data.get("safety_verdict")
        severity = data.get("severity")
        return cls(
            claim=Claim.from_dict(data.get("claim", {})),
            attributions=[
                ChunkAttribution.from_dict(a) for a in data.get("attributions", []) or []
            ],
            faithfulness=FaithfulnessVerdict(faithfulness) if faithfulness else None,
            safety_verdict=SafetyVerdict(safety_verdict) if safety_verdict else None,
            severity=SeverityLevel(severity) if severity else None,
            reason=data.get("reason"),
        )


# --------------------------------------------------------------------------- #
# EvaluationRecord
# --------------------------------------------------------------------------- #


@dataclass
class EvaluationRecord:
    """Aggregated evaluation output for a single (question, answer) pair.

    This is the top-level object produced by ``Evaluator.evaluate(...)`` and
    is intended to be serialized alongside the existing pipeline's JSON
    output (e.g. attached as an ``"evaluation"`` key).
    """

    question: str
    answer: str
    claim_evaluations: List[ClaimEvaluation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- Derived / summary helpers ---------------------------------------- #

    @property
    def num_claims(self) -> int:
        return len(self.claim_evaluations)

    @property
    def num_unfaithful_claims(self) -> int:
        return sum(
            1
            for ce in self.claim_evaluations
            if ce.faithfulness in (FaithfulnessVerdict.CONTRADICTED, FaithfulnessVerdict.UNSUPPORTED)
        )

    @property
    def num_unsafe_claims(self) -> int:
        return sum(
            1
            for ce in self.claim_evaluations
            if ce.safety_verdict == SafetyVerdict.UNSAFE
        )

    @property
    def is_faithful(self) -> bool:
        """True iff every claim resolved to SUPPORTED.

        Note: an answer with zero extractable claims (e.g. a refusal such
        as "I don't know based on the given context") returns False here,
        since there is nothing to have been faithfully supported. This is
        deliberate -- it does NOT mean the answer hallucinated. See
        `is_abstained` for that distinct case, and avoid treating a low
        `is_faithful` rate as equivalent to a high hallucination rate
        without also checking `is_abstained`.
        """
        return self.num_claims > 0 and self.num_unfaithful_claims == 0

    @property
    def is_abstained(self) -> bool:
        """True iff the answer contained no extractable factual claims at
        all (e.g. a refusal/"I don't know" response), as opposed to
        containing claims that were evaluated and found faithful or not.

        Abstention is a distinct outcome from both `is_faithful` and
        `is_unfaithful`-in-spirit: refusing to answer under insufficient
        or conflicting evidence is generally the SAFE behavior, not a
        hallucination. Report this separately rather than folding it into
        either faithfulness bucket, so that (for example) an increase in
        abstention rate under conflicting evidence is visible as its own
        finding rather than silently depressing the faithfulness rate.
        """
        return self.num_claims == 0

    @property
    def is_safe(self) -> bool:
        """True iff no claim was flagged UNSAFE."""
        return self.num_unsafe_claims == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "claim_evaluations": [ce.to_dict() for ce in self.claim_evaluations],
            "summary": {
                "num_claims": self.num_claims,
                "num_unfaithful_claims": self.num_unfaithful_claims,
                "num_unsafe_claims": self.num_unsafe_claims,
                "is_faithful": self.is_faithful,
                "is_abstained": self.is_abstained,
                "is_safe": self.is_safe,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationRecord":
        return cls(
            question=str(data.get("question", "")),
            answer=str(data.get("answer", "")),
            claim_evaluations=[
                ClaimEvaluation.from_dict(ce) for ce in data.get("claim_evaluations", []) or []
            ],
            metadata=dict(data.get("metadata", {}) or {}),
        )