"""
src/conflict_verifier.py
========================

WHY THIS MODULE EXISTS
-----------------------
When an LLM generates a conflicting version of a retrieved document, we cannot
assume the output is valid. The model might:
  - Produce an empty or truncated response
  - Change too much of the document (destroying the original context)
  - Change too little (failing to introduce a genuine contradiction)
  - Return garbled or non-medical text

This module provides a verification layer between evidence generation and
evidence caching. A generated conflict must pass verification before it is
stored in the cache and used in experiments.

WHERE IT FITS IN THE PIPELINE
-------------------------------
This module is OFFLINE ONLY. It is called inside build_cache.py, never
during runtime experiments.

Pipeline position:

    EvidenceGenerator.generate()
            ↓
    ConflictVerifier.verify()       ← this module
            ↓
    EvidenceCache.save()  (only if verification passes)

WHICH MODULES CALL IT
----------------------
    scripts/build_cache.py  → instantiates a ConflictVerifier and passes
                              it to EvidenceCache.

WHICH MODULES IT CALLS
-----------------------
    None. This module is a pure function layer with no dependencies on
    other project modules.

HOW IT CONTRIBUTES TO THE DISSERTATION
----------------------------------------
Verification serves two purposes:

1. Scientific integrity: It ensures that cached conflicts are actually
   usable as experimental stimuli. A conflict that is identical to the
   original (no change made) or radically different (style destroyed)
   would confound your results.

2. Methodological transparency: The metadata.json in each cache entry
   records which verifier was used and whether it passed. This allows
   you to report in your methodology chapter exactly what quality checks
   were applied to generated evidence.

DESIGN DECISION: PLUGGABLE VERIFICATION
-----------------------------------------
Verification is implemented as an abstract base class with concrete
subclasses. This allows you to:

  - Use a simple length-ratio check in Phase 2 (fast, no extra models)
  - Upgrade to semantic similarity verification later
  - Upgrade to MiniCheck-based contradiction verification in Phase 3
  - Add human annotation verification for a sample

None of these upgrades require changing EvidenceCache or build_cache.py.
You simply instantiate a different ConflictVerifier subclass in build_cache.py.

PHASE 2 DEFAULT: LengthRatioVerifier
--------------------------------------
For Phase 2, we use a simple word-count ratio check:
  - Generated text must be 80%–120% of the original word count
  - This ensures the model preserved document length as required
  - It is a necessary condition (right length) not a sufficient condition
    (right content). It catches catastrophic generation failures.
  - The verifier name and result are stored in metadata.json so the
    dissertation can report honestly that this is a lightweight check.

FUTURE VERIFIERS (examples)
-----------------------------
  SemanticSimilarityVerifier:
    Pass if cosine_similarity(original, generated) is between 0.7 and 0.98.
    Too low = too different. Too high = no change made.
    Requires sentence-transformers (already in requirements.txt).

  MiniCheckVerifier:
    Run MiniCheck on (generated_claim, original_document).
    Pass if MiniCheck reports the generated text is NOT supported by
    the original (i.e., a genuine contradiction exists).
    Requires MiniCheck (Phase 3 dependency).

  CompositeVerifier:
    Run multiple verifiers in sequence. Pass only if all pass.
    Easy to compose from existing verifiers.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.logger import get_logger

logger = get_logger(__name__)


# ── Verification Result ────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """
    Structured output of a ConflictVerifier.verify() call.

    Stored in metadata.json alongside each cached document so that
    the verification decision is transparent and reproducible.

    Attributes
    ----------
    passed          : True if verification passed and the generated text
                      is suitable for caching and experimental use.
    verifier_name   : Name of the verifier class that produced this result.
                      Recorded in metadata.json for methodological transparency.
    reason          : Human-readable explanation of why verification passed
                      or failed. Useful for debugging cache-building runs.
    score           : Optional numeric score from the verifier (e.g. ratio,
                      similarity score). None for verifiers that produce only
                      pass/fail without a numeric signal.
    """
    passed:         bool
    verifier_name:  str
    reason:         str
    score:          Optional[float] = None

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        score_str = f" (score={self.score:.3f})" if self.score is not None else ""
        return f"[{self.verifier_name}] {status}{score_str}: {self.reason}"


# ── Abstract Base Class ────────────────────────────────────────────────────────

class ConflictVerifier(ABC):
    """
    Abstract base class for conflict verification strategies.

    All concrete verifiers must implement verify(). This ensures that
    EvidenceCache can accept any verifier without knowing its implementation.

    This follows the Strategy Pattern: the verification algorithm is
    interchangeable without modifying the classes that use it.
    """

    @abstractmethod
    def verify(self, original_text: str, generated_text: str) -> VerificationResult:
        """
        Verify whether a generated conflicting document is suitable for use.

        Args:
            original_text  : The original retrieved document text.
            generated_text : The LLM-generated conflicting version.

        Returns:
            VerificationResult indicating pass/fail and the reason.
        """
        ...

    @property
    def name(self) -> str:
        """Return the class name for use in metadata and logging."""
        return self.__class__.__name__


# ── Phase 2 Default: LengthRatioVerifier ──────────────────────────────────────

class LengthRatioVerifier(ConflictVerifier):
    """
    Verifies that the generated document is approximately the same length
    as the original, measured in word count.

    SCIENTIFIC RATIONALE
    ---------------------
    The conflict generation prompt instructs the LLM to:
      - Preserve more than 90% of the original document
      - Preserve approximately the same length
      - Modify exactly one clinically important factual statement

    If the generated text is dramatically shorter than the original, the
    LLM likely truncated its output or produced a summary rather than a
    rewritten document. If it is dramatically longer, the LLM likely added
    unwanted explanations or commentary.

    LENGTH RATIO IS A NECESSARY BUT NOT SUFFICIENT CONDITION
    ---------------------------------------------------------
    This verifier catches catastrophic generation failures (empty output,
    truncated output, runaway generation). It does NOT verify that:
      - A genuine contradiction was introduced
      - The scientific writing style was preserved
      - Only one fact was changed

    These are desirable properties that require semantic verification
    (SemanticSimilarityVerifier, MiniCheckVerifier) which are planned
    for Phase 3.

    WHY 0.8–1.2 AS THE DEFAULT RANGE?
    -----------------------------------
    The generation prompt asks for approximately the same length.
    A 20% tolerance in either direction accommodates:
      - Natural variation in how the LLM paraphrases the unchanged content
      - Slight expansion when replacing a short claim with a longer one
      - Slight compression when replacing a longer claim with a shorter one

    Args:
        min_ratio : Minimum acceptable ratio (generated / original word count).
                    Default 0.8 (generated must be at least 80% as long).
        max_ratio : Maximum acceptable ratio.
                    Default 1.2 (generated must be at most 120% as long).
    """

    def __init__(
        self,
        min_ratio: float = 0.8,
        max_ratio: float = 1.2,
    ) -> None:
        if not (0.0 < min_ratio < max_ratio):
            raise ValueError(
                f"min_ratio must be positive and less than max_ratio. "
                f"Got min_ratio={min_ratio}, max_ratio={max_ratio}."
            )
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def verify(self, original_text: str, generated_text: str) -> VerificationResult:
        """
        Check that generated_text word count is within [min_ratio, max_ratio]
        of original_text word count.

        Args:
            original_text  : Original retrieved document text.
            generated_text : LLM-generated conflicting version.

        Returns:
            VerificationResult with pass/fail and the computed ratio.
        """
        # Guard: reject empty generated output immediately.
        # An empty string means the LLM returned nothing useful.
        if not generated_text or not generated_text.strip():
            logger.warning("Verification failed: generated text is empty.")
            return VerificationResult(
                passed=        False,
                verifier_name= self.name,
                reason=        "Generated text is empty or whitespace-only.",
                score=         0.0,
            )

        original_words  = len(original_text.split())
        generated_words = len(generated_text.split())

        # Guard: reject if original has no words (should not happen, but be safe).
        if original_words == 0:
            logger.warning("Verification skipped: original text is empty.")
            return VerificationResult(
                passed=        False,
                verifier_name= self.name,
                reason=        "Original text is empty — cannot compute ratio.",
                score=         None,
            )

        ratio = generated_words / original_words

        passed = self.min_ratio <= ratio <= self.max_ratio

        if passed:
            reason = (
                f"Length ratio {ratio:.3f} is within acceptable range "
                f"[{self.min_ratio}, {self.max_ratio}]. "
                f"Original: {original_words} words, Generated: {generated_words} words."
            )
        else:
            reason = (
                f"Length ratio {ratio:.3f} is outside acceptable range "
                f"[{self.min_ratio}, {self.max_ratio}]. "
                f"Original: {original_words} words, Generated: {generated_words} words."
            )

        logger.debug(f"LengthRatioVerifier: {reason}")

        return VerificationResult(
            passed=        passed,
            verifier_name= self.name,
            reason=        reason,
            score=         ratio,
        )


# ── Future Verifier Stubs ──────────────────────────────────────────────────────

class SemanticSimilarityVerifier(ConflictVerifier):
    """
    PLANNED FOR PHASE 3.

    Verifies that the generated document is semantically similar to the
    original (preserving topic and style) but not identical (confirming
    that a change was made).

    Uses cosine similarity between sentence embeddings:
      - Too low (< min_similarity): style or topic was destroyed
      - Too high (> max_similarity): no meaningful change was made
      - In range: genuine but targeted modification

    Requires sentence-transformers (already in requirements.txt).
    """

    def verify(self, original_text: str, generated_text: str) -> VerificationResult:
        raise NotImplementedError(
            "SemanticSimilarityVerifier is planned for Phase 3. "
            "Use LengthRatioVerifier for Phase 2."
        )


class CompositeVerifier(ConflictVerifier):
    """
    PLANNED FOR PHASE 3.

    Runs multiple ConflictVerifier instances in sequence.
    Passes only if ALL constituent verifiers pass.

    Example usage:
        verifier = CompositeVerifier([
            LengthRatioVerifier(min_ratio=0.8, max_ratio=1.2),
            SemanticSimilarityVerifier(min_similarity=0.7, max_similarity=0.98),
        ])

    The VerificationResult.reason will contain a summary from each verifier.
    """

    def __init__(self, verifiers: list[ConflictVerifier]) -> None:
        if not verifiers:
            raise ValueError("CompositeVerifier requires at least one verifier.")
        self.verifiers = verifiers

    def verify(self, original_text: str, generated_text: str) -> VerificationResult:
        raise NotImplementedError(
            "CompositeVerifier is planned for Phase 3. "
            "Use LengthRatioVerifier for Phase 2."
        )
