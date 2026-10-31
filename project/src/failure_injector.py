"""
src/failure_injector.py
=======================

WHY THIS MODULE EXISTS
-----------------------
The FailureInjector is the runtime component that introduces controlled
retrieval failures into the RAG pipeline. It sits between the Retriever
and the Generator, intercepting a RetrievalResult and optionally inserting
cached conflicting evidence before the result reaches the LLM.

This module is intentionally thin. It has exactly one responsibility:
reading pre-generated conflicting evidence from the EvidenceCache and
inserting it into the retrieved chunk list at the correct position.

It does NOT:
  - Generate conflicting evidence (that is EvidenceGenerator's job, offline)
  - Store or manage the evidence cache (that is EvidenceCache's job)
  - Know anything about the LLM, prompting, or generation

This separation keeps the runtime pipeline free of any API calls,
network requests, or LLM dependencies. All conflicting evidence is
generated offline by scripts/build_cache.py and stored in the cache
before any experiments are run.

WHERE IT FITS IN THE PIPELINE
-------------------------------

    Question
        ↓
    Retriever
        ↓  → RetrievalResult
        ↓       retrieved_chunks: List[str]   ← plain text strings
        ↓       retrieval_scores: List[float]
        ↓
  ★ FailureInjector.inject() ★          ← this module
        ↓  → modified RetrievalResult
        ↓       retrieved_chunks: List[str]   ← with conflict inserted
        ↓       retrieval_scores: List[float] ← with 0.0 for conflict
        ↓
    Generator.generate(retrieval_result, failure_mode=...)
        ↓  → GenerationResult
        ↓
    Faithfulness + Safety Assessment
        ↓
    Conflict Analysis + Evidence Refinement
        ↓
    Final Answer

WHICH MODULES CALL IT
----------------------
    src/main.py  →  instantiates FailureInjector and calls inject() per question

WHICH MODULES IT CALLS
-----------------------
    src/evidence_cache.py  →  EvidenceCache.get() for cached conflict text
    src/logger.py          →  get_logger()

HOW IT CONTRIBUTES TO THE DISSERTATION
----------------------------------------
The FailureInjector operationalises the controlled retrieval failure
conditions that are central to the dissertation's experimental design.

  Baseline condition:    inject() returns the original RetrievalResult
                         completely unchanged.

  Conflicting condition: inject() returns a new RetrievalResult where
                         retrieved_chunks and retrieval_scores have been
                         modified to include the cached conflicting passage.

This module only modifies two fields of RetrievalResult:
    retrieved_chunks  — the conflicting passage is inserted as a plain string
    retrieval_scores  — a score of 0.0 is inserted at the same position

RetrievalResult has no failure_mode field. The experiment's failure mode
is supplied separately by main.py when calling:

    generator.generate(
        retrieval_result,
        failure_mode=failure_cfg["mode"],
    )

This design keeps RetrievalResult a clean retrieval-only object and
delegates experiment labelling to the generation stage.

DATA MODEL
-----------
This module operates directly on List[str]. There is no Chunk dataclass
in this project. Each element of retrieved_chunks is a plain text string
representing one retrieved passage.

    retrieved_chunks: List[str]   ← list of passage text strings
    retrieval_scores: List[float] ← parallel list of cosine similarity scores

The injected conflicting passage is also a plain string returned by
EvidenceCache.get(). It is inserted into the list at the correct
position without wrapping in any object.

DESIGN DECISIONS
-----------------
IMMUTABILITY:
    inject() always returns a NEW RetrievalResult using dataclasses.replace()
    when modifications are made. The original RetrievalResult is never
    modified. In baseline mode (no changes), the original object is
    returned directly.

APPEND STRATEGY (Phase 2):
    The conflicting passage is inserted immediately after the original
    highest-ranked passage. This simulates a realistic scenario where
    both a correct document and a conflicting one are present in the
    context, forcing the generator to resolve the conflict.

    Before: [D1, D2, D3, D4, D5]
    After:  [D1, D1_conflicting, D2, D3, D4, D5]

TOP-RANK SELECTION:
    The top-ranked passage (index 0) is selected for injection.
    ASSUMPTION: The highest-ranked passage is most likely to influence
    the generator, consistent with retrieval scoring and position bias
    findings (Liu et al., 2023, "Lost in the Middle").

INTENSITY:
    intensity (0.0–1.0) controls what fraction of retrieved passages
    receive a conflicting counterpart. Default 0.2 = inject into 1 of 5.
    ceil() ensures at least 1 passage is always injected when enabled.

INJECTED SCORE:
    Injected passages receive a retrieval score of 0.0.
    This is distinguishable from all real retrieval scores (cosine
    similarity > 0) and allows downstream analysis to identify
    injected passages by their score alone.

CACHE KEY MISMATCH — IMPORTANT NOTE
-------------------------------------
This module derives cache keys from passage text using SHA-256 hashing
(_stable_key). However, scripts/build_cache.py stores cache entries using
document IDs (e.g. "pubmed_001"), not text hashes.

These two strategies are INCOMPATIBLE. The current _get_cached_conflicting_text()
method will always produce cache misses at runtime because the keys it
generates (SHA-256 of passage text) will never match the keys build_cache.py
writes (document_id strings).

Resolution options (choose one before running experiments):
  Option A — Change failure_injector to use document IDs:
      Requires main.py and retriever to pass document IDs alongside chunk
      texts, e.g. by adding a parallel retrieved_doc_ids: List[str] field
      to RetrievalResult.
  Option B — Change build_cache.py to store by text hash:
      Replace document_id keys in build_cache.py with _stable_key(doc.text)
      so both sides use the same hashing strategy.

Until this is resolved, set failure_injection.enabled: false in config.yaml.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Tuple
from types import SimpleNamespace

from src.evidence_cache import EvidenceCache, CacheMissError
from src.logger import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Score assigned to injected conflicting passages.
# 0.0 is distinguishable from all genuine retrieval scores.
INJECTED_CHUNK_SCORE: float = 0.0

# Supported failure modes — extend this set when adding new modes.
SUPPORTED_MODES: frozenset[str] = frozenset({
    "baseline",
    "conflicting",
})

# Supported injection strategies — extend when adding REPLACE etc.
SUPPORTED_STRATEGIES: frozenset[str] = frozenset({
    "append",
})

# Supported document selection strategies.
SUPPORTED_SELECTION_STRATEGIES: frozenset[str] = frozenset({
    "top_rank",
})


# ── FailureInjector ────────────────────────────────────────────────────────────

class FailureInjector:
    """
    Intercepts a RetrievalResult and injects cached conflicting passages
    according to the configured mode, strategy, and intensity.

    In "baseline" mode:    returns the original RetrievalResult unchanged.
    In "conflicting" mode: returns a new RetrievalResult with cached
                           conflicting text inserted into retrieved_chunks
                           and 0.0 inserted into retrieval_scores at the
                           same position.

    RetrievalResult has no failure_mode field. The experiment condition
    label is passed separately to generator.generate() by main.py.

    IMPORTANT: This class never calls EvidenceGenerator or makes any API
    calls. All conflicting evidence must be pre-generated offline by
    scripts/build_cache.py before experiments are run.

    Args:
        cfg   : The failure_injection section of config.yaml (SimpleNamespace).
        cache : Pre-initialised EvidenceCache instance (read-only at runtime).

    Expected config attributes:
        cfg.enabled              (bool)   — whether injection is active
        cfg.mode                 (str)    — "baseline" | "conflicting"
        cfg.strategy             (str)    — "append"
        cfg.selection_strategy   (str)    — "top_rank"
        cfg.intensity            (float)  — fraction of retrieved passages to inject into

    Public API (preserved for main.py compatibility):
        __init__(cfg, cache)
        inject(retrieval_result) -> RetrievalResult
    """

    def __init__(self, cfg: SimpleNamespace, cache: EvidenceCache) -> None:
        self.cfg                = cfg
        self.cache              = cache

        # Extract config values with safe fallbacks so old config files
        # that use different field names still work.
# Extract config values
        self.enabled = bool(cfg.get("enabled", False))
        self.mode = str(cfg.get("mode", "baseline")).lower().strip()
        self.strategy = str(cfg.get("strategy", "append")).lower().strip()
        self.selection_strategy = str(cfg.get("selection_strategy", "top_rank")).lower().strip()
        self.intensity = float(cfg.get("intensity", 0.2))

        self._validate_config()

        logger.info(
            f"FailureInjector initialised | "
            f"enabled={self.enabled} | "
            f"mode='{self.mode}' | "
            f"strategy='{self.strategy}' | "
            f"selection='{self.selection_strategy}' | "
            f"intensity={self.intensity}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def inject(self, retrieval_result) -> object:
        """
        Apply the configured failure injection to a RetrievalResult.

        This is the single entry point called by main.py for every question.

        In baseline mode or when disabled, the original RetrievalResult is
        returned unchanged. No copy is made.

        In conflicting mode, a new RetrievalResult is returned with only
        retrieved_chunks and retrieval_scores modified. All other fields
        (question_id, question, corpus_name, etc.) are preserved unchanged.

        RetrievalResult has no failure_mode field. The experiment condition
        is recorded by main.py when it calls:
            generator.generate(retrieval_result, failure_mode=failure_cfg["mode"])

        Args:
            retrieval_result : RetrievalResult from src/retriever.py.
                               retrieved_chunks is List[str].
                               retrieval_scores is List[float].

        Returns:
            Original RetrievalResult (baseline/disabled), or a new
            RetrievalResult with retrieved_chunks and retrieval_scores
            modified (conflicting mode). Original is never mutated.

        Raises:
            CacheMissError : If a required cache entry is missing and
                             EvidenceCache is configured in strict mode.
        """
        # ── Pass-through: baseline or disabled ────────────────────────────────
        if not self.enabled or self.mode == "baseline":
            logger.debug(
                f"[{retrieval_result.question_id}] "
                f"Injection skipped (enabled={self.enabled}, mode='{self.mode}'). "
                f"Returning original RetrievalResult unchanged."
            )
            return retrieval_result

        # ── Guard: empty retrieved chunks ──────────────────────────────────────
        if not retrieval_result.retrieved_chunks:
            logger.warning(
                f"[{retrieval_result.question_id}] "
                f"No retrieved chunks — skipping injection. "
                f"Returning original RetrievalResult unchanged."
            )
            return retrieval_result

        logger.info(
            f"[{retrieval_result.question_id}] "
            f"Injecting '{self.mode}' evidence | "
            f"strategy='{self.strategy}' | intensity={self.intensity} | "
            f"n_chunks={len(retrieval_result.retrieved_chunks)}"
        )

        # ── Step 1: decide how many passages to inject into ───────────────────
        n_to_inject = self._compute_injection_count(
            len(retrieval_result.retrieved_chunks)
        )

        # ── Step 2: select which passage indices to target ────────────────────
        target_indices = self._select_passages(
            retrieval_result.retrieved_chunks,
            retrieval_result.retrieval_scores,
            n_to_inject,
        )

        # ── Step 3: load cached conflicting text for each target index ─────────
        # Each element: (original_index, conflicting_text)
        injections: List[Tuple[int, str]] = []

        for idx in target_indices:
            text = self._get_cached_conflicting_text(
                original_text=  retrieval_result.retrieved_chunks[idx],
                question_id=    retrieval_result.question_id,
                passage_index=  idx,
            )
            if text is not None:
                injections.append((idx, text))
            else:
                logger.warning(
                    f"[{retrieval_result.question_id}] "
                    f"No cached conflict for passage at index {idx} — skipping."
                )

        # ── No injections available: return original unchanged ─────────────────
        if not injections:
            logger.warning(
                f"[{retrieval_result.question_id}] "
                f"No cached conflicts available for any selected passage. "
                f"Returning original RetrievalResult unchanged."
            )
            return retrieval_result

        # ── Step 4: apply append strategy ─────────────────────────────────────
        new_chunks, new_scores = self._apply_append(
            chunks=      retrieval_result.retrieved_chunks,
            scores=      retrieval_result.retrieval_scores,
            injections=  injections,
        )

        logger.info(
            f"[{retrieval_result.question_id}] "
            f"Injection complete: "
            f"{len(retrieval_result.retrieved_chunks)} → {len(new_chunks)} passages."
        )

        # Return a new RetrievalResult with only the two modified lists.
        # All other fields are carried over unchanged by dataclasses.replace().
        return replace(
            retrieval_result,
            retrieved_chunks= new_chunks,
            retrieval_scores= new_scores,
        )

    # ── Private: injection count ───────────────────────────────────────────────

    def _compute_injection_count(self, n_retrieved: int) -> int:
        """
        Compute how many passages to inject into, based on intensity.

        Uses ceil() to ensure at least 1 passage is always targeted
        when intensity > 0 and n_retrieved >= 1.

        Examples:
            intensity=0.2, n_retrieved=5  → ceil(1.0) = 1
            intensity=0.4, n_retrieved=5  → ceil(2.0) = 2
            intensity=1.0, n_retrieved=5  → ceil(5.0) = 5

        Args:
            n_retrieved : Total number of retrieved passages.

        Returns:
            Number of passages to inject into (min 1, max n_retrieved).
        """
        count = math.ceil(n_retrieved * self.intensity)
        # Clamp to [1, n_retrieved]
        count = max(1, min(count, n_retrieved))
        logger.debug(
            f"Injection count: ceil({n_retrieved} × {self.intensity}) = {count}"
        )
        return count

    # ── Private: passage selection ─────────────────────────────────────────────

    def _select_passages(
        self,
        chunks: List[str],
        scores: List[float],
        n: int,
    ) -> List[int]:
        """
        Select which passage indices to inject conflicting evidence into.

        Phase 2 implements only 'top_rank' strategy:
            Select the first n indices [0], [0, 1], [0, 1, 2], etc.
            These are the highest-ranked passages (already ordered by the
            Retriever from highest to lowest similarity score).

        ASSUMPTION (documented for dissertation):
            The top-ranked passage is most influential on the generator
            because it has the highest similarity score and appears first
            in the context window. This is consistent with retrieval score
            ordering and position bias in long-context LLMs
            (Liu et al., 2023, "Lost in the Middle").

        Future strategies to add without modifying inject():
            'random'     → randomly sample n indices
            'similarity' → target by similarity threshold
            'keyword'    → target passages containing specific terms

        Args:
            chunks : List of retrieved passage text strings.
            scores : Corresponding retrieval scores.
            n      : Number of passages to select.

        Returns:
            List of integer indices into chunks/scores.
        """
        if self.selection_strategy == "top_rank":
            # Retriever already orders chunks highest-score-first.
            return list(range(min(n, len(chunks))))

        # Defensive fallback — unreachable given _validate_config().
        raise ValueError(
            f"Unknown selection strategy: '{self.selection_strategy}'. "
            f"This should have been caught at init time."
        )

    # ── Private: inject append strategy ───────────────────────────────────────

    def _apply_append(
        self,
        chunks:     List[str],
        scores:     List[float],
        injections: List[Tuple[int, str]],
    ) -> Tuple[List[str], List[float]]:
        """
        Apply the APPEND injection strategy.

        For each (original_index, conflicting_text) pair, insert the
        conflicting text immediately after the original passage in the list.

        BEFORE: ["D1", "D2", "D3", "D4", "D5"]
        AFTER:  ["D1", "D1_conflicting", "D2", "D3", "D4", "D5"]

        This places both the correct passage and its conflicting counterpart
        in the generator's context window, forcing it to resolve the conflict.

        Injections are processed right-to-left (highest index first) so that
        earlier insertions do not shift the indices of later ones.

        Args:
            chunks     : Original retrieved passage strings.
            scores     : Original retrieval scores (parallel to chunks).
            injections : List of (original_index, conflicting_text) pairs.

        Returns:
            (new_chunks, new_scores) tuple with conflicting passages inserted.
        """
        # Work on copies — never mutate the originals.
        new_chunks: List[str]   = list(chunks)
        new_scores: List[float] = list(scores)

        # Process highest index first so earlier insertions don't shift later ones.
        sorted_injections = sorted(injections, key=lambda x: x[0], reverse=True)

        for original_idx, conflicting_text in sorted_injections:
            insert_position = original_idx + 1

            new_chunks.insert(insert_position, conflicting_text)
            new_scores.insert(insert_position, INJECTED_CHUNK_SCORE)

            logger.debug(
                f"Appended conflicting passage at position {insert_position} "
                f"(after original passage {original_idx}). "
                f"Injected score: {INJECTED_CHUNK_SCORE}."
            )

        return new_chunks, new_scores

    # ── Private: cache access ──────────────────────────────────────────────────

    def _get_cached_conflicting_text(
        self,
        original_text:  str,
        question_id:    str,
        passage_index:  int,
    ) -> Optional[str]:
        """
        Retrieve cached conflicting text for a given passage.

        CURRENT CACHE KEY STRATEGY (text hash):
            This method derives a cache key by hashing the passage text
            with SHA-256 and taking the first 16 hex characters.

            Example:
                original_text = "Metformin is first-line treatment..."
                doc_key       = "3a7f9c2b1d4e8f06"

        WARNING — KEY MISMATCH WITH build_cache.py:
            build_cache.py stores cache entries using document_id strings
            (e.g. "pubmed_001"), NOT text hashes.

            This means cache lookups here will always return None (cache miss)
            because the keys never match. The cache is technically populated
            but unreachable via this method.

            You must resolve this before running conflicting-mode experiments.
            Two options:

            Option A (recommended) — Use document IDs in the injector:
                Add retrieved_doc_ids: List[str] to RetrievalResult.
                The retriever populates it alongside retrieved_chunks.
                Replace _stable_key(original_text) below with the
                corresponding document_id from that list.

            Option B — Use text hashes in build_cache.py:
                In build_cache.py, replace document_id keys with
                _stable_key(doc.text) when calling cache.get_or_generate().
                Both sides then use the same hashing strategy.

            Until resolved, set failure_injection.enabled: false in config.yaml.

        Args:
            original_text  : The original passage text string.
            question_id    : Used for logging only.
            passage_index  : Position in retrieved list (for logging).

        Returns:
            Conflicting passage text string, or None if not available.
        """
        doc_key = _stable_key(original_text)

        print("=" * 60)
        print(f"[DEBUG] Question: {question_id}")
        print(f"[DEBUG] Passage index: {passage_index}")
        print(f"[DEBUG] Cache key: {doc_key}")

        try:
            result = self.cache.get(
                document_id=doc_key,
                failure_type=self.mode,
            )

            if result is None:
                print("[DEBUG] Cache MISS")
            else:
                print("[DEBUG] Cache HIT")
                print(f"[DEBUG] Retrieved {len(result)} characters")

            return result

        except CacheMissError:
            print("[DEBUG] Cache MISS (CacheMissError)")
            raise

        except Exception as e:
            print(f"[DEBUG] ERROR: {e}")
            logger.error(
                f"[{question_id}] Unexpected error loading cache for "
                f"passage index {passage_index} "
                f"(key={doc_key[:16]}...): {e}. "
                f"Skipping this passage."
            )
            return None

    # ── Private: validation ────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """
        Validate configuration at construction time.

        Raises ValueError immediately so misconfigured experiments fail
        loudly before any questions are processed, not silently mid-run.
        """
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Invalid failure injection mode: '{self.mode}'. "
                f"Supported modes: {sorted(SUPPORTED_MODES)}. "
                f"Check failure_injection.mode in config.yaml."
            )

        if self.enabled and self.mode != "baseline":
            if self.strategy not in SUPPORTED_STRATEGIES:
                raise ValueError(
                    f"Invalid injection strategy: '{self.strategy}'. "
                    f"Supported: {sorted(SUPPORTED_STRATEGIES)}. "
                    f"Check failure_injection.strategy in config.yaml."
                )

            if self.selection_strategy not in SUPPORTED_SELECTION_STRATEGIES:
                raise ValueError(
                    f"Invalid selection strategy: '{self.selection_strategy}'. "
                    f"Supported: {sorted(SUPPORTED_SELECTION_STRATEGIES)}. "
                    f"Check failure_injection.selection_strategy in config.yaml."
                )

            if not (0.0 < self.intensity <= 1.0):
                raise ValueError(
                    f"intensity must be in (0.0, 1.0]. Got {self.intensity}. "
                    f"Check failure_injection.intensity in config.yaml."
                )


# ── Module-level helper ────────────────────────────────────────────────────────

def _stable_key(text: str) -> str:
    """
    Derive a stable, short cache key from a passage text string.

    Used by _get_cached_conflicting_text() to look up cached evidence
    when only the passage text is available (no document_id).

    WARNING: build_cache.py currently stores cache entries using
    document_id strings, not text hashes. This function's output will
    not match those keys. See _get_cached_conflicting_text() docstring
    for the full explanation and resolution options.

    Args:
        text : The passage text to hash.

    Returns:
        A 16-character hex string (first 16 chars of SHA-256 digest).
    """
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]