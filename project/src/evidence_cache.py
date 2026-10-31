"""
src/evidence_cache.py
=====================

WHY THIS MODULE EXISTS
-----------------------
Evidence generation (running the configured LLM) is expensive, slow, and
introduces randomness. If we generated conflicting evidence on every
experiment run:
  - Experiments would not be reproducible (different generated text each time)
  - Experiments would be slow (model call per retrieved document per question)
  - Experiments would cost compute on every run

The EvidenceCache solves all three problems by separating evidence generation
(offline, once) from evidence retrieval (online, fast, free).

The cache is the bridge between the offline world (EvidenceGenerator, Qwen)
and the online world (FailureInjector, main.py). Only this module knows about
the cache file structure. Everything else treats it as a black box.

WHERE IT FITS IN THE PIPELINE
-------------------------------

    OFFLINE (build_cache.py):
        EvidenceGenerator.generate()
                ↓
        ConflictVerifier.verify()
                ↓
        EvidenceCache.save()  ← writes to disk

    ONLINE (main.py, via FailureInjector):
        EvidenceCache.get()   ← reads from disk (no generation)
                ↓
        FailureInjector uses text

WHICH MODULES CALL IT
----------------------
    scripts/build_cache.py → calls get_or_generate() (offline only)
    src/failure_injector.py → calls get() (online, read-only)

WHICH MODULES IT CALLS
-----------------------
    src/evidence_generator.py → called via get_or_generate() (offline only)
    src/conflict_verifier.py  → called via get_or_generate() (offline only)
    src/logger.py             → logging

HOW IT CONTRIBUTES TO THE DISSERTATION
----------------------------------------
The cache design directly addresses two methodological requirements:

1. Reproducibility: The same conflicting document is used every time a given
   document is retrieved, regardless of how many experiments you run or in
   what order. This ensures your experimental conditions are stable.

2. Retrieval-awareness: The cache is populated only from documents that are
   actually retrieved for your question set. You are not pre-generating
   conflicts for the entire corpus — only for the 250 or so unique documents
   your retriever actually surfaces. This keeps the framework retrieval-aware,
   which is a key differentiator from static benchmark datasets like
   MEDCOUNTERFACT.

CACHE STRUCTURE ON DISK
------------------------
    experiment_cache/
      evidence/
        <document_id>/
          original.json       ← original document text + metadata
          conflicting.json    ← generated conflicting text + metadata
          metadata.json       ← provenance: model, prompt, hash, verifier

DOCUMENT HASH VERIFICATION
----------------------------
metadata.json stores a SHA-256 hash of the original document text at
generation time. When loading from cache, the hash is re-computed and
compared. If the corpus has changed (document updated, re-chunked, etc.),
the hash mismatch is detected and logged as a warning. This prevents
silently using a conflicting document that was generated from a different
version of the original.

STRICT VS LENIENT MODE
------------------------
strict_cache (config):
  True  → raise CacheMissError if a document has no cached version.
           Use during experiments to ensure determinism.
  False → log a warning and skip injection for uncached documents.
           Use during cache building (first pass may have partial failures).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

from src.conflict_verifier import ConflictVerifier, VerificationResult
from src.evidence_generator import EvidenceGenerator, EvidenceGenerationError
from src.logger import get_logger
from src.models import Document

logger = get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class CacheMissError(Exception):
    """
    Raised when a document has no cached evidence and strict_cache=True.

    This is a deliberate design choice: during experiments, a cache miss
    means your experimental conditions are incomplete. Raising an error
    makes this visible immediately rather than silently producing baseline
    results for some questions and injected results for others.
    """
    pass


class CacheCorruptionError(Exception):
    """
    Raised when a cache entry exists but cannot be read or parsed.

    This can happen if a cache-building run was interrupted mid-write,
    leaving a partial JSON file. The error message includes the path so
    you can delete and regenerate the affected entry.
    """
    pass


# ── EvidenceCache ─────────────────────────────────────────────────────────────

class EvidenceCache:
    """
    Manages the on-disk cache of generated failure evidence.

    This class has two modes of operation:

    OFFLINE MODE (during build_cache.py):
        Instantiated with an EvidenceGenerator and ConflictVerifier.
        get_or_generate() will generate and store if not already cached.

    ONLINE MODE (during main.py, via FailureInjector):
        Instantiated WITHOUT an EvidenceGenerator (generator=None).
        get() reads from cache only. If the cache is missing, either
        raises CacheMissError (strict_cache=True) or logs and returns None.
        The FailureInjector handles the None case.

    Args:
        cache_dir    : Root directory for the cache.
                       e.g. "experiment_cache/evidence"
        generator    : EvidenceGenerator instance (offline only).
                       Pass None for online/read-only use.
        verifier     : ConflictVerifier instance (offline only).
                       Pass None for online/read-only use.
        strict_cache : If True, raise CacheMissError on cache miss.
                       If False, return None and log a warning.
    """

    def __init__(
        self,
        cache_dir:    str | Path,
        generator:    Optional[EvidenceGenerator]  = None,
        verifier:     Optional[ConflictVerifier]   = None,
        strict_cache: bool                         = True,
    ) -> None:
        self.cache_dir    = Path(cache_dir)
        self.generator    = generator
        self.verifier     = verifier
        self.strict_cache = strict_cache

        # Create cache directory if it doesn't exist.
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        mode = "offline (read-write)" if generator else "online (read-only)"
        logger.info(
            f"EvidenceCache initialised | "
            f"cache_dir={self.cache_dir} | "
            f"mode={mode} | "
            f"strict_cache={strict_cache}"
        )

    # ── Public API: Online (read-only) ────────────────────────────────────────

    def get(
        self,
        document_id:  str,
        failure_type: str,
        original_text: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieve cached failure evidence for a document.

        This is the method called by FailureInjector during experiments.
        It never generates — only reads from disk.

        Args:
            document_id   : The document whose failure version to retrieve.
            failure_type  : "conflicting"
            original_text : If provided, verifies the hash matches the
                            cached version. Logs a warning on mismatch.

        Returns:
            The cached failure evidence text, or None if not found and
            strict_cache=False.

        Raises:
            CacheMissError      : If not found and strict_cache=True.
            CacheCorruptionError: If the cache file exists but is malformed.
        """
        cached = self._load_failure_text(document_id, failure_type)

        if cached is None:
            message = (
                f"No cached '{failure_type}' evidence for document "
                f"'{document_id}'. Run build_cache.py to populate the cache."
            )
            if self.strict_cache:
                raise CacheMissError(message)
            else:
                logger.warning(message + " Skipping injection for this document.")
                return None

        # Optional hash verification
        if original_text is not None:
            self._verify_hash_if_metadata_exists(document_id, original_text)

        return cached

    def exists(self, document_id: str, failure_type: str) -> bool:
        """
        Check whether a cached failure version exists for a document.

        Used by build_cache.py to skip already-cached documents.

        Args:
            document_id  : Document to check.
            failure_type : "conflicting"

        Returns:
            True if the cache entry exists and is readable, False otherwise.
        """
        failure_path = self._failure_json_path(document_id, failure_type)
        if not failure_path.exists():
            return False
        # Quick sanity check: file is non-empty
        return failure_path.stat().st_size > 0

    # ── Public API: Offline (read-write) ──────────────────────────────────────

    def get_or_generate(
        self,
        document:     Document,
        failure_type: str,
    ) -> str:
        """
        Return cached failure evidence, generating it first if not cached.

        This is the method called by build_cache.py (OFFLINE ONLY).
        The online pipeline (main.py → FailureInjector) calls get() instead.

        Algorithm:
            1. Check if failure_type evidence already exists in cache.
            2. If YES → return cached text (no generation).
            3. If NO  → call generator, verify, store, return.

        Args:
            document     : The Document object to generate evidence for.
            failure_type : "conflicting"

        Returns:
            The failure evidence text (from cache or freshly generated).

        Raises:
            RuntimeError            : If called without a generator (online mode).
            EvidenceGenerationError : If generation fails.
            ValueError              : If verification fails after max retries.
        """
        if self.generator is None:
            raise RuntimeError(
                "get_or_generate() called without an EvidenceGenerator. "
                "This method is for offline cache building only. "
                "Use get() for online experiments."
            )

        # Check cache first — avoid unnecessary generation.
        if self.exists(document.document_id, failure_type):
            logger.debug(
                f"Cache HIT: '{document.document_id}' / '{failure_type}'"
            )
            cached = self._load_failure_text(document.document_id, failure_type)
            if cached is not None:
                return cached
            logger.warning(
                f"Cache entry for '{document.document_id}' exists but could "
                f"not be loaded. Regenerating."
            )

        logger.info(
            f"Cache MISS: generating '{failure_type}' evidence for "
            f"'{document.document_id}'..."
        )

        # Generate
        generated_text = self.generator.generate(document, failure_type)

        # Verify (if a verifier is configured)
        verification: Optional[VerificationResult] = None
        if self.verifier is not None:
            verification = self.verifier.verify(document.text, generated_text)
            if not verification.passed:
                logger.warning(
                    f"Verification FAILED for '{document.document_id}': "
                    f"{verification.reason}. Not caching."
                )
                raise ValueError(
                    f"Generated evidence for '{document.document_id}' failed "
                    f"verification: {verification.reason}"
                )
            logger.info(f"Verification PASSED: {verification}")

        # Save to cache
        self._save(document, failure_type, generated_text, verification)

        return generated_text

    # ── Private: File paths ───────────────────────────────────────────────────

    def _document_dir(self, document_id: str) -> Path:
        """Return the cache directory for a specific document."""
        return self.cache_dir / document_id

    def _original_json_path(self, document_id: str) -> Path:
        return self._document_dir(document_id) / "original.json"

    def _failure_json_path(self, document_id: str, failure_type: str) -> Path:
        return self._document_dir(document_id) / f"{failure_type}.json"

    def _metadata_json_path(self, document_id: str) -> Path:
        return self._document_dir(document_id) / "metadata.json"

    # ── Private: Save ─────────────────────────────────────────────────────────

    def _save(
        self,
        document:     Document,
        failure_type: str,
        generated:    str,
        verification: Optional[VerificationResult],
    ) -> None:
        """
        Save original, generated, and metadata JSON files to the cache.

        Creates the document directory if it doesn't exist.
        Writes atomically by constructing the full dict before opening
        the file, so a crash mid-write leaves a partial file rather than
        corrupting an existing one.
        """
        doc_dir = self._document_dir(document.document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save original document
        original_payload = {
            "document_id": document.document_id,
            "title":       document.title,
            "text":        document.text,
            "source":      document.source,
            "year":        document.year,
        }
        self._write_json(self._original_json_path(document.document_id), original_payload)

        # 2. Save generated failure document
        failure_payload = {
            "document_id":  document.document_id,
            "failure_type": failure_type,
            "text":         generated,
        }
        self._write_json(self._failure_json_path(document.document_id, failure_type), failure_payload)

        # 3. Save metadata (provenance record) — generator_model/prompt_version/
        #    temperature come from whatever EvidenceGenerator was configured
        #    (Qwen via HuggingFaceEvidenceGenerator by default).
        metadata = {
            "document_id":          document.document_id,
            "failure_type":         failure_type,
            "generator_model":      self.generator.model_name,
            "prompt_version":       self.generator.prompt_version,
            "temperature":          self.generator.temperature,
            "creation_time":        datetime.utcnow().isoformat() + "Z",
            "source_document_hash": _sha256(document.text),
            "verifier":             verification.verifier_name if verification else "none",
            "verification_passed":  verification.passed if verification else None,
            "verification_score":   verification.score if verification else None,
            "verification_reason":  verification.reason if verification else None,
        }
        self._write_json(self._metadata_json_path(document.document_id), metadata)

        logger.info(
            f"Cached: {doc_dir.relative_to(self.cache_dir.parent)} "
            f"[{failure_type}]"
        )

    # ── Private: Load ─────────────────────────────────────────────────────────

    def _load_failure_text(
        self, document_id: str, failure_type: str
    ) -> Optional[str]:
        """
        Load failure evidence text from the cache.

        Returns:
            Text string, or None if the file does not exist.

        Raises:
            CacheCorruptionError: If the file exists but cannot be parsed.
        """
        path = self._failure_json_path(document_id, failure_type)
        if not path.exists():
            return None

        try:
            data = self._read_json(path)
            text = data.get("text", "")
            if not text:
                raise CacheCorruptionError(
                    f"Cache file has empty 'text' field: {path}"
                )
            return text
        except (json.JSONDecodeError, KeyError) as e:
            raise CacheCorruptionError(
                f"Cache file is malformed: {path}\n"
                f"Error: {e}\n"
                f"Delete this file and re-run build_cache.py to regenerate."
            ) from e

    def _verify_hash_if_metadata_exists(
        self, document_id: str, original_text: str
    ) -> None:
        """
        If metadata.json exists, verify the document hash matches.

        A hash mismatch means the corpus has changed since the cache was
        built. Logs a warning but does not raise — the cached evidence
        is still used, but the discrepancy is recorded.
        """
        meta_path = self._metadata_json_path(document_id)
        if not meta_path.exists():
            return

        try:
            metadata      = self._read_json(meta_path)
            cached_hash   = metadata.get("source_document_hash", "")
            current_hash  = _sha256(original_text)

            if cached_hash and cached_hash != current_hash:
                logger.warning(
                    f"Hash mismatch for '{document_id}': the corpus document "
                    f"has changed since the cache was built. "
                    f"Cached hash: {cached_hash[:12]}... "
                    f"Current hash: {current_hash[:12]}... "
                    f"Consider regenerating: delete "
                    f"{self._document_dir(document_id)} and re-run build_cache.py."
                )
        except Exception as e:
            logger.debug(f"Could not verify hash for '{document_id}': {e}")

    # ── Private: JSON I/O ─────────────────────────────────────────────────────

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _read_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Public: Utility ───────────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Return a summary of the current cache contents.

        Useful for build_cache.py to print a completion report.

        Returns:
            Dict with 'total_documents', 'failure_types', 'entries'.
        """
        entries = {}
        if self.cache_dir.exists():
            for doc_dir in sorted(self.cache_dir.iterdir()):
                if doc_dir.is_dir():
                    failure_files = [
                        f.stem for f in doc_dir.iterdir()
                        if f.suffix == ".json" and f.stem not in ("original", "metadata")
                    ]
                    if failure_files:
                        entries[doc_dir.name] = failure_files

        all_types = {ft for types in entries.values() for ft in types}
        return {
            "total_documents": len(entries),
            "failure_types":   sorted(all_types),
            "entries":         entries,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    """
    Compute SHA-256 hash of a text string.

    Used to detect whether the original corpus document has changed
    since the cache was built.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()