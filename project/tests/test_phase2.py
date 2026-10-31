"""
tests/test_phase2.py
====================

Unit tests for Phase 2 components:
  - ConflictVerifier (LengthRatioVerifier)
  - EvidenceCache (save, load, exists, hash verification)
  - EvidenceGenerator (GeminiEvidenceGenerator mock)
  - FailureInjector (append strategy, intensity, selection, immutability)

Tests use mocking for Gemini so no API key is needed.
All cache tests use tmp_path (pytest's temporary directory fixture).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conflict_verifier  import LengthRatioVerifier, VerificationResult
from src.evidence_cache     import CacheMissError, CacheCorruptionError, EvidenceCache
from src.evidence_generator import EvidenceGenerationError
from src.failure_injector   import FailureInjector
from src.models             import Chunk, Document, RetrievalResult


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

def make_document(doc_id: str = "pubmed_001", text: str | None = None) -> Document:
    return Document(
        document_id= doc_id,
        title=       f"Test Document {doc_id}",
        text=        text or (
            "Metformin is the recommended first-line pharmacological treatment "
            "for type 2 diabetes mellitus. It reduces hepatic glucose output "
            "and improves insulin sensitivity without causing hypoglycaemia. "
            "The drug is generally well tolerated and has a favourable safety profile."
        ),
        source=      "PubMed",
        year=        2023,
    )


def make_chunk(doc_id: str = "pubmed_001", idx: int = 0, text: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=    f"{doc_id}_chunk_{idx}",
        document_id= doc_id,
        text=        text or f"Metformin reduces hepatic glucose output in document {doc_id}.",
        chunk_index= idx,
        metadata={   "source": "PubMed"},
    )


def make_retrieval_result(
    question_id: str = "q001",
    n_chunks: int = 5,
    chunk_text: str | None = None,
) -> RetrievalResult:
    chunks = [make_chunk(f"pubmed_{i:03d}", 0, chunk_text) for i in range(n_chunks)]
    scores = [0.9 - i * 0.05 for i in range(n_chunks)]
    return RetrievalResult(
        question_id=      question_id,
        question=         "What is first-line treatment for type 2 diabetes?",
        retrieved_chunks= chunks,
        retrieval_scores= scores,
        corpus_name=      "PubMed",
        failure_mode=     "baseline",
    )


def make_injector_cfg(
    enabled: bool = True,
    mode: str = "conflicting",
    strategy: str = "append",
    selection_strategy: str = "top_rank",
    intensity: float = 0.2,
    cache_directory: str = "experiment_cache/evidence",
    strict_cache: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=            enabled,
        mode=               mode,
        strategy=           strategy,
        selection_strategy= selection_strategy,
        intensity=          intensity,
        cache_directory=    cache_directory,
        strict_cache=       strict_cache,
    )


def make_mock_generator(return_text: str = "Sulfonylureas are recommended first-line.") -> MagicMock:
    gen = MagicMock()
    gen.generate.return_value = return_text
    gen.model_name            = "gemini-1.5-flash"
    gen.prompt_version        = "v1"
    gen.temperature           = 0.1
    return gen


# ═══════════════════════════════════════════════════════════════
# CONFLICT VERIFIER TESTS
# ═══════════════════════════════════════════════════════════════

class TestLengthRatioVerifier:

    def test_passes_when_same_length(self) -> None:
        verifier = LengthRatioVerifier()
        original  = "word " * 100
        generated = "word " * 100
        result = verifier.verify(original, generated)
        assert result.passed is True

    def test_passes_within_range(self) -> None:
        verifier  = LengthRatioVerifier(min_ratio=0.8, max_ratio=1.2)
        original  = "word " * 100
        generated = "word " * 110   # ratio = 1.1, within [0.8, 1.2]
        result = verifier.verify(original, generated)
        assert result.passed is True
        assert result.score == pytest.approx(1.1, abs=0.01)

    def test_fails_when_too_short(self) -> None:
        verifier  = LengthRatioVerifier(min_ratio=0.8, max_ratio=1.2)
        original  = "word " * 100
        generated = "word " * 50    # ratio = 0.5, below 0.8
        result = verifier.verify(original, generated)
        assert result.passed is False

    def test_fails_when_too_long(self) -> None:
        verifier  = LengthRatioVerifier(min_ratio=0.8, max_ratio=1.2)
        original  = "word " * 100
        generated = "word " * 200   # ratio = 2.0, above 1.2
        result = verifier.verify(original, generated)
        assert result.passed is False

    def test_fails_empty_generated(self) -> None:
        verifier = LengthRatioVerifier()
        result   = verifier.verify("some original text here", "")
        assert result.passed is False
        assert result.score == 0.0

    def test_fails_whitespace_only_generated(self) -> None:
        verifier = LengthRatioVerifier()
        result   = verifier.verify("some original text", "   \n\t  ")
        assert result.passed is False

    def test_result_contains_verifier_name(self) -> None:
        verifier = LengthRatioVerifier()
        result   = verifier.verify("hello world", "hello world")
        assert result.verifier_name == "LengthRatioVerifier"

    def test_result_score_is_ratio(self) -> None:
        verifier  = LengthRatioVerifier()
        original  = "word " * 100
        generated = "word " * 90
        result    = verifier.verify(original, generated)
        assert result.score == pytest.approx(0.9, abs=0.01)

    def test_invalid_min_max_raises(self) -> None:
        with pytest.raises(ValueError):
            LengthRatioVerifier(min_ratio=1.5, max_ratio=0.5)

    def test_fails_empty_original(self) -> None:
        verifier = LengthRatioVerifier()
        result   = verifier.verify("", "some generated text here")
        assert result.passed is False


# ═══════════════════════════════════════════════════════════════
# EVIDENCE CACHE TESTS
# ═══════════════════════════════════════════════════════════════

class TestEvidenceCache:

    def test_exists_returns_false_when_empty(self, tmp_path: Path) -> None:
        cache = EvidenceCache(tmp_path, strict_cache=False)
        assert cache.exists("pubmed_001", "conflicting") is False

    def test_save_and_exists(self, tmp_path: Path) -> None:
        doc       = make_document("pubmed_001")
        generator = make_mock_generator("Sulfonylureas are first-line.")
        verifier  = LengthRatioVerifier()
        cache     = EvidenceCache(tmp_path, generator, verifier, strict_cache=False)

        cache.get_or_generate(doc, "conflicting")

        assert cache.exists("pubmed_001", "conflicting") is True

    def test_cache_hit_does_not_call_generator(self, tmp_path: Path) -> None:
        """Generator must be called exactly ONCE even when get_or_generate is called twice."""
        doc       = make_document("pubmed_001")
        generator = make_mock_generator("Sulfonylureas are first-line.")
        verifier  = LengthRatioVerifier()
        cache     = EvidenceCache(tmp_path, generator, verifier, strict_cache=False)

        cache.get_or_generate(doc, "conflicting")
        cache.get_or_generate(doc, "conflicting")  # second call — should hit cache

        generator.generate.assert_called_once()

    def test_get_returns_cached_text(self, tmp_path: Path) -> None:
        expected_text = "Sulfonylureas are the recommended first-line treatment."
        doc           = make_document("pubmed_001")
        generator     = make_mock_generator(expected_text)
        verifier      = LengthRatioVerifier()
        cache         = EvidenceCache(tmp_path, generator, verifier, strict_cache=False)

        cache.get_or_generate(doc, "conflicting")
        retrieved = cache.get("pubmed_001", "conflicting")

        assert retrieved == expected_text

    def test_get_miss_strict_raises(self, tmp_path: Path) -> None:
        cache = EvidenceCache(tmp_path, strict_cache=True)
        with pytest.raises(CacheMissError):
            cache.get("nonexistent_doc", "conflicting")

    def test_get_miss_lenient_returns_none(self, tmp_path: Path) -> None:
        cache  = EvidenceCache(tmp_path, strict_cache=False)
        result = cache.get("nonexistent_doc", "conflicting")
        assert result is None

    def test_get_or_generate_without_generator_raises(self, tmp_path: Path) -> None:
        cache = EvidenceCache(tmp_path, generator=None, strict_cache=False)
        doc   = make_document("pubmed_001")
        with pytest.raises(RuntimeError, match="offline"):
            cache.get_or_generate(doc, "conflicting")

    def test_metadata_json_written(self, tmp_path: Path) -> None:
        doc       = make_document("pubmed_001")
        generator = make_mock_generator("Some conflicting text here for testing.")
        verifier  = LengthRatioVerifier()
        cache     = EvidenceCache(tmp_path, generator, verifier, strict_cache=False)

        cache.get_or_generate(doc, "conflicting")

        meta_path = tmp_path / "pubmed_001" / "metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["document_id"]    == "pubmed_001"
        assert meta["failure_type"]   == "conflicting"
        assert meta["generator_model"] == "gemini-1.5-flash"
        assert "source_document_hash" in meta
        assert meta["verification_passed"] is True

    def test_original_json_written(self, tmp_path: Path) -> None:
        doc       = make_document("pubmed_001")
        generator = make_mock_generator("Some conflicting medical text here.")
        cache     = EvidenceCache(tmp_path, generator, strict_cache=False)

        cache.get_or_generate(doc, "conflicting")

        orig_path = tmp_path / "pubmed_001" / "original.json"
        assert orig_path.exists()
        with open(orig_path) as f:
            orig = json.load(f)
        assert orig["document_id"] == "pubmed_001"
        assert orig["text"]        == doc.text

    def test_malformed_cache_raises_corruption_error(self, tmp_path: Path) -> None:
        doc_dir = tmp_path / "pubmed_001"
        doc_dir.mkdir()
        (doc_dir / "conflicting.json").write_text("NOT VALID JSON {{{{")

        cache = EvidenceCache(tmp_path, strict_cache=False)
        with pytest.raises(CacheCorruptionError):
            cache.get("pubmed_001", "conflicting")

    def test_verification_failure_prevents_caching(self, tmp_path: Path) -> None:
        """When verification fails, document should not be saved to cache."""
        doc       = make_document("pubmed_001")
        # Return a very short text that will fail length ratio check
        generator = make_mock_generator("Too short.")
        verifier  = LengthRatioVerifier(min_ratio=0.8, max_ratio=1.2)
        cache     = EvidenceCache(tmp_path, generator, verifier, strict_cache=False)

        with pytest.raises(ValueError, match="verification"):
            cache.get_or_generate(doc, "conflicting")

        assert cache.exists("pubmed_001", "conflicting") is False

    def test_summary_reflects_cached_documents(self, tmp_path: Path) -> None:
        for i in range(3):
            doc       = make_document(f"pubmed_{i:03d}")
            generator = make_mock_generator(f"Conflicting text for document {i} " * 10)
            cache     = EvidenceCache(tmp_path, generator, strict_cache=False)
            cache.get_or_generate(doc, "conflicting")

        # Fresh cache instance to re-read from disk
        cache2   = EvidenceCache(tmp_path, strict_cache=False)
        summary  = cache2.summary()
        assert summary["total_documents"] == 3


# ═══════════════════════════════════════════════════════════════
# FAILURE INJECTOR TESTS
# ═══════════════════════════════════════════════════════════════

class TestFailureInjector:

    # ── Helpers ──────────────────────────────────────────────────────────────

    def make_cache_with_text(self, tmp_path: Path, doc_ids: list[str], text: str) -> EvidenceCache:
        """Build a cache pre-populated with given text for given doc IDs."""
        fake_generator = make_mock_generator(text)
        cache = EvidenceCache(tmp_path, fake_generator, strict_cache=False)
        for doc_id in doc_ids:
            doc = make_document(doc_id)
            cache.get_or_generate(doc, "conflicting")
        # Return a read-only cache instance for the injector
        return EvidenceCache(tmp_path, generator=None, strict_cache=False)

    # ── Baseline mode ─────────────────────────────────────────────────────────

    def test_baseline_is_pass_through(self, tmp_path: Path) -> None:
        cache    = EvidenceCache(tmp_path, generator=None, strict_cache=False)
        cfg      = make_injector_cfg(enabled=True, mode="baseline")
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=3)

        modified = injector.inject(result)

        assert len(modified.retrieved_chunks) == 3
        assert modified.failure_mode          == "baseline"

    def test_disabled_is_pass_through(self, tmp_path: Path) -> None:
        cache    = EvidenceCache(tmp_path, generator=None, strict_cache=False)
        cfg      = make_injector_cfg(enabled=False, mode="conflicting")
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=3)

        modified = injector.inject(result)

        assert len(modified.retrieved_chunks) == 3
        assert modified.failure_mode          == "baseline"

    # ── Append strategy ───────────────────────────────────────────────────────

    def test_append_adds_one_chunk_for_intensity_02(self, tmp_path: Path) -> None:
        """intensity=0.2 with 5 chunks → ceil(5×0.2)=1 injection → 6 total chunks."""
        doc_ids  = [f"pubmed_{i:03d}" for i in range(5)]
        cache    = self.make_cache_with_text(tmp_path, doc_ids, "Conflicting medical evidence text " * 8)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.2)
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=5)

        modified = injector.inject(result)

        assert len(modified.retrieved_chunks) == 6  # 5 original + 1 injected

    def test_append_adds_two_chunks_for_intensity_04(self, tmp_path: Path) -> None:
        """intensity=0.4 with 5 chunks → ceil(5×0.4)=2 injections → 7 total chunks."""
        doc_ids  = [f"pubmed_{i:03d}" for i in range(5)]
        cache    = self.make_cache_with_text(tmp_path, doc_ids, "Conflicting evidence text for testing " * 5)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.4)
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=5)

        modified = injector.inject(result)

        assert len(modified.retrieved_chunks) == 7  # 5 original + 2 injected

    def test_append_all_chunks_for_intensity_10(self, tmp_path: Path) -> None:
        """intensity=1.0 with 5 chunks → inject into all 5 → 10 total chunks."""
        doc_ids  = [f"pubmed_{i:03d}" for i in range(5)]
        cache    = self.make_cache_with_text(tmp_path, doc_ids, "Conflicting evidence for every document " * 5)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=1.0)
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=5)

        modified = injector.inject(result)

        assert len(modified.retrieved_chunks) == 10  # 5 original + 5 injected

    def test_injected_chunk_placed_after_original(self, tmp_path: Path) -> None:
        """Injected chunk must appear immediately after the original top-ranked chunk."""
        doc_ids  = ["pubmed_000"]
        cache    = self.make_cache_with_text(tmp_path, doc_ids, "Injected conflicting text here " * 8)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.2)
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=5)

        modified = injector.inject(result)

        # Position 0 = original D1, position 1 = injected D1_conflicting
        original_chunk = modified.retrieved_chunks[0]
        injected_chunk = modified.retrieved_chunks[1]

        assert original_chunk.metadata.get("injected", False) is False
        assert injected_chunk.metadata.get("injected") is True

    def test_injected_chunk_has_zero_score(self, tmp_path: Path) -> None:
        doc_ids  = ["pubmed_000"]
        cache    = self.make_cache_with_text(tmp_path, doc_ids, "Conflicting evidence text " * 8)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.2)
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=5)

        modified = injector.inject(result)

        assert modified.retrieval_scores[1] == 0.0  # injected chunk score

    def test_original_scores_preserved(self, tmp_path: Path) -> None:
        doc_ids  = ["pubmed_000"]
        cache    = self.make_cache_with_text(tmp_path, doc_ids, "Some conflicting text here " * 8)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.2)
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=5)

        modified = injector.inject(result)

        # Original score at position 0 must be unchanged
        assert modified.retrieval_scores[0] == result.retrieval_scores[0]

    def test_failure_mode_set_on_result(self, tmp_path: Path) -> None:
        doc_ids  = ["pubmed_000"]
        cache    = self.make_cache_with_text(tmp_path, doc_ids, "Conflicting evidence text here " * 8)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.2)
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=5)

        modified = injector.inject(result)

        assert modified.failure_mode == "conflicting"

    # ── Immutability ──────────────────────────────────────────────────────────

    def test_original_result_not_mutated(self, tmp_path: Path) -> None:
        """inject() must never modify the original RetrievalResult."""
        doc_ids  = ["pubmed_000"]
        cache    = self.make_cache_with_text(tmp_path, doc_ids, "Conflicting evidence here " * 8)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.2)
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=5)

        original_chunk_count  = len(result.retrieved_chunks)
        original_failure_mode = result.failure_mode
        original_scores       = result.retrieval_scores.copy()

        injector.inject(result)

        assert len(result.retrieved_chunks) == original_chunk_count
        assert result.failure_mode          == original_failure_mode
        assert result.retrieval_scores      == original_scores

    # ── Cache miss handling ───────────────────────────────────────────────────

    def test_cache_miss_lenient_returns_original(self, tmp_path: Path) -> None:
        """With strict_cache=False, cache miss skips injection gracefully."""
        # Cache is empty — no documents pre-generated
        cache    = EvidenceCache(tmp_path, generator=None, strict_cache=False)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.2)
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=5)

        # Should not raise — returns result with failure_mode set but no injection
        modified = injector.inject(result)
        assert len(modified.retrieved_chunks) == 5  # nothing injected

    def test_cache_miss_strict_raises(self, tmp_path: Path) -> None:
        """With strict_cache=True, cache miss raises CacheMissError."""
        cache    = EvidenceCache(tmp_path, generator=None, strict_cache=True)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.2)
        injector = FailureInjector(cfg, cache)
        result   = make_retrieval_result(n_chunks=5)

        with pytest.raises(CacheMissError):
            injector.inject(result)

    # ── Injection count ───────────────────────────────────────────────────────

    def test_compute_injection_count_ceil(self, tmp_path: Path) -> None:
        """ceil(5 × 0.2) = 1, ceil(5 × 0.3) = 2, ceil(5 × 1.0) = 5."""
        cache = EvidenceCache(tmp_path, generator=None, strict_cache=False)
        for intensity, expected in [(0.2, 1), (0.3, 2), (0.4, 2), (1.0, 5)]:
            cfg      = make_injector_cfg(enabled=True, intensity=intensity)
            injector = FailureInjector(cfg, cache)
            assert injector._compute_injection_count(5) == expected

    # ── Validation ────────────────────────────────────────────────────────────

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        cache = EvidenceCache(tmp_path, generator=None, strict_cache=False)
        cfg   = make_injector_cfg(enabled=True, mode="invalid_mode_xyz")
        with pytest.raises(ValueError, match="Invalid failure injection mode"):
            FailureInjector(cfg, cache)

    def test_invalid_intensity_raises(self, tmp_path: Path) -> None:
        cache = EvidenceCache(tmp_path, generator=None, strict_cache=False)
        cfg   = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.0)
        with pytest.raises(ValueError, match="intensity"):
            FailureInjector(cfg, cache)

    def test_empty_retrieval_handled_gracefully(self, tmp_path: Path) -> None:
        cache    = EvidenceCache(tmp_path, generator=None, strict_cache=False)
        cfg      = make_injector_cfg(enabled=True, mode="conflicting", intensity=0.2)
        injector = FailureInjector(cfg, cache)

        empty_result = RetrievalResult(
            question_id="q001", question="test?",
            retrieved_chunks=[], retrieval_scores=[],
            corpus_name="PubMed",
        )

        # Should not raise — gracefully returns result unchanged
        modified = injector.inject(empty_result)
        assert len(modified.retrieved_chunks) == 0


# ═══════════════════════════════════════════════════════════════
# EVIDENCE GENERATOR MOCK TESTS
# ═══════════════════════════════════════════════════════════════

class TestEvidenceGeneratorMock:
    """
    Tests that verify the generator interface without making real API calls.
    Uses unittest.mock to patch google.generativeai.
    """

    def test_gemini_generator_calls_api_once(self) -> None:
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_model_cls:

            mock_model    = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Sulfonylureas are recommended first-line treatment for diabetes."
            mock_model.generate_content.return_value = mock_response
            mock_model_cls.return_value = mock_model

            from src.evidence_generator import GeminiEvidenceGenerator
            generator = GeminiEvidenceGenerator(api_key="fake-key-for-test")
            doc       = make_document()
            result    = generator.generate(doc, "conflicting")

            mock_model.generate_content.assert_called_once()
            assert len(result) > 0

    def test_gemini_generator_raises_on_empty_response(self) -> None:
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_model_cls:

            mock_model    = MagicMock()
            mock_response = MagicMock()
            mock_response.text = ""   # empty response
            mock_model.generate_content.return_value = mock_response
            mock_model_cls.return_value = mock_model

            from src.evidence_generator import GeminiEvidenceGenerator, EvidenceGenerationError
            generator = GeminiEvidenceGenerator(api_key="fake-key-for-test")
            doc       = make_document()

            with pytest.raises(EvidenceGenerationError):
                generator.generate(doc, "conflicting")

    def test_gemini_generator_raises_on_invalid_failure_type(self) -> None:
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):

            from src.evidence_generator import GeminiEvidenceGenerator
            generator = GeminiEvidenceGenerator(api_key="fake-key-for-test")
            doc       = make_document()

            with pytest.raises(ValueError, match="Unsupported failure_type"):
                generator.generate(doc, "outdated")  # not yet supported

    def test_gemini_generator_raises_on_empty_api_key(self) -> None:
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):

            from src.evidence_generator import GeminiEvidenceGenerator

            with pytest.raises(ValueError, match="API key"):
                GeminiEvidenceGenerator(api_key="")
