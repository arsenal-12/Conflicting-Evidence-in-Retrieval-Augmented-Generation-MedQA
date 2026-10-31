"""
scripts/build_cache.py
======================

WHY THIS SCRIPT EXISTS
-----------------------
This is the OFFLINE cache-building script. It is the ONLY place in the
entire project where EvidenceGenerator (and therefore the local Qwen model)
is called.

Running this script before experiments ensures that:
  1. All evidence generation happens offline, once, reproducibly.
  2. Experiment runs (main.py) never load the generation model — they only
     read cache.
  3. Cache contents are stable across experiment runs.
  4. Compute cost (model inference) is paid once, not per experiment.

WHERE IT FITS IN THE WORKFLOW
-------------------------------

    [ONE-TIME SETUP]
    python scripts/build_cache.py
            │
            ├── Loads all questions
            ├── Runs retriever on all questions (read-only)
            ├── Collects unique passages (hashed for cache keys)
            └── For each uncached passage:
                    HuggingFaceEvidenceGenerator.generate()
                    LengthRatioVerifier.verify()
                    EvidenceCache.save()

    [REPEATED EXPERIMENTS]
    python -m src.main
            │
            └── FailureInjector reads from cache — no model calls

CACHE KEY STRATEGY
-------------------
retrieved_chunks in RetrievalResult are plain strings (List[str]).
There are no document IDs attached to chunks at retrieval time.

Both this script and FailureInjector therefore use the same keying
strategy: SHA-256 hash of the passage text, truncated to 16 hex chars.

    cache key = SHA-256(passage_text)[:16]

This ensures that when build_cache.py stores a conflicting version of a
passage, FailureInjector can look it up at runtime using the same key
derived from the same passage text.

CONFIGURATION
-------------
This script uses the same config.yaml as main.py.
Configuration is loaded as a plain Python dict via load_config().
All config access uses dict syntax: cfg["section"]["key"].

No cfg.paths, cfg.embedder, or other attribute-style access is used.

The evidence generator itself is selected via cfg["evidence_generator"]
(provider/model/device) and built by create_evidence_generator() — see
src/evidence_generator.py.

USAGE
------
    python scripts/build_cache.py
    python scripts/build_cache.py --config config/config.yaml
    python scripts/build_cache.py --dry-run
    python scripts/build_cache.py --force-regenerate
    python scripts/build_cache.py --max-questions 10

ENVIRONMENT VARIABLES REQUIRED
--------------------------------
    None. Evidence generation runs on a local Hugging Face model
    (Qwen2.5-1.5B-Instruct by default) — no API key is required.
"""

from __future__ import annotations

import argparse
import hashlib

def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is on sys.path for src.* imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conflict_verifier  import LengthRatioVerifier
from src.document_loader    import DocumentLoader
from src.embedder           import Embedder
from src.evidence_cache     import EvidenceCache
from src.evidence_generator import create_evidence_generator
from src.logger             import get_logger
from src.models             import Document
from src.question_loader    import QuestionLoader
from src.retriever          import Retriever
from src.utils              import load_config
from src.vector_store       import VectorStore

logger = get_logger("build_cache")


# ── Cache key helper ───────────────────────────────────────────────────────────
# Must produce identical output to _stable_key() in src/failure_injector.py.
# Both the write side (here) and the read side (FailureInjector) must use
# the same function so that keys match at runtime.
#
# This function is intentionally duplicated rather than imported from
# failure_injector.py to avoid a dependency between the offline script and
# a runtime module. Both copies must remain byte-for-byte identical.

def _stable_key(text: str) -> str:
    """
    Derive a stable 16-character cache key from passage text.

    SHA-256 is deterministic, collision-resistant, and fast for the
    corpus sizes used in this dissertation (hundreds of documents).

    Args:
        text : The passage text to hash.

    Returns:
        First 16 hex characters of the SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the offline evidence cache for failure injection experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running the evidence generator.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate evidence even if cache entries already exist.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit to first N questions (for quick testing).",
    )
    parser.add_argument(
        "--failure-type",
        default="conflicting",
        choices=["conflicting"],
        help="Which failure type to generate (default: conflicting).",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load configuration ─────────────────────────────────────────────────────
    # load_config() returns a plain Python dict.
    # All access below uses dict syntax: cfg["section"]["key"].
    # There is no cfg.paths, cfg.embedder, or any attribute-style access.
    cfg: dict = load_config(args.config)

    logger.info("=" * 65)
    logger.info("  RAG Dissertation — Offline Cache Builder")
    logger.info("=" * 65)
    logger.info(f"Config        : {args.config}")
    logger.info(f"Failure type  : {args.failure_type}")
    logger.info(f"Dry run       : {args.dry_run}")
    logger.info(f"Force regen   : {args.force_regenerate}")
    logger.info(f"Max questions : {args.max_questions or 'all'}")

    # ── Step 1: Load questions ─────────────────────────────────────────────────
    # QuestionLoader(config) receives the full config dict, matching how
    # main.py calls it. It reads cfg["dataset"]["path"] etc. internally.
    # We apply max_questions by passing a shallow-copied dataset section
    # so the original cfg dict is never mutated.
    logger.info("\n[1/6] Loading question dataset...")

    if args.max_questions is not None:
        # Build an overridden config with max_questions applied.
        # Shallow-copy only the dataset section to avoid mutating the original.
        overridden_cfg = {
            **cfg,
            "dataset": {**cfg["dataset"], "max_questions": args.max_questions},
        }
        questions = QuestionLoader(overridden_cfg).load()
    else:
        questions = QuestionLoader(cfg).load()

    logger.info(f"       → {len(questions)} questions loaded.")

    # ── Step 2: Load corpus documents ──────────────────────────────────────────
    # DocumentLoader(config) receives the full config dict, matching main.py.
    # It reads cfg["corpus"]["path"] internally and returns List[Document].
    logger.info("\n[2/6] Loading document corpus...")
    documents: List[Document] = DocumentLoader(cfg).load()

    # Build a text → Document lookup so we can find the full Document object
    # for a given retrieved passage text during cache generation.
    doc_by_text: Dict[str, Document] = {doc.text: doc for doc in documents}

    logger.info(f"       → {len(documents)} documents loaded.")

    # ── Step 3: Embed corpus and initialise retriever ──────────────────────────
    # All three classes (Embedder, VectorStore, Retriever) receive the full
    # config dict, exactly as main.py instantiates them.
    logger.info("\n[3/6] Initialising retrieval pipeline...")

    # Embedder reads cfg["embedding"]["model_name"], cfg["embedding"]["device"] etc.
    vector_store = VectorStore(cfg)

    try:
        vector_store.load()
        logger.info("Using existing Chroma collection.")

        embedder = Embedder(cfg)

    except ValueError:
        logger.info("Building new Chroma collection...")

        embedder = Embedder(cfg)
        document_embeddings = embedder.embed_documents(
            [doc.text for doc in documents]
        )

        vector_store.build(documents, document_embeddings)
    # Retriever reads cfg["retriever"]["top_k"] etc.
    retriever = Retriever(cfg, embedder=embedder, vector_store=vector_store)

    logger.info(f"       → Retrieval pipeline ready.")

    # ── Step 4: Discover all unique retrieved passages ─────────────────────────
    # result.retrieved_chunks is List[str] — plain passage text strings.
    # We collect unique passage texts and hash each one to produce the
    # cache key. This is the same key FailureInjector uses at runtime.
    logger.info("\n[4/6] Discovering retrieved passages for all questions...")

    # Maps cache_key (SHA-256[:16]) → original passage text string.
    unique_passages: Dict[str, str] = {}

    for i, question in enumerate(questions, 1):
        result = retriever.retrieve(question)
        if i == 1:
            print("BUILD CACHE FIRST PASSAGE")
            print(result.retrieved_chunks[0][:300])
            print(_stable_key(result.retrieved_chunks[0]))

        for passage_text in result.retrieved_chunks:
            # retrieved_chunks is List[str] — access as plain string, never
            # as an object. There is no .document_id, .text, or .metadata.
            key = _stable_key(passage_text)
            if key not in unique_passages:
                unique_passages[key] = passage_text

        if i % 10 == 0:
            logger.info(f"       Processed {i}/{len(questions)} questions...")

    logger.info(
        f"       → {len(unique_passages)} unique passages discovered "
        f"across {len(questions)} questions."
    )

    if args.dry_run:
        logger.info("\n[DRY RUN] Would generate evidence for:")
        for key in sorted(unique_passages.keys()):
            preview = unique_passages[key][:80].replace("\n", " ")
            logger.info(f"  {key}  →  {preview}...")
        logger.info("\nDry run complete. No model was loaded and no evidence was generated.")
        return

    # ── Step 5: Initialise generator, verifier, cache ─────────────────────────
    logger.info("\n[5/6] Initialising evidence generator and cache...")

    # create_evidence_generator() reads cfg["evidence_generator"] and returns
    # the configured backend (HuggingFaceEvidenceGenerator / Qwen by default).
    generator = create_evidence_generator(cfg)

    fail_cfg = cfg.get("failure_injection",  {})
    ver_cfg  = cfg.get("conflict_verifier",  {})

    verifier = LengthRatioVerifier(
        min_ratio= float(ver_cfg.get("min_ratio", 0.8)),
        max_ratio= float(ver_cfg.get("max_ratio", 1.2)),
    )

    # Cache directory from cfg["failure_injection"]["cache_directory"].
    cache_dir = fail_cfg.get("cache_directory", "experiment_cache/evidence")

    # strict_cache=False during cache building so that individual passage
    # failures are logged and skipped rather than crashing the whole run.
    cache = EvidenceCache(
        cache_dir=    cache_dir,
        generator=    generator,
        verifier=     verifier,
        strict_cache= False,
    )
    logger.info(f"       → Cache directory: {cache_dir}")

    # ── Step 6: Generate and cache conflicting evidence ────────────────────────
    # For each unique passage:
    #   1. Derive cache key from passage text (same key FailureInjector uses).
    #   2. If already cached and not force-regenerating, skip.
    #   3. Look up the full Document object by matching the passage text.
    #   4. If no exact match (chunk is a sub-passage of a longer document),
    #      wrap the passage text in a minimal Document for generation.
    #   5. Call EvidenceCache.get_or_generate() which calls the Qwen model
    #      once, verifies the output, and saves to cache_dir/<cache_key>/.
    logger.info(
        f"\n[6/6] Generating cache for {len(unique_passages)} unique passages..."
    )

    stats = {
        "cached_new":      0,
        "cached_existing": 0,
        "failed":          0,
    }

    for cache_key, passage_text in sorted(unique_passages.items()):
        if not args.force_regenerate and cache.exists(cache_key, args.failure_type):
            logger.info(f"  SKIP (exists){cache_key}")
            stats["cached_existing"] += 1
            continue

        document = Document(
            document_id=cache_key,
            title="",
            text=passage_text,
            source="retrieved_chunk",
            year=0,
        )

        try:
            cache.get_or_generate(document, args.failure_type)

            preview = passage_text[:60].replace("\n", " ").strip()
            logger.info(f"  OK : {cache_key}  ({preview}...)")
            stats["cached_new"] += 1

        except Exception as e:
            logger.error(f"  FAIL (generation) : {cache_key} — {e}")
            stats["failed"] += 1
    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  Cache Building Complete")
    logger.info("=" * 65)
    logger.info(f"  Unique passages discovered : {len(unique_passages)}")
    logger.info(f"  Newly generated            : {stats['cached_new']}")
    logger.info(f"  Already existed            : {stats['cached_existing']}")
    logger.info(f"  Failed (skipped)           : {stats['failed']}")
    logger.info(f"  Cache directory            : {cache_dir}")

    cache_summary = cache.summary()
    logger.info(
        f"  Total entries in cache     : {cache_summary['total_documents']}"
    )
    logger.info("=" * 65)

    if stats["failed"] > 0:
        logger.warning(
            f"{stats['failed']} passages failed generation or verification. "
            f"Re-run build_cache.py to retry — existing entries will be skipped."
        )


if __name__ == "__main__":
    main()