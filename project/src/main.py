"""
===============================================================================
main.py
===============================================================================

Load configuration
        ↓
Load questions
        ↓
Load PubMed documents
        ↓
Load/build ChromaDB
        ↓
Create retriever
        ↓
Create conflict injector + cache
        ↓
Load Qwen
        ↓
Create evaluator
        ↓
Create Decision Engine
        ↓
Create query builder + Evidence Refiner
        ↓
Create Answer Refiner
        ↓
For each question:
    ↓
    retrieve original Top-5
    ↓
    inject conflict
    ↓
    generate answer
    ↓
    evaluate answer
    ↓
    Decision Engine
       ├── NO_ACTION
       ├── ANSWER_REFINEMENT
       └── EVIDENCE_REFINEMENT
    ↓
    optionally refine + regenerate + re-evaluate
    ↓
    save JSONL record
    
Module	What it does
QuestionLoader	Loads MedQA questions
DocumentLoader	Loads PubMed documents
Embedder	Converts text to vectors
VectorStore	Builds/loads ChromaDB
Retriever	Retrieves top-k passages
FailureInjector	Adds the controlled conflict
Generator	Generates the answer using Qwen
Evaluator	Claims + attribution + faithfulness + safety
DecisionEngine	Chooses the corrective action
ConflictAwareQueryBuilder	Builds a query for replacement evidence
EvidenceRefiner	Replaces problematic evidence
AnswerRefiner	Rewrites the answer
io_utils	Builds/saves experiment records
"""

from __future__ import annotations
from time import perf_counter

import argparse
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger
import hashlib

logger = get_logger(__name__)
from src.decision_engine import DecisionAction, DecisionEngine
from src.document_loader import DocumentLoader
from src.embedder import Embedder
from src.evaluator import Evaluator
from src.evidence_cache import EvidenceCache
from src.evidence_refiner import EvidenceRefinementError, EvidenceRefiner
from src.evaluator.answer_refiner import AnswerRefiner
from src.failure_injector import FailureInjector
from src.generator import Generator
from src.io_utils import build_experiment_record, append_jsonl_result, ensure_dir
from src.logger import configure_logging, get_logger
from src.query_builder import ConflictAwareQueryBuilder
from src.question_loader import QuestionLoader
from src.retriever import Retriever
from src.utils import load_config, load_environment, timer
from src.vector_store import VectorStore
from src.evaluator.claim_extractor import LLMClient

class GeneratorLLMClientAdapter:
    """
    The adapter allows the evaluator to reuse the existing Qwen generator without
    coupling the evaluator to the RAG-specific Generator class. It converts the
    evaluator's system/user prompt interface into the interface expected by the
    generator backend.
    """
 
    def __init__(self, generator: Generator, max_new_tokens: int = 512) -> None:
 
        self._generator = generator
        self._max_new_tokens = max_new_tokens
 
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        start = perf_counter()
 
        logger.info(
            "[LLM CALL] Prompt size = %d characters",
            len(combined_prompt),
        )
 
        result = self._generator._backend.generate(
            prompt=combined_prompt,
            temperature=self._generator.temperature,
            max_new_tokens=self._max_new_tokens,
            top_p=self._generator.top_p,
        )
 
        elapsed = perf_counter() - start
        logger.info(
            "[LLM] %.2f sec | Prompt chars=%d | Output words=%d",
            elapsed,
            len(combined_prompt),
            len(result.split()),
        )
 
        return result
 
 
def _chunks_to_evaluator_format(chunks: list[str]) -> list[dict[str, str]]:
    return [{"chunk_id": f"chunk_{idx}", "text": text} for idx, text in enumerate(chunks)]
 
 
def _set_record_field(record: dict, key: str, value: object, logger) -> None:
    if key in record:
        disambiguated_key = f"{key}__refinement_stage"
        logger.warning(
            "build_experiment_record() already produced a %r field; "
            "storing this pipeline's value under %r instead to avoid "
            "overwriting it.",
            key,
            disambiguated_key,
        )
        key = disambiguated_key
    record[key] = value
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline with evaluator-driven evidence refinement.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file (default: config/config.yaml).",
    )
    return parser.parse_args()
 
 
def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    load_environment()
    configure_logging(config)
 
    logger = get_logger(__name__)
    logger.info("=" * 70)
    logger.info("Starting RAG pipeline with evaluator-driven evidence refinement.")
    logger.info("=" * 70)
 
    # ------------------------------------------------------------------
    # Stage 1: Load questions.
    # ------------------------------------------------------------------
    logger.info("Loading questions...")
    questions = QuestionLoader(config).load()
 
    # ------------------------------------------------------------------
    # Stage 2: Load corpus documents.
    # ------------------------------------------------------------------
    logger.info("Loading corpus documents...")
    documents = DocumentLoader(config).load()
    print("Number of documents:", len(documents))
 
    print("\nFirst document:")
    print(documents[0].title)
    print(documents[0].text[:500])
 
    print("\nDocument 100:")
    print(documents[100].title)
    print(documents[100].text[:500])
 
    # ------------------------------------------------------------------
    # Stage 3: Prepare the vector store.
    # ------------------------------------------------------------------
    logger.info("Preparing vector store...")
    vector_store = VectorStore(config)
 
    try:
        vector_store.load()
        logger.info("Using existing Chroma collection.")
 
        embedder = Embedder(config)
 
    except ValueError:
        logger.info("Building new Chroma collection...")
 
        embedder = Embedder(config)
        document_embeddings = embedder.embed_documents(
            [doc.text for doc in documents]
        )
 
        vector_store.build(documents, document_embeddings)
 
    # ------------------------------------------------------------------
    # Stage 5: Construct the retriever.
    # ------------------------------------------------------------------
    retriever = Retriever(config, embedder=embedder, vector_store=vector_store)
 
    # ------------------------------------------------------------------
    # Stage 6: Initialise Phase 2 failure injection components.
    # ------------------------------------------------------------------
    failure_cfg = config["failure_injection"]
 
    cache = EvidenceCache(
        cache_dir=failure_cfg["cache_directory"],
        generator=None,
        verifier=None,
        strict_cache=failure_cfg.get("strict_cache", True),
    )
 
    failure_injector = FailureInjector(
        failure_cfg,
        cache,
    )
 
    # ------------------------------------------------------------------
    # Stage 7: Load the generator model.
    # ------------------------------------------------------------------
    logger.info("Loading generator model...")
    generator = Generator(config)
 
    # ------------------------------------------------------------------
    # Stage 7.5: Initialise the Evaluator.
    # ------------------------------------------------------------------
    logger.info("Initialising evaluator...")
    evaluator_cfg = config.get("evaluator", {})
    llm_client = GeneratorLLMClientAdapter(
        generator,
        max_new_tokens=evaluator_cfg.get("max_new_tokens", 512),
    )
    evaluator = Evaluator.from_llm_client(
        llm_client,
        max_retries=evaluator_cfg.get("max_retries", 0),
        fail_fast=evaluator_cfg.get("fail_fast", False),
    )
 
    # ------------------------------------------------------------------
    # Stage 7.6: Initialise DecisionEngine, ConflictAwareQueryBuilder,
    # and EvidenceRefiner.
    # ------------------------------------------------------------------
    logger.info("Initialising decision engine and evidence refinement components...")
    decision_engine = DecisionEngine()
 
    query_builder_cfg = config.get("query_builder", {})
    query_builder = ConflictAwareQueryBuilder(
        llm_client,
        max_retries=query_builder_cfg.get("max_retries", 0),
    )
 
    evidence_refiner = EvidenceRefiner(
        query_builder=query_builder,
        retriever=retriever,
    )
    answer_refiner = AnswerRefiner(
        llm_client,
        max_retries=evaluator_cfg.get("max_retries", 2),
    )
 
    # ------------------------------------------------------------------
    # Stage 8: Run retrieval + generation + evaluation for every question.
    # ------------------------------------------------------------------
    output_dir = config["experiment"]["output_dir"]
    results_filename = config["experiment"]["results_filename"]
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, results_filename)
 
    dataset_name = config["dataset"]["name"]
    corpus_name = config["corpus"]["name"]
    embedding_model_name = config["embedding"]["model_name"]
 
    logger.info("Running retrieval + generation + evaluation for %d questions...", len(questions))
    for question in tqdm(questions, desc="Running RAG pipeline", unit="question"):
        original_retrieval = retriever.retrieve(question)
 
        print("=" * 60)
        print("MAIN FIRST PASSAGE")
        print(repr(original_retrieval.retrieved_chunks[0][:300]))
        print(hashlib.sha256(original_retrieval.retrieved_chunks[0].encode()).hexdigest()[:16])
 
        modified_retrieval = failure_injector.inject(original_retrieval)
 
        generation_result = generator.generate(
            modified_retrieval,
            failure_mode=failure_cfg["mode"],
        )
 
        evaluator_chunks = _chunks_to_evaluator_format(
            modified_retrieval.retrieved_chunks
        )
 
        with timer() as first_evaluation_timer:
            evaluation_record = evaluator.evaluate(
                question=question.question,
                answer=generation_result.answer,
                retrieved_chunks=evaluator_chunks,
            )
 
        record = build_experiment_record(
            result=generation_result,
            original_retrieval=original_retrieval,
            modified_retrieval=modified_retrieval,
            dataset_name=dataset_name,
            corpus_name=corpus_name,
            embedding_model=embedding_model_name,
            evaluation=evaluation_record.to_dict(),
        )
        decision = decision_engine.decide(evaluation_record)
        _set_record_field(record, "evidence_refinement_decision", decision.to_dict(), logger)
 
        if decision.action is DecisionAction.EVIDENCE_REFINEMENT:
            try:
                refined_retrieval, refinement_result = evidence_refiner.refine(
                    question=question,
                    original_retrieval=modified_retrieval,
                    decision=decision,
                )
 
            except EvidenceRefinementError as exc:
                logger.warning(
                    "Evidence refinement failed for question_id=%s: %s. "
                    "Recording the failure; no second-pass generation or "
                    "evaluation will be run for this question.",
                    question.question_id,
                    exc,
                )
                _set_record_field(
                    record,
                    "evidence_refinement_metadata",
                    {
                        "action": DecisionAction.EVIDENCE_REFINEMENT.value,
                        "status": "failed",
                        "error": str(exc),
                    },
                    logger,
                )
 
            else:
                refined_failure_mode = f"{failure_cfg['mode']}_refined"
 
                with timer() as second_generation_timer:
                    refinement_generation_result = generator.generate(
                        refined_retrieval,
                        failure_mode=refined_failure_mode,
                    )
 
                with timer() as second_evaluation_timer:
                    refinement_evaluation_record = evaluator.evaluate_changed_claims(
                        question=question.question,
                        old_record=evaluation_record,
                        new_answer=refinement_generation_result.answer,
                        retrieved_chunks=_chunks_to_evaluator_format(
                            refined_retrieval.retrieved_chunks
                        ),
                    )
 
                post_refinement_decision = decision_engine.decide(refinement_evaluation_record)
                if post_refinement_decision.action is DecisionAction.NO_ACTION:
                    logger.info("Refinement resolved all unsupported/unsafe claims.")
 
                elif post_refinement_decision.action is DecisionAction.EVIDENCE_REFINEMENT:
                    logger.info(
                        "Refinement improved the answer but further refinement may still be possible. "
                        "Stopping after one refinement iteration."
                    )
 
                elif post_refinement_decision.action is DecisionAction.ANSWER_REFINEMENT:
                    logger.info(
                        "Post-refinement answer still contains unsupported claims requiring "
                        "answer refinement (out of dissertation scope)."
                    )
 
                total_refinement_seconds = (
                    refinement_result.query_building_seconds
                    + refinement_result.corrective_retrieval_seconds
                    + second_generation_timer["elapsed"]
                    + second_evaluation_timer["elapsed"]
                )
 
                answer_changed = (
                    refinement_generation_result.answer != generation_result.answer
                )
 
                refinement_metadata = refinement_result.to_dict()
                refinement_metadata["second_generation_seconds"] = second_generation_timer["elapsed"]
                refinement_metadata["second_evaluation_seconds"] = second_evaluation_timer["elapsed"]
                refinement_metadata["total_refinement_seconds"] = total_refinement_seconds
                refinement_metadata["answer_changed"] = answer_changed
 
                before = evaluation_record
                after = refinement_evaluation_record
 
                before_supported = (
                    before.num_claims - before.num_unfaithful_claims
                )
 
                after_supported = (
                    after.num_claims - after.num_unfaithful_claims
                )
 
                faithfulness_improvement = (
                    before.num_unfaithful_claims
                    - after.num_unfaithful_claims
                )
 
                safety_improvement = (
                    before.num_unsafe_claims
                    - after.num_unsafe_claims
                )
 
                if before.num_unfaithful_claims > 0:
                    faithfulness_improvement_pct = (
                        faithfulness_improvement
                        / before.num_unfaithful_claims
                    ) * 100
                else:
                    faithfulness_improvement_pct = 0.0
 
                comparison = {
                    "claims_before": before.num_claims,
                    "claims_after": after.num_claims,
 
                    "supported_before": before_supported,
                    "supported_after": after_supported,
 
                    "unfaithful_before": before.num_unfaithful_claims,
                    "unfaithful_after": after.num_unfaithful_claims,
 
                    "unsafe_before": before.num_unsafe_claims,
                    "unsafe_after": after.num_unsafe_claims,
 
                    "faithfulness_improvement": faithfulness_improvement,
                    "faithfulness_improvement_percent": round(
                        faithfulness_improvement_pct,
                        2,
                    ),
 
                    "safety_improvement": safety_improvement,
                    "answer_changed": answer_changed,
 
                    "chunks_replaced": len(refinement_result.replaced_indices),
                    "chunks_skipped": len(refinement_result.skipped_indices),
                    "retrieval_changed": (
                        len(refinement_result.replaced_indices) > 0
                    ),
 
                    "query_build_time": refinement_result.query_building_seconds,
                    "retrieval_time": refinement_result.corrective_retrieval_seconds,
 
                    "generation_time_before": generation_result.generation_time,
                    "generation_time_after": refinement_generation_result.generation_time,
 
                    "generation_time_difference": (
                        generation_result.generation_time
                        - refinement_generation_result.generation_time
                    ),
 
                    "evaluation_time_before": first_evaluation_timer["elapsed"],
                    "evaluation_time_after": second_evaluation_timer["elapsed"],
 
                    "total_refinement_time": total_refinement_seconds,
 
                    "refinement_success": (
                        post_refinement_decision.action is DecisionAction.NO_ACTION
                    ),
                    "post_refinement_action": (
                        post_refinement_decision.action.name
                    ),
                }
 
                _set_record_field(record, "evidence_refinement_metadata", refinement_metadata, logger)
                _set_record_field(
                    record, "evidence_refinement_generation", refinement_generation_result.to_dict(), logger
                )
                _set_record_field(
                    record, "evidence_refinement_evaluation", refinement_evaluation_record.to_dict(), logger
                )
                _set_record_field(
                    record, "evidence_refinement_resolution", post_refinement_decision.to_dict(), logger
                )
                _set_record_field(
                    record,
                    "refinement_comparison",
                    comparison,
                    logger,
                )
 
        elif decision.action is DecisionAction.ANSWER_REFINEMENT:
 
            logger.info(
                "Running answer refinement for question_id=%s",
                question.question_id,
            )
 
            refined_answer = answer_refiner.refine(
                question=question.question,
                original_answer=generation_result.answer,
                evaluation_record=evaluation_record,
                retrieved_chunks=_chunks_to_evaluator_format(
                    modified_retrieval.retrieved_chunks
                ),
            )
 
            logger.info(
                "Original answer:\n%s",
                generation_result.answer,
            )
 
            logger.info(
                "Refined answer:\n%s",
                refined_answer,
            )
 
            with timer() as second_evaluation_timer:
                refinement_evaluation_record = evaluator.evaluate(
                    question=question.question,
                    answer=refined_answer,
                    retrieved_chunks=_chunks_to_evaluator_format(
                        modified_retrieval.retrieved_chunks
                    ),
                )
 
            post_refinement_decision = decision_engine.decide(
                refinement_evaluation_record
            )
 
            logger.info(
                "Post-refinement action: %s",
                post_refinement_decision.action.name,
            )
 
            _set_record_field(
                record,
                "answer_refinement_answer",
                refined_answer,
                logger,
            )
 
            _set_record_field(
                record,
                "answer_refinement_evaluation",
                refinement_evaluation_record.to_dict(),
                logger,
            )
 
            _set_record_field(
                record,
                "answer_refinement_resolution",
                post_refinement_decision.to_dict(),
                logger,
            )
 
        append_jsonl_result(
            record,
            output_path,
        )
 
    logger.info(
        "Pipeline complete. Results written to %s",
        output_path,
    )
 
 
def main() -> None:
    args = parse_args()
    run_pipeline(args.config)
 
 
if __name__ == "__main__":
    main()