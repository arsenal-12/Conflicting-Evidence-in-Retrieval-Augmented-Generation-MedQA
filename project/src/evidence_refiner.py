"""
===============================================================================
evidence_refiner.py
===============================================================================

Purpose
-------
Implements the EVIDENCE_REFINEMENT path of the evaluator-driven refinement
loop: given a RefinementDecision of type EVIDENCE_REFINEMENT, resolve the
contradicting/supporting chunk indices into text, build one conflict-aware
retrieval query, retrieve replacement evidence from the existing PubMed
corpus via the unchanged Retriever, and surgically replace only the
contradicting chunk(s) in the original RetrievalResult.

Design constraints (per project architecture)
-----------------------------------------------
- Never rewrites or fabricates evidence; every replacement chunk comes from
  Retriever.retrieve() against the real corpus.
- Only replaces the chunk(s) identified by DecisionEngine as CONTRADICTS;
  every other chunk in the original RetrievalResult is left untouched.
- The total number of retrieved chunks is unchanged.
- Only one refinement iteration; this module performs exactly one
  QueryBuilder call and one Retriever call per invocation.
- ANSWER_REFINEMENT and NO_ACTION are out of scope for this module -- the
  caller (main.py) is responsible for only invoking EvidenceRefiner when
  DecisionEngine.decide() returned EVIDENCE_REFINEMENT.
- Does not modify DecisionEngine, ConflictAwareQueryBuilder, Retriever, or
  RetrievalResult's/Question's schemas; it only calls their existing
  public interfaces.

Chunk index convention
-----------------------
`RefinementDecision.contradicting_chunk_indices` /
`.supporting_chunk_indices` are positions within the exact
`retrieved_chunks` list that was passed to `Evaluator.evaluate()` for the
answer being refined (see decision_engine.py's module docstring and
main.py's `_chunks_to_evaluator_format()`). This module assumes the
`original_retrieval` it is given is that same RetrievalResult (i.e. the
post-FailureInjector `modified_retrieval` in main.py's current loop), so
indices resolve directly against `original_retrieval.retrieved_chunks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.decision_engine import DecisionAction, RefinementDecision
from src.logger import get_logger
from src.models import Question, RetrievalResult
from src.query_builder import ConflictAwareQueryBuilder, QueryBuildError
from src.retriever import Retriever
from src.utils import timer

logger = get_logger(__name__)


class EvidenceRefinementError(RuntimeError):
    """Raised when the EVIDENCE_REFINEMENT path cannot be completed.

    Mirrors the fail-loudly convention already used by
    `ConflictAwareQueryBuilder` (`QueryBuildError`) and the evaluator's
    LLM-backed stages: refinement is expected to be reliable enough that a
    failure should surface to the caller rather than be silently
    swallowed into a degraded/guessed result.
    """


# `RefinementResult.status` values. Kept as plain string constants (not an
# Enum) to match the lightweight style of this module's other status-ish
# fields; both are checked by value, not by identity.
STATUS_SUCCESS = "success"
STATUS_NO_VALID_REPLACEMENT = "no_valid_replacement"


@dataclass
class RefinementResult:
    """Metadata describing one EVIDENCE_REFINEMENT attempt.

    Kept separate from `RetrievalResult` rather than added as new fields
    on it, since `RetrievalResult`'s schema is an existing interface this
    project's architecture says must not be modified. Callers that want
    per-question refinement provenance for analysis/logging should
    persist this object alongside the returned `RetrievalResult`, not
    inside it.
    """

    action: DecisionAction
    status: str
    refined_query: str
    replaced_indices: List[int]
    skipped_indices: List[int]
    original_chunks_replaced: List[str]
    original_scores_replaced: List[float]
    replacement_chunks: List[str]
    replacement_scores: List[float]
    reason: str
    query_building_seconds: float
    corrective_retrieval_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "status": self.status,
            "refined_query": self.refined_query,
            "replaced_indices": self.replaced_indices,
            "skipped_indices": self.skipped_indices,
            "original_chunks_replaced": self.original_chunks_replaced,
            "original_scores_replaced": self.original_scores_replaced,
            "replacement_chunks": self.replacement_chunks,
            "replacement_scores": self.replacement_scores,
            "reason": self.reason,
            "query_building_seconds": self.query_building_seconds,
            "corrective_retrieval_seconds": self.corrective_retrieval_seconds,
        }


class EvidenceRefiner:
    """Executes the EVIDENCE_REFINEMENT corrective-retrieval path.

    Wires together the already-completed `ConflictAwareQueryBuilder` and
    the unchanged `Retriever` to perform surgical, corpus-grounded
    evidence replacement. Contains no LLM calls of its own and no
    retrieval logic of its own -- it only resolves indices to text,
    delegates query construction and retrieval to the existing modules,
    and reassembles the result.
    """

    def __init__(
        self,
        query_builder: ConflictAwareQueryBuilder,
        retriever: Retriever,
    ) -> None:
        self._query_builder = query_builder
        self._retriever = retriever

    def refine(
        self,
        question: Question,
        original_retrieval: RetrievalResult,
        decision: RefinementDecision,
    ) -> Tuple[RetrievalResult, RefinementResult]:
        """Perform one surgical evidence-refinement pass.

        Args:
            question: The original `Question` (its `.question` text is
                passed to the QueryBuilder as context; `.question_id` and
                `.ground_truth_answer` are carried into the synthetic
                refinement query).
            original_retrieval: The `RetrievalResult` whose
                `retrieved_chunks` the decision's chunk indices were
                computed against (i.e. the same chunk list that was
                passed to `Evaluator.evaluate()` for this answer).
            decision: A `RefinementDecision` with
                `action == DecisionAction.EVIDENCE_REFINEMENT`.

        Returns:
            A tuple of:
              - a new `RetrievalResult`, identical to `original_retrieval`
                except that the contradicting chunk(s) (and their scores)
                have been replaced with newly retrieved evidence. Chunk
                count is unchanged. `original_retrieval` itself is not
                mutated.
              - a `RefinementResult` recording what was replaced and with
                what, for logging/analysis.

        Raises:
            ValueError: If `decision.action` is not EVIDENCE_REFINEMENT,
                or if it has no contradicting chunk indices to replace.
            EvidenceRefinementError: If chunk indices are out of range,
                query building fails, or corrective retrieval returns no
                chunks.
        """
        if decision.action is not DecisionAction.EVIDENCE_REFINEMENT:
            raise ValueError(
                "EvidenceRefiner.refine() requires a decision with action "
                f"EVIDENCE_REFINEMENT, got {decision.action.value!r}. "
                "ANSWER_REFINEMENT and NO_ACTION are out of scope for this "
                "module -- the caller must branch on DecisionEngine's "
                "output before invoking EvidenceRefiner."
            )

        if not decision.contradicting_chunk_indices:
            raise ValueError(
                "RefinementDecision has action EVIDENCE_REFINEMENT but no "
                "contradicting_chunk_indices to replace; nothing to refine."
            )

        total_chunks = len(original_retrieval.retrieved_chunks)
        contradicting_indices = list(dict.fromkeys(decision.contradicting_chunk_indices))
        for idx in contradicting_indices:
            if not (0 <= idx < total_chunks):
                raise EvidenceRefinementError(
                    f"contradicting_chunk_indices contains index {idx}, "
                    f"which is out of range for retrieved_chunks of length "
                    f"{total_chunks}. This means the DecisionEngine's "
                    "chunk_<index> convention and the retrieved_chunks "
                    "list passed here are out of sync -- check that "
                    "`original_retrieval` is the exact RetrievalResult "
                    "that was passed to Evaluator.evaluate() for this "
                    "answer."
                )

        contradicting_texts = [
            original_retrieval.retrieved_chunks[idx] for idx in contradicting_indices
        ]
        supporting_texts = [
            original_retrieval.retrieved_chunks[idx]
            for idx in dict.fromkeys(decision.supporting_chunk_indices)
            if 0 <= idx < total_chunks
        ]

        # Steps 1-3: build ONE conflict-aware query.
        try:
            with timer() as query_building_timer:
                refined_query = self._query_builder.build(
                    question=question.question,
                    contradicted_claims=decision.triggering_claims,
                    contradicting_chunks=contradicting_texts,
                    supporting_chunks=supporting_texts,
                )
        except QueryBuildError as exc:
            raise EvidenceRefinementError(
                f"Conflict-aware query building failed: {exc}"
            ) from exc

        # Steps 4-5: retrieve replacement evidence via the unchanged
        # Retriever, from the same PubMed corpus (Retriever.retrieve()
        # searches the same VectorStore/corpus it was constructed with --
        # there is no separate index for refinement).
        refinement_question = Question(
            question_id=f"{question.question_id}_refinement",
            question=refined_query,
            # Retriever.retrieve() never reads ground_truth_answer -- it
            # is a required field on Question with no default, so it is
            # carried over from the original question purely to satisfy
            # the dataclass constructor, not because it plays any role in
            # retrieval.
            ground_truth_answer=question.ground_truth_answer,
        )
        with timer() as corrective_retrieval_timer:
            replacement_retrieval = self._retriever.retrieve(refinement_question)

        if not replacement_retrieval.retrieved_chunks:
            raise EvidenceRefinementError(
                "Corrective retrieval returned zero chunks for refined "
                f"query {refined_query!r}; cannot perform surgical "
                "replacement."
            )

        # Steps 6-8: surgical replacement -- replace ONLY the
        # contradicting chunk(s), keep every other chunk and the total
        # count unchanged.
        #
        # For each contradicting slot, walk the already-retrieved
        # candidate list (in retrieval-rank order; no re-querying the
        # Retriever) and take the first candidate that is:
        #   - not identical to the chunk currently occupying that slot
        #     (identical == a no-op "replacement" that isn't really one), and
        #   - not already present anywhere else in the working chunk set
        #     (prevents introducing a duplicate into the evidence set).
        # If no such candidate exists among what was retrieved, that slot
        # is left unchanged and recorded as skipped -- this module never
        # issues a second retrieval call to find a better one.
        new_chunks = list(original_retrieval.retrieved_chunks)
        new_scores = list(original_retrieval.retrieval_scores)
        replaced_indices: List[int] = []
        skipped_indices: List[int] = []
        original_scores_replaced: List[float] = []
        replacement_chunks_used: List[str] = []
        replacement_scores_used: List[float] = []

        candidates = list(
            zip(
                replacement_retrieval.retrieved_chunks,
                replacement_retrieval.retrieval_scores,
            )
        )

        for target_idx in contradicting_indices:
            original_text_at_slot = new_chunks[target_idx]
            chosen: tuple[str, float] | None = None

            for candidate_text, candidate_score in candidates:
                if candidate_text == original_text_at_slot:
                    # Would-be no-op replacement -- try the next candidate.
                    continue
                if candidate_text in new_chunks:
                    # Already present elsewhere in the evidence set --
                    # would introduce a duplicate. Try the next candidate.
                    continue
                chosen = (candidate_text, candidate_score)
                break

            if chosen is None:
                skipped_indices.append(target_idx)
                logger.info(
                    "EvidenceRefiner: no valid replacement for chunk index "
                    "%d (question_id=%s) -- every retrieved candidate was "
                    "either identical to the existing chunk or already "
                    "present elsewhere in the evidence set. Slot left "
                    "unchanged.",
                    target_idx,
                    question.question_id,
                )
                continue

            replacement_text, replacement_score = chosen
            original_scores_replaced.append(new_scores[target_idx])
            new_chunks[target_idx] = replacement_text
            new_scores[target_idx] = replacement_score
            replaced_indices.append(target_idx)
            replacement_chunks_used.append(replacement_text)
            replacement_scores_used.append(replacement_score)

        status = STATUS_SUCCESS if replaced_indices else STATUS_NO_VALID_REPLACEMENT

        # Step 9: return a new RetrievalResult -- same shape, same length;
        # only the successfully-replaced indices changed. `original_retrieval`
        # itself is left untouched.
        new_retrieval_result = RetrievalResult(
            question_id=original_retrieval.question_id,
            question=original_retrieval.question,
            retrieved_chunks=new_chunks,
            retrieval_scores=new_scores,
            corpus_name=original_retrieval.corpus_name,
        )

        refinement_result = RefinementResult(
            action=decision.action,
            status=status,
            refined_query=refined_query,
            replaced_indices=replaced_indices,
            skipped_indices=skipped_indices,
            original_chunks_replaced=[
                original_retrieval.retrieved_chunks[idx] for idx in replaced_indices
            ],
            original_scores_replaced=original_scores_replaced,
            replacement_chunks=replacement_chunks_used,
            replacement_scores=replacement_scores_used,
            reason=decision.reason,
            query_building_seconds=query_building_timer["elapsed"],
            corrective_retrieval_seconds=corrective_retrieval_timer["elapsed"],
        )

        logger.info(
            "EvidenceRefiner: status=%s -- replaced %d chunk(s) at indices "
            "%s, skipped %d slot(s) at indices %s for question_id=%s using "
            "refined query %r.",
            status,
            len(replaced_indices),
            replaced_indices,
            len(skipped_indices),
            skipped_indices,
            question.question_id,
            refined_query,
        )

        return new_retrieval_result, refinement_result