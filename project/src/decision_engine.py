"""
decision_engine.py
==================

Reads the (frozen) evaluator's output and decides what, if anything, the
pipeline should do next:

    NO_ACTION            -- nothing to correct.
    EVIDENCE_REFINEMENT  -- at least one claim is CONTRADICTED by retrieved
                            evidence; corrective retrieval has a concrete
                            conflict to search around.
    ANSWER_REFINEMENT    -- no claim is CONTRADICTED, but at least one claim
                            is UNSUPPORTED; there is no contradicting chunk
                            to target with a new retrieval query, so the
                            answer itself needs correction.

This module performs NO LLM calls and contains NO retrieval, generation, or
evidence-rewriting logic. It only reads the already-public output of
src.evaluator (EvaluationRecord, ClaimEvaluation, ChunkAttribution, and
the associated enums) and returns a decision.

It never mutates anything it reads, and it does not import or touch any
evaluator internals beyond that public surface -- the evaluator package
remains completely untouched.

Contract with main.py
---------------------

ChunkAttribution.chunk_id values are expected to follow the
"chunk_<index>" convention produced by main.py's
_chunks_to_evaluator_format() helper, where <index> is the position of that
chunk within the exact retrieved_chunks list passed into
Evaluator.evaluate() for this answer.

This module parses that convention to recover chunk indices; if that
convention ever changes, update _chunk_index_from_id accordingly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.evaluator import (
    AttributionLabel,
    Claim,
    ClaimEvaluation,
    EvaluationRecord,
    FaithfulnessVerdict,
)


logger = logging.getLogger(__name__)


_CHUNK_ID_RE = re.compile(r"^chunk_(\d+)$")


class DecisionAction(str, Enum):
    """The three corrective actions the pipeline can take after evaluation."""

    NO_ACTION = "NO_ACTION"
    EVIDENCE_REFINEMENT = "EVIDENCE_REFINEMENT"
    ANSWER_REFINEMENT = "ANSWER_REFINEMENT"


@dataclass
class RefinementDecision:
    """The DecisionEngine's output: what action to take and why.

    `contradicting_chunk_indices` / `supporting_chunk_indices` are only
    populated for EVIDENCE_REFINEMENT decisions.

    They are the positions within the retrieved_chunks list originally
    passed to Evaluator.evaluate() of chunks that respectively
    contradicted or supported the triggering claim(s).

    ANSWER_REFINEMENT and NO_ACTION leave both lists empty.
    """

    action: DecisionAction
    reason: str
    triggering_claims: List[Claim] = field(default_factory=list)
    contradicting_chunk_indices: List[int] = field(default_factory=list)
    supporting_chunk_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "triggering_claims": [
                c.to_dict() for c in self.triggering_claims
            ],
            "contradicting_chunk_indices": self.contradicting_chunk_indices,
            "supporting_chunk_indices": self.supporting_chunk_indices,
        }


def _chunk_index_from_id(chunk_id: str) -> Optional[int]:
    """Recover a positional chunk index from a 'chunk_<index>' id.

    Returns None and logs a warning if chunk_id does not match the expected
    convention rather than raising. A single unparsable id should not abort
    the whole decision.
    """

    match = _CHUNK_ID_RE.match(chunk_id)

    if match is None:
        logger.warning(
            "chunk_id %r does not match the expected 'chunk_<index>' "
            "convention produced by main.py's "
            "_chunks_to_evaluator_format(); "
            "skipping this attribution for index-based lookups.",
            chunk_id,
        )
        return None

    return int(match.group(1))


class DecisionEngine:
    """Decides the pipeline's next corrective action from an EvaluationRecord.

    Decision policy, evaluated in priority order:

    1. EVIDENCE_REFINEMENT

       Any claim whose faithfulness verdict is CONTRADICTED triggers
       evidence refinement.

       The specific contradicting and supporting chunks are identified
       from that claim's per-chunk attributions.

    2. ANSWER_REFINEMENT

       If there are no CONTRADICTED claims, any UNSUPPORTED claim triggers
       answer refinement.

       Safety is still assessed for UNSUPPORTED claims, but an UNSUPPORTED
       claim should not be left unchanged merely because it was assessed
       SAFE.

       Since there is no contradicting chunk to target with corrective
       retrieval, the answer itself requires correction.

    3. NO_ACTION

       If there are no CONTRADICTED or UNSUPPORTED claims, no refinement
       is required.

       In practice, this means all extractable claims are SUPPORTED.

    The decision is made once per EvaluationRecord (i.e. per answer), not
    per claim.

    If any claim triggers EVIDENCE_REFINEMENT, that is the decision for
    the whole record, matching the pipeline's single-pass
    corrective-retrieval design.
    """

    def decide(
        self,
        evaluation_record: EvaluationRecord,
    ) -> RefinementDecision:
        """Inspect evaluation_record and return the corrective action.

        Args:
            evaluation_record:
                The EvaluationRecord produced by Evaluator.evaluate()
                for one generated answer.

        Returns:
            A RefinementDecision describing the action, why it was chosen,
            the triggering claims, and, for EVIDENCE_REFINEMENT, the
            relevant chunk indices.
        """

        # ---------------------------------------------------------------
        # 1. CONTRADICTED -> EVIDENCE_REFINEMENT
        # ---------------------------------------------------------------

        contradicted = [
            ce
            for ce in evaluation_record.claim_evaluations
            if ce.faithfulness == FaithfulnessVerdict.CONTRADICTED
        ]

        if contradicted:
            return self._build_evidence_refinement_decision(
                contradicted
            )

        # ---------------------------------------------------------------
        # 2. UNSUPPORTED -> ANSWER_REFINEMENT
        # ---------------------------------------------------------------

        unsupported = [
            ce
            for ce in evaluation_record.claim_evaluations
            if ce.faithfulness == FaithfulnessVerdict.UNSUPPORTED
        ]

        if unsupported:
            return self._build_answer_refinement_decision(
                unsupported
            )

        # ---------------------------------------------------------------
        # 3. NO CLAIMS -> NO_ACTION
        # ---------------------------------------------------------------

        if evaluation_record.num_claims == 0:
            logger.info(
                "DecisionEngine: NO_ACTION for question %r "
                "(answer contained no extractable claims -- likely "
                "an abstention/refusal).",
                evaluation_record.question,
            )

            return RefinementDecision(
                action=DecisionAction.NO_ACTION,
                reason=(
                    "Answer contained no extractable claims "
                    "(abstention/refusal); nothing to correct."
                ),
            )

        # ---------------------------------------------------------------
        # 4. ALL CLAIMS SUPPORTED -> NO_ACTION
        # ---------------------------------------------------------------

        logger.info(
            "DecisionEngine: NO_ACTION for question %r "
            "(all %d claim(s) are SUPPORTED).",
            evaluation_record.question,
            evaluation_record.num_claims,
        )

        return RefinementDecision(
            action=DecisionAction.NO_ACTION,
            reason="All claims are SUPPORTED.",
        )

    def _build_evidence_refinement_decision(
        self,
        contradicted: List[ClaimEvaluation],
    ) -> RefinementDecision:
        """Build an EVIDENCE_REFINEMENT decision from CONTRADICTED claims."""

        triggering_claims = [
            ce.claim for ce in contradicted
        ]

        contradicting_indices: List[int] = []
        supporting_indices: List[int] = []

        for ce in contradicted:
            for attribution in ce.attributions:

                idx = _chunk_index_from_id(
                    attribution.chunk_id
                )

                if idx is None:
                    continue

                if attribution.label == AttributionLabel.CONTRADICTS:
                    contradicting_indices.append(idx)

                elif attribution.label == AttributionLabel.SUPPORTS:
                    supporting_indices.append(idx)

        # De-duplicate while preserving first-seen order.
        #
        # A single chunk can contradict more than one triggering claim.
        contradicting_indices = list(
            dict.fromkeys(contradicting_indices)
        )

        supporting_indices = list(
            dict.fromkeys(supporting_indices)
        )

        logger.info(
            "DecisionEngine: EVIDENCE_REFINEMENT for %d claim(s); "
            "%d contradicting chunk(s), %d supporting chunk(s).",
            len(triggering_claims),
            len(contradicting_indices),
            len(supporting_indices),
        )

        return RefinementDecision(
            action=DecisionAction.EVIDENCE_REFINEMENT,
            reason=(
                f"{len(triggering_claims)} claim(s) were CONTRADICTED "
                "by retrieved evidence; corrective retrieval can "
                "target the contradicting chunk(s)."
            ),
            triggering_claims=triggering_claims,
            contradicting_chunk_indices=contradicting_indices,
            supporting_chunk_indices=supporting_indices,
        )

    def _build_answer_refinement_decision(
        self,
        unsupported: List[ClaimEvaluation],
    ) -> RefinementDecision:
        """Build an ANSWER_REFINEMENT decision from UNSUPPORTED claims."""

        triggering_claims = [
            ce.claim for ce in unsupported
        ]

        logger.info(
            "DecisionEngine: ANSWER_REFINEMENT for %d "
            "unsupported claim(s).",
            len(triggering_claims),
        )

        return RefinementDecision(
            action=DecisionAction.ANSWER_REFINEMENT,
            reason=(
                f"{len(triggering_claims)} claim(s) are UNSUPPORTED "
                "and cannot be grounded in the retrieved evidence; "
                "the answer itself requires correction rather than "
                "evidence refinement."
            ),
            triggering_claims=triggering_claims,
        )