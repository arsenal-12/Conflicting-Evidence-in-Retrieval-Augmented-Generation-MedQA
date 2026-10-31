"""
query_builder.py
==================

Conflict-Aware Query Builder: constructs a single refined retrieval query
for corrective retrieval, given the original question and the specific
claim(s)/chunk(s) a `DecisionEngine` has identified as being in conflict.

Boundary this module enforces
-------------------------------
This module's ENTIRE output contract is one plain-text search query
string. It never produces, rewrites, or fabricates evidence. The LLM used
here is only ever asked to phrase a *query*; whatever it returns is used
solely to call `Retriever.retrieve()` (wrapped as a synthetic `Question`)
and is then discarded -- only genuine corpus search results flow
downstream from there. `retriever.py` is neither imported nor modified by
this module.

Modularity
-----------
This module has no dependency on `DecisionEngine`, `ChunkAttribution`, or
any other evaluator dataclass beyond `Claim` (used only for its `.text`)
and the shared `LLMClient` protocol. Its inputs are plain strings, so it
can be reused by any caller that can supply a question, some claim texts,
and some chunk texts -- it does not need to know how those were decided.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from src.evaluator import Claim, LLMClient

logger = logging.getLogger(__name__)


class QueryBuildError(RuntimeError):
    """Raised when no usable query could be produced after all retries.

    Mirrors the error-handling convention already used by the evaluator's
    LLM-backed stages (`ClaimExtractionError`, `ClaimAttributionError`,
    `SafetyAssessmentError`): fail loudly and let the caller decide what
    to do, rather than silently degrading to a guessed query.
    """


_SYSTEM_PROMPT = """You are a retrieval query rewriting assistant.

Your ONLY task is to write one short search query that will help find \
passages in a document corpus that confirm or resolve a factual \
contradiction. You do not answer the original question, you do not state \
facts yourself, and you do not invent information -- you only produce a \
search query string suitable for a semantic similarity search over a \
corpus of passages.

Rules:
- Output ONLY the query text. No quotation marks, no explanation, no \
markdown, no labels, no preamble, no trailing commentary.
- The query must name the specific fact in dispute, phrased so a \
retrieval system can find passages that directly address it.
- Use the original question as the primary context.\
Use the contradicted claim to identify the disputed fact.\
Construct a query that retrieves evidence capable of verifying or correcting the disputed claim.

Do not bias the query toward the wording of the contradicting chunk.
- Keep the query concise: Prefer 8–15 keywords or a short search phrase.Do not exceed 20 words.
- Respond with a single line of plain text and nothing else.
"""

_USER_PROMPT_TEMPLATE = """Original question:
\"\"\"
{question}
\"\"\"

Claim(s) contradicted by retrieved evidence:
{claims_block}

Chunk(s) that contradict the claim(s) above:
{contradicting_block}

Chunk(s) that already support the claim(s) above (if any):
{supporting_block}

Write ONE refined retrieval query that will help retrieve authoritative corpus passages capable of verifying or correcting the contradicted claim.
Return ONLY the query.
"""


def _format_claims(claims: Sequence[Claim]) -> str:
    if not claims:
        return "(none)"
    return "\n".join(f"- {claim.text}" for claim in claims)


def _format_chunk_texts(chunk_texts: Sequence[str]) -> str:
    if not chunk_texts:
        return "(none)"
    return "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(chunk_texts))


class ConflictAwareQueryBuilder:
    """Builds one refined retrieval query string for corrective retrieval.

    The returned string is intended to be passed straight into
    `Retriever.retrieve()` (wrapped as a synthetic `Question`) by the
    caller -- this class does not perform retrieval itself.
    """

    def __init__(self, llm_client: LLMClient, max_retries: int = 2) -> None:
        """
        Args:
            llm_client: Any object implementing the shared `LLMClient`
                protocol (`generate(system_prompt, user_prompt) -> str`).
                In this project this is the same `GeneratorLLMClientAdapter`
                instance already used to drive the evaluator's LLM-backed
                stages -- no second model or adapter is required.
            max_retries: Number of additional attempts if the LLM response
                is empty or otherwise unusable as a query.
        """
        self._llm_client = llm_client
        self._max_retries = max(0, max_retries)

    def build(
        self,
        question: str,
        contradicted_claims: Sequence[Claim],
        contradicting_chunks: Sequence[str],
        supporting_chunks: Sequence[str] = (),
    ) -> str:
        """Construct a refined retrieval query for corrective retrieval.

        Args:
            question: The original user question, as plain text.
            contradicted_claims: The `Claim`(s) a `DecisionEngine` flagged
                as CONTRADICTED. Must be non-empty -- a conflict-aware
                query cannot be built without knowing what is in conflict.
            contradicting_chunks: Plain chunk text(s) that contradicted
                the claim(s) above.
            supporting_chunks: Plain chunk text(s) that already supported
                the claim(s) above, if any. Optional context that helps
                the LLM phrase a query that distinguishes the two sides of
                the contradiction.

        Returns:
            A single-line, non-empty query string.

        Raises:
            ValueError: If `contradicted_claims` is empty.
            QueryBuildError: If the LLM fails to produce a usable query
                after `max_retries` additional attempts.
        """
        if not contradicted_claims:
            raise ValueError(
                "ConflictAwareQueryBuilder.build() requires at least one "
                "contradicted claim to build a conflict-aware query."
            )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            question=question,
            claims_block=_format_claims(contradicted_claims),
            contradicting_block=_format_chunk_texts(contradicting_chunks),
            supporting_block=_format_chunk_texts(supporting_chunks),
        )

        last_error: Optional[str] = None
        for attempt in range(self._max_retries + 1):
            logger.debug(
                "ConflictAwareQueryBuilder: attempt %d/%d",
                attempt + 1,
                self._max_retries + 1,
            )
            try:
                raw_response = self._llm_client.generate(_SYSTEM_PROMPT, user_prompt)
            except Exception as exc:  # pragma: no cover - defensive, client-specific
                last_error = f"LLM call raised an exception: {exc}"
                logger.warning(last_error)
                continue

            query = self._clean_query(raw_response)
            if not query:
                last_error = f"LLM returned an empty or unusable query: {raw_response!r}"
                logger.warning(last_error)
                continue

            logger.info("ConflictAwareQueryBuilder produced query: %r", query)
            return query

        raise QueryBuildError(
            f"Query building failed after {self._max_retries + 1} attempt(s): {last_error}"
        )

    @staticmethod
    def _clean_query(raw_response: str) -> str:
        """Normalize a raw LLM response into a single-line query string.

        Takes the first non-empty line (defensive against a model
        prepending stray commentary despite the system prompt), then
        strips a single layer of wrapping quotes from that line, if
        present.
        """
        for line in raw_response.splitlines():
            line = line.strip()
            if not line:
                continue
            if len(line) >= 2 and line[0] in "\"'" and line[-1] == line[0]:
                line = line[1:-1].strip()
            return line
        return ""