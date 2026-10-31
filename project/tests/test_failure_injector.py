from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.failure_injector import FailureInjector
from src.models import RetrievalResult


# ------------------------------------------------------------------
# Fake cache (instead of reading the real cache)
# ------------------------------------------------------------------
class FakeCache:
    def get(self, document_id, failure_type):
        return "THIS IS A CONFLICTING PASSAGE"


cfg = SimpleNamespace(
    enabled=True,
    mode="conflicting",
    strategy="append",
    selection_strategy="top_rank",
    intensity=0.2,
)

injector = FailureInjector(cfg, FakeCache())

retrieval = RetrievalResult(
    question_id="q1",
    question="What causes diabetes?",
    retrieved_chunks=[
        "Original passage 1",
        "Original passage 2",
        "Original passage 3",
    ],
    retrieval_scores=[
        0.95,
        0.87,
        0.80,
    ],
    corpus_name="pubmed",
)

new_result = injector.inject(retrieval)

print("\nOriginal chunks:")
for c in retrieval.retrieved_chunks:
    print("-", c)

print("\nInjected chunks:")
for c in new_result.retrieved_chunks:
    print("-", c)

print("\nScores:")
print(new_result.retrieval_scores)