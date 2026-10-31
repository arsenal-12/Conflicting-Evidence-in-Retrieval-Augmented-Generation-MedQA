import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main import _chunks_to_evaluator_format

chunks = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany."
]

converted = _chunks_to_evaluator_format(chunks)

print(converted)