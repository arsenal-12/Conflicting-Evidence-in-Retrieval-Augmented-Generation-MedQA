import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pprint import pprint

from src.utils import load_config, load_environment
from src.generator import Generator
from src.main import GeneratorLLMClientAdapter
from src.evaluator import Evaluator

load_environment()

config = load_config("config/config.yaml")

generator = Generator(config)
adapter = GeneratorLLMClientAdapter(generator)
evaluator = Evaluator.from_llm_client(adapter)

chunks = [
    {
        "chunk_id": "chunk_0",
        "text": "Paris is the capital city of France."
    }
]

result = evaluator.evaluate(
    question="What is the capital of France?",
    answer="The capital of France is Paris.",
    retrieved_chunks=chunks,
)

pprint(result.to_dict())