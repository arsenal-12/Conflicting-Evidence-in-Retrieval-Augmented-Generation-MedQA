import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config, load_environment
from src.generator import Generator
from src.evaluator import Evaluator
from src.main import GeneratorLLMClientAdapter

load_environment()

config = load_config("config/config.yaml")

print("Loading Generator...")
generator = Generator(config)
print("✓ Generator loaded")

print("Creating Adapter...")
adapter = GeneratorLLMClientAdapter(generator)
print("✓ Adapter created")

print("Creating Evaluator...")
evaluator = Evaluator.from_llm_client(adapter)

print("✓ Evaluator initialized successfully")