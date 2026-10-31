import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config, load_environment
from src.generator import Generator
from src.main import GeneratorLLMClientAdapter

load_environment()

config = load_config("config/config.yaml")

generator = Generator(config)
adapter = GeneratorLLMClientAdapter(generator)

response = adapter.generate(
    "You are a helpful assistant.",
    "Reply with exactly one word: hello"
)

print(response)