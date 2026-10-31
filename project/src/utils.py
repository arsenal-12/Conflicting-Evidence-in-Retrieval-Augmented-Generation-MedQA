"""
| Function                   | Purpose                           | Used By                 |
| -------------------------- | --------------------------------- | ----------------------- |
| `load_config()`            | Reads and validates `config.yaml` | `main.py`               |
| `load_environment()`       | Loads variables from `.env`       | `main.py`               |
| `resolve_device()`         | Chooses CPU or GPU                | `Embedder`, `Generator` |
| `generate_experiment_id()` | Creates a unique experiment ID    | `io_utils.py`           |
| `utc_timestamp()`          | Records the experiment time       | `io_utils.py`           |
| `timer()`                  | Measures execution time           | `Generator`             |
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator

import yaml
from dotenv import load_dotenv

from src.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_CONFIG_SECTIONS = (
    "dataset",
    "corpus",
    "embedding",
    "vector_store",
    "retriever",
    "generator",
    "experiment",
    "logging",
)


def load_config(config_path: str) -> Dict[str, Any]:

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            # Re-raise as ValueError so callers only need to catch one
            # exception type for "the config is broken" scenarios.
            raise ValueError(f"Failed to parse YAML config at {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} did not parse to a dictionary.")

    missing_sections = [s for s in _REQUIRED_CONFIG_SECTIONS if s not in config]
    if missing_sections:
        raise ValueError(
            f"Config at {config_path} is missing required section(s): {missing_sections}"
        )

    logger.debug("Loaded configuration from %s", config_path)
    return config


def load_environment(env_path: str = ".env") -> None:

    if os.path.isfile(env_path):
        load_dotenv(dotenv_path=env_path)
        logger.debug("Loaded environment variables from %s", env_path)
    else:
        logger.debug("No .env file found at %s; continuing with system environment only.", env_path)


def resolve_device(preferred: str = "auto") -> str:

    force_cpu = os.getenv("FORCE_CPU", "false").strip().lower() == "true"

    if preferred == "cpu" or force_cpu:
        return "cpu"

    if preferred in ("auto", "cuda"):
        try:
            # Imported lazily so that modules which never touch the GPU
            # (e.g. question_loader) do not pay the cost of importing torch.
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            logger.warning("torch is not installed; falling back to CPU.")

    return "cpu"


def generate_experiment_id() -> str:

    return uuid.uuid4().hex[:12]


def utc_timestamp() -> str:

    return datetime.now(timezone.utc).isoformat()


@contextmanager
def timer() -> Generator[Dict[str, float], None, None]:

    result: Dict[str, float] = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
