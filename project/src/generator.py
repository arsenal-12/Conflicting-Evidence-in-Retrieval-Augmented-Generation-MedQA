"""
===============================================================================
generator.py
===============================================================================

Purpose
-------
Generate an answer using the retrieved context.

This module builds the final prompt, sends it to the configured language
model, and packages the generated response into a GenerationResult object.

Pipeline Position
-----------------
RetrievalResult
        ↓
Build Context
        ↓
Build Prompt
        ↓
LLM (Mock / HuggingFace)
        ↓
GenerationResult

Responsibilities
----------------
• Convert retrieved chunks into a readable context.
• Build the final prompt using the prompt template.
• Generate an answer using the selected backend.
• Measure generation time.
• Return a GenerationResult object.

Why this file exists
--------------------
The Retriever only finds relevant information.

This module is responsible for turning that information into a prompt,
asking the language model to answer the question, and recording everything
needed for later evaluation.

When should I modify this file?
-------------------------------
Modify this file if you want to:

• Change the prompt format.
• Use a different LLM provider.
• Add prompt engineering.
• Add reasoning strategies.
• Add failure injection before generation.

Do NOT modify this file if you want to change:

• Retrieval
• Embedding
• Vector search

Those belong to their respective modules.

Software Engineering
--------------------
✓ Single Responsibility Principle
    Responsible only for answer generation.

✓ Strategy Pattern
    Supports multiple generation backends without changing Generator.

✓ Loose Coupling
    Generator does not know how HuggingFace or Mock backends work
    internally.

✓ Extensible Design
    New providers (OpenAI, Gemini, Claude, etc.) can be added without
    modifying Generator.generate().
"""

from typing import Any, Dict, Optional
import os
from src.logger import get_logger
from src.models import GenerationResult, RetrievalResult
from src.utils import resolve_device, timer
from time import perf_counter
from src.logger import get_logger

logger = get_logger(__name__)


def _build_context(retrieved_chunks: list) -> str:
    """
    Convert multiple retrieved document chunks into one formatted context.

    This helper prepares the retrieved documents before they are inserted into
    the prompt template.

    Each chunk is numbered to make the prompt easier for both the language
    model and the developer to read.
    """
    return "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(retrieved_chunks))


class _MockGenerationBackend:
    """
    A lightweight fake language model.

    Used only for testing the pipeline without downloading or running a real
    LLM.

    This allows the complete RAG pipeline to be tested on any machine.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float, max_new_tokens: int, top_p: float) -> str:
        """Returns a short, deterministic placeholder answer.

        Args:
            prompt: The fully-rendered prompt (unused beyond length
                reporting, since this backend does not actually call a
                model).
            temperature: Unused; accepted for interface parity with the
                real backend.
            max_new_tokens: Unused; accepted for interface parity with the
                real backend.
            top_p: Unused; accepted for interface parity with the real
                backend.

        Returns:
            A placeholder answer string.
        """
        return (
            "[MOCK GENERATION] This is a placeholder answer produced by the "
            f"mock generator backend ({self.model_name}) because "
            "generator.provider is set to 'mock' in config.yaml. Prompt "
            f"length was {len(prompt)} characters."
        )


class _HuggingFaceGenerationBackend:
    """
    Handles real text generation using a Hugging Face model.

    This backend loads the tokenizer and model once, then reuses them for
    every generation request.
    """

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(os.cpu_count())

        logger.info(
            "Loading generator model '%s' on device '%s'...",
            model_name,
            device,
        )

        token = os.getenv("HUGGINGFACE_TOKEN")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            local_files_only=False,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            local_files_only=False,
        )

        logger.info("Generator model loaded successfully.")

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
    ) -> str:
        """
        Generate an answer using the loaded Hugging Face model.
        """

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # Tokenize using the model's chat template.
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move tensors onto the configured device.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        import torch

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
            "use_cache": True,
        }

        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = max(temperature, 1e-5)
            generate_kwargs["top_p"] = top_p
        else:
            generate_kwargs["do_sample"] = False

        with torch.inference_mode():

            start = perf_counter()

            output_ids = self._model.generate(
                **inputs,
                **generate_kwargs,
            )

            elapsed = perf_counter() - start

            generated_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]

            text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            logger.info(
                "[GENERATION] %.1f sec | %d output words",
                elapsed,
                len(text.split()),
            )

            return text

class Generator:
    """
    Coordinates the complete answer generation stage.

    This class prepares the prompt, selects the configured backend, and
    returns the final GenerationResult.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Loads the generator backend configured in `config.yaml`.

        Args:
            config: The full parsed application config.

        Raises:
            ValueError: If `generator.provider` is not "huggingface" or
                "mock".
        """
        gen_cfg = config["generator"]

        self.model_name: str = gen_cfg["model_name"]
        self.provider: str = gen_cfg.get("provider", "huggingface")
        self.temperature: float = gen_cfg.get("temperature", 0.2)
        self.max_new_tokens: int = gen_cfg.get("max_new_tokens", 256)
        self.top_p: float = gen_cfg.get("top_p", 0.9)
        self.prompt_template: str = gen_cfg["prompt_template"]

        if self.provider == "mock":
            self._backend = _MockGenerationBackend(self.model_name)

        elif self.provider == "huggingface":
            device = resolve_device(gen_cfg.get("device", "auto"))
            self._backend = _HuggingFaceGenerationBackend(
                self.model_name,
                device,
            )

        else:
            raise ValueError(
                f"Unknown generator provider '{self.provider}'. "
                "Must be 'huggingface' or 'mock'."
            )

    def generate(
        self,
        retrieval_result: RetrievalResult,
        failure_mode: str = "baseline",
    ) -> GenerationResult:
        """
        Generate an answer from a RetrievalResult.

        Args:
            retrieval_result:
                Output from the Retriever containing the retrieved context.

            failure_mode:
                Name of the retrieval condition (baseline, conflicting,
            ).

        Returns:
            A populated GenerationResult.
        """

        context = _build_context(
            retrieval_result.retrieved_chunks
        )

        prompt = self.prompt_template.format(
            context=context,
            question=retrieval_result.question,
        )

        with timer() as t:
            answer = self._backend.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
            )

        logger.debug(
            "Generated answer for question_id=%s in %.2fs.",
            retrieval_result.question_id,
            t["elapsed"],
        )

        return GenerationResult(
            question_id=retrieval_result.question_id,
            question=retrieval_result.question,
            retrieved_chunks=retrieval_result.retrieved_chunks,
            prompt=prompt,
            generator_model=self.model_name,
            answer=answer,
            generation_time=t["elapsed"],
            failure_mode=failure_mode,
            faithfulness_score=None,
            safety_label=None,
        )