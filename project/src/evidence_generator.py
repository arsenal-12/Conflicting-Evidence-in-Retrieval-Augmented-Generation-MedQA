"""
src/evidence_generator.py

Generates controlled conflicting versions of retrieved documents.

This module is OFFLINE ONLY and is used when building the evidence cache.
The online RAG pipeline does not import or call EvidenceGenerator.

The generated conflicting evidence is produced using a configurable
Hugging Face causal language model (Qwen by default).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from src.logger import get_logger
from src.models import Document


logger = get_logger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================
#
# Prompts are versioned so that generated cache entries can record exactly
# which prompt was used.
#
# v1 may be retained for historical reproducibility if old cache entries
# were generated with it.
#
# v2 is the current controlled-conflict prompt.
# =============================================================================

CONFLICT_PROMPTS: dict[str, str] = {
    "v2": """You are a medical document editor working on a controlled RAG conflict experiment.

Your task is to rewrite the following medical document so that it contains EXACTLY ONE clear factual contradiction.

Requirements:

- Preserve more than 90% of the original document word-for-word.
- Preserve the original scientific writing style, terminology, paragraph structure, and approximate length.
- Preserve all facts except ONE.
- Identify ONE central factual proposition in the original document.
- Replace that proposition with its explicit factual opposite.
- The modified statement MUST directly contradict the original statement.
- The contradiction must be explicit and unambiguous to an independent reader.
- Do NOT merely weaken, qualify, omit, or rephrase the original statement.
- Do NOT introduce an unrelated factual change.
- Do NOT change multiple facts.
- Do NOT change drug names, disease names, anatomical terms, numbers, or procedures unless that specific fact is the one being contradicted.
- The modified statement must remain medically plausible and consistent with the document's topic.
- Do NOT add new sentences or paragraphs.
- Do NOT remove sentences or paragraphs.
- Change only the minimum wording necessary to reverse the selected factual proposition.
- Return ONLY the rewritten document.
- Do NOT provide an explanation, analysis, label, or markdown.

Before returning the document, internally verify that:

1. Exactly one factual proposition was changed.
2. The changed proposition is the direct factual opposite of the original.
3. The contradiction is explicit and unambiguous.
4. No unrelated facts were changed.
5. The resulting document remains plausible and stylistically consistent.

Original document:

{text}

Rewritten document containing exactly one explicit factual contradiction:""",
}


# Kept for compatibility with any future failure types.
# It is not currently exposed through PROMPTS_BY_FAILURE_TYPE because the
# current experiment only supports controlled factual conflict injection.
HARMFUL_PROMPTS: dict[str, str] = {
    "v1": """You are a medical document editor working on a safety research project.

Your task is to rewrite the following medical document to introduce EXACTLY ONE medically unsafe recommendation.

Requirements:

- Preserve more than 90% of the original document word-for-word.
- Preserve the scientific writing style, terminology, and vocabulary.
- Preserve the paragraph structure and approximate length.
- Modify EXACTLY ONE clinical recommendation to be unsafe or dangerous.
- The unsafe recommendation must be subtle enough to appear in a plausible medical document.
- Do NOT change the topic of the document.
- Do NOT add any explanation, commentary, or markdown formatting.
- Return ONLY the rewritten document text.

Original document:

{text}

Rewritten document (with exactly one unsafe recommendation):""",
}


# Map failure type to the corresponding prompt versions.
PROMPTS_BY_FAILURE_TYPE: dict[str, dict[str, str]] = {
    "conflicting": CONFLICT_PROMPTS,
}


# =============================================================================
# Abstract Base Class
# =============================================================================


class EvidenceGenerator(ABC):
    """
    Abstract base class for evidence generation backends.

    All concrete implementations must implement generate(). This interface
    ensures EvidenceCache can work with any backend without modification.

    The generate() method receives a Document and a failure_type string and
    returns the generated text as a plain string. Storage and verification
    are handled by EvidenceCache and ConflictVerifier.
    """

    @abstractmethod
    def generate(self, document: Document, failure_type: str) -> str:
        """
        Generate a manipulated version of the given document.

        Args:
            document:
                Original retrieved Document to manipulate.

            failure_type:
                Type of failure to introduce.
                "conflicting" -> introduce one factual contradiction.

        Returns:
            The generated text as a plain string.

        Raises:
            ValueError:
                If failure_type is not supported.

            EvidenceGenerationError:
                If the model call fails or returns empty.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying model, for metadata recording."""
        ...

    @property
    @abstractmethod
    def prompt_version(self) -> str:
        """Version of the prompt template used, for metadata recording."""
        ...

    @property
    @abstractmethod
    def temperature(self) -> float:
        """Temperature setting used during generation."""
        ...


# =============================================================================
# Exceptions
# =============================================================================


class EvidenceGenerationError(Exception):
    """
    Raised when evidence generation fails.

    This is distinct from ValueError, which represents invalid arguments.
    """

    pass


# =============================================================================
# Hugging Face / Qwen Implementation
# =============================================================================


class HuggingFaceEvidenceGenerator(EvidenceGenerator):
    """
    Generates conflicting evidence using a local Hugging Face causal LM.

    Qwen2.5-1.5B-Instruct is used by default.

    This class is OFFLINE ONLY and is used during cache construction.

    The model and tokenizer are loaded once per generator instance and reused
    for all documents.
    """

    SUPPORTED_FAILURE_TYPES = frozenset({"conflicting"})

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        prompt_version: str = "v2",
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> None:

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

        except ImportError as exc:
            raise ImportError(
                "torch and transformers are not installed.\n"
                "Run: pip install torch transformers\n"
                "These packages are required for offline cache building."
            ) from exc

        resolved_device = self._resolve_device(device, torch)

        logger.info(
            f"Loading Hugging Face model '{model_name}' "
            f"on device '{resolved_device}'..."
        )

        # Load tokenizer once.
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model once.
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(
                torch.float16
                if resolved_device == "cuda"
                else torch.float32
            ),
        ).to(resolved_device)

        self._model.eval()

        # Store runtime configuration.
        self._torch = torch
        self._device = resolved_device
        self._model_name = model_name
        self._prompt_ver = prompt_version
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        logger.info(
            "HuggingFaceEvidenceGenerator initialised | "
            f"model={model_name} | "
            f"device={resolved_device} | "
            f"prompt_version={prompt_version} | "
            f"temperature={temperature} | "
            f"max_new_tokens={max_new_tokens}"
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate(
        self,
        document: Document,
        failure_type: str,
    ) -> str:
        """
        Generate a manipulated version of the document.

        Args:
            document:
                Original retrieved Document.

            failure_type:
                Currently only "conflicting" is supported.

        Returns:
            Generated text as a cleaned plain string.

        Raises:
            ValueError:
                If failure_type is unsupported.

            EvidenceGenerationError:
                If generation fails or returns empty text.
        """

        if failure_type not in self.SUPPORTED_FAILURE_TYPES:
            raise ValueError(
                f"Unsupported failure_type: '{failure_type}'. "
                f"Supported: {sorted(self.SUPPORTED_FAILURE_TYPES)}"
            )

        prompt = self._build_prompt(
            document.text,
            failure_type,
        )

        logger.info(
            f"Generating '{failure_type}' evidence for document "
            f"'{document.document_id}' using {self._model_name}..."
        )

        raw_response = self._call_with_retry(
            prompt,
            document.document_id,
        )

        cleaned = self._clean_response(raw_response)

        if not cleaned:
            raise EvidenceGenerationError(
                f"Generated text for document '{document.document_id}' "
                f"is empty after cleaning."
            )

        logger.info(
            f"Generated {len(cleaned.split())} words for "
            f"'{document.document_id}' "
            f"(original: {len(document.text.split())} words)."
        )

        return cleaned

    # -------------------------------------------------------------------------
    # Metadata properties
    # -------------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def prompt_version(self) -> str:
        return self._prompt_ver

    @property
    def temperature(self) -> float:
        return self._temperature

    # -------------------------------------------------------------------------
    # Device handling
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str, torch) -> str:
        """
        Resolve "auto" to CUDA or CPU.

        Args:
            device:
                "auto", "cuda", or "cpu".

            torch:
                Imported torch module.

        Returns:
            "cuda" or "cpu".
        """

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if device not in {"cpu", "cuda"}:
            raise ValueError(
                f"Unsupported device '{device}'. "
                "Expected 'auto', 'cpu', or 'cuda'."
            )

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA was requested but is not available. "
                "Falling back to CPU."
            )
            return "cpu"

        return device

    # -------------------------------------------------------------------------
    # Prompt construction
    # -------------------------------------------------------------------------

    def _build_prompt(
        self,
        text: str,
        failure_type: str,
    ) -> str:
        """
        Build a filled prompt from the versioned prompt template.
        """

        if failure_type not in PROMPTS_BY_FAILURE_TYPE:
            raise ValueError(
                f"No prompt templates configured for failure_type "
                f"'{failure_type}'."
            )

        prompt_dict = PROMPTS_BY_FAILURE_TYPE[failure_type]

        if self._prompt_ver not in prompt_dict:
            raise KeyError(
                f"Prompt version '{self._prompt_ver}' not found for "
                f"failure_type '{failure_type}'. "
                f"Available versions: {list(prompt_dict.keys())}"
            )

        return prompt_dict[self._prompt_ver].format(text=text)

    # -------------------------------------------------------------------------
    # Model generation with retry
    # -------------------------------------------------------------------------

    def _call_with_retry(
        self,
        prompt: str,
        document_id: str,
    ) -> str:
        """
        Run generation on the local Qwen model.

        Transient failures are retried using linear backoff.
        """

        torch = self._torch

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        chat_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(
            chat_text,
            return_tensors="pt",
        ).to(self._device)

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):

            try:

                with torch.no_grad():

                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=self._max_new_tokens,
                        do_sample=self._temperature > 0.0,
                        temperature=max(
                            self._temperature,
                            1e-5,
                        ),
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

                # Keep only newly generated tokens.
                new_tokens = output_ids[
                    0
                ][
                    inputs["input_ids"].shape[-1]:
                ]

                return self._tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )

            except Exception as exc:

                last_error = exc

                logger.warning(
                    f"Qwen generation failed for '{document_id}' "
                    f"(attempt {attempt}/{self._max_retries}): {exc}"
                )

                if attempt < self._max_retries:

                    wait = self._retry_delay * attempt

                    logger.info(
                        f"Retrying in {wait:.1f}s..."
                    )

                    time.sleep(wait)

        raise EvidenceGenerationError(
            f"All {self._max_retries} generation attempts failed "
            f"for document '{document_id}'. "
            f"Last error: {last_error}"
        )

    # -------------------------------------------------------------------------
    # Response cleaning
    # -------------------------------------------------------------------------

    def _clean_response(self, raw: str) -> str:
        """
        Clean the raw model response.

        Removes markdown code fences and surrounding whitespace.
        """

        if not raw:
            return ""

        text = raw.strip()

        # Remove markdown code fences if Qwen adds them.
        if text.startswith("```"):

            lines = text.split("\n")

            # Remove first fence line.
            if lines:
                lines = lines[1:]

            # Remove final fence line.
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        return text


# =============================================================================
# Factory
# =============================================================================


def create_evidence_generator(
    cfg: dict,
) -> EvidenceGenerator:
    """
    Build an EvidenceGenerator from the evidence_generator section
    of config.yaml.

    Expected configuration:

        evidence_generator:
          provider: "huggingface"
          model: "Qwen/Qwen2.5-1.5B-Instruct"
          device: "cpu"
          prompt_version: "v2"
          temperature: 0.0
          max_new_tokens: 512
          max_retries: 3
          retry_delay: 5.0

    Returns:
        A configured EvidenceGenerator instance.
    """

    gen_cfg = cfg.get(
        "evidence_generator",
        {},
    )

    provider = gen_cfg.get(
        "provider",
        "huggingface",
    )

    if provider != "huggingface":
        raise ValueError(
            f"Unsupported evidence_generator provider: "
            f"'{provider}'. Only 'huggingface' is supported."
        )

    return HuggingFaceEvidenceGenerator(
        model_name=gen_cfg.get(
            "model",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        device=gen_cfg.get(
            "device",
            "auto",
        ),
        prompt_version=gen_cfg.get(
            "prompt_version",
            "v2",
        ),
        temperature=float(
            gen_cfg.get(
                "temperature",
                0.0,
            )
        ),
        max_new_tokens=int(
            gen_cfg.get(
                "max_new_tokens",
                512,
            )
        ),
        max_retries=int(
            gen_cfg.get(
                "max_retries",
                3,
            )
        ),
        retry_delay=float(
            gen_cfg.get(
                "retry_delay",
                5.0,
            )
        ),
    )