"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompletionResult:
    """Result from an LLM text completion call."""

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class TranscriptionResult:
    """Result from a speech-to-text transcription call."""

    text: str
    language: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ImageAnalysisResult:
    """Result from an image analysis call."""

    content: str
    model: str = ""


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All providers must implement the ``complete`` method.  The ``transcribe``
    and ``analyze_image`` methods are optional and should raise
    ``NotImplementedError`` when the provider does not support the capability.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Send a chat-completion request to the LLM.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: Override model ID for this call.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            A ``CompletionResult`` containing the generated text.

        Raises:
            LLMException: On any provider-side error.
        """

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file.
            language: Optional ISO language code hint.

        Returns:
            A ``TranscriptionResult`` containing the transcribed text.

        Raises:
            NotImplementedError: If the provider does not support transcription.
            LLMException: On any provider-side error.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support transcription")

    def analyze_image(
        self, prompt: str, image_source: str, *, model: str | None = None, max_tokens: int = 4096
    ) -> ImageAnalysisResult:
        """Analyse an image with a vision-capable model.

        Args:
            prompt: Text prompt describing the analysis task.
            image_source: URL or local file path of the image.
            model: Override model ID for this call.
            max_tokens: Maximum tokens in the response.

        Returns:
            An ``ImageAnalysisResult`` containing the analysis text.

        Raises:
            NotImplementedError: If the provider does not support image analysis.
            LLMException: On any provider-side error.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support image analysis")

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. ``'openai'``, ``'anthropic'``)."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default chat model ID for this provider."""
