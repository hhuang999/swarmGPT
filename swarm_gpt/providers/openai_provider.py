"""OpenAI LLM provider implementation (GPT-4o + Whisper + Vision)."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

from openai import OpenAI

from swarm_gpt.exception import LLMException
from swarm_gpt.providers.base import (
    CompletionResult,
    ImageAnalysisResult,
    LLMProvider,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models.

    Supports chat completion (GPT-4o), speech-to-text (Whisper), and
    image analysis (GPT-4o Vision).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        default_model: str = "gpt-4o-2024-05-13",
        default_vision_model: str = "gpt-4o-2024-05-13",
        default_stt_model: str = "whisper-1",
        organization: str | None = None,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
            default_model: Default chat completion model.
            default_vision_model: Default vision model.
            default_stt_model: Default speech-to-text model.
            organization: Optional OpenAI organisation ID.
        """
        self._client = OpenAI(api_key=api_key, organization=organization)
        self._default_model = default_model
        self._default_vision_model = default_vision_model
        self._default_stt_model = default_stt_model

    # -- LLMProvider interface -------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return "openai"

    @property
    def default_model(self) -> str:
        """Default chat model ID for this provider."""
        return self._default_model

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Send a chat-completion request to OpenAI.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: Override model ID for this call.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            A ``CompletionResult`` containing the generated text.

        Raises:
            LLMException: On any OpenAI API error.
        """
        try:
            response = self._client.chat.completions.create(
                model=model or self._default_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            usage: dict[str, int] = {}
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return CompletionResult(content=content, model=response.model, usage=usage)
        except LLMException:
            raise
        except Exception as exc:
            raise LLMException(f"OpenAI completion failed: {exc}") from exc

    def transcribe(self, audio_path: str, *, language: str | None = None) -> TranscriptionResult:
        """Transcribe an audio file using OpenAI Whisper.

        Args:
            audio_path: Path to the audio file.
            language: Optional ISO language code hint.

        Returns:
            A ``TranscriptionResult`` containing the transcribed text.

        Raises:
            LLMException: On any OpenAI API error.
        """
        try:
            with open(audio_path, "rb") as audio_file:
                kwargs: dict[str, Any] = {"model": self._default_stt_model, "file": audio_file}
                if language is not None:
                    kwargs["language"] = language
                response = self._client.audio.transcriptions.create(**kwargs)
            return TranscriptionResult(text=response.text, language=language or "")
        except Exception as exc:
            raise LLMException(f"OpenAI transcription failed: {exc}") from exc

    def analyze_image(
        self,
        prompt: str,
        image_source: str,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> ImageAnalysisResult:
        """Analyse an image using OpenAI GPT-4o Vision.

        Args:
            prompt: Text prompt describing the analysis task.
            image_source: URL or local file path of the image.
            model: Override model ID for this call.
            max_tokens: Maximum tokens in the response.

        Returns:
            An ``ImageAnalysisResult`` containing the analysis text.

        Raises:
            LLMException: On any OpenAI API error.
        """
        image_content = self._build_image_content(image_source)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content,
                ],
            }
        ]
        result = self.complete(
            messages, model=model or self._default_vision_model, max_tokens=max_tokens
        )
        return ImageAnalysisResult(content=result.content, model=result.model)

    # -- Private helpers -------------------------------------------------------

    @staticmethod
    def _build_image_content(image_source: str) -> dict[str, Any]:
        """Build the image content dict for the Vision API.

        If *image_source* looks like a URL it is passed through directly.
        Otherwise it is treated as a local file and base64-encoded.

        Args:
            image_source: URL or local file path.

        Returns:
            A content dict suitable for the OpenAI Vision API.
        """
        if image_source.startswith(("http://", "https://")):
            return {"type": "image_url", "image_url": {"url": image_source}}

        path = Path(image_source)
        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        with open(path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_data}"
        return {"type": "image_url", "image_url": {"url": data_url}}
