"""Anthropic LLM provider implementation (Claude + Vision)."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

from swarm_gpt.exception import LLMException
from swarm_gpt.providers.base import CompletionResult, ImageAnalysisResult, LLMProvider

logger = logging.getLogger(__name__)

# Anthropic SDK is optional -- only required when this provider is used.
try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude models.

    Supports chat completion (Claude) and image analysis (Claude Vision).
    Does **not** support speech-to-text transcription.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        default_model: str = "claude-sonnet-4-20250514",
        default_vision_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY``
                env var.
            default_model: Default chat completion model.
            default_vision_model: Default vision model.
        """
        if anthropic is None:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider.  "
                "Install it with: pip install anthropic"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._default_model = default_model
        self._default_vision_model = default_vision_model

    # -- LLMProvider interface -------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return "anthropic"

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
        """Send a chat-completion request to Anthropic Claude.

        The Anthropic API expects the last element of *messages* to be a
        single ``user`` turn.  This method extracts a ``system`` message (if
        present as the first element) and passes the rest to the API.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: Override model ID for this call.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            A ``CompletionResult`` containing the generated text.

        Raises:
            LLMException: On any Anthropic API error.
        """
        try:
            system_prompt, api_messages = self._split_messages(messages)
            kwargs: dict[str, Any] = {
                "model": model or self._default_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": api_messages,
            }
            if system_prompt is not None:
                kwargs["system"] = system_prompt
            response = self._client.messages.create(**kwargs)
            content = response.content[0].text if response.content else ""
            usage: dict[str, int] = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }
            return CompletionResult(content=content, model=response.model, usage=usage)
        except LLMException:
            raise
        except Exception as exc:
            raise LLMException(f"Anthropic completion failed: {exc}") from exc

    def analyze_image(
        self,
        prompt: str,
        image_source: str,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> ImageAnalysisResult:
        """Analyse an image using Anthropic Claude Vision.

        Args:
            prompt: Text prompt describing the analysis task.
            image_source: URL or local file path of the image.
            model: Override model ID for this call.
            max_tokens: Maximum tokens in the response.

        Returns:
            An ``ImageAnalysisResult`` containing the analysis text.

        Raises:
            LLMException: On any Anthropic API error.
        """
        media_block = self._build_media_block(image_source)
        user_content: list[dict[str, Any]] = [
            {"type": "image", "source": media_block},
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": user_content}]
        result = self.complete(
            messages, model=model or self._default_vision_model, max_tokens=max_tokens
        )
        return ImageAnalysisResult(content=result.content, model=result.model)

    # -- Private helpers -------------------------------------------------------

    @staticmethod
    def _split_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Extract a system message and return the remaining conversation.

        Anthropic expects the system prompt as a separate parameter, not
        inside the messages list.  This method extracts the first ``system``
        message (if any) and returns it alongside the filtered messages.

        Args:
            messages: Raw message list that may contain system messages.

        Returns:
            A ``(system_prompt, api_messages)`` tuple.
        """
        system_prompt: str | None = None
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system" and system_prompt is None:
                system_prompt = msg["content"]
            else:
                # Map "system" role messages (after the first) to "user" so
                # they still reach the model without breaking the API contract.
                api_messages.append(
                    {"role": "user" if msg.get("role") == "system" else msg["role"], "content": msg["content"]}  # type: ignore[dict-item]
                )
        return system_prompt, api_messages

    @staticmethod
    def _build_media_block(image_source: str) -> dict[str, Any]:
        """Build the media block dict for the Anthropic Vision API.

        If *image_source* looks like a URL it is passed through directly.
        Otherwise it is treated as a local file and base64-encoded.

        Args:
            image_source: URL or local file path.

        Returns:
            A source dict suitable for the Anthropic messages API.
        """
        if image_source.startswith(("http://", "https://")):
            return {"type": "url", "url": image_source}

        path = Path(image_source)
        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        with open(path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return {"type": "base64", "media_type": mime_type, "data": b64_data}
