"""Multi-provider LLM abstraction layer for SwarmGPT."""

from __future__ import annotations

from swarm_gpt.providers.base import (
    CompletionResult,
    ImageAnalysisResult,
    LLMProvider,
    TranscriptionResult,
)
from swarm_gpt.providers.openai_provider import OpenAIProvider

try:
    from swarm_gpt.providers.anthropic_provider import AnthropicProvider
except ImportError:
    AnthropicProvider = None  # type: ignore[assignment,misc]


def get_provider(name: str, *, api_key: str | None = None, **kwargs: object) -> LLMProvider:
    """Factory function that returns the appropriate LLM provider.

    Args:
        name: Provider name, one of ``"openai"`` or ``"anthropic"``.
        api_key: Optional API key.  When *None* the provider falls back to
            its standard environment variable (``OPENAI_API_KEY`` or
            ``ANTHROPIC_API_KEY``).
        **kwargs: Additional keyword arguments forwarded to the provider
            constructor.

    Returns:
        An ``LLMProvider`` instance.

    Raises:
        ValueError: If *name* is not a recognised provider.
        ImportError: If the required SDK for the provider is not installed.
    """
    providers: dict[str, type[LLMProvider]] = {"openai": OpenAIProvider}
    if AnthropicProvider is not None:
        providers["anthropic"] = AnthropicProvider

    name_lower = name.lower().strip()
    if name_lower not in providers:
        raise ValueError(f"Unknown provider '{name}'. Available providers: {', '.join(providers)}")

    provider_cls = providers[name_lower]
    return provider_cls(api_key=api_key, **kwargs)  # type: ignore[call-arg]


__all__ = [
    "AnthropicProvider",
    "CompletionResult",
    "ImageAnalysisResult",
    "LLMProvider",
    "OpenAIProvider",
    "TranscriptionResult",
    "get_provider",
]
