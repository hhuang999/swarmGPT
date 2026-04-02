"""Tests for AnthropicProvider."""

from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

import pytest

from swarm_gpt.exception import LLMException
from swarm_gpt.providers.base import CompletionResult, ImageAnalysisResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> object:
    """Return an AnthropicProvider with a fully mocked client.

    We patch ``anthropic_provider.anthropic`` so that the constructor can
    instantiate a mock client, then overwrite ``_client`` with a fresh
    ``MagicMock`` so each test starts clean.
    """
    with patch("swarm_gpt.providers.anthropic_provider.anthropic") as mock_sdk:
        mock_sdk.Anthropic.return_value = MagicMock()
        # Delay the import so the patch is active at class-creation time
        # (only needed on first call; subsequent calls use the cached class).
        from swarm_gpt.providers.anthropic_provider import AnthropicProvider

        inst = AnthropicProvider(api_key="test-key")
        inst._client = MagicMock()  # fresh mock for each test
        yield inst


# We need AnthropicProvider in scope for type hints / non-fixture tests.
from swarm_gpt.providers.anthropic_provider import AnthropicProvider  # noqa: E402

# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for provider properties."""

    def test_name(self, provider: AnthropicProvider) -> None:
        assert provider.name == "anthropic"

    def test_default_model(self, provider: AnthropicProvider) -> None:
        assert provider.default_model == "claude-sonnet-4-20250514"

    def test_custom_default_model(self) -> None:
        with patch("swarm_gpt.providers.anthropic_provider.anthropic") as mock_sdk:
            mock_sdk.Anthropic.return_value = MagicMock()
            p = AnthropicProvider(api_key="k", default_model="claude-3-opus-20240229")
            assert p.default_model == "claude-3-opus-20240229"


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


class TestComplete:
    """Tests for the complete() method."""

    def _make_mock_response(
        self, text: str = "Hello", model: str = "claude-sonnet-4-20250514"
    ) -> MagicMock:
        """Build a mock Anthropic messages response."""
        mock_content = [MagicMock()]
        mock_content[0].text = text
        mock_response = MagicMock()
        mock_response.content = mock_content
        mock_response.model = model
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        return mock_response

    def test_basic_completion(self, provider: AnthropicProvider) -> None:
        mock_resp = self._make_mock_response("Hello world")
        provider._client.messages.create.return_value = mock_resp

        result = provider.complete([{"role": "user", "content": "Hi"}])

        assert isinstance(result, CompletionResult)
        assert result.content == "Hello world"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        provider._client.messages.create.assert_called_once()
        call_kwargs = provider._client.messages.create.call_args
        assert call_kwargs.kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    def test_system_message_extraction(self, provider: AnthropicProvider) -> None:
        mock_resp = self._make_mock_response("ok")
        provider._client.messages.create.return_value = mock_resp

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        provider.complete(messages)

        call_kwargs = provider._client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "You are helpful."
        assert call_kwargs.kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_extra_system_messages_become_user(self, provider: AnthropicProvider) -> None:
        """Second system message should be remapped to user role."""
        mock_resp = self._make_mock_response("ok")
        provider._client.messages.create.return_value = mock_resp

        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
            {"role": "user", "content": "Hi"},
        ]
        provider.complete(messages)

        call_kwargs = provider._client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "System 1"
        api_msgs = call_kwargs.kwargs["messages"]
        assert api_msgs[0]["role"] == "user"
        assert api_msgs[0]["content"] == "System 2"
        assert api_msgs[1]["role"] == "user"
        assert api_msgs[1]["content"] == "Hi"

    def test_override_model(self, provider: AnthropicProvider) -> None:
        mock_resp = self._make_mock_response("ok")
        provider._client.messages.create.return_value = mock_resp

        provider.complete(
            [{"role": "user", "content": "Hi"}], model="claude-3-opus-20240229"
        )

        call_kwargs = provider._client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-3-opus-20240229"

    def test_max_tokens_and_temperature(self, provider: AnthropicProvider) -> None:
        mock_resp = self._make_mock_response("ok")
        provider._client.messages.create.return_value = mock_resp

        provider.complete(
            [{"role": "user", "content": "Hi"}], max_tokens=100, temperature=0.7
        )

        call_kwargs = provider._client.messages.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 100
        assert call_kwargs.kwargs["temperature"] == 0.7

    def test_empty_content(self, provider: AnthropicProvider) -> None:
        mock_resp = self._make_mock_response("ok")
        mock_resp.content = []
        provider._client.messages.create.return_value = mock_resp

        result = provider.complete([{"role": "user", "content": "Hi"}])
        assert result.content == ""

    def test_api_error_raises_llm_exception(self, provider: AnthropicProvider) -> None:
        provider._client.messages.create.side_effect = Exception("API down")

        with pytest.raises(LLMException, match="Anthropic completion failed"):
            provider.complete([{"role": "user", "content": "Hi"}])


# ---------------------------------------------------------------------------
# transcribe() -- not supported
# ---------------------------------------------------------------------------


class TestTranscribe:
    """Tests for the transcribe() method (unsupported)."""

    def test_transcribe_not_implemented(self, provider: AnthropicProvider) -> None:
        with pytest.raises(NotImplementedError, match="does not support transcription"):
            provider.transcribe("/tmp/audio.mp3")


# ---------------------------------------------------------------------------
# analyze_image()
# ---------------------------------------------------------------------------


class TestAnalyzeImage:
    """Tests for the analyze_image() method."""

    def test_analyze_url(self, provider: AnthropicProvider) -> None:
        mock_resp = MagicMock()
        mock_content = [MagicMock()]
        mock_content[0].text = "A picture of a cat"
        mock_resp.content = mock_content
        mock_resp.model = "claude-sonnet-4-20250514"
        provider._client.messages.create.return_value = mock_resp

        result = provider.analyze_image("Describe this", "https://example.com/cat.png")

        assert isinstance(result, ImageAnalysisResult)
        assert result.content == "A picture of a cat"
        call_kwargs = provider._client.messages.create.call_args
        user_content = call_kwargs.kwargs["messages"][0]["content"]
        assert user_content[0]["type"] == "image"
        assert user_content[0]["source"]["type"] == "url"

    def test_analyze_local_file(self, provider: AnthropicProvider) -> None:
        mock_resp = MagicMock()
        mock_content = [MagicMock()]
        mock_content[0].text = "An image"
        mock_resp.content = mock_content
        mock_resp.model = "claude-sonnet-4-20250514"
        provider._client.messages.create.return_value = mock_resp

        fake_b64 = "aW1hZ2UgZGF0YQ=="
        with patch("builtins.open", mock_open(read_data=b"image data")):
            with patch("base64.b64encode", return_value=fake_b64.encode()):
                with patch("mimetypes.guess_type", return_value=("image/png", None)):
                    result = provider.analyze_image("Describe", "/tmp/img.png")

        assert result.content == "An image"
        call_kwargs = provider._client.messages.create.call_args
        user_content = call_kwargs.kwargs["messages"][0]["content"]
        assert user_content[0]["type"] == "image"
        assert user_content[0]["source"]["type"] == "base64"
        assert user_content[0]["source"]["media_type"] == "image/png"

    def test_analyze_image_error(self, provider: AnthropicProvider) -> None:
        provider._client.messages.create.side_effect = Exception("fail")

        with pytest.raises(LLMException, match="Anthropic completion failed"):
            provider.analyze_image("Describe", "https://example.com/img.png")


# ---------------------------------------------------------------------------
# Missing SDK
# ---------------------------------------------------------------------------


class TestMissingSDK:
    """Tests for behaviour when the anthropic SDK is not installed."""

    def test_raises_import_error_when_sdk_missing(self) -> None:
        """Verify AnthropicProvider.__init__ raises ImportError when SDK is None."""
        import swarm_gpt.providers.anthropic_provider as mod

        original = mod.anthropic
        try:
            mod.anthropic = None
            with pytest.raises(ImportError, match="anthropic"):
                AnthropicProvider(api_key="test")
        finally:
            mod.anthropic = original
