"""Tests for OpenAIProvider."""

from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

import pytest

from swarm_gpt.exception import LLMException
from swarm_gpt.providers.base import CompletionResult, ImageAnalysisResult, TranscriptionResult
from swarm_gpt.providers.openai_provider import OpenAIProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> OpenAIProvider:
    """Return an OpenAIProvider with a mocked client."""
    with patch("swarm_gpt.providers.openai_provider.OpenAI"):
        inst = OpenAIProvider(api_key="test-key")
        inst._client = MagicMock()
        yield inst


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for provider properties."""

    def test_name(self, provider: OpenAIProvider) -> None:
        assert provider.name == "openai"

    def test_default_model(self, provider: OpenAIProvider) -> None:
        assert provider.default_model == "gpt-4o-2024-05-13"

    def test_custom_default_model(self) -> None:
        with patch("swarm_gpt.providers.openai_provider.OpenAI"):
            p = OpenAIProvider(api_key="k", default_model="gpt-4o-mini")
            assert p.default_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


class TestComplete:
    """Tests for the complete() method."""

    def test_basic_completion(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello world"
        mock_response.model = "gpt-4o-2024-05-13"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        provider._client.chat.completions.create.return_value = mock_response

        result = provider.complete([{"role": "user", "content": "Hi"}])

        assert isinstance(result, CompletionResult)
        assert result.content == "Hello world"
        assert result.model == "gpt-4o-2024-05-13"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["total_tokens"] == 15
        provider._client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-2024-05-13",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=4096,
            temperature=0.0,
        )

    def test_override_model(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = None
        provider._client.chat.completions.create.return_value = mock_response

        result = provider.complete([{"role": "user", "content": "Hi"}], model="gpt-4o-mini")

        assert result.model == "gpt-4o-mini"
        provider._client.chat.completions.create.assert_called_once()
        call_kwargs = provider._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"

    def test_max_tokens_and_temperature(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        provider._client.chat.completions.create.return_value = mock_response

        provider.complete(
            [{"role": "user", "content": "Hi"}], max_tokens=100, temperature=0.7
        )

        call_kwargs = provider._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 100
        assert call_kwargs.kwargs["temperature"] == 0.7

    def test_none_content_falls_back_to_empty(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        provider._client.chat.completions.create.return_value = mock_response

        result = provider.complete([{"role": "user", "content": "Hi"}])
        assert result.content == ""

    def test_api_error_raises_llm_exception(self, provider: OpenAIProvider) -> None:
        provider._client.chat.completions.create.side_effect = Exception("API down")

        with pytest.raises(LLMException, match="OpenAI completion failed"):
            provider.complete([{"role": "user", "content": "Hi"}])

    def test_multi_turn_messages(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_response.model = "gpt-4o"
        mock_response.usage = None
        provider._client.chat.completions.create.return_value = mock_response

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        provider.complete(messages)

        call_kwargs = provider._client.chat.completions.create.call_args
        assert call_kwargs.kwargs["messages"] == messages


# ---------------------------------------------------------------------------
# transcribe()
# ---------------------------------------------------------------------------


class TestTranscribe:
    """Tests for the transcribe() method."""

    def test_basic_transcription(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        provider._client.audio.transcriptions.create.return_value = mock_response

        with patch("builtins.open", mock_open(read_data=b"fake_audio")):
            result = provider.transcribe("/tmp/audio.mp3")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"

    def test_transcription_with_language(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.text = "Bonjour"
        provider._client.audio.transcriptions.create.return_value = mock_response

        with patch("builtins.open", mock_open(read_data=b"fake_audio")):
            result = provider.transcribe("/tmp/audio.mp3", language="fr")

        assert result.text == "Bonjour"
        assert result.language == "fr"
        call_kwargs = provider._client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["language"] == "fr"

    def test_transcription_error(self, provider: OpenAIProvider) -> None:
        provider._client.audio.transcriptions.create.side_effect = Exception("fail")

        with patch("builtins.open", mock_open(read_data=b"fake_audio")):
            with pytest.raises(LLMException, match="OpenAI transcription failed"):
                provider.transcribe("/tmp/audio.mp3")


# ---------------------------------------------------------------------------
# analyze_image()
# ---------------------------------------------------------------------------


class TestAnalyzeImage:
    """Tests for the analyze_image() method."""

    def test_analyze_url(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A picture of a cat"
        mock_response.model = "gpt-4o-2024-05-13"
        mock_response.usage = None
        provider._client.chat.completions.create.return_value = mock_response

        result = provider.analyze_image("Describe this image", "https://example.com/cat.png")

        assert isinstance(result, ImageAnalysisResult)
        assert result.content == "A picture of a cat"
        call_kwargs = provider._client.chat.completions.create.call_args
        sent_messages = call_kwargs.kwargs["messages"]
        content_block = sent_messages[0]["content"]
        assert any("image_url" in str(item) for item in content_block)
        assert any("https://example.com/cat.png" in str(item) for item in content_block)

    def test_analyze_local_file(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "An image"
        mock_response.model = "gpt-4o-2024-05-13"
        mock_response.usage = None
        provider._client.chat.completions.create.return_value = mock_response

        fake_b64 = "aW1hZ2UgZGF0YQ=="
        with patch("builtins.open", mock_open(read_data=b"image data")):
            with patch("base64.b64encode", return_value=fake_b64.encode()):
                with patch("mimetypes.guess_type", return_value=("image/png", None)):
                    result = provider.analyze_image("Describe", "/tmp/img.png")

        assert result.content == "An image"

    def test_analyze_image_error(self, provider: OpenAIProvider) -> None:
        provider._client.chat.completions.create.side_effect = Exception("fail")

        with pytest.raises(LLMException, match="OpenAI completion failed"):
            provider.analyze_image("Describe", "https://example.com/img.png")
