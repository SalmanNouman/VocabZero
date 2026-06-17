from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from openai import OpenAIError

from vocab_zero.core.llm_client import GemmaClient, OpenAICompatibleClient
from vocab_zero.core.models import TranslationConfig


@pytest.fixture
def config() -> TranslationConfig:
    return TranslationConfig(
        api_key="placeholder-value",
        model_name="gpt-4o-mini",
        timeout_seconds=30.0,
        retry_count=1,
    )


@pytest.fixture
def client(config: TranslationConfig) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(config)


def test_translate_valid_response(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"translation": "hola", "reasoning": "test", "confidence": 0.9}'

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello")
        
        assert result is not None
        assert result.translation == "hola"
        assert result.reasoning == "test"
        assert result.confidence == 0.9


def test_translate_malformed_json(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "not json"

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello")
        
        assert result is None


def test_translate_missing_fields(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"translation": "hola"}'

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello")
        
        assert result is None


def test_translate_invalid_confidence(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"translation": "hola", "reasoning": "test", "confidence": 1.5}'

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello")
        
        assert result is None


def test_translate_empty_translation(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"translation": "", "reasoning": "test", "confidence": 0.9}'

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello")
        
        assert result is None


def test_translate_no_api_key() -> None:
    config = TranslationConfig(api_key=None)
    client = OpenAICompatibleClient(config)
    
    result = client.translate("hello")
    
    assert result is None


def test_translate_provider_error_retries(client: OpenAICompatibleClient) -> None:
    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.side_effect = [OpenAIError("error"), OpenAIError("error")]
        
        result = client.translate("hello")
        
        assert result is None
        assert mock_chat.completions.create.call_count == 2


def test_translate_provider_error_succeeds_on_retry(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"translation": "hola", "reasoning": "test", "confidence": 0.9}'

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.side_effect = [OpenAIError("error"), mock_response]
        
        result = client.translate("hello")
        
        assert result is not None
        assert result.translation == "hola"


def test_translate_with_context(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"translation": "hola", "reasoning": "test", "confidence": 0.9}'

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello", context="greeting")
        
        assert result is not None
        assert result.translation == "hola"
        
        call_args = mock_chat.completions.create.call_args
        user_message = call_args[1]["messages"][1]["content"]
        assert "greeting" in user_message


def test_translate_with_examples(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"translation": "hola", "reasoning": "test", "confidence": 0.9}'

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello", examples=["bonjour", "hallo"])
        
        assert result is not None
        assert result.translation == "hola"
        
        call_args = mock_chat.completions.create.call_args
        user_message = call_args[1]["messages"][1]["content"]
        assert "bonjour" in user_message
        assert "hallo" in user_message


def test_translate_empty_content(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = None

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello")
        
        assert result is None


def test_system_prompt_marks_untrusted_data(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"translation": "hola", "reasoning": "test", "confidence": 0.9}'

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        client.translate("hello")
        
        call_args = mock_chat.completions.create.call_args
        system_message = call_args[1]["messages"][0]["content"]
        assert "untrusted" in system_message.lower()


def test_no_api_key_in_logs(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"translation": "hola", "reasoning": "test", "confidence": 0.9}'

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello")
        
        assert result is not None
        assert "placeholder-value" not in str(result)


def test_empty_choices_returns_none(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = []

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello")
        
        assert result is None


def test_missing_message_content_returns_none(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = None

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        
        result = client.translate("hello")
        
        assert result is None



def test_missing_message_attribute_returns_none(client: OpenAICompatibleClient) -> None:
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = None

    with patch.object(client._client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response

        result = client.translate("hello")

        assert result is None


# ---------------------------------------------------------------------------
# GemmaClient tests
# ---------------------------------------------------------------------------


@pytest.fixture
def gemma_config() -> TranslationConfig:
    return TranslationConfig(
        api_key=None,
        base_url=None,
        model_name="google/gemma-2b-it",
    )


@pytest.fixture
def gemma_client(gemma_config: TranslationConfig) -> GemmaClient:
    return GemmaClient(config=gemma_config)


def _make_mock_gemma_pipeline(response_json: str) -> Mock:
    """Return a callable mock that mimics the transformers text-generation pipeline output."""
    mock_pipe = Mock()
    mock_pipe.return_value = [{"generated_text": response_json}]
    return mock_pipe


def test_gemma_default_model() -> None:
    client = GemmaClient()
    assert client.model_name == "google/gemma-2b-it"


def test_gemma_custom_model() -> None:
    client = GemmaClient(model_name="my-custom-model")
    assert client.model_name == "my-custom-model"


def test_gemma_openai_compatible_init() -> None:
    config = TranslationConfig(base_url="http://localhost:11434/v1", api_key="some-key")
    client = GemmaClient(config=config)
    assert client._openai_client is not None
    assert client.model_name == "google/gemma-2b-it"  # Fallback from default gpt-4o-mini

    # Verify custom model in configuration is respected
    config_custom = TranslationConfig(
        base_url="http://localhost:11434/v1",
        api_key="some-key",
        model_name="gemma-2-9b-it"
    )
    client_custom = GemmaClient(config=config_custom)
    assert client_custom.model_name == "gemma-2-9b-it"


def test_gemma_translate_success_hf(gemma_client: GemmaClient) -> None:
    gemma_client._pipeline = _make_mock_gemma_pipeline(
        '{"translation": "water", "reasoning": "context matches", "confidence": 0.9}'
    )
    result = gemma_client.translate("maay")
    assert result is not None
    assert result.translation == "water"
    assert result.reasoning == "context matches"
    assert result.confidence == 0.9


def test_gemma_translate_success_openai() -> None:
    config = TranslationConfig(base_url="http://localhost:11434/v1", api_key="some-key")
    client = GemmaClient(config=config)
    
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = (
        '{"translation": "hello", "reasoning": "openai compat", "confidence": 0.85}'
    )
    
    with patch.object(client._openai_client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        result = client.translate("hello")
        
        assert result is not None
        assert result.translation == "hello"
        assert result.reasoning == "openai compat"
        assert result.confidence == 0.85


def test_gemma_translate_masked_sentence_completion_openai() -> None:
    config = TranslationConfig(base_url="http://localhost:11434/v1", api_key="some-key")
    client = GemmaClient(config=config)
    
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = (
        '{"translation": "four, plastic, three", "reasoning": "masked fill", "confidence": 0.95}'
    )
    
    with patch.object(client._openai_client, "chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_response
        result = client.translate("I bought [unknown] bags")
        
        assert result is not None
        assert result.translation == "four, plastic, three"
        assert result.reasoning == "masked fill"
        assert result.confidence == 0.95
        
        # Verify the prompt contained instructions for masked completion
        args = mock_chat.completions.create.call_args[1]
        messages = args["messages"]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "masked/unknown word" in system_msg
        assert "comma-separated list" in system_msg


def test_gemma_translate_masked_sentence_completion_hf(gemma_client: GemmaClient) -> None:
    gemma_client._pipeline = _make_mock_gemma_pipeline(
        '{"translation": "four, plastic, three", "reasoning": "masked fill", "confidence": 0.95}'
    )
    result = gemma_client.translate("I bought [mask] bags")
    assert result is not None
    assert result.translation == "four, plastic, three"
    assert result.reasoning == "masked fill"
    assert result.confidence == 0.95


def test_gemma_translate_empty_term_returns_none(gemma_client: GemmaClient) -> None:
    gemma_client._pipeline = _make_mock_gemma_pipeline("{}")
    result = gemma_client.translate("   ")
    assert result is None


def test_gemma_translate_pipeline_load_failure_returns_none(gemma_client: GemmaClient) -> None:
    with patch("vocab_zero.core.llm_client.GemmaClient._load_pipeline", return_value=False):
        result = gemma_client.translate("maay")
    assert result is None


def test_gemma_translate_pipeline_raises_returns_none(gemma_client: GemmaClient) -> None:
    mock_pipe = Mock(side_effect=RuntimeError("GPU OOM"))
    gemma_client._pipeline = mock_pipe
    result = gemma_client.translate("maay")
    assert result is None


def test_gemma_translate_invalid_json_returns_none(gemma_client: GemmaClient) -> None:
    gemma_client._pipeline = _make_mock_gemma_pipeline("not json")
    result = gemma_client.translate("maay")
    assert result is None


def test_gemma_frequency_fingerprint_returns_none(gemma_client: GemmaClient) -> None:
    gemma_client._pipeline = _make_mock_gemma_pipeline('{"translation": "water", "reasoning": "test", "confidence": 0.8}')
    for freq_key in ("440", "220_440", "100_200_400"):
        result = gemma_client.translate(freq_key)
        assert result is None
    gemma_client._pipeline.assert_not_called()


def test_gemma_lazy_pipeline_loads_on_first_call(gemma_client: GemmaClient) -> None:
    assert gemma_client._pipeline is None
    mock_pipe = _make_mock_gemma_pipeline('{"translation": "hello", "reasoning": "test", "confidence": 0.8}')
    def _fake_load() -> bool:
        gemma_client._pipeline = mock_pipe
        return True
    with patch.object(gemma_client, '_load_pipeline', side_effect=_fake_load):
        result = gemma_client.translate("test")
    assert result is not None
