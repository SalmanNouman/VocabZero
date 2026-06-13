from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from openai import OpenAIError

from vocab_zero.core.models import TranslationConfig
from vocab_zero.core.llm_client import OpenAICompatibleClient


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
