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


# ---------------------------------------------------------------------------
# NLLBClient tests
# ---------------------------------------------------------------------------

from vocab_zero.core.llm_client import NLLBClient  # noqa: E402


@pytest.fixture
def nllb_client() -> NLLBClient:
    return NLLBClient(src_lang="rhg_Latn", tgt_lang="eng_Latn")


def _make_mock_pipeline(translation: str = "water") -> Mock:
    """Return a callable mock that mimics the transformers pipeline output."""
    mock_pipe = Mock()
    mock_pipe.return_value = [{"translation_text": translation}]
    return mock_pipe


def test_nllb_translate_success(nllb_client: NLLBClient) -> None:
    nllb_client._pipeline = _make_mock_pipeline("water")
    result = nllb_client.translate("maay")
    assert result is not None
    assert result.translation == "water"
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning != ""


def test_nllb_translate_empty_term_returns_none(nllb_client: NLLBClient) -> None:
    nllb_client._pipeline = _make_mock_pipeline()
    result = nllb_client.translate("   ")
    assert result is None


def test_nllb_translate_pipeline_load_failure_returns_none(nllb_client: NLLBClient) -> None:
    # Force _load_pipeline to fail by making the import raise
    with patch("vocab_zero.core.llm_client.NLLBClient._load_pipeline", return_value=False):
        result = nllb_client.translate("maay")
    assert result is None


def test_nllb_translate_pipeline_raises_returns_none(nllb_client: NLLBClient) -> None:
    mock_pipe = Mock(side_effect=RuntimeError("GPU OOM"))
    nllb_client._pipeline = mock_pipe
    result = nllb_client.translate("maay")
    assert result is None


def test_nllb_translate_empty_result_list_returns_none(nllb_client: NLLBClient) -> None:
    mock_pipe = Mock(return_value=[])
    nllb_client._pipeline = mock_pipe
    result = nllb_client.translate("maay")
    assert result is None


def test_nllb_translate_empty_translation_text_returns_none(nllb_client: NLLBClient) -> None:
    mock_pipe = Mock(return_value=[{"translation_text": "   "}])
    nllb_client._pipeline = mock_pipe
    result = nllb_client.translate("maay")
    assert result is None


def test_nllb_translate_non_list_result_returns_none(nllb_client: NLLBClient) -> None:
    mock_pipe = Mock(return_value=None)
    nllb_client._pipeline = mock_pipe
    result = nllb_client.translate("maay")
    assert result is None


def test_nllb_translate_context_prepended(nllb_client: NLLBClient) -> None:
    """Verify that context is prepended to the source term before calling the pipeline."""
    mock_pipe = _make_mock_pipeline("food")
    nllb_client._pipeline = mock_pipe
    nllb_client.translate("khana", context="meal")
    call_text = mock_pipe.call_args[0][0]
    assert "meal" in call_text
    assert "khana" in call_text


def test_nllb_frequency_fingerprint_returns_none(nllb_client: NLLBClient) -> None:
    """Frequency keys must never be passed to NLLB — it cannot translate them."""
    nllb_client._pipeline = _make_mock_pipeline("water")
    for freq_key in ("440", "220_440", "100_200_400", "440_880_1760"):
        result = nllb_client.translate(freq_key)
        assert result is None, f"Expected None for frequency key {freq_key!r}"
    # Pipeline should not have been called at all
    nllb_client._pipeline.assert_not_called()


def test_nllb_real_text_not_rejected(nllb_client: NLLBClient) -> None:
    """Regular Rohingya text like 'maay' must still be passed to NLLB."""
    nllb_client._pipeline = _make_mock_pipeline("water")
    result = nllb_client.translate("maay")
    assert result is not None
    nllb_client._pipeline.assert_called_once()


def test_nllb_default_lang_codes() -> None:
    client = NLLBClient()
    assert client.src_lang == "rhg_Latn"
    assert client.tgt_lang == "eng_Latn"


def test_nllb_lazy_pipeline_loads_on_first_call(nllb_client: NLLBClient) -> None:
    """Pipeline should be None at construction time and initialised lazily."""
    assert nllb_client._pipeline is None

    mock_pipe = _make_mock_pipeline("hello")
    with patch("vocab_zero.core.llm_client.pipeline", mock_pipe, create=True):
        # Patch the import inside _load_pipeline
        with patch.dict("sys.modules", {"transformers": Mock(pipeline=mock_pipe)}):
            nllb_client._pipeline = mock_pipe  # Simulate successful lazy load
            result = nllb_client.translate("test")

    assert result is not None

