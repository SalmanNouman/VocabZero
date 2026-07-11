from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from vocab_zero.core.dictionary import DictionaryManager, LexiconEntry
from vocab_zero.core.engine import TranslationEngine
from vocab_zero.core.models import (
    AudioConfig,
    FeedbackRequest,
    LLMResponse,
    TranslationConfig,
)
from vocab_zero.core.vector_db import SearchResult, VectorStoreClient


@pytest.fixture
def temp_dictionary(tmp_path: Path) -> DictionaryManager:
    return DictionaryManager(path=tmp_path / "lexicon.json")


@pytest.fixture
def config() -> TranslationConfig:
    return TranslationConfig(
        high_threshold=0.8,
        low_threshold=0.4,
        rag_k=5,
        timeout_seconds=30.0,
        retry_count=1,
    )


@pytest.fixture
def mock_llm_client() -> Mock:
    client = Mock()
    client.translate.return_value = LLMResponse(
        translation="hola",
        reasoning="test",
        confidence=0.9,
    )
    return client


@pytest.fixture
def mock_vector_store() -> Mock:
    store = Mock(spec=VectorStoreClient)
    store.search.return_value = []
    store.add_entry.return_value = True
    return store


@pytest.fixture
def engine(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> TranslationEngine:
    return TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        audio_config=AudioConfig(use_deltas=False, dtw_threshold_12=15.0, min_confidence_gate=0.0),
    )


def test_high_confidence_exact_match_no_llm_call(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    temp_dictionary.upsert(LexiconEntry(source_term="hello", target_term="hola", confidence=0.9))
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("hello")
    
    assert result.translated_text == "hola"
    assert result.confidence == 0.9
    assert result.source == "dictionary"
    assert result.status == "translated"
    mock_llm_client.translate.assert_not_called()
    mock_vector_store.search.assert_not_called()


def test_medium_confidence_exact_match_calls_llm(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    temp_dictionary.upsert(LexiconEntry(source_term="hello", target_term="hola", confidence=0.6))
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("hello")
    
    assert result.translated_text == "hola"
    assert result.source == "llm_inference"
    assert result.status == "translated"
    mock_llm_client.translate.assert_called_once()
    mock_vector_store.search.assert_called_once()


def test_medium_confidence_llm_fails_returns_stored(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    temp_dictionary.upsert(LexiconEntry(source_term="hello", target_term="hola", confidence=0.6))
    mock_llm_client.translate.return_value = None
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("hello")
    
    assert result.translated_text == "hola"
    assert result.source == "dictionary"
    assert result.status == "low_confidence"
    assert result.confidence == 0.6


def test_low_confidence_exact_match_requests_feedback(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    temp_dictionary.upsert(LexiconEntry(source_term="hello", target_term="hola", confidence=0.3))
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("hello")
    
    assert result.translated_text == "hola"
    assert result.source == "dictionary"
    assert result.status == "requires_feedback"
    assert result.feedback_request is not None
    assert result.feedback_request.source_term == "hello"


def test_low_confidence_with_callback_persists_entry(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    temp_dictionary.upsert(LexiconEntry(source_term="hello", target_term="hola", confidence=0.3))

    feedback_entry = LexiconEntry(source_term="hello", target_term="hi", confidence=0.95)

    def feedback_callback(request: FeedbackRequest) -> LexiconEntry:
        return feedback_entry

    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        on_feedback_required=feedback_callback,
    )

    result = engine.translate("hello")

    assert result.translated_text == "hi"
    assert result.status == "learned"
    assert result.source == "human_feedback"

    updated_entry = temp_dictionary.lookup("hello")
    assert updated_entry is not None
    assert updated_entry.target_term == "hi"
    assert updated_entry.confidence == 0.95

    mock_vector_store.add_entry.assert_called_once_with(feedback_entry)


def test_low_confidence_with_callback_declined_returns_declined_status(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    temp_dictionary.upsert(LexiconEntry(source_term="hello", target_term="hola", confidence=0.3))

    def feedback_callback(request: FeedbackRequest) -> LexiconEntry | None:
        return None

    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        on_feedback_required=feedback_callback,
    )

    result = engine.translate("hello")

    assert result.translated_text == "hola"
    assert result.status == "feedback_declined"
    assert result.source == "dictionary"
    assert result.feedback_request is None


def test_unknown_term_with_rag_invokes_llm(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    mock_vector_store.search.return_value = [
        SearchResult(
            entry=LexiconEntry(source_term="greeting", target_term="saludo", confidence=0.9),
            score=0.85,
        )
    ]
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("hello")
    
    assert result.translated_text == "hola"
    assert result.source == "llm_inference"
    assert result.status == "low_confidence"
    mock_llm_client.translate.assert_called_once()
    mock_vector_store.search.assert_called_once()


def test_unknown_term_empty_rag_still_calls_llm(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    mock_vector_store.search.return_value = []
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("hello")
    
    assert result.translated_text == "hola"
    assert result.source == "llm_inference"
    mock_llm_client.translate.assert_called_once()


def test_unknown_term_no_llm_requests_feedback(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    config: TranslationConfig,
) -> None:
    mock_vector_store.search.return_value = []
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=None,
        config=config,
    )
    
    result = engine.translate("hello")
    
    assert result.translated_text == ""
    assert result.status == "requires_feedback"
    assert result.feedback_request is not None


def test_unknown_term_llm_low_confidence_requests_feedback(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    mock_llm_client.translate.return_value = LLMResponse(
        translation="hola",
        reasoning="test",
        confidence=0.3,
    )
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("hello")
    
    assert result.status == "requires_feedback"
    assert result.feedback_request is not None


def test_unknown_term_with_callback_persists_entry(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    mock_llm_client.translate.return_value = None

    feedback_entry = LexiconEntry(source_term="hello", target_term="hi", confidence=0.95)

    def feedback_callback(request: FeedbackRequest) -> LexiconEntry:
        return feedback_entry

    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        on_feedback_required=feedback_callback,
    )

    result = engine.translate("hello")

    assert result.translated_text == "hi"
    assert result.status == "learned"

    updated_entry = temp_dictionary.lookup("hello")
    assert updated_entry is not None
    assert updated_entry.target_term == "hi"

    mock_vector_store.add_entry.assert_called_once_with(feedback_entry)


def test_unknown_term_with_callback_declined_returns_declined_status(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    mock_llm_client.translate.return_value = None

    def feedback_callback(request: FeedbackRequest) -> LexiconEntry | None:
        return None

    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        on_feedback_required=feedback_callback,
    )

    result = engine.translate("hello")

    assert result.translated_text == ""
    assert result.status == "feedback_declined"
    assert result.source == "none"
    assert result.feedback_request is None


def test_vector_upsert_failure_preserves_dictionary_learning(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    mock_llm_client.translate.return_value = None
    mock_vector_store.add_entry.return_value = False
    
    feedback_entry = LexiconEntry(source_term="hello", target_term="hi", confidence=0.95)
    
    def feedback_callback(request: FeedbackRequest) -> LexiconEntry:
        return feedback_entry
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        on_feedback_required=feedback_callback,
    )
    
    result = engine.translate("hello")
    
    assert result.translated_text == "hi"
    assert result.status == "learned"
    assert result.error_code == "semantic_index_update_failed"
    
    updated_entry = temp_dictionary.lookup("hello")
    assert updated_entry is not None
    assert updated_entry.target_term == "hi"


def test_empty_source_term_returns_error(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("")
    
    assert result.status == "error"
    assert result.error_code == "invalid_input"
    assert result.error_message is not None


def test_confidence_calculation_with_vector_results(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    mock_vector_store.search.return_value = [
        SearchResult(
            entry=LexiconEntry(source_term="greeting", target_term="saludo", confidence=0.9),
            score=0.8,
        )
    ]
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("hello")
    
    expected_confidence = (0.6 * 0.9) + (0.4 * 0.8)
    assert abs(result.confidence - expected_confidence) < 0.01


def test_confidence_calculation_without_vector_results(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    mock_vector_store.search.return_value = []
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("hello")
    
    assert result.confidence == 0.9




def test_no_vector_store(
    temp_dictionary: DictionaryManager,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=None,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("hello")
    
    assert result.translated_text == "hola"
    assert result.source == "llm_inference"


def test_context_length_capping(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    config.max_context_length = 10
    long_context = "a" * 100
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    engine.translate("hello", context=long_context)
    
    assert mock_llm_client.translate.called
    call_args = mock_llm_client.translate.call_args
    if len(call_args.args) > 1:
        context_arg = call_args.args[1]
    else:
        context_arg = call_args[1].get("context")
    assert context_arg is not None
    assert len(context_arg) <= 10


def test_no_api_key_or_secrets_in_errors(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
) -> None:
    config.api_key = "placeholder-value"
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
    )
    
    result = engine.translate("")
    
    assert result.status == "error"
    result_text = str(result).lower()
    assert "placeholder-value" not in result_text
    assert "secret" not in result_text
    assert "credential" not in result_text


def test_learned_entry_saves_dictionary(
    temp_dictionary: DictionaryManager,
    mock_vector_store: Mock,
    mock_llm_client: Mock,
    config: TranslationConfig,
    tmp_path: Path,
) -> None:
    mock_llm_client.translate.return_value = None
    
    feedback_entry = LexiconEntry(source_term="hello", target_term="hi", confidence=0.95)
    
    def feedback_callback(request: FeedbackRequest) -> LexiconEntry:
        return feedback_entry
    
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        on_feedback_required=feedback_callback,
    )
    
    engine.translate("hello")

    new_dict = DictionaryManager(path=tmp_path / "lexicon.json")
    assert new_dict.lookup("hello") is not None
    assert new_dict.lookup("hello").target_term == "hi"


def test_rerank_acoustic_candidates(engine):
    entry, conf = engine.rerank_acoustic_candidates([])
    assert entry is None
    assert conf == 0.0

    t1 = [[0.1] * 12 for _ in range(5)]
    entry1 = LexiconEntry(source_term="term1", target_term="target1", confidence=1.0, mfcc_templates=[t1])

    entry, conf = engine.rerank_acoustic_candidates([(entry1, 5.0)])
    assert entry == entry1
    assert conf == pytest.approx(0.67, abs=0.01)

    entry, conf = engine.rerank_acoustic_candidates([(entry1, 20.0)])
    assert entry is None
    assert conf == 0.0

    entry2 = LexiconEntry(source_term="term2", target_term="target2", confidence=1.0, mfcc_templates=[t1])
    entry, conf = engine.rerank_acoustic_candidates([(entry1, 4.0), (entry2, 10.0)])
    assert entry == entry1
    assert conf == pytest.approx(0.73, abs=0.01)

    engine.llm_client.select_best_candidate.return_value = ("target2", 0.9)
    entry, conf = engine.rerank_acoustic_candidates([(entry1, 7.0), (entry2, 8.0)], context="some context")
    assert entry == entry2
    assert conf > 0.5


def test_rerank_threshold_comes_from_audio_config(temp_dictionary, mock_vector_store, mock_llm_client, config):
    # 36-dim config with a high threshold should accept a candidate that the 12-dim fallback would reject
    audio_cfg = AudioConfig(use_deltas=True, dtw_threshold_36=100.0)
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        audio_config=audio_cfg,
    )
    t1 = [[0.1] * 12 for _ in range(5)]
    entry1 = LexiconEntry(source_term="term1", target_term="target1", confidence=1.0, mfcc_templates=[t1])
    # dist=20.0 would be rejected under threshold=15.0 but accepted under threshold=100.0
    entry, conf = engine.rerank_acoustic_candidates([(entry1, 20.0)])
    assert entry == entry1
    assert conf > 0.0


def test_rerank_single_candidate_no_margin_rejection(temp_dictionary, mock_vector_store, mock_llm_client, config):
    """Single candidate with dist < threshold should be accepted, even if dist > 70% of threshold."""
    audio_cfg = AudioConfig(use_deltas=False, dtw_threshold_12=10.0, min_confidence_gate=0.0)
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        audio_config=audio_cfg,
    )
    t1 = [[0.1] * 12 for _ in range(5)]
    entry1 = LexiconEntry(source_term="term1", target_term="target1", confidence=1.0, mfcc_templates=[t1])

    # dist=8.0 is 80% of threshold 10.0 → accepted and confidence is 0.2
    entry, conf = engine.rerank_acoustic_candidates([(entry1, 8.0)])
    assert entry == entry1
    assert conf == pytest.approx(0.2, abs=0.01)

    # dist=6.0 is 60% of threshold 10.0 → accepted and confidence is 0.4
    entry, conf = engine.rerank_acoustic_candidates([(entry1, 6.0)])
    assert entry == entry1
    assert conf == pytest.approx(0.4, abs=0.01)



def test_rerank_confidence_has_no_floor(temp_dictionary, mock_vector_store, mock_llm_client, config):
    """Confidence should go below 0.5 for high-distance matches (no floor)."""
    audio_cfg = AudioConfig(use_deltas=False, dtw_threshold_12=10.0, min_confidence_gate=0.0)
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        audio_config=audio_cfg,
    )
    t1 = [[0.1] * 12 for _ in range(5)]
    entry1 = LexiconEntry(source_term="term1", target_term="target1", confidence=1.0, mfcc_templates=[t1])

    # dist=6.5 → conf = 1.0 - 6.5/10.0 = 0.35, which is below old floor of 0.5
    entry, conf = engine.rerank_acoustic_candidates([(entry1, 6.5)])
    assert entry == entry1
    assert conf == pytest.approx(0.35, abs=0.01)
    assert conf < 0.5


def test_rerank_min_confidence_gate_rejects_single_candidate(temp_dictionary, mock_vector_store, mock_llm_client, config):
    """Single candidate with confidence below min_confidence_gate is rejected."""
    audio_cfg = AudioConfig(use_deltas=False, dtw_threshold_12=10.0, min_confidence_gate=0.5)
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        audio_config=audio_cfg,
    )
    t1 = [[0.1] * 12 for _ in range(5)]
    entry1 = LexiconEntry(source_term="term1", target_term="target1", confidence=1.0, mfcc_templates=[t1])

    # dist=8.0 → conf = 0.2, below gate 0.5 → rejected
    entry, conf = engine.rerank_acoustic_candidates([(entry1, 8.0)])
    assert entry is None
    assert conf == 0.0

    # dist=4.0 → conf = 0.6, above gate 0.5 → accepted
    entry, conf = engine.rerank_acoustic_candidates([(entry1, 4.0)])
    assert entry == entry1
    assert conf == pytest.approx(0.6, abs=0.01)


def test_rerank_min_confidence_gate_rejects_reranked_result(temp_dictionary, mock_vector_store, mock_llm_client, config):
    """Reranked result with final confidence below min_confidence_gate is rejected."""
    audio_cfg = AudioConfig(use_deltas=False, dtw_threshold_12=10.0, min_confidence_gate=0.5)
    engine = TranslationEngine(
        dictionary=temp_dictionary,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
        config=config,
        audio_config=audio_cfg,
    )
    t1 = [[0.1] * 12 for _ in range(5)]
    entry1 = LexiconEntry(source_term="term1", target_term="target1", confidence=1.0, mfcc_templates=[t1])
    entry2 = LexiconEntry(source_term="term2", target_term="target2", confidence=1.0, mfcc_templates=[t1])

    # Both candidates have high DTW dist → low acoustic conf, no LLM → below gate
    # dist=8.0 → acoustic conf = 0.2; dist=9.0 → 0.1. Winner is entry1 (0.2) < gate 0.5 → rejected
    entry, conf = engine.rerank_acoustic_candidates([(entry1, 8.0), (entry2, 9.0)])
    assert entry is None
    assert conf == 0.0

    # Lower distances → above gate → accepted
    entry, conf = engine.rerank_acoustic_candidates([(entry1, 3.0), (entry2, 9.0)])
    assert entry == entry1
    assert conf == pytest.approx(0.7, abs=0.01)
