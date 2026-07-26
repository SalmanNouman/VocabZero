from __future__ import annotations

from vocab_zero.core import engine_factory as factory_module
from vocab_zero.core.engine_factory import build_engine


class FakeVectorStore:
    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = persist_dir


class FakeOpenAIClient:
    def __init__(self, config: object) -> None:
        self.config = config


class FakeGemmaClient:
    def __init__(self, config: object) -> None:
        self.config = config


def test_build_engine_with_dictionary_only(tmp_path, monkeypatch):
    """build_engine without vector DB or LLM returns a bare engine."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    dict_path = tmp_path / "lexicon.json"
    dict_path.write_text("{}")

    engine = build_engine(dictionary_path=str(dict_path), vector_db_path=None)

    assert engine is not None
    assert engine.dictionary is not None
    assert engine.vector_store is None
    assert engine.llm_client is None


def test_build_engine_with_vector_db(tmp_path, monkeypatch):
    """build_engine wires VectorStoreClient when vector_db_path is provided."""
    dict_path = tmp_path / "lexicon.json"
    dict_path.write_text("{}")
    vector_path = tmp_path / "vector_db"

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(factory_module, "VectorStoreClient", FakeVectorStore)

    engine = build_engine(dictionary_path=str(dict_path), vector_db_path=str(vector_path))

    assert engine is not None
    assert engine.dictionary is not None
    assert engine.vector_store is not None
    assert isinstance(engine.vector_store, FakeVectorStore)
    assert engine.vector_store.persist_dir == str(vector_path)
    assert engine.llm_client is None


def test_build_engine_with_openai_llm(tmp_path, monkeypatch):
    """build_engine creates OpenAICompatibleClient when OPENAI_API_KEY is set."""
    dict_path = tmp_path / "lexicon.json"
    dict_path.write_text("{}")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(factory_module, "OpenAICompatibleClient", FakeOpenAIClient)

    engine = build_engine(dictionary_path=str(dict_path), vector_db_path=None)

    assert engine is not None
    assert engine.dictionary is not None
    assert engine.vector_store is None
    assert engine.llm_client is not None
    assert isinstance(engine.llm_client, FakeOpenAIClient)


def test_build_engine_with_gemma_provider(tmp_path, monkeypatch):
    """build_engine creates GemmaClient when LLM_PROVIDER=gemma."""
    dict_path = tmp_path / "lexicon.json"
    dict_path.write_text("{}")

    monkeypatch.setenv("LLM_PROVIDER", "gemma")
    monkeypatch.setattr(factory_module, "GemmaClient", FakeGemmaClient)

    engine = build_engine(dictionary_path=str(dict_path), vector_db_path=None)

    assert engine is not None
    assert engine.llm_client is not None
    assert isinstance(engine.llm_client, FakeGemmaClient)


def test_build_engine_reads_audio_config_from_env(tmp_path, monkeypatch):
    """build_engine sources Whisper matching thresholds from the environment."""
    dict_path = tmp_path / "lexicon.json"
    dict_path.write_text("{}")

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("VOCABZERO_MATCH_DISTANCE_THRESHOLD", "0.42")
    monkeypatch.setenv("VOCABZERO_MIN_CONFIDENCE", "0.7")

    engine = build_engine(dictionary_path=str(dict_path), vector_db_path=None)

    assert engine.audio_config.match_distance_threshold == 0.42
    assert engine.audio_config.min_confidence_gate == 0.7
