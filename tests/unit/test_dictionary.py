import json
from pathlib import Path

import pytest

from vocab_zero.core.dictionary import DictionaryManager, JsonObject, LexiconEntry


class FakeYamlLikeSerializer:
    def dumps(self, data: JsonObject) -> str:
        return json.dumps(data)

    def loads(self, text: str) -> JsonObject:
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            return {}
        return loaded


def make_entry(
    source_term: str = "hello",
    target_term: str = "hola",
    confidence: float = 0.5,
    context_examples: list[str] | None = None,
) -> LexiconEntry:
    return LexiconEntry(
        source_term=source_term,
        target_term=target_term,
        confidence=confidence,
        context_examples=context_examples or [],
    )


def make_manager(tmp_path: Path) -> DictionaryManager:
    return DictionaryManager(tmp_path / "lexicon.json")


def test_lookup_existing_term(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    entry = make_entry()

    assert manager.insert(entry) is True

    found = manager.lookup("hello")
    assert found == entry


def test_lookup_missing_term_returns_none(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    assert manager.lookup("missing") is None


def test_insert_duplicate_returns_false(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    entry = make_entry()

    assert manager.insert(entry) is True
    assert manager.insert(entry) is False


def test_upsert_replaces_existing_translation(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.insert(make_entry(target_term="hola", confidence=0.4))
    corrected = make_entry(target_term="buenas", confidence=0.9, context_examples=["hello there"])

    result = manager.upsert(corrected)

    found = manager.lookup("hello")
    assert result == corrected
    assert found == corrected


def test_update_confidence_increases(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.insert(make_entry(confidence=0.4))

    updated = manager.update_confidence("hello", 0.2)

    assert updated is not None
    assert updated.confidence == pytest.approx(0.6)


def test_update_confidence_clamps_to_one(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.insert(make_entry(confidence=0.8))

    updated = manager.update_confidence("hello", 0.5)

    assert updated is not None
    assert updated.confidence == 1.0


def test_update_confidence_clamps_to_zero(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.insert(make_entry(confidence=0.2))

    updated = manager.update_confidence("hello", -0.5)

    assert updated is not None
    assert updated.confidence == 0.0


def test_update_confidence_missing_returns_none(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    assert manager.update_confidence("missing", 0.1) is None


def test_delete_existing(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    manager.insert(make_entry())

    assert manager.delete("hello") is True
    assert manager.lookup("hello") is None


def test_delete_missing_returns_false(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    assert manager.delete("missing") is False


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    manager = DictionaryManager(path)
    entry = make_entry(context_examples=["hello world", "say hello"])
    manager.insert(entry)

    manager.save()
    loaded = DictionaryManager(path)

    assert loaded.lookup("hello") == entry


def test_init_loads_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    path.write_text(
        json.dumps({"entries": [make_entry(source_term="cat", target_term="gato").model_dump()]}),
        encoding="utf-8",
    )

    manager = DictionaryManager(path)

    assert manager.lookup("cat") == make_entry(source_term="cat", target_term="gato")


def test_load_nonexistent_file_is_noop(tmp_path: Path) -> None:
    manager = DictionaryManager(tmp_path / "missing.json")

    assert manager.lookup("hello") is None


def test_load_invalid_json_returns_false_and_preserves_store(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    manager = DictionaryManager(path)
    entry = make_entry()
    manager.insert(entry)
    path.write_text("{not valid json", encoding="utf-8")

    assert manager.load() is False
    assert manager.lookup("hello") == entry


def test_load_invalid_entry_returns_false_and_preserves_store(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    manager = DictionaryManager(path)
    entry = make_entry()
    manager.insert(entry)
    path.write_text(json.dumps({"entries": [{"source_term": "bad"}]}), encoding="utf-8")

    assert manager.load() is False
    assert manager.lookup("hello") == entry


def test_context_examples_persisted(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    entry = make_entry(context_examples=["hello world", "hello again"])
    manager = DictionaryManager(path)
    manager.insert(entry)

    manager.save()
    loaded = DictionaryManager(path)

    assert loaded.lookup("hello") == entry


def test_custom_serializer_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.yaml"
    serializer = FakeYamlLikeSerializer()
    manager = DictionaryManager(path, serializer=serializer)
    entry = make_entry()
    manager.insert(entry)

    manager.save()
    loaded = DictionaryManager(path, serializer=serializer)

    assert loaded.lookup("hello") == entry


def test_load_synchronizes_vector_index(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    vector = [1.0] + [0.0] * 383
    entry = make_entry(source_term="hi", target_term="hola")
    entry.embeddings = [vector]

    manager = DictionaryManager(path)
    manager.insert(entry)
    manager.save()

    # A fresh manager reads the JSON and must rebuild the vector index so the
    # stored embedding is retrievable via similarity search.
    reloaded = DictionaryManager(path)
    results = reloaded.vector_store.search_by_vector(vector, k=1)
    assert len(results) == 1
    assert results[0].entry.source_term == "hi"


def test_prune_templates_reduces_embeddings(tmp_path: Path) -> None:
    manager = DictionaryManager(tmp_path / "lexicon.json")
    entry = make_entry(source_term="hi", target_term="hola")
    entry.embeddings = [[float(i)] + [0.0] * 383 for i in range(8)]
    manager.upsert(entry)

    manager.prune_templates(max_templates=5)

    assert len(manager.lookup("hi").embeddings) == 5


def test_embeddings_validation_rejects_ragged_and_non_finite() -> None:
    with pytest.raises(ValueError):
        LexiconEntry(
            source_term="hi",
            target_term="hola",
            embeddings=[[1.0, 2.0], [1.0, 2.0, 3.0]],
        )

    with pytest.raises(ValueError):
        LexiconEntry(
            source_term="hi",
            target_term="hola",
            embeddings=[[float("nan")] + [0.0] * 383],
        )

    with pytest.raises(ValueError):
        LexiconEntry(source_term="hi", target_term="hola", embeddings=[[]])


def test_load_failure_clears_stale_vector_index(tmp_path: Path) -> None:
    path = tmp_path / "lexicon.json"
    vector = [1.0] + [0.0] * 383
    entry = make_entry(source_term="hi", target_term="hola")
    entry.embeddings = [vector]

    manager = DictionaryManager(path)
    manager.insert(entry)
    manager.save()
    assert manager.vector_store.search_by_vector(vector, k=1)

    # Corrupt the lexicon on disk, then reload: the derived index must be cleared
    # so a failed load can't serve stale rows from the previous lexicon.
    path.write_text("{ not valid json", encoding="utf-8")
    assert manager.load() is False
    assert manager.vector_store.search_by_vector(vector, k=1) == []
