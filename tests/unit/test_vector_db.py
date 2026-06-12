from __future__ import annotations

import hashlib
from typing import TypeAlias

import chromadb.errors
import pytest
from chromadb.utils.embedding_functions import EmbeddingFunction

from vocab_zero.core.dictionary import LexiconEntry
from vocab_zero.core.vector_db import SearchResult, VectorStoreClient

ConfigValue: TypeAlias = str | int | float | bool | None


class DeterministicEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        self._call_count = 0

    @staticmethod
    def name() -> str:
        return "deterministic-test-embedding"

    def get_config(self) -> dict[str, ConfigValue]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, ConfigValue]) -> DeterministicEmbeddingFunction:
        return DeterministicEmbeddingFunction()

    def __call__(self, input: list[str]) -> list[list[float]]:
        self._call_count += 1
        embeddings: list[list[float]] = []
        for text in input:
            hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
            embedding = []
            for i in range(384):
                byte_idx = i % 32
                embedding.append(hash_bytes[byte_idx] / 255.0)
            embeddings.append(embedding)
        return embeddings


@pytest.fixture
def test_embedding_function() -> DeterministicEmbeddingFunction:
    return DeterministicEmbeddingFunction()


@pytest.fixture
def sample_entries() -> list[LexiconEntry]:
    return [
        LexiconEntry(
            source_term="hello",
            target_term="hola",
            confidence=0.9,
            context_examples=["Hello, how are you?", "Say hello to everyone"],
        ),
        LexiconEntry(
            source_term="goodbye",
            target_term="adios",
            confidence=0.85,
            context_examples=["Goodbye, see you later"],
        ),
        LexiconEntry(
            source_term="cat",
            target_term="gato",
            confidence=0.95,
            context_examples=["The cat is sleeping"],
        ),
    ]


def test_add_entry_success(tmp_path, test_embedding_function, sample_entries):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    assert client.add_entry(sample_entries[0]) is True
    assert client.add_entry(sample_entries[1]) is True
    
    results = client.search("hello", k=5)
    assert len(results) >= 1


def test_add_entry_upsert(tmp_path, test_embedding_function, sample_entries):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    entry = sample_entries[0]
    assert client.add_entry(entry) is True
    
    updated_entry = LexiconEntry(
        source_term="hello",
        target_term="bonjour",
        confidence=0.8,
        context_examples=["Updated context"],
    )
    assert client.add_entry(updated_entry) is True
    
    results = client.search("hello", k=1)
    assert len(results) == 1
    assert results[0].entry.target_term == "bonjour"
    assert results[0].entry.confidence == 0.8


def test_search_retrieves_nearest_entries(tmp_path, test_embedding_function, sample_entries):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    for entry in sample_entries:
        client.add_entry(entry)
    
    results = client.search("hello", k=5)
    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_search_empty_query(tmp_path, test_embedding_function):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    assert client.search("", k=5) == []
    assert client.search("   ", k=5) == []


def test_search_invalid_k(tmp_path, test_embedding_function):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    assert client.search("hello", k=0) == []
    assert client.search("hello", k=-1) == []
    assert client.search("hello", k="invalid") == []  # type: ignore[arg-type]


def test_delete_removes_entry(tmp_path, test_embedding_function, sample_entries):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    client.add_entry(sample_entries[0])
    assert client.delete("hello") is True
    assert client.delete("hello") is False
    assert client.search("hello", k=5) == []


def test_delete_nonexistent(tmp_path, test_embedding_function):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    assert client.delete("nonexistent") is False


def test_clear_empties_collection(tmp_path, test_embedding_function, sample_entries):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    for entry in sample_entries:
        client.add_entry(entry)
    
    assert client.clear() is True
    assert client.search("hello", k=5) == []


def test_persistence_across_reinstantiation(tmp_path, test_embedding_function, sample_entries):
    client1 = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    client1.add_entry(sample_entries[0])
    client1.add_entry(sample_entries[1])
    
    client2 = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    results = client2.search("hello", k=5)
    assert len(results) >= 2


def test_malformed_metadata_handling(tmp_path, test_embedding_function, sample_entries):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    client.add_entry(sample_entries[0])
    
    client._collection.upsert(
        ids=["bad_entry"],
        documents=["test document"],
        metadatas=[{"source_term": 123, "target_term": None}],  # type: ignore[dict-item]
    )
    
    results = client.search("test", k=5)
    assert len(results) == 1
    assert results[0].entry.source_term == "hello"


def test_chroma_write_failure_handling(tmp_path, monkeypatch, test_embedding_function, sample_entries):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    def failing_upsert(*args, **kwargs):
        raise chromadb.errors.ChromaError("ChromaDB write failed")
    
    monkeypatch.setattr(client._collection, "upsert", failing_upsert)
    
    assert client.add_entry(sample_entries[0]) is False


def test_search_order_preserved(tmp_path, test_embedding_function, sample_entries):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    for entry in sample_entries:
        client.add_entry(entry)
    
    results = client.search("hello", k=3)
    assert len(results) >= 2, "Need at least 2 results to test ordering"
    if len(results) > 1:
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


def test_score_conversion(tmp_path, test_embedding_function, sample_entries):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    client.add_entry(sample_entries[0])
    results = client.search("hello", k=1)
    
    assert len(results) == 1
    assert 0.0 <= results[0].score <= 1.0




def test_multiple_collections_isolated(tmp_path, test_embedding_function, sample_entries):
    client1 = VectorStoreClient(
        persist_dir=tmp_path,
        collection_name="lexicon1",
        embedding_function=test_embedding_function,
    )
    
    client2 = VectorStoreClient(
        persist_dir=tmp_path,
        collection_name="lexicon2",
        embedding_function=test_embedding_function,
    )
    
    client1.add_entry(sample_entries[0])
    client2.add_entry(sample_entries[1])
    
    results1_hello = client1.search("hello", k=5)
    results2_goodbye = client2.search("goodbye", k=5)
    
    assert len(results1_hello) == 1
    assert len(results2_goodbye) == 1
    assert results1_hello[0].entry.source_term == "hello"
    assert results2_goodbye[0].entry.source_term == "goodbye"
    
    results1_goodbye = client1.search("goodbye", k=5)
    results2_hello = client2.search("hello", k=5)
    
    assert all(r.entry.source_term != "goodbye" for r in results1_goodbye)
    assert all(r.entry.source_term != "hello" for r in results2_hello)

def test_entry_with_empty_context(tmp_path, test_embedding_function):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    entry = LexiconEntry(
        source_term="test",
        target_term="prueba",
        confidence=0.7,
        context_examples=[],
    )
    
    assert client.add_entry(entry) is True
    results = client.search("test", k=1)
    assert len(results) == 1
    assert results[0].entry.context_examples == []


def test_entry_with_unicode(tmp_path, test_embedding_function):
    client = VectorStoreClient(
        persist_dir=tmp_path,
        embedding_function=test_embedding_function,
    )
    
    entry = LexiconEntry(
        source_term="café",
        target_term="咖啡",
        confidence=0.9,
        context_examples=["I'd like a café", "咖啡 is delicious"],
    )
    
    assert client.add_entry(entry) is True
    results = client.search("café", k=1)
    assert len(results) == 1
    assert results[0].entry.source_term == "café"
