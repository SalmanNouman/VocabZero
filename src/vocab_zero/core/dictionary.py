from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Iterator, Protocol, TypeAlias

from pydantic import BaseModel, Field, ValidationError

from vocab_zero.utils.audio import k_medoids

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class LexiconEntry(BaseModel):
    source_term: str
    target_term: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    context_examples: list[str] = Field(default_factory=list)
    # One Whisper-tiny encoder embedding (384 floats) per taught recording.
    embeddings: list[list[float]] = Field(default_factory=list)


class NGramModel:
    def __init__(self) -> None:
        self.bigram_counts: dict[str, dict[str, int]] = {}
        self.unigram_counts: dict[str, int] = {}

    def train_on_sentence(self, sentence: str) -> None:
        if not sentence:
            return
        words = [w.strip().strip(".,?!;:()\"'").lower() for w in sentence.split() if w.strip()]
        if not words:
            return
        for i in range(len(words)):
            curr = words[i]
            self.unigram_counts[curr] = self.unigram_counts.get(curr, 0) + 1
            if i > 0:
                prev = words[i - 1]
                if prev not in self.bigram_counts:
                    self.bigram_counts[prev] = {}
                self.bigram_counts[prev][curr] = self.bigram_counts[prev].get(curr, 0) + 1

    def get_transition_probability(self, prev_word: str | None, curr_word: str) -> float:
        curr = curr_word.strip().lower()
        if not prev_word:
            count = self.unigram_counts.get(curr, 0)
            total = sum(self.unigram_counts.values())
            v = len(self.unigram_counts)
            return (count + 0.1) / (total + 0.1 * v) if total > 0 else 1.0 / (v if v > 0 else 100)

        prev = prev_word.strip().lower()
        prev_count = self.unigram_counts.get(prev, 0)
        if prev_count == 0 or prev not in self.bigram_counts:
            return self.get_transition_probability(None, curr)

        curr_count = self.bigram_counts[prev].get(curr, 0)
        v = len(self.unigram_counts)
        return (curr_count + 0.1) / (prev_count + 0.1 * v)


class Serializer(Protocol):
    def dumps(self, data: JsonObject) -> str: ...

    def loads(self, text: str) -> JsonObject: ...


class JsonSerializer:
    def dumps(self, data: JsonObject) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)

    def loads(self, text: str) -> JsonObject:
        loaded = json.loads(text)
        if not isinstance(loaded, dict):
            raise ValueError("Top-level JSON must be an object")
        return loaded


class DictionaryManager:
    def __init__(self, path: Path | str = "lexicon.json", serializer: Serializer | None = None) -> None:
        self.path = Path(path)
        self.serializer = serializer or JsonSerializer()
        self._entries: dict[str, LexiconEntry] = {}
        
        # Initialize VectorStoreClient pointing to a persistent directory in the same path
        from vocab_zero.core.vector_db import VectorStoreClient
        persist_dir = self.path.parent / ".chroma"
        self.vector_store = VectorStoreClient(persist_dir=persist_dir)
        self.ngram_model = NGramModel()
        self._lock = RLock()
        self.load()

    def lookup(self, source_term: str) -> LexiconEntry | None:
        with self._lock:
            return self._entries.get(source_term)

    def lookup_by_hash(self, acoustic_hash: str) -> LexiconEntry | None:
        return self.lookup(acoustic_hash)

    def insert(self, entry: LexiconEntry) -> bool:
        with self._lock:
            if entry.source_term in self._entries:
                return False
            self._entries[entry.source_term] = entry
            self.vector_store.add_entry(entry)
            for example in entry.context_examples:
                self.ngram_model.train_on_sentence(example)
            self.save()
            return True

    def upsert(self, entry: LexiconEntry) -> LexiconEntry:
        with self._lock:
            self._entries[entry.source_term] = entry
            self.vector_store.add_entry(entry)
            self.ngram_model = NGramModel()
            for stored in self._entries.values():
                for example in stored.context_examples:
                    self.ngram_model.train_on_sentence(example)
            self.save()
            return entry

    def update_confidence(self, source_term: str, delta: float) -> LexiconEntry | None:
        with self._lock:
            entry = self.lookup(source_term)
            if entry is None:
                return None
            confidence = min(1.0, max(0.0, entry.confidence + delta))
            updated = entry.model_copy(update={"confidence": confidence})
            self._entries[source_term] = updated
            self.vector_store.add_entry(updated)
            self.save()
            return updated

    def delete(self, source_term: str) -> bool:
        with self._lock:
            if source_term not in self._entries:
                return False
            del self._entries[source_term]
            self.vector_store.delete(source_term)
            self.save()
            return True

    def has(self, source_term: str) -> bool:
        with self._lock:
            return source_term in self._entries

    def iter_entries(self) -> Iterator[LexiconEntry]:
        with self._lock:
            return iter(list(self._entries.values()))

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_name(f"{self.path.name}.tmp")
            temp_path.write_text(self.serializer.dumps(self._dump_entries()), encoding="utf-8")
            temp_path.replace(self.path)

    def load(self) -> bool:
        with self._lock:
            if not self.path.exists():
                return False

            try:
                loaded = self.serializer.loads(self.path.read_text(encoding="utf-8"))
                entries = self._parse_entries(loaded)
            except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError):
                return False

            self._entries = entries
            self.vector_store.clear()
            for entry in self._entries.values():
                self.vector_store.add_entry(entry)

            self.ngram_model = NGramModel()
            for entry in self._entries.values():
                for example in entry.context_examples:
                    self.ngram_model.train_on_sentence(example)
            return True

    def prune_templates(self, max_templates: int = 5) -> None:
        with self._lock:
            changed = False
            for entry in self._entries.values():
                if len(entry.embeddings) > max_templates:
                    entry.embeddings = k_medoids(entry.embeddings, max_templates)
                    self.vector_store.add_entry(entry)
                    changed = True
            if changed:
                self.save()

    def _dump_entries(self) -> JsonObject:
        return {
            "entries": [entry.model_dump() for entry in self._entries.values()],
        }

    def _parse_entries(self, data: JsonObject) -> dict[str, LexiconEntry]:
        entries_data = data.get("entries")
        if not isinstance(entries_data, list):
            raise ValueError("Invalid lexicon storage format")

        entries: dict[str, LexiconEntry] = {}
        for entry_data in entries_data:
            entry = LexiconEntry.model_validate(entry_data)
            entries[entry.source_term] = entry
        return entries


