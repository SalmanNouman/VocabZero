from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, TypeAlias

from pydantic import BaseModel, Field, ValidationError

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class LexiconEntry(BaseModel):
    source_term: str
    target_term: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    context_examples: list[str] = Field(default_factory=list)


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
        self.load()

    def lookup(self, source_term: str) -> LexiconEntry | None:
        return self._entries.get(source_term)

    def insert(self, entry: LexiconEntry) -> bool:
        if entry.source_term in self._entries:
            return False
        self._entries[entry.source_term] = entry
        return True

    def upsert(self, entry: LexiconEntry) -> LexiconEntry:
        self._entries[entry.source_term] = entry
        return entry

    def update_confidence(self, source_term: str, delta: float) -> LexiconEntry | None:
        entry = self.lookup(source_term)
        if entry is None:
            return None
        confidence = min(1.0, max(0.0, entry.confidence + delta))
        updated = entry.model_copy(update={"confidence": confidence})
        self._entries[source_term] = updated
        return updated

    def delete(self, source_term: str) -> bool:
        if source_term not in self._entries:
            return False
        del self._entries[source_term]
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f"{self.path.name}.tmp")
        temp_path.write_text(self.serializer.dumps(self._dump_entries()), encoding="utf-8")
        temp_path.replace(self.path)

    def load(self) -> bool:
        if not self.path.exists():
            return False

        try:
            loaded = self.serializer.loads(self.path.read_text(encoding="utf-8"))
            entries = self._parse_entries(loaded)
        except (OSError, json.JSONDecodeError, ValidationError, TypeError, ValueError):
            return False

        self._entries = entries
        return True

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
