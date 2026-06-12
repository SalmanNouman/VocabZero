from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from pydantic import BaseModel

from vocab_zero.core.dictionary import LexiconEntry

if TYPE_CHECKING:
    from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction


class SearchResult(BaseModel):
    entry: LexiconEntry
    score: float


class VectorStoreClient:
    def __init__(
        self,
        persist_dir: Path | str = ".chroma",
        collection_name: str = "lexicon",
        embedding_function: EmbeddingFunction | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        if embedding_function is not None:
            self._embedding_function: ChromaEmbeddingFunction = embedding_function
        else:
            self._embedding_function = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_function,
        )

    def add_entry(self, entry: LexiconEntry) -> bool:
        try:
            document_text = self._build_document_text(entry)
            metadata = {
                "source_term": entry.source_term,
                "target_term": entry.target_term,
                "confidence": str(entry.confidence),
                "context_examples_json": json.dumps(entry.context_examples, ensure_ascii=False),
            }
            
            self._collection.upsert(
                ids=[entry.source_term],
                documents=[document_text],
                metadatas=[metadata],
            )
            return True
        except (ValueError, TypeError, chromadb.errors.ChromaError):
            return False

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            return []
        if not isinstance(k, int) or k <= 0:
            return []
        
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
            )
            
            if not results or not results["ids"] or not results["ids"][0]:
                return []
            
            search_results: list[SearchResult] = []
            for idx, _ in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][idx]
                distance = results["distances"][0][idx]
                score = max(0.0, min(1.0, 1.0 - distance))
                
                entry = self._metadata_to_entry(metadata)
                if entry is None:
                    continue
                
                search_results.append(SearchResult(entry=entry, score=score))
            
            return search_results
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, chromadb.errors.ChromaError):
            return []

    def delete(self, source_term: str) -> bool:
        if not source_term or not isinstance(source_term, str):
            return False
        
        try:
            existing = self._collection.get(ids=[source_term])
            if not existing or not existing["ids"]:
                return False
            
            self._collection.delete(ids=[source_term])
            return True
        except (ValueError, TypeError, chromadb.errors.ChromaError):
            return False

    def clear(self) -> bool:
        try:
            existing = self._collection.get()
            if existing and existing["ids"]:
                self._collection.delete(ids=existing["ids"])
            return True
        except (ValueError, TypeError, chromadb.errors.ChromaError):
            return False

    def _build_document_text(self, entry: LexiconEntry) -> str:
        parts = [entry.source_term, entry.target_term]
        if entry.context_examples:
            parts.extend(entry.context_examples)
        return " ".join(parts)

    def _metadata_to_entry(self, metadata: dict[str, object]) -> LexiconEntry | None:
        try:
            source_term = metadata.get("source_term")
            target_term = metadata.get("target_term")
            confidence_str = metadata.get("confidence", "0.5")
            context_examples_json = metadata.get("context_examples_json", "[]")
            
            if not isinstance(source_term, str) or not isinstance(target_term, str):
                return None
            
            confidence = float(confidence_str)
            context_examples = json.loads(context_examples_json)
            if not isinstance(context_examples, list) or not all(isinstance(item, str) for item in context_examples):
                context_examples = []
            
            return LexiconEntry(
                source_term=source_term,
                target_term=target_term,
                confidence=confidence,
                context_examples=context_examples,
            )
        except Exception:
            return None
