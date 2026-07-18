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
            
            # Delete any existing entries for this source_term first to avoid duplicates
            self.delete(entry.source_term)
            
            if entry.mfcc_templates:
                for idx, template in enumerate(entry.mfcc_templates):
                    if not template or not template[0]:
                        continue
                    vector = template[0]
                    metadata = {
                        "source_term": entry.source_term,
                        "target_term": entry.target_term,
                        "confidence": str(entry.confidence),
                        "context_examples_json": json.dumps(entry.context_examples, ensure_ascii=False),
                        "template_idx": str(idx),
                    }
                    self._collection.upsert(
                        ids=[f"{entry.source_term}_t{idx}"],
                        embeddings=[vector],
                        documents=[document_text],
                        metadatas=[metadata],
                    )
            else:
                metadata = {
                    "source_term": entry.source_term,
                    "target_term": entry.target_term,
                    "confidence": str(entry.confidence),
                    "context_examples_json": json.dumps(entry.context_examples, ensure_ascii=False),
                    "template_idx": "none",
                }
                self._collection.upsert(
                    ids=[entry.source_term],
                    documents=[document_text],
                    metadatas=[metadata],
                )
            return True
        except Exception:
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
            seen_source_terms = set()
            for idx, _ in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][idx]
                distance = results["distances"][0][idx]
                score = max(0.0, min(1.0, 1.0 - distance))
                
                entry = self._metadata_to_entry(metadata)
                if entry is None:
                    continue
                
                # De-duplicate entries by source_term in standard text search results
                if entry.source_term in seen_source_terms:
                    continue
                seen_source_terms.add(entry.source_term)
                
                search_results.append(SearchResult(entry=entry, score=score))
            
            return search_results
        except Exception:
            return []

    def search_by_vector(self, query_vector: list[float], k: int = 5) -> list[SearchResult]:
        if not query_vector:
            return []
        if not isinstance(k, int) or k <= 0:
            return []
        
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=k,
            )
            
            if not results or not results["ids"] or not results["ids"][0]:
                return []
            
            search_results: list[SearchResult] = []
            seen_source_terms = set()
            for idx, _ in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][idx]
                distance = results["distances"][0][idx]
                score = max(0.0, min(1.0, 1.0 - distance))
                
                entry = self._metadata_to_entry(metadata)
                if entry is None:
                    continue
                
                # De-duplicate templates of the same source_term in search results
                if entry.source_term in seen_source_terms:
                    continue
                seen_source_terms.add(entry.source_term)
                
                search_results.append(SearchResult(entry=entry, score=score))
            
            return search_results
        except Exception:
            return []

    def delete(self, source_term: str) -> bool:
        if not source_term or not isinstance(source_term, str):
            return False
        
        try:
            # Delete by querying metadatas
            existing = self._collection.get(where={"source_term": source_term})
            if existing and existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                return True
            
            # Also try deleting direct id
            existing_direct = self._collection.get(ids=[source_term])
            if existing_direct and existing_direct["ids"]:
                self._collection.delete(ids=[source_term])
                return True
                
            return False
        except Exception:
            return False

    def clear(self) -> bool:
        try:
            existing = self._collection.get()
            if existing and existing["ids"]:
                self._collection.delete(ids=existing["ids"])
            return True
        except Exception:
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
