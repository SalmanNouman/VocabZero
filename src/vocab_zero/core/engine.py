from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from vocab_zero.core.dictionary import DictionaryManager, LexiconEntry
from vocab_zero.core.llm_client import LLMClient
from vocab_zero.core.models import (
    AudioConfig,
    FeedbackRequest,
    TranslationConfig,
    TranslationResult,
)
from vocab_zero.core.vector_db import SearchResult, VectorStoreClient

if TYPE_CHECKING:
    from vocab_zero.core.models import LLMResponse


class TranslationEngine:
    def __init__(
        self,
        dictionary: DictionaryManager,
        vector_store: VectorStoreClient | None = None,
        llm_client: LLMClient | None = None,
        config: TranslationConfig | None = None,
        on_feedback_required: Callable[[FeedbackRequest], LexiconEntry | None] | None = None,
        audio_config: AudioConfig | None = None,
    ) -> None:
        self.dictionary = dictionary
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.config = config or TranslationConfig()
        self.on_feedback_required = on_feedback_required
        self.audio_config = audio_config or AudioConfig()

    def translate(
        self,
        source_term: str,
        context: str | None = None,
    ) -> TranslationResult:
        if not source_term or not source_term.strip():
            return TranslationResult(
                translated_text="",
                status="error",
                error_code="invalid_input",
                error_message="Source term cannot be empty",
            )

        normalized_term = source_term.strip()
        entry = self.dictionary.lookup(normalized_term)

        if entry is not None:
            return self._handle_known_term(entry, normalized_term, context)
        
        return self._handle_unknown_term(normalized_term, context)

    def _handle_known_term(
        self,
        entry: LexiconEntry,
        normalized_term: str,
        context: str | None,
    ) -> TranslationResult:
        if entry.confidence >= self.config.high_threshold:
            return TranslationResult(
                translated_text=entry.target_term,
                confidence=entry.confidence,
                source="dictionary",
                status="translated",
            )

        if entry.confidence >= self.config.low_threshold:
            return self._handle_medium_confidence(entry, normalized_term, context)

        return self._handle_low_confidence(entry, normalized_term, context)

    def _handle_medium_confidence(
        self,
        entry: LexiconEntry,
        normalized_term: str,
        context: str | None,
    ) -> TranslationResult:
        if self.llm_client is not None:
            vector_results = self._query_vector_store(source_term=normalized_term, context=context)
            llm_response = self._call_llm(normalized_term, context, vector_results)
            
            if llm_response is not None:
                final_confidence = self._calculate_confidence(llm_response, vector_results)
                
                if final_confidence >= self.config.high_threshold:
                    return TranslationResult(
                        translated_text=llm_response.translation,
                        confidence=final_confidence,
                        source="llm_inference",
                        status="translated",
                        context_used=[r.entry.target_term for r in vector_results],
                    )
                
                return TranslationResult(
                    translated_text=llm_response.translation,
                    confidence=final_confidence,
                    source="llm_inference",
                    status="low_confidence",
                    context_used=[r.entry.target_term for r in vector_results],
                )

        return TranslationResult(
            translated_text=entry.target_term,
            confidence=entry.confidence,
            source="dictionary",
            status="low_confidence",
        )

    def _handle_low_confidence(
        self,
        entry: LexiconEntry,
        normalized_term: str,
        context: str | None,
    ) -> TranslationResult:
        if self.on_feedback_required is not None:
            feedback_request = FeedbackRequest(
                source_term=normalized_term,
                context=context,
                candidate_matches=[entry.target_term],
                reason="Stored translation has low confidence",
            )
            
            new_entry = self.on_feedback_required(feedback_request)
            
            if new_entry is not None:
                return self.persist_learned_entry(new_entry)

            return TranslationResult(
                translated_text=entry.target_term,
                confidence=entry.confidence,
                source="dictionary",
                status="feedback_declined",
            )

        return TranslationResult(
            translated_text=entry.target_term,
            confidence=entry.confidence,
            source="dictionary",
            status="requires_feedback",
            feedback_request=FeedbackRequest(
                source_term=normalized_term,
                context=context,
                candidate_matches=[entry.target_term],
                reason="Stored translation has low confidence",
            ),
        )

    def _handle_unknown_term(
        self,
        source_term: str,
        context: str | None,
    ) -> TranslationResult:
        vector_results = self._query_vector_store(source_term, context)
        
        if self.llm_client is not None:
            llm_response = self._call_llm(source_term, context, vector_results)
            
            if llm_response is not None:
                final_confidence = self._calculate_confidence(llm_response, vector_results)
                
                if final_confidence >= self.config.low_threshold:
                    return TranslationResult(
                        translated_text=llm_response.translation,
                        confidence=final_confidence,
                        source="llm_inference",
                        status="low_confidence",
                        context_used=[r.entry.target_term for r in vector_results],
                    )

        return self._request_feedback(source_term, context, vector_results)

    def _query_vector_store(
        self,
        source_term: str,
        context: str | None,
    ) -> list[SearchResult]:
        if self.vector_store is None:
            return []

        query_parts = [source_term]
        if context:
            query_parts.append(context.strip()[:self.config.max_context_length])
        
        query = " ".join(query_parts)
        
        return self.vector_store.search(query, k=self.config.rag_k)

    def _call_llm(
        self,
        source_term: str,
        context: str | None,
        vector_results: list[SearchResult],
    ) -> LLMResponse | None:
        if self.llm_client is None:
            return None

        examples = [r.entry.target_term for r in vector_results]
        
        if context:
            context = context[:self.config.max_context_length]
        
        return self.llm_client.translate(source_term, context, examples, self.config)

    def _calculate_confidence(
        self,
        llm_response: LLMResponse,
        vector_results: list[SearchResult],
    ) -> float:
        if not vector_results:
            return max(0.0, min(1.0, llm_response.confidence))
        
        max_vector_similarity = max((r.score for r in vector_results), default=0.0)
        
        total_weight = self.config.llm_confidence_weight + self.config.vector_confidence_weight
        if total_weight > 0:
            normalized_llm_weight = self.config.llm_confidence_weight / total_weight
            normalized_vector_weight = self.config.vector_confidence_weight / total_weight
        else:
            normalized_llm_weight = 0.5
            normalized_vector_weight = 0.5
        
        final_confidence = (
            (normalized_llm_weight * llm_response.confidence)
            + (normalized_vector_weight * max_vector_similarity)
        )
        
        return max(0.0, min(1.0, final_confidence))

    def _request_feedback(
        self,
        source_term: str,
        context: str | None,
        vector_results: list[SearchResult],
    ) -> TranslationResult:
        candidates = [r.entry.target_term for r in vector_results]
        
        if self.on_feedback_required is not None:
            feedback_request = FeedbackRequest(
                source_term=source_term,
                context=context,
                candidate_matches=candidates,
                reason="Term not found in dictionary",
            )
            
            new_entry = self.on_feedback_required(feedback_request)
            
            if new_entry is not None:
                return self.persist_learned_entry(new_entry)

            return TranslationResult(
                translated_text="",
                confidence=0.0,
                source="none",
                status="feedback_declined",
            )

        return TranslationResult(
            translated_text="",
            confidence=0.0,
            source="none",
            status="requires_feedback",
            feedback_request=FeedbackRequest(
                source_term=source_term,
                context=context,
                candidate_matches=candidates,
                reason="Term not found in dictionary",
            ),
        )

    def persist_learned_entry(self, entry: LexiconEntry) -> TranslationResult:
        self.dictionary.upsert(entry)
        
        try:
            self.dictionary.save()
        except (OSError, IOError):
            self.dictionary.delete(entry.source_term)
            return TranslationResult(
                translated_text=entry.target_term,
                confidence=entry.confidence,
                source="human_feedback",
                status="error",
                error_code="persistence_failed",
                error_message="Failed to save dictionary entry",
            )
        
        vector_error = None
        if self.vector_store is not None:
            success = self.vector_store.add_entry(entry)
            if not success:
                vector_error = "semantic_index_update_failed"

        return TranslationResult(
            translated_text=entry.target_term,
            confidence=entry.confidence,
            source="human_feedback",
            status="learned",
            error_code=vector_error,
            error_message="Semantic index update failed" if vector_error else None,
        )

    def rerank_acoustic_candidates(
        self,
        candidates: list[tuple[LexiconEntry, float]],
        context: str | None = None,
    ) -> tuple[LexiconEntry | None, float]:
        """Rerank candidates using acoustic DTW, N-Gram probability, and optionally LLM semantic ranking.

        Returns:
            Tuple of (selected_entry, confidence).
        """
        if not candidates:
            return None, 0.0

        # DTW distance threshold sourced from AudioConfig (dimension-aware)
        dtw_threshold = self.audio_config.dtw_threshold

        # Sort candidates by DTW distance (ascending)
        candidates = sorted(candidates, key=lambda x: x[1])

        best_entry, best_dist = candidates[0]

        # Check if the acoustic distance is below the threshold. If even the best is above, it's not a match.
        if best_dist >= dtw_threshold:
            return None, 0.0

        if len(candidates) == 1:
            conf = float(max(0.0, min(1.0, 1.0 - (best_dist / dtw_threshold))))
            if conf < self.audio_config.min_confidence_gate:
                return None, 0.0
            return best_entry, conf

        second_entry, second_dist = candidates[1]


        # Phonetic ambiguity threshold (proportional to DTW threshold)
        dtw_diff_threshold = dtw_threshold * self.audio_config.ambiguity_margin_ratio
        ambiguous = (second_dist - best_dist) < dtw_diff_threshold

        # 1. Apply N-gram language model
        last_word = None
        if context:
            words = [w.strip().lower() for w in context.split() if w.strip()]
            if words:
                last_word = words[-1]

        # Calculate N-gram transition probabilities
        ngram_probs = {}
        for entry, dist in candidates:
            prob = self.dictionary.ngram_model.get_transition_probability(last_word, entry.target_term)
            ngram_probs[entry.source_term] = prob

        # 2. Apply LLM reranking if ambiguous and LLM is available
        llm_selected = None
        llm_conf = 0.0
        if ambiguous and self.llm_client is not None and context:
            # Get target terms of top 5 candidates
            target_options = [entry.target_term for entry, _ in candidates[:5]]
            try:
                llm_result = self.llm_client.select_best_candidate(target_options, context)
                if llm_result:
                    selected_term, confidence = llm_result
                    # Find matching entry
                    for entry, _ in candidates:
                        if entry.target_term.strip().lower() == selected_term.strip().lower():
                            llm_selected = entry
                            llm_conf = confidence
                            break
            except Exception:
                logging.getLogger(__name__).warning(
                    "LLM select_best_candidate failed; falling back to acoustic + ngram scoring",
                    exc_info=True,
                )

        # Combined scoring to choose the winner
        best_ranked_entry = best_entry
        highest_score = -1.0

        for entry, dist in candidates:
            # Acoustic score: 1.0 is perfect, 0.0 is at/above threshold
            s_acoustic = max(0.0, 1.0 - (dist / dtw_threshold))

            # N-gram score: transition probability (0.0 to 1.0)
            s_ngram = ngram_probs.get(entry.source_term, 0.0)

            # LLM score
            s_llm = 0.0
            if llm_selected is not None and entry.source_term == llm_selected.source_term:
                s_llm = llm_conf

            # Weighted combined score (acoustic 0.5, ngram 0.2, llm 0.3)
            score = 0.5 * s_acoustic + 0.2 * s_ngram + 0.3 * s_llm

            if score > highest_score:
                highest_score = score
                best_ranked_entry = entry

        # Calculate final confidence
        winner_dist = next(dist for entry, dist in candidates if entry.source_term == best_ranked_entry.source_term)
        winner_acoustic_conf = float(max(0.0, min(1.0, 1.0 - (winner_dist / dtw_threshold))))

        if llm_selected is not None and best_ranked_entry.source_term == llm_selected.source_term:
            final_conf = 0.7 * winner_acoustic_conf + 0.3 * llm_conf
        else:
            margin = max(0.0, (second_dist - best_dist) / dtw_threshold)
            blend = self.audio_config.ambiguity_confidence_floor + (
                1.0 - self.audio_config.ambiguity_confidence_floor
            ) * min(1.0, margin / self.audio_config.ambiguity_margin_ratio)
            final_conf = winner_acoustic_conf * blend

        if final_conf < self.audio_config.min_confidence_gate:
            return None, 0.0
        return best_ranked_entry, final_conf
