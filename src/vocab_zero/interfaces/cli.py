from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING, Callable

from vocab_zero.core.dictionary import DictionaryManager, LexiconEntry
from vocab_zero.core.engine import TranslationEngine
from vocab_zero.core.llm_client import GemmaClient, OpenAICompatibleClient
from vocab_zero.core.models import FeedbackRequest, TranslationConfig, TranslationResult
from vocab_zero.core.vector_db import VectorStoreClient
from vocab_zero.interfaces.base import BaseInterface

if TYPE_CHECKING:
    from vocab_zero.core.models import TranslationConfig


class CLIInterface(BaseInterface):
    """Terminal CLI adapter for translation interaction."""

    def __init__(
        self,
        engine: TranslationEngine,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ) -> None:
        self.engine = engine
        self.input_func = input_func
        self.output_func = output_func
        self.engine.on_feedback_required = self.request_human_feedback

    def run(self) -> None:
        """Run the interactive CLI loop."""
        self.output_func("VocabZero CLI - Type a term to translate, 'quit' or 'exit' to leave")

        while True:
            try:
                user_input = self.input_func("> ").strip()

                if not user_input:
                    self.output_func("(empty input - skipped)")
                    continue

                if user_input.lower() in ("quit", "exit"):
                    self.output_func("Goodbye!")
                    break

                result = self.engine.translate(user_input)
                self.display_translation(result)

            except (KeyboardInterrupt, EOFError):
                self.output_func("\nGoodbye!")
                break

    def display_translation(self, result: TranslationResult) -> None:
        """Display a translation result to the user."""
        if result.status == "translated":
            self.output_func(f"Translation: {result.translated_text}")
            if result.confidence > 0:
                self.output_func(f"Confidence: {result.confidence:.2f}")
            if result.source:
                self.output_func(f"Source: {result.source}")
            if result.context_used:
                self.output_func(f"Context used: {', '.join(result.context_used)}")

        elif result.status == "low_confidence":
            self.output_func(f"Translation (low confidence): {result.translated_text}")
            self.output_func(f"Confidence: {result.confidence:.2f}")
            if result.source:
                self.output_func(f"Source: {result.source}")
            if result.context_used:
                self.output_func(f"Context used: {', '.join(result.context_used)}")

        elif result.status == "requires_feedback":
            self.output_func(f"Feedback required: {result.feedback_request.reason if result.feedback_request else 'Unknown reason'}")
            if result.feedback_request and result.feedback_request.candidate_matches:
                self.output_func(f"Candidates: {', '.join(result.feedback_request.candidate_matches)}")

        elif result.status == "feedback_declined":
            self.output_func("Feedback skipped - no translation provided")

        elif result.status == "learned":
            self.output_func(f"Learned: {result.translated_text}")
            if result.error_code:
                self.output_func(f"Warning: {result.error_code}")

        elif result.status == "error":
            self.output_func(f"Error: {result.error_message or 'Unknown error'}")
            if result.error_code:
                self.output_func(f"Error code: {result.error_code}")
        else:
            self.output_func(
                f"Unhandled status: '{result.status}'."
                f"Message: {result.error_message or result.translated_text or 'No details available.'}"
            )

    def request_human_feedback(self, request: FeedbackRequest) -> LexiconEntry | None:
        """Request human feedback and return a new lexicon entry or None if rejected."""
        self.output_func(f"Feedback needed for: {request.source_term}")
        if request.context:
            self.output_func(f"Context: {request.context}")
        if request.reason:
            self.output_func(f"Reason: {request.reason}")
        if request.candidate_matches:
            self.output_func(f"Candidate translations: {', '.join(request.candidate_matches)}")

        self.output_func("Enter the correct translation (or press Enter to skip):")
        user_input = self.input_func("> ").strip()

        if not user_input:
            return None

        return LexiconEntry(
            source_term=request.source_term,
            target_term=user_input,
            confidence=1.0,
            context_examples=[request.context] if request.context else [],
        )


def build_engine(
    dictionary_path: str = "lexicon.json",
    vector_db_path: str | None = None,
) -> TranslationEngine:
    """Factory function to build a TranslationEngine with configured components.

    Respects the following environment variables:

    - ``LLM_PROVIDER``: ``"gemma"`` to use the local Gemma model or OpenAI-compatible
      endpoint, or ``"openai"`` (default) to use any OpenAI-compatible endpoint.
    - ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``LLM_MODEL_NAME``: used when
      provider is ``"openai"`` or ``"gemma"`` with an OpenAI-compatible endpoint.
    """
    dictionary = DictionaryManager(path=dictionary_path)

    vector_store: VectorStoreClient | None = None
    if vector_db_path:
        vector_store = VectorStoreClient(persist_dir=vector_db_path)

    config = TranslationConfig.from_env()

    llm_client: GemmaClient | OpenAICompatibleClient | None = None
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "gemma":
        llm_client = GemmaClient(config=config)
    elif config.api_key or config.base_url:
        llm_client = OpenAICompatibleClient(config)

    return TranslationEngine(
        dictionary=dictionary,
        vector_store=vector_store,
        llm_client=llm_client,
        config=config,
    )


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="VocabZero CLI - Dynamic Lexicon-Learning Translator")
    parser.add_argument(
        "--dictionary",
        default="lexicon.json",
        help="Path to the dictionary JSON file (default: lexicon.json)",
    )
    parser.add_argument(
        "--vector-db",
        default=None,
        help="Path to the vector database (optional, enables semantic search)",
    )

    args = parser.parse_args()

    engine = build_engine(dictionary_path=args.dictionary, vector_db_path=args.vector_db)
    cli = CLIInterface(engine)
    cli.run()
