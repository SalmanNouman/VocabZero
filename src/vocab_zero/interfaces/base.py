from __future__ import annotations

from abc import ABC, abstractmethod

from vocab_zero.core.dictionary import LexiconEntry
from vocab_zero.core.models import FeedbackRequest, TranslationResult


class BaseInterface(ABC):
    """Abstract base class for translation interfaces."""

    @abstractmethod
    def run(self) -> None:
        """Run the interface's main interaction loop."""
        pass

    @abstractmethod
    def display_translation(self, result: TranslationResult) -> None:
        """Display a translation result to the user."""
        pass

    @abstractmethod
    def request_human_feedback(self, request: FeedbackRequest) -> LexiconEntry | None:
        """Request human feedback and return a new lexicon entry or None if rejected."""
        pass
