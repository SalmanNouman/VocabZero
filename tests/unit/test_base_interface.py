from __future__ import annotations

import pytest

from vocab_zero.core.dictionary import LexiconEntry
from vocab_zero.core.models import FeedbackRequest, TranslationResult
from vocab_zero.interfaces.base import BaseInterface


def test_base_interface_is_abstract():
    """Test that BaseInterface cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseInterface()


def test_concrete_interface_must_implement_abstract_methods():
    """Test that concrete classes must implement all abstract methods."""

    class IncompleteInterface(BaseInterface):
        pass

    with pytest.raises(TypeError):
        IncompleteInterface()


def test_concrete_interface_with_all_methods():
    """Test that a concrete class implementing all methods can be instantiated."""

    class CompleteInterface(BaseInterface):
        def run(self) -> None:
            pass

        def display_translation(self, result: TranslationResult) -> None:
            pass

        def request_human_feedback(self, request: FeedbackRequest) -> LexiconEntry | None:
            return None

    interface = CompleteInterface()
    assert interface is not None
    interface.run()
    interface.display_translation(TranslationResult(translated_text="test"))
    result = interface.request_human_feedback(FeedbackRequest(source_term="test"))
    assert result is None
