from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vocab_zero.core.dictionary import LexiconEntry
from vocab_zero.core.models import FeedbackRequest, TranslationResult
from vocab_zero.interfaces.api import app

MFCC_NUM_COEFFICIENTS = 12


class MockEngine:
    def __init__(self) -> None:
        self.dictionary = MagicMock()
        self.dictionary._entries = {}
        self.dictionary.iter_entries = lambda: iter(self.dictionary._entries.values())
        self.dictionary.has = lambda x: x in self.dictionary._entries
        self.translate = MagicMock()
        self.persist_learned_entry = MagicMock()


@pytest.fixture
def client():
    with patch("vocab_zero.interfaces.api.build_engine") as mock_build:
        engine = MockEngine()
        mock_build.return_value = engine
        with TestClient(app) as c:
            c.app.state.engine = engine
            yield c


def test_get_index_exists(client):
    with patch("vocab_zero.interfaces.api.INDEX_HTML") as mock_html, patch(
        "vocab_zero.interfaces.api.FileResponse"
    ) as mock_response:
        mock_html.exists.return_value = True
        mock_response.return_value = "file_content"
        response = client.get("/")
        assert response.status_code == 200
        mock_response.assert_called_once_with(mock_html)


def test_get_index_missing(client):
    with patch("vocab_zero.interfaces.api.INDEX_HTML") as mock_html:
        mock_html.exists.return_value = False
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {
            "ok": False,
            "error": {
                "code": "missing_static",
                "message": "Frontend static HTML file is missing",
            },
        }


def test_translate_success(client):
    engine = client.app.state.engine
    engine.translate.return_value = TranslationResult(
        translated_text="hello",
        confidence=0.9,
        source="dictionary",
        status="translated",
        context_used=["context1"],
    )

    response = client.post(
        "/api/translate", json={"source_term": "hola", "context": "greet"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "data": {
            "translated_text": "hello",
            "confidence": 0.9,
            "source": "dictionary",
            "status": "translated",
            "context_used": ["context1"],
            "error_code": None,
            "error_message": None,
            "feedback_request": None,
        },
    }
    engine.translate.assert_called_once_with("hola", "greet")


def test_translate_requires_feedback(client):
    engine = client.app.state.engine
    engine.translate.return_value = TranslationResult(
        translated_text="",
        confidence=0.0,
        source="none",
        status="requires_feedback",
        feedback_request=FeedbackRequest(
            source_term="hola",
            context="greet",
            candidate_matches=["hello"],
            reason="not found",
        ),
    )

    response = client.post(
        "/api/translate", json={"source_term": "hola", "context": "greet"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "data": {
            "translated_text": "",
            "confidence": 0.0,
            "source": "none",
            "status": "requires_feedback",
            "context_used": [],
            "error_code": None,
            "error_message": None,
            "feedback_request": {
                "source_term": "hola",
                "context": "greet",
                "candidate_matches": ["hello"],
                "reason": "not found",
            },
        },
    }


def test_translate_error(client):
    engine = client.app.state.engine
    engine.translate.return_value = TranslationResult(
        translated_text="",
        status="error",
        error_code="invalid_input",
        error_message="Source term cannot be empty",
    )

    response = client.post(
        "/api/translate", json={"source_term": "", "context": ""}
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "invalid_input",
            "message": "Source term cannot be empty",
        },
    }


def test_feedback_success(client):
    engine = client.app.state.engine
    engine.persist_learned_entry.return_value = TranslationResult(
        translated_text="hello",
        confidence=1.0,
        source="human_feedback",
        status="learned",
    )

    response = client.post(
        "/api/feedback",
        json={"source_term": "hola", "target_term": "hello", "context": "greet"},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "data": {"status": "learned"}}

    args, _ = engine.persist_learned_entry.call_args
    entry = args[0]
    assert entry.source_term == "hola"
    assert entry.target_term == "hello"
    assert entry.context_examples == ["greet"]


def test_feedback_error(client):
    engine = client.app.state.engine
    engine.persist_learned_entry.return_value = TranslationResult(
        translated_text="hello",
        confidence=1.0,
        source="human_feedback",
        status="error",
        error_code="persistence_failed",
        error_message="Failed to save entry",
    )

    response = client.post(
        "/api/feedback",
        json={"source_term": "hola", "target_term": "hello", "context": "greet"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "persistence_failed",
            "message": "Failed to save entry",
        },
    }


def test_get_lexicon(client):
    engine = client.app.state.engine
    engine.dictionary._entries = {
        "hola": LexiconEntry(
            source_term="hola",
            target_term="hello",
            confidence=1.0,
            context_examples=["greet"],
        )
    }

    response = client.get("/api/lexicon")
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "data": [
            {
                "source_term": "hola",
                "target_term": "hello",
                "confidence": 1.0,
                "context_examples": ["greet"],
            }
        ],
    }


def test_delete_lexicon_entry_success(client):
    engine = client.app.state.engine
    engine.dictionary._entries = {
        "hola": LexiconEntry(
            source_term="hola", target_term="hello", confidence=1.0
        )
    }

    response = client.delete("/api/lexicon/hola")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "data": {"deleted": True}}
    engine.dictionary.delete.assert_called_once_with("hola")
    engine.dictionary.save.assert_called_once()


def test_delete_lexicon_entry_not_found(client):
    engine = client.app.state.engine
    engine.dictionary._entries = {}

    response = client.delete("/api/lexicon/missing")
    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "not_found",
            "message": "Entry 'missing' not found in lexicon",
        },
    }


def test_translate_audio_match(client):
    engine = client.app.state.engine
    # Setup database entry with a stored MFCC template
    dummy_mfcc = [[0.1] * MFCC_NUM_COEFFICIENTS for _ in range(5)]
    engine.dictionary._entries = {
        "200_500": LexiconEntry(
            source_term="200_500",
            target_term="hello",
            confidence=1.0,
            mfcc_template=dummy_mfcc,
        )
    }

    dummy_audio = [0.0] * 1000

    with patch("vocab_zero.interfaces.api.dtw_distance", return_value=0.0):
        response = client.post(
            "/api/translate",
            json={"source_term": "200_500", "audio_data": dummy_audio},
        )

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["ok"] is True
    assert res_data["data"]["translated_text"] == "hello"
    assert res_data["data"]["source"] == "acoustic_matching"
    assert res_data["data"]["status"] == "translated"


def test_translate_audio_no_match(client):
    engine = client.app.state.engine
    # Setup database with high DTW distance target
    # query will be all zeros, target is all ones
    dummy_mfcc = [[1.0] * MFCC_NUM_COEFFICIENTS for _ in range(5)]
    engine.dictionary._entries = {
        "200_500": LexiconEntry(
            source_term="200_500",
            target_term="hello",
            confidence=1.0,
            mfcc_template=dummy_mfcc,
        )
    }

    dummy_audio = [0.0] * 1000

    with patch("vocab_zero.interfaces.api.dtw_distance", return_value=99.0):
        response = client.post(
            "/api/translate",
            json={"source_term": "200_500", "audio_data": dummy_audio},
        )

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["ok"] is True
    assert res_data["data"]["status"] == "requires_feedback"
    assert res_data["data"]["feedback_request"]["source_term"] == "200_500"
    assert res_data["data"]["feedback_request"]["candidate_matches"] == ["hello"]


def test_feedback_audio_save(client):
    engine = client.app.state.engine
    engine.persist_learned_entry.return_value = TranslationResult(
        translated_text="hello",
        confidence=1.0,
        source="human_feedback",
        status="learned",
    )

    dummy_audio = [0.0] * 1000
    response = client.post(
        "/api/feedback",
        json={
            "source_term": "200_500",
            "target_term": "hello",
            "audio_data": dummy_audio,
            "context": "greet",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "data": {"status": "learned"}}

    # Verify the template was generated and passed to dictionary entry
    args, _ = engine.persist_learned_entry.call_args
    entry = args[0]
    assert entry.source_term == "200_500"
    assert entry.target_term == "hello"
    assert entry.mfcc_template is not None
    assert len(entry.mfcc_template) > 0
    assert len(entry.mfcc_template[0]) == MFCC_NUM_COEFFICIENTS


def test_websocket_translate_text(client):
    engine = client.app.state.engine
    engine.translate.return_value = TranslationResult(
        translated_text="hello",
        confidence=0.9,
        source="dictionary",
        status="translated",
        context_used=["context1"],
    )

    with client.websocket_connect("/stream") as websocket:
        websocket.send_json({"type": "translate", "source_term": "hola", "context": "greet"})
        response = websocket.receive_json()

    assert response["type"] == "translation"
    assert response["translated_text"] == "hello"
    assert response["confidence"] == 0.9
    assert response["status"] == "translated"
    engine.translate.assert_called_once_with("hola", "greet")


def test_websocket_translate_requires_feedback(client):
    engine = client.app.state.engine
    engine.translate.return_value = TranslationResult(
        translated_text="",
        confidence=0.0,
        source="none",
        status="requires_feedback",
        feedback_request=FeedbackRequest(
            source_term="hola",
            context="greet",
            candidate_matches=["hello"],
            reason="not found",
        ),
    )

    with client.websocket_connect("/stream") as websocket:
        websocket.send_json({"type": "translate", "source_term": "hola", "context": "greet"})
        response = websocket.receive_json()

    assert response["type"] == "translation"
    assert response["status"] == "requires_feedback"
    assert response["feedback_request"]["source_term"] == "hola"
    assert response["feedback_request"]["candidate_matches"] == ["hello"]


def test_websocket_feedback(client):
    engine = client.app.state.engine
    engine.persist_learned_entry.return_value = TranslationResult(
        translated_text="hello",
        confidence=1.0,
        source="human_feedback",
        status="learned",
    )

    with client.websocket_connect("/stream") as websocket:
        websocket.send_json(
            {"type": "feedback", "source_term": "hola", "target_term": "hello", "context": "greet"}
        )
        response = websocket.receive_json()

    assert response["type"] == "feedback_acknowledged"
    assert response["status"] == "learned"

    args, _ = engine.persist_learned_entry.call_args
    entry = args[0]
    assert entry.source_term == "hola"
    assert entry.target_term == "hello"


def test_websocket_invalid_input(client):
    with client.websocket_connect("/stream") as websocket:
        websocket.send_json({"type": "translate"})
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "invalid_input"


def test_websocket_unknown_message_type(client):
    with client.websocket_connect("/stream") as websocket:
        websocket.send_json({"type": "unknown_type"})
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "unknown_message_type"
