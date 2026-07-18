from __future__ import annotations



import asyncio

import json

from pathlib import Path

from unittest.mock import MagicMock, patch



import pytest

from fastapi.testclient import TestClient



from vocab_zero.core.dictionary import DictionaryManager, LexiconEntry

from vocab_zero.core.models import AudioConfig, FeedbackRequest, TranslationResult, LLMResponse

from vocab_zero.interfaces.api import app, periodic_pruning_task, perform_audio_matching_candidates




class MockEngine:

    def __init__(self) -> None:

        self.dictionary = MagicMock()

        self.dictionary._entries = {}

        self.dictionary.iter_entries = lambda: iter(self.dictionary._entries.values())

        self.dictionary.has = lambda x: x in self.dictionary._entries

        self.dictionary.lookup = lambda x: self.dictionary._entries.get(x)

        self.dictionary.delete.side_effect = lambda x: x in self.dictionary._entries and (self.dictionary._entries.pop(x, None) is not None)

        self.translate = MagicMock()

        self.persist_learned_entry = MagicMock()

        self.rerank_acoustic_candidates = MagicMock()

        self.audio_config = AudioConfig()





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

    assert response.json() == {

        "ok": True,

        "data": {"status": "learned", "source_term": "hola"},

    }



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


    entry = LexiconEntry(

        source_term="200_500",

        target_term="hello",

        confidence=1.0,

        embeddings=[[0.1] * 384],

    )

    engine.dictionary._entries = {"200_500": entry}

    engine.dictionary.vector_store.search_by_vector.return_value = []

    engine.rerank_acoustic_candidates.return_value = (entry, 0.9)



    dummy_audio = [0.0] * 1000



    with patch("vocab_zero.interfaces.api.extract_whisper_embedding", return_value=[0.1] * 384):

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


    entry = LexiconEntry(

        source_term="200_500",

        target_term="hello",

        confidence=1.0,

        embeddings=[[0.1] * 384],

    )

    engine.dictionary._entries = {"200_500": entry}

    engine.dictionary.vector_store.search_by_vector.return_value = []

    engine.rerank_acoustic_candidates.return_value = (None, 0.0)



    dummy_audio = [0.0] * 1000



    with patch("vocab_zero.interfaces.api.extract_whisper_embedding", return_value=[0.1] * 384):

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

    # Pre-populate dictionary so feedback updates the existing entry

    # (otherwise source_term becomes an auto-generated acoustic hash)

    existing = LexiconEntry(

        source_term="200_500",

        target_term="hello",

        confidence=0.5,

        context_examples=[],

        embeddings=[],

    )

    engine.dictionary._entries = {"200_500": existing}

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

    assert response.json() == {

        "ok": True,

        "data": {"status": "learned", "source_term": "200_500"},

    }



    # Verify the template was generated and passed to dictionary entry

    args, _ = engine.persist_learned_entry.call_args

    entry = args[0]

    assert entry.source_term == "200_500"

    assert entry.target_term == "hello"

    assert entry.embeddings

    assert len(entry.embeddings) > 0

    assert len(entry.embeddings[0]) == 384

    # Default AudioConfig produces 36-dim frames (12 static + 12 delta + 12 delta-delta)

    assert all(isinstance(v, float) for v in entry.embeddings[0])





def test_api_autocomplete(client):

    engine = client.app.state.engine

    engine.llm_client = MagicMock()

    engine.llm_client.translate.return_value = LLMResponse(

        translation="four, plastic, three",

        reasoning="autocomplete",

        confidence=0.9,

    )



    response = client.post(

        "/api/autocomplete",

        json={"sentence": "I bought [unknown] bags"},

    )

    assert response.status_code == 200

    res_data = response.json()

    assert res_data["ok"] is True

    assert res_data["data"]["suggestions"] == ["four", "plastic", "three"]









def test_feedback_unknown_sound_auto_generates_acoustic_hash(client):

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

            "source_term": "unknown_sound",

            "target_term": "hello",

            "audio_data": dummy_audio,

            "context": "greet",

        },

    )



    assert response.status_code == 200

    response_data = response.json()

    assert response_data["ok"] is True

    assert response_data["data"]["status"] == "learned"

    assert response_data["data"]["source_term"].startswith("sound_")



    args, _ = engine.persist_learned_entry.call_args

    entry = args[0]

    # When the source_term is unknown and audio is provided, source_term should

    # be auto-generated as an acoustic hash with the "sound_" prefix.

    assert entry.source_term.startswith("sound_")

    assert len(entry.source_term) == len("sound_") + 8

    assert entry.target_term == "hello"


def test_feedback_empty_embedding_fails_loudly(client):
    engine = client.app.state.engine

    with patch(
        "vocab_zero.interfaces.api.extract_whisper_embedding", return_value=[]
    ):
        response = client.post(
            "/api/feedback",
            json={
                "source_term": "unknown_sound",
                "target_term": "hello",
                "audio_data": [0.0] * 1000,
            },
        )

    data = response.json()
    assert data["ok"] is False
    assert data["error"]["code"] == "extraction_failed"
    # Nothing should have been persisted when embedding extraction fails.
    engine.persist_learned_entry.assert_not_called()





def test_periodic_pruning_task_prunes_without_blocking(tmp_path: Path):

    dictionary_path = tmp_path / "lexicon.json"

    dictionary = DictionaryManager(path=dictionary_path)



    templates = [[float(i) * 0.01 + float(j) for j in range(384)] for i in range(8)]

    entry = LexiconEntry(

        source_term="term1",

        target_term="target1",

        confidence=1.0,

        embeddings=templates,

    )

    dictionary.upsert(entry)

    assert len(dictionary.lookup("term1").embeddings) == 8



    async def _run() -> None:

        task = asyncio.create_task(periodic_pruning_task(dictionary, interval=0.01))

        try:

            # Prove the loop is non-blocking: this sleep must complete even while

            # the pruning task is scheduled. Use wait_for to keep it deterministic.

            await asyncio.wait_for(asyncio.sleep(0.05), timeout=1.0)

            # Allow at least one pruning cycle to fire.

            await asyncio.wait_for(asyncio.sleep(0.05), timeout=1.0)

        finally:

            task.cancel()

            try:

                await task

            except asyncio.CancelledError:

                pass



    asyncio.run(_run())



    pruned = dictionary.lookup("term1")

    assert pruned is not None

    assert len(pruned.embeddings) <= 5

    assert dictionary_path.exists()

    saved = json.loads(dictionary_path.read_text(encoding="utf-8"))

    saved_entry = next(e for e in saved["entries"] if e["source_term"] == "term1")

    assert len(saved_entry["embeddings"]) <= 5





def test_get_audio_config(client):

    response = client.get("/api/audio_config")

    assert response.status_code == 200

    data = response.json()

    assert data["ok"] is True

    assert data["data"]["match_distance_threshold"] == pytest.approx(0.30)
    assert data["data"]["min_confidence_gate"] == pytest.approx(0.6)
    assert data["data"]["ambiguity_margin_ratio"] == pytest.approx(0.15)
    assert data["data"]["ambiguity_confidence_floor"] == pytest.approx(0.4)
    assert data["data"]["sample_rate"] == 16000


def test_audio_matching_whisper_embedding_retrieval(monkeypatch, tmp_path):
    dictionary = DictionaryManager(path=tmp_path / "lexicon.json")
    
    # Store a dummy 384-dimensional Whisper embedding wrapped in [whisper_vector]
    dummy_vector = [0.1] * 384
    entry = LexiconEntry(
        source_term="word",
        target_term="word",
        confidence=1.0,
        embeddings=[dummy_vector],
    )
    dictionary.upsert(entry)

    # Mock extract_whisper_embedding to return the same vector
    monkeypatch.setattr(
        "vocab_zero.interfaces.api.extract_whisper_embedding",
        lambda *_args, **_kwargs: dummy_vector,
    )

    candidates = perform_audio_matching_candidates(
        [0.0],
        dictionary,
        AudioConfig(),
    )

    assert len(candidates) == 1
    assert candidates[0][0].target_term == "word"
    # Cosine distance should be 0.0 (exact match)
    assert candidates[0][1] == pytest.approx(0.0, abs=1e-5)






def test_acoustic_match_accepted_regardless_of_confidence(client):

    """The API should accept any match returned by rerank_acoustic_candidates."""

    engine = client.app.state.engine

    entry = LexiconEntry(

        source_term="test_term",

        target_term="test_target",

        confidence=1.0,

        embeddings=[[0.1] * 384],

    )

    engine.rerank_acoustic_candidates.return_value = (entry, 0.3)



    with patch("vocab_zero.interfaces.api.perform_audio_matching_candidates") as mock_match:

        mock_match.return_value = [(entry, 1.0)]

        response = client.post("/api/translate", json={

            "audio_data": [0.1] * 4000,

        })

        data = response.json()

        assert data["ok"] is True

        assert data["data"]["status"] == "translated"

        assert data["data"]["translated_text"] == "test_target"
