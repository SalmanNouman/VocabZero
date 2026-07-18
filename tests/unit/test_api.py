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



MFCC_NUM_COEFFICIENTS = 12

MFCC_36_DIM = 36





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

    dummy_mfcc = [[0.1] * MFCC_36_DIM for _ in range(5)]

    entry = LexiconEntry(

        source_term="200_500",

        target_term="hello",

        confidence=1.0,

        mfcc_templates=[dummy_mfcc],

    )

    engine.dictionary._entries = {"200_500": entry}

    engine.rerank_acoustic_candidates.return_value = (entry, 0.9)



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

    dummy_mfcc = [[1.0] * MFCC_36_DIM for _ in range(5)]

    entry = LexiconEntry(

        source_term="200_500",

        target_term="hello",

        confidence=1.0,

        mfcc_templates=[dummy_mfcc],

    )

    engine.dictionary._entries = {"200_500": entry}

    engine.rerank_acoustic_candidates.return_value = (None, 0.0)



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

    # Pre-populate dictionary so feedback updates the existing entry

    # (otherwise source_term becomes an auto-generated acoustic hash)

    existing = LexiconEntry(

        source_term="200_500",

        target_term="hello",

        confidence=0.5,

        context_examples=[],

        mfcc_templates=[],

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

    assert entry.mfcc_templates

    assert len(entry.mfcc_templates) > 0

    assert len(entry.mfcc_templates[0]) > 0

    # Default AudioConfig produces 36-dim frames (12 static + 12 delta + 12 delta-delta)

    assert len(entry.mfcc_templates[0][0]) == MFCC_36_DIM





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





def test_feedback_12_dim_fallback_matches(client):

    engine = client.app.state.engine

    # Configure engine for 12-dim fallback

    engine.audio_config = AudioConfig(use_deltas=False)

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

        },

    )



    assert response.status_code == 200

    args, _ = engine.persist_learned_entry.call_args

    entry = args[0]

    assert entry.source_term.startswith("sound_")

    # 12-dim fallback produces 12-dim templates

    assert len(entry.mfcc_templates[0][0]) == MFCC_NUM_COEFFICIENTS





def test_periodic_pruning_task_prunes_without_blocking(tmp_path: Path):

    dictionary_path = tmp_path / "lexicon.json"

    dictionary = DictionaryManager(path=dictionary_path)



    templates = [[[float(i + j) * 0.01 for j in range(12)] for _ in range(5)] for i in range(8)]

    entry = LexiconEntry(

        source_term="term1",

        target_term="target1",

        confidence=1.0,

        mfcc_templates=templates,

    )

    dictionary.upsert(entry)

    assert len(dictionary.lookup("term1").mfcc_templates) == 8



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

    assert len(pruned.mfcc_templates) <= 5

    assert dictionary_path.exists()

    saved = json.loads(dictionary_path.read_text(encoding="utf-8"))

    saved_entry = next(e for e in saved["entries"] if e["source_term"] == "term1")

    assert len(saved_entry["mfcc_templates"]) <= 5





def test_get_audio_config(client):

    response = client.get("/api/audio_config")

    assert response.status_code == 200

    data = response.json()

    assert data["ok"] is True

    assert "dtw_threshold_36" in data["data"]

    assert "dtw_threshold_12" in data["data"]

    assert "dtw_threshold" in data["data"]

    assert "min_confidence_gate" in data["data"]
    assert data["data"]["ambiguity_margin_ratio"] == pytest.approx(0.15)
    assert data["data"]["ambiguity_confidence_floor"] == pytest.approx(0.4)
    assert data["data"]["dtw_band_ratio"] == pytest.approx(0.2)
    assert data["data"]["max_length_ratio"] == pytest.approx(2.5)
    assert data["data"]["template_agg_k"] == 3

    assert data["data"]["dtw_threshold_36"] == pytest.approx(1.8)


def test_audio_matching_whisper_embedding_retrieval(monkeypatch, tmp_path):
    dictionary = DictionaryManager(path=tmp_path / "lexicon.json")
    
    # Store a dummy 384-dimensional Whisper embedding wrapped in [whisper_vector]
    dummy_vector = [0.1] * 384
    entry = LexiconEntry(
        source_term="word",
        target_term="word",
        confidence=1.0,
        mfcc_templates=[[dummy_vector]],
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
        AudioConfig(use_deltas=False),
    )

    assert len(candidates) == 1
    assert candidates[0][0].target_term == "word"
    # Cosine distance should be 0.0 (exact match)
    assert candidates[0][1] == pytest.approx(0.0, abs=1e-5)






def test_calibrate_sample_and_clear(client):

    # Generate a short sine wave as test audio

    import numpy as np

    t = np.linspace(0, 0.5, 8000, endpoint=False)

    audio = np.sin(2 * np.pi * 440 * t).tolist()



    client.app.state.calibration_samples = {}



    with patch("vocab_zero.interfaces.api.extract_mfcc") as mock_mfcc:

        mock_mfcc.return_value = [[0.1] * 12 for _ in range(10)]



        response = client.post("/api/calibrate/sample", json={

            "label": "phrase_1",

            "audio_data": audio,

        })

        assert response.status_code == 200

        data = response.json()

        assert data["ok"] is True

        assert data["data"]["label"] == "phrase_1"

        assert data["data"]["sample_count"] == 1



    # Clear

    response = client.delete("/api/calibrate/samples")

    assert response.status_code == 200

    assert response.json()["ok"] is True

    assert client.app.state.calibration_samples == {}





def test_calibrate_sample_empty_label(client):

    response = client.post("/api/calibrate/sample", json={

        "label": "  ",

        "audio_data": [0.1] * 3000,

    })

    data = response.json()

    assert data["ok"] is False

    assert data["error"]["code"] == "invalid_input"





def test_calibrate_sample_too_short(client):

    response = client.post("/api/calibrate/sample", json={

        "label": "test",

        "audio_data": [0.1] * 100,

    })

    data = response.json()

    assert data["ok"] is False

    assert data["error"]["code"] == "audio_too_short"





def test_calibrate_compute_insufficient_data(client):

    client.app.state.calibration_samples = {}

    response = client.post("/api/calibrate/compute")

    data = response.json()

    assert data["ok"] is False

    assert data["error"]["code"] == "insufficient_data"





def test_calibrate_compute_success(client):

    # Pre-populate with 2 labels, 2 templates each

    client.app.state.calibration_samples = {

        "phrase_1": [

            [[0.1] * 12 for _ in range(10)],

            [[0.12] * 12 for _ in range(10)],

        ],

        "phrase_2": [

            [[0.9] * 12 for _ in range(10)],

            [[0.88] * 12 for _ in range(10)],

        ],

    }



    with patch("vocab_zero.interfaces.api.dtw_distance") as mock_dtw:

        call_count = [0]



        def side_effect(a, b):

            call_count[0] += 1

            a_val = a[0][0]

            b_val = b[0][0]

            return abs(a_val - b_val) * 5.0



        mock_dtw.side_effect = side_effect



        response = client.post("/api/calibrate/compute")

        data = response.json()

        assert data["ok"] is True

        assert "intra_class" in data["data"]

        assert "inter_class" in data["data"]

        assert "suggested_threshold" in data["data"]

        assert "separation_ratio" in data["data"]





def test_calibrate_apply(client):

    engine = client.app.state.engine



    response = client.post("/api/calibrate/apply", json={

        "dtw_threshold_36": 2.5,

        "persist": False,

    })

    data = response.json()

    assert data["ok"] is True

    assert data["data"]["dtw_threshold_36"] == 2.5

    assert engine.audio_config.dtw_threshold_36 == 2.5





def test_calibrate_apply_invalid_value(client):

    response = client.post("/api/calibrate/apply", json={

        "dtw_threshold_36": -1.0,

    })

    data = response.json()

    assert data["ok"] is False

    assert data["error"]["code"] == "invalid_value"





def test_calibrate_apply_no_changes(client):

    response = client.post("/api/calibrate/apply", json={

        "persist": False,

    })

    data = response.json()

    assert data["ok"] is False

    assert data["error"]["code"] == "no_changes"





def test_acoustic_match_accepted_regardless_of_confidence(client):

    """The API should accept any match returned by rerank_acoustic_candidates."""

    engine = client.app.state.engine

    entry = LexiconEntry(

        source_term="test_term",

        target_term="test_target",

        confidence=1.0,

        mfcc_templates=[[[0.1] * 12 for _ in range(5)]],

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
