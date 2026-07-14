from __future__ import annotations

import math
import pytest
from fastapi.testclient import TestClient
from vocab_zero.interfaces.api import app


@pytest.fixture
def temp_env(tmp_path, monkeypatch):
    """Fixture to set up a temporary environment for integration testing."""
    temp_dict = tmp_path / "lexicon_temp.json"
    if temp_dict.exists():
        temp_dict.unlink()
    
    # Configure API to use our temporary dictionary and no vector DB
    monkeypatch.setenv("VOCABZERO_DICTIONARY", str(temp_dict))
    monkeypatch.setenv("VOCABZERO_VECTOR_DB", "")
    
    # Prevent initializing a real local LLM (avoid GPU/bitsandbytes requirements)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    # Use 12-dim fallback for integration tests (threshold=15.0) to preserve
    # the acoustic-distinction semantics the test was designed for. The 36-dim
    # pipeline is covered by unit tests in tests/unit/test_audio.py.
    monkeypatch.setenv("VOCABZERO_DELTAS", "0")
    monkeypatch.setenv("VOCABZERO_CMVN", "0")
    monkeypatch.setenv("VOCABZERO_VTLN", "0")
    monkeypatch.setenv("VOCABZERO_LIFTERING", "0")
    return temp_dict


def generate_sine_wave(frequency: float = 440.0, duration: float = 0.5, sample_rate: int = 16000) -> list[float]:
    """Generates a simple synthetic PCM audio signal (sine wave) for matching."""
    num_samples = int(duration * sample_rate)
    return [0.5 * math.sin(2 * math.pi * frequency * i / sample_rate) for i in range(num_samples)]


def generate_white_noise(duration: float = 0.6, sample_rate: int = 16000, seed: int = 42) -> list[float]:
    """Generates white noise audio that is acoustically distinct from sine waves."""
    import random
    rng = random.Random(seed)
    num_samples = int(duration * sample_rate)
    return [rng.uniform(-0.5, 0.5) for _ in range(num_samples)]


def test_integration_flow(temp_env):
    """E2E Integration test for dictionary CRUD, translation, feedback, and audio matching."""
    # Build synthetic audio signals
    audio_signal_1 = generate_sine_wave(frequency=440.0, duration=0.6)  # ~9600 samples
    audio_signal_2 = [0.0] * int(0.6 * 16000)  # silence is acoustically distinct from sine
    
    with TestClient(app) as client:
        # 1. Get index file (bundled frontend HTML)
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        
        # 2. Get lexicon (initially empty)
        response = client.get("/api/lexicon")
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert response.json()["data"] == []
        
        # 3. Translate unknown word text-fallback
        response = client.post("/api/translate", json={"source_term": "unknown_word"})
        assert response.status_code == 200
        res_json = response.json()
        assert res_json["ok"] is True
        assert res_json["data"]["status"] == "requires_feedback"
        assert "feedback_request" in res_json["data"]
        
        # 4. Translate unknown audio
        response = client.post(
            "/api/translate",
            json={
                "audio_data": audio_signal_1,
                "context": "greeting"
            }
        )
        assert response.status_code == 200
        res_json = response.json()
        assert res_json["ok"] is True
        assert res_json["data"]["status"] == "requires_feedback"
        
        # 5. Teach the engine a word mapping via feedback
        response = client.post(
            "/api/feedback",
            json={
                "source_term": "rohingya_sine_440",
                "target_term": "hello_sine",
                "audio_data": audio_signal_1,
                "context": "greeting"
            }
        )
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["ok"] is True
        assert response_data["data"]["status"] == "learned"
        assert response_data["data"]["source_term"].startswith("sound_")
        
        # 6. Verify entry is in lexicon (source_term is now an auto-generated acoustic hash)
        response = client.get("/api/lexicon")
        assert response.status_code == 200
        lexicon = response.json()["data"]
        assert len(lexicon) == 1
        assert lexicon[0]["source_term"].startswith("sound_")
        assert lexicon[0]["target_term"] == "hello_sine"
        acoustic_source = lexicon[0]["source_term"]
        
        # Check dictionary manager directly to verify templates are saved
        entry = client.app.state.engine.dictionary.lookup(acoustic_source)
        assert entry is not None
        assert len(entry.mfcc_templates) == 1
        
        # 7. Test successful translation using the trained audio signal
        response = client.post(
            "/api/translate",
            json={
                "audio_data": audio_signal_1,
                "context": "greeting"
            }
        )
        assert response.status_code == 200
        res_json = response.json()
        assert res_json["ok"] is True
        assert res_json["data"]["status"] == "translated"
        assert res_json["data"]["translated_text"] == "hello_sine"
        assert res_json["data"]["source"] == "acoustic_matching"
        assert res_json["data"]["confidence"] > 0.8
        
        # 8. Test translation of a DIFFERENT audio signal (should not match or require feedback)
        response = client.post(
            "/api/translate",
            json={
                "audio_data": audio_signal_2,
                "context": "greeting"
            }
        )
        assert response.status_code == 200
        res_json = response.json()
        assert res_json["ok"] is True
        assert res_json["data"]["status"] == "requires_feedback"
        
        # 9. Test autocomplete suggestions (should return empty since no LLM provider)
        response = client.post(
            "/api/autocomplete",
            json={
                "sentence": "rohingya_sine_440",
                "context": "greeting"
            }
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert response.json()["data"] == {"suggestions": []}
        
        # 10. Delete the lexicon entry
        response = client.delete(f"/api/lexicon/{acoustic_source}")
        assert response.status_code == 200
        assert response.json() == {"ok": True, "data": {"deleted": True}}
        
        # Verify it is deleted
        response = client.get("/api/lexicon")
        assert response.json()["data"] == []

