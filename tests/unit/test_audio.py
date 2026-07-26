from __future__ import annotations

import numpy as np
from dotenv import dotenv_values

from vocab_zero.core.models import AudioConfig
from vocab_zero.utils.audio import acoustic_hash, extract_whisper_embedding, k_medoids


def test_acoustic_hash_deterministic():
    features = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    h1 = acoustic_hash(features)
    h2 = acoustic_hash(features)
    assert h1 == h2
    assert h1.startswith("sound_")
    assert len(h1) == len("sound_") + 8


def test_acoustic_hash_differs_for_different_features():
    f1 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    f2 = [[0.1, 0.2, 0.9], [0.4, 0.5, 0.6]]
    assert acoustic_hash(f1) != acoustic_hash(f2)


def test_acoustic_hash_empty():
    assert acoustic_hash([]) == "sound_00000000"
    assert acoustic_hash([[]]) == "sound_00000000"


def test_k_medoids_selects_k_representative_vectors():
    v1 = [0.1] * 384
    v2 = [0.9] * 384
    v3 = [0.11] * 384
    v4 = [0.89] * 384

    vectors = [v1, v2, v3, v4]
    medoids = k_medoids(vectors, 2)
    assert len(medoids) == 2
    # Each returned medoid must be one of the input vectors.
    assert all(m in vectors for m in medoids)


def test_k_medoids_returns_all_when_fewer_than_k():
    vectors = [[0.1] * 384, [0.2] * 384]
    assert k_medoids(vectors, 4) == vectors


def test_audio_config_defaults():
    cfg = AudioConfig()
    assert cfg.sample_rate == 16000
    assert cfg.match_distance_threshold == 0.30
    assert cfg.min_confidence_gate == 0.6
    assert cfg.ambiguity_margin_ratio == 0.15
    assert cfg.ambiguity_confidence_floor == 0.4


def test_audio_config_from_env_thresholds(monkeypatch):
    monkeypatch.setenv("VOCABZERO_MATCH_DISTANCE_THRESHOLD", "0.42")
    monkeypatch.setenv("VOCABZERO_MIN_CONFIDENCE", "0.75")
    monkeypatch.setenv("VOCABZERO_AMBIGUITY_MARGIN_RATIO", "0.2")
    monkeypatch.setenv("VOCABZERO_AMBIGUITY_CONFIDENCE_FLOOR", "0.5")
    cfg = AudioConfig.from_env()
    assert cfg.match_distance_threshold == 0.42
    assert cfg.min_confidence_gate == 0.75
    assert cfg.ambiguity_margin_ratio == 0.2
    assert cfg.ambiguity_confidence_floor == 0.5


def test_audio_config_reads_dotenv_file(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "VOCABZERO_MATCH_DISTANCE_THRESHOLD=0.27\nVOCABZERO_MIN_CONFIDENCE=0.8\n",
        encoding="utf-8",
    )
    for key, value in dotenv_values(env_path).items():
        if value is not None:
            monkeypatch.setenv(key, value)
    cfg = AudioConfig.from_env()

    assert cfg.match_distance_threshold == 0.27
    assert cfg.min_confidence_gate == 0.8


def test_extract_whisper_embedding():
    # Empty signal returns an empty embedding.
    assert extract_whisper_embedding([]) == []

    # A short signal returns a 384-dim mean-pooled encoder embedding.
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000, endpoint=False))
    embedding = extract_whisper_embedding(signal)

    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(val, float) for val in embedding)
