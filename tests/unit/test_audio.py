from __future__ import annotations

import numpy as np

from vocab_zero.core.models import AudioConfig
from vocab_zero.utils.audio import dtw_distance, extract_mfcc


def test_extract_mfcc_empty():
    assert extract_mfcc([]) == []
    assert extract_mfcc(np.array([])) == []


def test_extract_mfcc_dims():
    # Generate 1 second of a 440Hz sine wave at 16kHz sample rate
    sample_rate = 16000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)

    mfccs = extract_mfcc(signal, sample_rate=sample_rate)

    assert isinstance(mfccs, list)
    assert len(mfccs) > 0
    assert isinstance(mfccs[0], list)
    assert len(mfccs[0]) == 12  # 13 coefficients minus 1 (energy coefficient discarded)


def test_dtw_distance_identical():
    m1 = [[0.1 * i] * 12 for i in range(10)]
    # DTW distance between identical matrices should be exactly 0
    assert dtw_distance(m1, m1) == 0.0


def test_dtw_distance_different():
    m1 = [[0.1 * i] * 12 for i in range(10)]
    m2 = [[0.2 * i] * 12 for i in range(10)]
    dist = dtw_distance(m1, m2)
    assert dist > 0.0


def test_dtw_distance_empty():
    m1 = [[0.1] * 12]
    assert dtw_distance([], m1) == float("inf")
    assert dtw_distance(m1, []) == float("inf")


def test_audio_config_save_and_load_calibration(tmp_path):
    cal_path = tmp_path / "calibration.json"

    cfg = AudioConfig(dtw_threshold_36=2.3, dtw_threshold_12=1.5, min_confidence_gate=0.65)
    cfg.save_calibration(cal_path)

    assert cal_path.exists()

    base = AudioConfig()
    loaded = AudioConfig.load_calibration(cal_path, base)
    assert loaded.dtw_threshold_36 == 2.3
    assert loaded.dtw_threshold_12 == 1.5
    assert loaded.min_confidence_gate == 0.65
    # Non-calibration fields preserved from base
    assert loaded.sample_rate == base.sample_rate
    assert loaded.use_cmvn == base.use_cmvn


def test_audio_config_load_calibration_missing_file(tmp_path):
    cal_path = tmp_path / "nonexistent.json"
    base = AudioConfig(dtw_threshold_36=1.8)
    loaded = AudioConfig.load_calibration(cal_path, base)
    assert loaded.dtw_threshold_36 == 1.8


def test_audio_config_load_calibration_corrupt_file(tmp_path):
    cal_path = tmp_path / "calibration.json"
    cal_path.write_text("not valid json", encoding="utf-8")
    base = AudioConfig(dtw_threshold_36=1.8)
    loaded = AudioConfig.load_calibration(cal_path, base)
    assert loaded.dtw_threshold_36 == 1.8


def test_audio_config_min_confidence_gate_default():
    cfg = AudioConfig()
    assert cfg.min_confidence_gate == 0.6


def test_audio_config_from_env_thresholds(monkeypatch):
    monkeypatch.setenv("VOCABZERO_DTW_THRESHOLD_36", "2.5")
    monkeypatch.setenv("VOCABZERO_DTW_THRESHOLD_12", "1.7")
    monkeypatch.setenv("VOCABZERO_MIN_CONFIDENCE", "0.75")
    cfg = AudioConfig.from_env()
    assert cfg.dtw_threshold_36 == 2.5
    assert cfg.dtw_threshold_12 == 1.7
    assert cfg.min_confidence_gate == 0.75
