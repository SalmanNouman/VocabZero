from __future__ import annotations

import numpy as np

from vocab_zero.core.models import AudioConfig
from vocab_zero.utils.audio import acoustic_hash, dtw_distance, extract_mfcc, subsequence_dtw


def test_extract_mfcc_empty():
    assert extract_mfcc([]) == []
    assert extract_mfcc(np.array([])) == []


def _sine_signal(sample_rate: int = 16000, duration: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def test_extract_mfcc_dims():
    sample_rate = 16000
    signal = _sine_signal(sample_rate=sample_rate)

    mfccs = extract_mfcc(signal, sample_rate=sample_rate)

    assert isinstance(mfccs, list)
    assert len(mfccs) > 0
    assert isinstance(mfccs[0], list)
    # Default AudioConfig has use_deltas=True -> 36 dims (12 static + 12 delta + 12 delta-delta)
    assert len(mfccs[0]) == 36


def test_deltas_produce_36_dims():
    signal = _sine_signal()
    cfg = AudioConfig(use_deltas=True, use_cmvn=False, use_vtln=False, use_liftering=False)
    mfccs = extract_mfcc(signal, audio_config=cfg)
    assert len(mfccs) > 0
    assert len(mfccs[0]) == 36


def test_no_deltas_fallback_12_dims():
    signal = _sine_signal()
    cfg = AudioConfig(use_deltas=False, use_cmvn=False, use_vtln=False, use_liftering=False)
    mfccs = extract_mfcc(signal, audio_config=cfg)
    assert len(mfccs) > 0
    assert len(mfccs[0]) == 12


def test_cmvn_normalizes_mean_to_zero():
    signal = _sine_signal()
    cfg = AudioConfig(use_cmvn=True, use_vtln=False, use_liftering=False, use_deltas=False)
    mfccs = extract_mfcc(signal, audio_config=cfg)
    arr = np.array(mfccs)
    means = arr.mean(axis=0)
    assert np.allclose(means, 0.0, atol=1e-5)


def test_vtln_warping_changes_filterbank():
    signal = _sine_signal(freq=300.0)
    cfg_off = AudioConfig(use_cmvn=False, use_vtln=False, use_liftering=False, use_deltas=False)
    cfg_on = AudioConfig(use_cmvn=False, use_vtln=True, use_liftering=False, use_deltas=False)
    # Force distinct alpha by monkeypatching the estimator.
    import vocab_zero.utils.audio as audio_mod

    orig = audio_mod._estimate_vtln_alpha
    audio_mod._estimate_vtln_alpha = lambda sig, sr: 0.8
    try:
        off = extract_mfcc(signal, audio_config=cfg_off)
        on = extract_mfcc(signal, audio_config=cfg_on)
    finally:
        audio_mod._estimate_vtln_alpha = orig
    assert off != on


def test_liftering_suppresses_low_quefrency():
    signal = _sine_signal(freq=200.0)
    cfg_off = AudioConfig(use_cmvn=False, use_vtln=False, use_liftering=False, use_deltas=False)
    cfg_on = AudioConfig(use_cmvn=False, use_vtln=False, use_liftering=True, lifter_coef=22, use_deltas=False)
    off = extract_mfcc(signal, audio_config=cfg_off)
    on = extract_mfcc(signal, audio_config=cfg_on)
    # C1 (index 0 after C0 drop) magnitude should differ under liftering
    assert abs(on[0][0]) != abs(off[0][0])


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


def test_dtw_distance_identical():
    m1 = [[0.1 * i] * 12 for i in range(10)]
    assert dtw_distance(m1, m1) == 0.0


def test_dtw_distance_different():
    m1 = [[0.1 * i] * 12 for i in range(10)]
    m2 = [[0.2 * i] * 12 for i in range(10)]
    dist = dtw_distance(m1, m2)
    assert dist > 0.0


def test_dtw_distance_banded_identical():
    m1 = [[0.1 * i] * 12 for i in range(10)]
    assert dtw_distance(m1, m1, band_radius=0) == 0.0


def test_dtw_distance_banded_is_not_lower_than_unconstrained():
    m1 = [[0.0] * 12, [10.0] * 12, [0.0] * 12]
    m2 = [[0.0] * 12, [0.0] * 12, [10.0] * 12]
    unconstrained = dtw_distance(m1, m2)
    banded = dtw_distance(m1, m2, band_radius=0)
    assert banded >= unconstrained


def test_dtw_distance_banded_radius_clamps_to_length_difference():
    short = [[0.1] * 12 for _ in range(2)]
    long = [[0.1] * 12 for _ in range(10)]
    assert dtw_distance(short, long, band_radius=0) < float("inf")


def test_dtw_distance_empty():
    m1 = [[0.1] * 12]
    assert dtw_distance([], m1) == float("inf")
    assert dtw_distance(m1, []) == float("inf")


def test_subsequence_dtw():
    template = [[0.1 * i] * 12 for i in range(1, 6)]
    stream = [[0.9] * 12 for _ in range(3)] + template + [[0.9] * 12 for _ in range(4)]

    detections = subsequence_dtw(template, stream, threshold=15.0)
    assert len(detections) > 0
    dist, start, end = detections[0]
    assert dist == 0.0
    assert start == 3
    assert end == 7


def test_subsequence_dtw_empty():
    assert subsequence_dtw([], [], threshold=15.0) == []
    assert subsequence_dtw([[0.1] * 12], [], threshold=15.0) == []
    assert subsequence_dtw([], [[0.1] * 12], threshold=15.0) == []


def test_k_medoids():
    from vocab_zero.utils.audio import k_medoids
    t1 = [[0.1] * 12 for _ in range(5)]
    t2 = [[0.2] * 12 for _ in range(5)]
    t3 = [[0.11] * 12 for _ in range(5)]

    templates = [t1, t2, t3]
    medoids = k_medoids(templates, 2)
    assert len(medoids) == 2
    assert len(k_medoids(templates, 4)) == 3


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
    monkeypatch.setenv("VOCABZERO_AMBIGUITY_MARGIN_RATIO", "0.2")
    monkeypatch.setenv("VOCABZERO_AMBIGUITY_CONFIDENCE_FLOOR", "0.5")
    monkeypatch.setenv("VOCABZERO_DTW_BAND_RATIO", "0.3")
    monkeypatch.setenv("VOCABZERO_MAX_LENGTH_RATIO", "3.0")
    monkeypatch.setenv("VOCABZERO_TEMPLATE_AGG_K", "4")
    cfg = AudioConfig.from_env()
    assert cfg.dtw_threshold_36 == 2.5
    assert cfg.dtw_threshold_12 == 1.7
    assert cfg.min_confidence_gate == 0.75
    assert cfg.ambiguity_margin_ratio == 0.2
    assert cfg.ambiguity_confidence_floor == 0.5
    assert cfg.dtw_band_ratio == 0.3
    assert cfg.max_length_ratio == 3.0
    assert cfg.template_agg_k == 4
