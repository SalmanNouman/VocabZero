from __future__ import annotations

import numpy as np

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
