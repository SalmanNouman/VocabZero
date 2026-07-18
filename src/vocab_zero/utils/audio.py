"""Audio feature utilities for the VocabZero acoustic matching pipeline.

Provides Whisper-tiny encoder embedding extraction, deterministic acoustic
hashing, and k-medoids clustering used to prune redundant embeddings.
"""

from __future__ import annotations

import hashlib

import numpy as np


def acoustic_hash(features: list[list[float]]) -> str:
    """Deterministic acoustic hash of an embedding matrix.

    Rounds to 6 decimals to avoid cross-platform float drift, then takes the
    first 8 hex chars of a SHA-1 digest. Returns ``"sound_<8 hex chars>"``.
    """
    if not features or not features[0]:
        return "sound_00000000"

    arr = np.array(features, dtype=np.float64)
    rounded = np.round(arr, decimals=6)
    digest = hashlib.sha1(rounded.tobytes()).hexdigest()
    return f"sound_{digest[:8]}"


def k_medoids(vectors: list[list[float]], k: int) -> list[list[float]]:
    """Cluster embedding vectors with a simplified k-medoids algorithm.

    Args:
        vectors: List of embedding vectors (each a list of floats).
        k: Number of representative medoids to select.

    Returns:
        List of ``k`` selected representative vectors.
    """
    n = len(vectors)
    if n <= k:
        return vectors

    arr = np.array(vectors, dtype=np.float64)

    # Pairwise Euclidean distance matrix.
    diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))

    # Seed with the most central point, then greedily add the farthest ones.
    medoids = [int(np.argmin(dist_matrix.sum(axis=1)))]
    while len(medoids) < k:
        min_dists = np.min(dist_matrix[:, medoids], axis=1)
        medoids.append(int(np.argmax(min_dists)))

    best_medoids = list(medoids)

    def compute_cost(meds: list[int]) -> float:
        return float(np.sum(np.min(dist_matrix[:, meds], axis=1)))

    best_cost = compute_cost(best_medoids)

    # PAM-like refinement.
    for _ in range(50):
        changed = False
        for m_idx in range(k):
            for o in range(n):
                if o in best_medoids:
                    continue
                test_medoids = list(best_medoids)
                test_medoids[m_idx] = o
                test_cost = compute_cost(test_medoids)
                if test_cost < best_cost:
                    best_cost = test_cost
                    best_medoids = test_medoids
                    changed = True
        if not changed:
            break

    return [vectors[i] for i in best_medoids]


_whisper_model = None
_whisper_processor = None


def extract_whisper_embedding(signal: list[float] | np.ndarray) -> list[float]:
    """Extract a 384-dimensional Whisper-tiny encoder embedding from raw audio.

    The encoder always emits 1500 frames (30s padded to zeros), so pooling is
    restricted to the frames covering the actual signal to avoid the silence
    padding dominating the mean-pooled vector.

    Args:
        signal: 1D NumPy array or list of raw audio samples (float).

    Returns:
        A list of 384 floats representing the mean-pooled encoder output, or an
        empty list when the signal is empty.
    """
    global _whisper_model, _whisper_processor

    if len(signal) == 0:
        return []

    import logging

    import torch
    import torch._dynamo
    from transformers import WhisperModel, WhisperProcessor

    if _whisper_model is None or _whisper_processor is None:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        try:
            _whisper_processor = WhisperProcessor.from_pretrained(
                "openai/whisper-tiny", local_files_only=True
            )
            _whisper_model = WhisperModel.from_pretrained(
                "openai/whisper-tiny", local_files_only=True
            )
        except OSError:
            _whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            _whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
        _whisper_model.eval()

    sig = (
        np.array(signal, dtype=np.float32)
        if not isinstance(signal, np.ndarray)
        else signal.astype(np.float32)
    )

    inputs = _whisper_processor(sig, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    # Whisper's encoder downsamples 16kHz audio to a 50 fps frame rate.
    active_len = max(1, min(1500, int(np.ceil((len(sig) / 16000.0) * 50.0))))

    def _forward() -> list[float]:
        with torch.no_grad():
            encoder_outputs = _whisper_model.encoder(input_features)
            last_hidden_state = encoder_outputs.last_hidden_state  # (1, 1500, 384)
            mean_pooled = torch.mean(last_hidden_state[:, :active_len, :], dim=1)
            return mean_pooled[0].tolist()

    return torch._dynamo.disable(_forward)()
