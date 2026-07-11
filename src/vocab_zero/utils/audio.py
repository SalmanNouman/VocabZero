from __future__ import annotations

import hashlib

import numpy as np

from vocab_zero.core.models import AudioConfig

_VTLN_REFERENCE_F3 = 2500.0
_VTLN_ALPHA_MIN = 0.8
_VTLN_ALPHA_MAX = 1.2
_DELTA_WINDOW = 2


def _estimate_vtln_alpha(
    preemphasized: np.ndarray,
    sample_rate: int,
) -> float:
    """Estimate VTLN warping factor from speaker F3 via Praat formant analysis.

    Returns alpha = reference_F3 / speaker_F3 clamped to [0.8, 1.2].
    Falls back to 1.0 (no warp) if formant estimation fails.
    """
    try:
        import parselmouth  # local import: heavy optional dep
    except Exception:
        return 1.0

    try:
        snd = parselmouth.Sound(preemphasized.astype(np.float64), sampling_frequency=sample_rate)
        formant = snd.to_formant_burg(max_number_of_formants=3)
        duration = float(snd.duration)
        if duration <= 0:
            return 1.0

        n_points = 10
        f3_values: list[float] = []
        for i in range(n_points):
            t = (i + 0.5) * duration / n_points
            value = formant.get_value_at_time(3, t)
            if value is not None and not np.isnan(value) and value > 0:
                f3_values.append(float(value))

        if not f3_values:
            return 1.0

        speaker_f3 = float(np.mean(f3_values))
        if speaker_f3 <= 0:
            return 1.0

        alpha = _VTLN_REFERENCE_F3 / speaker_f3
        return float(max(_VTLN_ALPHA_MIN, min(_VTLN_ALPHA_MAX, alpha)))
    except Exception:
        return 1.0


def _build_mel_filterbank(
    num_filters: int,
    nfft: int,
    sample_rate: int,
    alpha: float,
) -> np.ndarray:
    """Build a (possibly VTLN-warped) Mel filterbank matrix.

    When alpha != 1.0, filterbank center frequencies are warped by 1/alpha so
    that a speaker with higher formants (alpha < 1) is matched to the reference.
    """
    low_mel = 0.0
    high_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700.0)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595.0) - 1)

    if alpha != 1.0:
        hz_points = hz_points / alpha

    bins = np.floor((nfft + 1) * hz_points / sample_rate).astype(np.int32)
    max_bin = int(nfft / 2)
    bins = np.clip(bins, 0, max_bin)

    fbank = np.zeros((num_filters, int(nfft / 2 + 1)))
    for m in range(1, num_filters + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        if f_m > f_m_minus:
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        if f_m_plus > f_m:
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

    return fbank


def _apply_liftering(cepstral: np.ndarray, lifter_coef: int) -> np.ndarray:
    """Apply cepstral liftering to suppress pitch harmonics in low cepstral coefficients."""
    n = cepstral.shape[1]
    idx = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 + (lifter_coef / 2.0) * np.sin(np.pi * idx / lifter_coef)
    return cepstral * weights[np.newaxis, :]


def _cmvn(cepstral: np.ndarray) -> np.ndarray:
    """Per-coefficient cepstral mean and variance normalization across the utterance."""
    mean = cepstral.mean(axis=0, keepdims=True)
    std = cepstral.std(axis=0, keepdims=True)
    eps = np.finfo(np.float64).eps
    return (cepstral - mean) / (std + eps)


def _compute_deltas(features: np.ndarray, n: int = _DELTA_WINDOW) -> np.ndarray:
    """Compute delta features via the regression formula with window size N."""
    num_frames, dim = features.shape
    denom = 2.0 * sum(k * k for k in range(1, n + 1))
    deltas = np.zeros_like(features, dtype=np.float64)

    for t in range(num_frames):
        acc = np.zeros(dim, dtype=np.float64)
        for k in range(1, n + 1):
            tp = min(t + k, num_frames - 1)
            tm = max(t - k, 0)
            acc += k * (features[tp] - features[tm])
        deltas[t] = acc / denom

    return deltas


def extract_mfcc(
    signal: np.ndarray | list[float],
    sample_rate: int = 16000,
    win_len: float = 0.025,
    win_step: float = 0.010,
    num_cepstrum: int = 13,
    num_filters: int = 26,
    nfft: int = 512,
    preemph: float = 0.97,
    audio_config: AudioConfig | None = None,
) -> list[list[float]]:
    """Extract speaker-agnostic MFCC features from a raw audio signal.

    Applies a four-layer normalization pipeline (CMVN, VTLN, liftering, deltas)
    controlled by ``AudioConfig``. When ``audio_config`` is None a default
    ``AudioConfig()`` is used (36-dim output with all layers enabled).

    Args:
        signal: 1D NumPy array or list of raw audio samples (float).
        sample_rate: Audio sample rate in Hz.
        win_len: Analysis window length in seconds.
        win_step: Hop size between windows in seconds.
        num_cepstrum: Number of cepstral coefficients before C0 drop.
        num_filters: Number of Mel filters to apply.
        nfft: FFT size.
        preemph: Pre-emphasis coefficient.
        audio_config: Optional ``AudioConfig`` overriding normalization flags,
            thresholds, and feature dimensionality.

    Returns:
        List of lists of floats with shape (num_frames, 12) when deltas are
        disabled, or (num_frames, 36) when deltas are enabled.
    """
    cfg = audio_config or AudioConfig()
    sample_rate = cfg.sample_rate if audio_config is not None else sample_rate
    num_cepstrum = cfg.num_cepstrum if audio_config is not None else num_cepstrum
    num_filters = cfg.num_filters if audio_config is not None else num_filters
    nfft = cfg.nfft if audio_config is not None else nfft
    preemph = cfg.preemph if audio_config is not None else preemph

    sig = np.array(signal, dtype=np.float32) if not isinstance(signal, np.ndarray) else signal.astype(np.float32)
    if len(sig) == 0:
        return []

    # 1. Pre-emphasis filter
    sig = np.append(sig[0], sig[1:] - preemph * sig[:-1])

    # 2. Framing
    frame_len = int(round(win_len * sample_rate))
    frame_step = int(round(win_step * sample_rate))
    signal_len = len(sig)

    if signal_len <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((signal_len - frame_len) / frame_step))

    pad_signal_len = int((num_frames - 1) * frame_step + frame_len)
    pad_signal = np.append(sig, np.zeros(pad_signal_len - signal_len))

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)
    ).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # 3. Windowing (Hamming window)
    frames *= np.hamming(frame_len)

    # 4. Fast Fourier Transform & Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = (1.0 / nfft) * (mag_frames**2)

    # 5. VTLN warping factor (Layer 2) applied to Mel filterbank
    alpha = 1.0
    if cfg.use_vtln:
        alpha = _estimate_vtln_alpha(sig, sample_rate)

    fbank = _build_mel_filterbank(num_filters, nfft, sample_rate, alpha)

    # 6. Apply filterbank & compute log-Mel energies
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_mel_energies = np.log(filter_banks)

    # 7. Discrete Cosine Transform (DCT-II)
    dct_matrix = np.zeros((num_cepstrum, num_filters))
    for i in range(num_cepstrum):
        for j in range(num_filters):
            dct_matrix[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * num_filters))

    cepstral_coefficients = np.dot(log_mel_energies, dct_matrix.T)

    # 8. Discard 0-th coefficient (overall energy/volume) for loudness invariance
    cepstral_coefficients = cepstral_coefficients[:, 1:]

    # 9. Liftering (Layer 3) — suppress pitch harmonics in low cepstral coefficients
    if cfg.use_liftering:
        cepstral_coefficients = _apply_liftering(cepstral_coefficients, cfg.lifter_coef)

    # 10. CMVN (Layer 1) — normalize per-coefficient mean/variance across utterance
    if cfg.use_cmvn:
        cepstral_coefficients = _cmvn(cepstral_coefficients)

    # 11. Deltas (Layer 4) — append delta and delta-delta features
    if cfg.use_deltas:
        delta = _compute_deltas(cepstral_coefficients)
        delta_delta = _compute_deltas(delta)
        cepstral_coefficients = np.concatenate(
            [cepstral_coefficients, delta, delta_delta], axis=1
        )

    return cepstral_coefficients.tolist()


def acoustic_hash(features: list[list[float]]) -> str:
    """Deterministic acoustic hash of a normalized feature matrix.

    Rounds to 6 decimals to avoid cross-platform float drift, then takes the
    first 8 hex chars of a SHA-1 digest. Returns ``"sound_<8 hex chars>"``.
    """
    if not features or not features[0]:
        return "sound_00000000"

    arr = np.array(features, dtype=np.float64)
    rounded = np.round(arr, decimals=6)
    digest = hashlib.sha1(rounded.tobytes()).hexdigest()
    return f"sound_{digest[:8]}"


def dtw_distance(mfcc1: list[list[float]], mfcc2: list[list[float]]) -> float:
    """Compute the duration-normalized Dynamic Time Warping (DTW) distance.

    Uses dynamic programming to align two MFCC sequence matrices.

    Args:
        mfcc1: Matrix (M, 13) of cepstral coefficients.
        mfcc2: Matrix (N, 13) of cepstral coefficients.

    Returns:
        The alignment distance normalized by the sum of sequence lengths.
    """
    m, n = len(mfcc1), len(mfcc2)
    if m == 0 or n == 0:
        return float("inf")

    s1 = np.array(mfcc1)
    s2 = np.array(mfcc2)

    cost = np.zeros((m, n))

    def frame_dist(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.linalg.norm(x - y))

    cost[0, 0] = frame_dist(s1[0], s2[0])

    for i in range(1, m):
        cost[i, 0] = cost[i - 1, 0] + frame_dist(s1[i], s2[0])

    for j in range(1, n):
        cost[0, j] = cost[0, j - 1] + frame_dist(s1[0], s2[j])

    for i in range(1, m):
        for j in range(1, n):
            cost[i, j] = frame_dist(s1[i], s2[j]) + min(
                cost[i - 1, j],  # insertion
                cost[i, j - 1],  # deletion
                cost[i - 1, j - 1],  # match
            )

    return float(cost[m - 1, n - 1] / (m + n))


def subsequence_dtw(
    template: list[list[float]],
    stream: list[list[float]],
    threshold: float,
) -> list[tuple[float, int, int]]:
    """Perform Subsequence Dynamic Time Warping (sDTW).

    Searches for a short template (length N) within a continuous stream (length M).

    Args:
        template: Matrix (N, D) of cepstral coefficients.
        stream: Matrix (M, D) of cepstral coefficients.
        threshold: Normalized DTW distance threshold for detection. Must be
            supplied by the caller from ``AudioConfig.dtw_threshold``.

    Returns:
        List of tuples: (normalized_distance, start_frame, end_frame) for detection events.
    """
    n, m = len(template), len(stream)
    if n == 0 or m == 0:
        return []

    t_arr = np.array(template)
    s_arr = np.array(stream)

    D = np.zeros((n, m))
    start_tracker = np.zeros((n, m), dtype=np.int32)

    def frame_dist(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.linalg.norm(x - y))

    # Initialize first row (i=0): alignment can start at any j
    for j in range(m):
        D[0, j] = frame_dist(t_arr[0], s_arr[j])
        start_tracker[0, j] = j

    # DP updates
    for i in range(1, n):
        # j = 0
        D[i, 0] = D[i - 1, 0] + frame_dist(t_arr[i], s_arr[0])
        start_tracker[i, 0] = start_tracker[i - 1, 0]

        for j in range(1, m):
            dist = frame_dist(t_arr[i], s_arr[j])

            prev_costs = [
                D[i - 1, j],      # insertion
                D[i, j - 1],      # deletion
                D[i - 1, j - 1]   # match
            ]
            min_idx = int(np.argmin(prev_costs))

            D[i, j] = dist + prev_costs[min_idx]

            if min_idx == 0:
                start_tracker[i, j] = start_tracker[i - 1, j]
            elif min_idx == 1:
                start_tracker[i, j] = start_tracker[i, j - 1]
            else:
                start_tracker[i, j] = start_tracker[i - 1, j - 1]

    # Find local minima in the last row D[n-1, :]
    detections = []
    last_row = D[n - 1, :]

    for j in range(m):
        dist = float(last_row[j] / n)  # Normalize by template length
        if dist < threshold:
            # Check if it's a local minimum in a window around j
            is_local_min = True
            window = 5
            start_w = max(0, j - window)
            end_w = min(m, j + window + 1)
            for k in range(start_w, end_w):
                if last_row[k] < last_row[j]:
                    is_local_min = False
                    break
            if is_local_min:
                start_idx = int(start_tracker[n - 1, j])
                detections.append((dist, start_idx, j))

    return detections


def k_medoids(
    templates: list[list[list[float]]],
    k: int,
) -> list[list[list[float]]]:
    """Cluster templates using a simplified k-medoids algorithm based on DTW distance.

    Args:
        templates: List of templates, each a list of list of float (MFCC matrix).
        k: Number of medoids to select.

    Returns:
        List of k selected representative templates.
    """
    n = len(templates)
    if n <= k:
        return templates

    # Compute distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = dtw_distance(templates[i], templates[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Initialize medoids: pick the first one with the minimum sum of distances,
    # then iteratively pick points that are farthest from existing medoids.
    medoids = [int(np.argmin(dist_matrix.sum(axis=1)))]
    while len(medoids) < k:
        min_dists = np.min(dist_matrix[:, medoids], axis=1)
        next_medoid = int(np.argmax(min_dists))
        medoids.append(next_medoid)

    # Optimization loop (PAM-like)
    best_medoids = list(medoids)

    def compute_cost(meds: list[int]) -> float:
        return float(np.sum(np.min(dist_matrix[:, meds], axis=1)))

    best_cost = compute_cost(best_medoids)

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

    return [templates[i] for i in best_medoids]
