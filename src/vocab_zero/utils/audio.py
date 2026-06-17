from __future__ import annotations

import numpy as np


def extract_mfcc(
    signal: np.ndarray | list[float],
    sample_rate: int = 16000,
    win_len: float = 0.025,
    win_step: float = 0.010,
    num_cepstrum: int = 13,
    num_filters: int = 26,
    nfft: int = 512,
    preemph: float = 0.97,
) -> list[list[float]]:
    """Extract Mel-Frequency Cepstral Coefficients (MFCC) from a raw audio signal.

    Args:
        signal: 1D NumPy array or list of raw audio samples (float).
        sample_rate: Audio sample rate in Hz.
        win_len: Analysis window length in seconds.
        win_step: Hop size between windows in seconds.
        num_cepstrum: Number of cepstral coefficients to return.
        num_filters: Number of Mel filters to apply.
        nfft: FFT size.
        preemph: Pre-emphasis coefficient.

    Returns:
        List of lists of floats representing the (num_frames, num_cepstrum) MFCC matrix.
    """
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

    # 5. Mel Filterbank Generation
    low_mel = 0.0
    high_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700.0)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595.0) - 1)

    bins = np.floor((nfft + 1) * hz_points / sample_rate).astype(np.int32)

    fbank = np.zeros((num_filters, int(nfft / 2 + 1)))
    for m in range(1, num_filters + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        # Left slope
        if f_m > f_m_minus:
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        # Right slope
        if f_m_plus > f_m:
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

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
    # Discard 0-th coefficient (overall energy/volume) for loudness invariance,
    # keeping the remaining coefficients representing vocal tract timbre.
    return cepstral_coefficients[:, 1:].tolist()


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
