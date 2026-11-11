"""Preserved stage-one spectrum formula, without plotting side effects."""

import numpy as np


def amplitude_spectrum(x, sample_rate, window="hann"):
    """One-sided peak-amplitude spectrum; demean only the FFT input.

    sample_rate=1 gives cycles/sample. No padding, filtering or resampling.
    Coherent-gain correction calibrates on-bin isolated harmonics; it does
    not make every off-bin maximum an unbiased harmonic-amplitude estimate.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("Expected a finite, real, one-dimensional signal")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    n = len(x)
    w = (
        0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
        if window == "hann"
        else np.ones(n)
    )
    a = np.abs(np.fft.rfft((x - x.mean()) * w)) / w.sum()
    if n % 2 == 0:
        a[1:-1] *= 2
    else:
        a[1:] *= 2
    return np.fft.rfftfreq(n, 1 / sample_rate), a
