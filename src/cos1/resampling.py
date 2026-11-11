import numpy as np
from scipy.signal import resample
from .time_grid import plan


def transform(x, fs, target=51200):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite 1D samples")
    grid = plan(len(x), fs, target)
    return resample(x, grid["M"], axis=0, domain="time", window=None), grid
