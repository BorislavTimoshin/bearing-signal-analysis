import numpy as np


def statistics(x):
    c = x - x.mean()
    s = np.sqrt(np.mean(c * c))
    rms = np.sqrt(np.mean(x * x))
    return dict(
        mean=float(x.mean()),
        rms_raw=float(rms),
        std_centered=float(s),
        max_abs=float(np.max(abs(x))),
        crest_raw=float(np.max(abs(x)) / rms) if rms else None,
        crest_centered=float(np.max(abs(c)) / s) if s else None,
        excess=float(np.mean((c / s) ** 4) - 3) if s else None,
    )


def check(x, y, g):
    n = len(x)
    m = len(y)
    k = min(n, m) // 2
    a = np.fft.rfft(x) / n
    b = np.fft.rfft(y) / m
    scale = max(float(np.max(abs(x))), 1e-12)
    common_error = float(np.max(abs(a[:k] - b[:k]))) / scale
    mean_error = float(abs(x.mean() - y.mean())) / scale
    p = abs(a) ** 2
    weights = np.full(len(p), 2.0)
    weights[0] = 1
    if n % 2 == 0:
        weights[-1] = 1
    f = np.fft.rfftfreq(n, 1 / g["fs_in"])
    removed = float(np.sum((p * weights)[f > g["fs_out"] / 2]) / np.sum(p * weights))
    high = np.fft.rfftfreq(m, 1 / g["fs_out"]) > g["fs_in"] / 2
    high_error = float(np.max(abs(b[high]))) / scale if high.any() else 0.0
    ok = bool(
        len(y) == g["M"]
        and np.isfinite(y).all()
        and mean_error < 1e-10
        and common_error < 1e-10
        and high_error < 1e-10
    )
    result = dict(
        passed=ok,
        mean_relative_error=mean_error,
        common_complex_error=common_error,
        new_high_frequency_error=high_error,
        input_energy_above_output_nyquist_fraction=removed,
        input_endpoint_jump=float(x[0] - x[-1]),
        output_endpoint_jump=float(y[0] - y[-1]),
        duration_error_s=len(y) / g["fs_out"] - n / g["fs_in"],
        last_time_s=(m - 1) / g["fs_out"],
    )
    for prefix, z in [("before", x), ("after", y)]:
        result.update({prefix + "_" + key: val for key, val in statistics(z).items()})
    return result
