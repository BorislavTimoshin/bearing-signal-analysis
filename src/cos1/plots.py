import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .spectrum import amplitude_spectrum


def plot(x, y, g, row, path):
    tx = np.arange(len(x)) / g["fs_in"]
    ty = np.arange(len(y)) / g["fs_out"]
    T = g["duration"]
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    for ax, (start, end, title) in zip(
        [axs[0, 0], axs[0, 1], axs[2, 0], axs[2, 1]],
        [
            (0, min(0.05, T), "Waveform"),
            (0.01, 0.012, "Sample grid"),
            (0, 0.003, "Beginning"),
            (T - 0.003, T, "End"),
        ],
    ):
        for t, z, label in [(tx, x, "Input"), (ty, y, "Output")]:
            keep = (t >= start) & (t < end)
            ax.plot(t[keep], z[keep], ".-", ms=2, lw=0.7, label=label)
        ax.set(
            title=title,
            xlabel="Time, s",
            ylabel=row["signal_unit"] or "Native units (unverified)",
        )
        ax.legend()
    for ax, window, title in [
        (axs[1, 0], "rectangular", "DFT amplitude (demeaned, rectangular)"),
        (axs[1, 1], "hann", "Common band (Hann, coherent gain corrected)"),
    ]:
        for z, fs, label in [(x, g["fs_in"], "Input"), (y, g["fs_out"], "Output")]:
            f, a = amplitude_spectrum(z, fs, window)
            ax.plot(f, a, lw=0.6, label=label)
        ax.axvline(g["fs_out"] / 2, color="black", ls="--", lw=0.6)
        if window == "hann":
            ax.set_xlim(0, min(g["fs_in"], g["fs_out"]) / 2)
        ax.set(
            title=title,
            xlabel="Frequency, Hz",
            ylabel=row["signal_unit"] or "Native amplitude",
        )
        ax.legend()
    fig.suptitle(
        f"{row['dataset']} | {row['source_filename']} | {row['channel']} | {row['fault_type']}\n{g['fs_in']} → {g['fs_out']} Hz; T={T:g} s"
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)
