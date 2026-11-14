"""Affine normalization with explicit, reusable parameters (no resampling)."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

METHODS = ("none", "individual_z", "max_abs", "rms", "group_z", "reference_rms")


def signal(x):
    """Validate a finite, real, nonempty 1-D signal without changing it."""
    if np.iscomplexobj(x):
        raise ValueError("Complex input is not supported")
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError("Expected a finite, nonempty real vector")
    return a


@dataclass(frozen=True)
class Parameters:
    """y = (x - offset) / scale; IDs document fitting, not training status."""

    method: str
    offset: float
    scale: float
    fit_ids: tuple[str, ...] = ()
    weighting: str = "not_applicable"
    scope: str = "descriptive_only"

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError("Unknown method")
        if (
            not np.isfinite(self.offset)
            or not np.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError(
                "Finite offset and strictly positive finite scale required"
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        fields = dict(obj)
        fields["fit_ids"] = tuple(fields.get("fit_ids", ()))
        return cls(**fields)


def fit_individual(x, method, record_id=""):
    a = signal(x)
    ids = (record_id,) if record_id else ()
    if method == "none":
        return Parameters(method, 0.0, 1.0)
    if method == "individual_z":
        return Parameters(method, float(a.mean()), float(a.std(ddof=0)), ids)
    if method == "max_abs":
        return Parameters(method, 0.0, float(np.max(np.abs(a))), ids)
    if method == "rms":
        return Parameters(method, 0.0, float(np.sqrt(np.mean(a * a))), ids)
    raise ValueError("Use fit_group_z / fit_reference for common parameters")


def fit_group_z(records):
    """Equal record weights, not equal sample weights. Includes between means."""
    if not records:
        raise ValueError("Empty fitting group")
    arrays = [signal(x) for x in records.values()]
    means = np.array([a.mean() for a in arrays])
    mean = float(means.mean())
    variance = float(np.mean([np.mean((a - mean) ** 2) for a in arrays]))
    return Parameters(
        "group_z",
        mean,
        float(np.sqrt(variance)),
        tuple(records),
        "equal_record_mixture",
    )


def fit_reference(healthy_records):
    """Median of record RMS; comparability/healthy status enforced by caller."""
    if not healthy_records:
        raise ValueError("No justified healthy reference")
    rms = [float(np.sqrt(np.mean(signal(x) ** 2))) for x in healthy_records.values()]
    return Parameters(
        "reference_rms",
        0.0,
        float(np.median(rms)),
        tuple(healthy_records),
        "median_of_record_RMS",
    )


def apply(x, parameters):
    """Return new data; never fit on new input and never modify the input."""
    y = (signal(x) - parameters.offset) / parameters.scale
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite normalization output")
    return y


def save_parameters(path, parameters):
    Path(path).write_text(
        json.dumps(parameters.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
    )


def load_parameters(path):
    return Parameters.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def mean_square_spectrum(x, fs, center=True):
    """Rectangular one-sided power PER BIN, not PSD nor squared peak amplitude.

    Sum equals mean square (variance if center=True), by Parseval.
    Divide by df=fs/N for a periodogram density in unit**2/Hz.
    """
    a = signal(x)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Positive finite fs required")
    if center:
        a = a - a.mean()
    power = np.abs(np.fft.rfft(a) / len(a)) ** 2
    power[1 : -1 if len(a) % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(len(a), 1 / fs), power
